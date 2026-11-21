#!/usr/bin/env python3
"""C2, the reward classifier of arm G4-7 (protocol-g4.md §14, G4-7.6): its data, the texts of its check, and the check
that it can serve as the reward (c2cal7). C2 itself is fine-tuned by naturalness.py classifier, unchanged, on dmbert v2
(--max-length 1024), and scores the policy's samples as a judge (g4judge.py serve --role c2).

    python g4c2.py data7 --bt DIR --c1-data classifier.parquet --evalsets evalsets.parquet --blind blind100.jsonl \
        --flan-eval eval-rows.parquet --checkpoint V2 --out DIR
    python naturalness.py classifier --dumb-code DUMB --checkpoint V2 --data DIR/classifier.parquet --max-length 1024 \
        --out MODEL
    python g4c2.py cal7-texts --evalsets evalsets.parquet --g40 'eval/G4-0/pairs/pairs-w*.parquet' --sampling DIR \
        --c2-data DIR/classifier.parquet --checkpoint V2 [--calib NAME=texts.parquet ...] --out texts.parquet
    python naturalness.py classify --dumb-code DUMB --model MODEL --texts texts.parquet --out predictions.parquet
    python g4c2.py cal7 --results MODEL/results.json --texts texts.parquet --predictions predictions.parquet \
        --c2-data DIR/classifier.parquet --sampling DIR --g40-scores 'eval/G4-0/scores/metricx-w*.parquet' \
        [--c1 NAME=predictions.parquet ...] --out DIR

data7: Dutch originals (nl, label 1) against Gemma 4's round trip of the same passage (nl_back, label 0) from the train
split of the five <source>-nl-en-gemma4 sets at G4-3's revisions; a passage qualifies when both texts are there and
whole in C2 (at most 1,022 tokens of its tokenizer without the special tokens), hard_fails is empty and the round trip
ended without a loop or English left; out: the documents and the texts of C1's 2,000 passages, a 13-gram of the
evaluation sets, the FLAN evaluation rows or the blind passages, and a text already drawn. Per source 1,000 documents in
a seeded order with one seeded passage each; the first 100 of a second seeded order are the test split. classifier.parquet
has C1's columns (pid, id, source, rank, split, label, text) plus tokens; docs.txt the documents, for g4rl.py pool7.

cal7: the four conditions of G4-7.6, all must hold (C2CAL7_OK, exit 0; else C2CAL7_FAIL, exit 1): accuracy on the test
split >= 0.75; "original" on Europarl-direct's Dutch originals >= 0.75; G4-0's greedy share "original" on WMT24++ (good)
+ FLAN in [0.02, 0.90]; on G4-0's 8 samples per source of the sampling probe, per set, the median over the sources of
the within-group sd (ddof 0) of clip(logit, -6, 6) >= 0.25 and >= the median sd of 2 max(0, QE - QE0). The rest is
reported and decides nothing.
"""

import argparse
import collections
import glob
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g4arm  # noqa: E402

SEED7 = "en-nl-g4arm/0/G4-7/c2"
PER_SOURCE = 1000
TEST_PER_SOURCE = 100
C2_MAX_TOKENS = 1022  # Yeb 21 Nov
DEFAULT_TOKENIZER = "yhavinga/dutch-llama-tokenizer"  # finetune.load_pretrained_params' default, as naturalness.py
BT_EXTRA = ("nl_back", "back_ended", "back_decode_loop", "back_english_left")
# the last field is the passage: k, or p<a>-<b> for the long passages of G4-3's revisions (paragraphs a..b)
DOC_RE = re.compile(r"^(?P<doc>[a-z]+:.+):(?:d|t\d+):(?:\d+|p\d+-\d+)$")
# the bounds of c2cal7 (spec §5.3.3, Yeb 21 Nov: agreed)
ACCURACY_MIN = 0.75
EUROPARL_MIN = 0.75
SHARE_RANGE = (0.02, 0.90)
SPREAD_MIN = 0.25
LOGIT_CLIP = 6.0
QE_WEIGHT = 2.0
SPEARMAN_HIGH = 0.5
SAMPLES = tuple(f"t10s{k}" for k in range(1, 9))
SETS = ("wmt24pp", "flan")
N_EUROPARL = g4arm.FOUR_SETS["europarl"]


def doc_key(pid):
    """<source>:<document> of a passage id <source>:<document>:<d|t<n>>:<k> (a long passage: <k> = p<a>-<b>)"""
    m = DOC_RE.match(pid or "")
    if not m:
        raise SystemExit(f"{pid!r}: not <source>:<document>:<d|t<n>>:<k or p<a>-<b>>")
    return m.group("doc")


def tokenizer_of(checkpoint):
    """C2's tokenizer as naturalness.py resolves it: config.json's tokenizer_name, else the DUMB code's default"""
    from transformers import AutoTokenizer

    config = json.loads((Path(checkpoint) / "config.json").read_text())
    return AutoTokenizer.from_pretrained(config.get("tokenizer_name") or DEFAULT_TOKENIZER)


def token_counter(tokenizer):
    return lambda texts: [len(i) for i in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


# ---------------------------------------------------------------------------------------------------- data7

def build7(rows, c1_rows, blocked, count, sources=tuple(g4arm.BT_REPOS3)):
    """(data rows, documents, report) from the BT train rows (read_bt with BT_EXTRA). Lazy per document: a document is
    drawn when one of its passages, in a seeded order, passes every filter; the tokens are counted only there."""
    c1_docs = {doc_key(r["pid"]) for r in c1_rows}
    c1_texts = {(r["text"] or "").strip() for r in c1_rows if int(r["label"]) == 1}
    by_doc = collections.defaultdict(list)
    for r in rows:
        by_doc[(r["source"], doc_key(r["pid"]))].append(r)
    seen = set()
    out, docs, report = [], [], {"seed": SEED7, "per_source": PER_SOURCE, "test_per_source": TEST_PER_SOURCE,
                                 "max_tokens": C2_MAX_TOKENS, "c1_documents": len(c1_docs), "funnel": {}}
    for source in sources:
        funnel = collections.Counter()
        order = sorted(d for s, d in by_doc if s == source)
        funnel["documents"] = len(order)
        random.Random(f"{SEED7}/{source}").shuffle(order)
        drawn = []
        for doc in order:
            if len(drawn) == PER_SOURCE:
                break
            funnel["documents_examined"] += 1
            if doc in c1_docs:
                funnel["doc_of_c1"] += 1
                continue
            passages = sorted(by_doc[(source, doc)], key=lambda r: r["pid"])
            random.Random(f"{SEED7}/{source}/{doc}").shuffle(passages)
            for r in passages:
                funnel["passages_examined"] += 1
                nl, back = (r["nl"] or "").strip(), (r["nl_back"] or "").strip()
                reason = None
                if not nl or not back:
                    reason = "empty"
                elif r["hard_fails"]:
                    reason = "hard_fails"
                elif not r["back_ended"]:
                    reason = "back_not_ended"
                elif r["back_decode_loop"]:
                    reason = "back_decode_loop"
                elif r["back_english_left"]:
                    reason = "back_english_left"
                elif nl in c1_texts:
                    reason = "text_of_c1"
                elif nl in seen or back in seen:
                    reason = "text_drawn"
                elif g4arm.shares_gram(r, blocked):
                    reason = "eval_13gram"
                if reason is None:
                    n_nl, n_back = count([nl, back])
                    if max(n_nl, n_back) > C2_MAX_TOKENS:
                        reason = "over_1022_tokens"
                if reason:
                    funnel[reason] += 1
                    continue
                drawn.append({"pid": r["pid"], "doc": doc, "nl": nl, "back": back, "tokens": (n_nl, n_back)})
                seen |= {nl, back}
                break
            else:
                funnel["documents_without_passage"] += 1
        if len(drawn) < PER_SOURCE:
            raise SystemExit(f"{source}: {len(drawn)} documents qualify, {PER_SOURCE} needed ({dict(funnel)})")
        test_order = sorted(d["doc"] for d in drawn)
        random.Random(f"{SEED7}/{source}/test").shuffle(test_order)
        test = set(test_order[:TEST_PER_SOURCE])
        for rank, d in enumerate(drawn):
            split = "test" if d["doc"] in test else "train"
            for label, text, n in ((1, d["nl"], d["tokens"][0]), (0, d["back"], d["tokens"][1])):
                out.append({"pid": d["pid"], "id": d["pid"], "source": source, "rank": rank, "split": split,
                            "label": label, "text": text, "tokens": n})
        docs += [d["doc"] for d in drawn]
        funnel["drawn"] = len(drawn)
        report["funnel"][source] = dict(funnel)
    counts = collections.Counter((r["split"], r["label"]) for r in out)
    report["counts"] = {f"{s}/{lab}": n for (s, lab), n in sorted(counts.items())}
    report["tokens_max"] = {f"{s}/{lab}": max(r["tokens"] for r in out if (r["split"], r["label"]) == (s, lab))
                            for s, lab in counts}
    return out, sorted(docs), report


def data7(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    revisions = g4arm.revisions3(args.bt)
    rows = g4arm.read_bt(args.bt, revisions, split="train", extra=BT_EXTRA)
    c1 = pq.read_table(args.c1_data, columns=["pid", "label", "text"]).to_pylist()
    if len({r["pid"] for r in c1}) != 2000:
        raise SystemExit(f"{args.c1_data}: {len({r['pid'] for r in c1})} passages, C1 has 2,000")
    blocked = g4arm.checked_blocked(args.evalsets, args.blind)
    import sft_build

    blocked |= sft_build.flan_eval_ngrams(pq.read_table(args.flan_eval, columns=["inputs", "targets"]).to_pylist())
    out, docs, report = build7(rows, c1, blocked, token_counter(tokenizer_of(args.checkpoint)))
    held = {doc_key(r["pid"]) for r in g4arm.read_bt(args.bt_validation, g4arm.revisions3(args.bt_validation),
                                                     split="validation")} if args.bt_validation else set()
    if held & set(docs):
        raise SystemExit(f"{len(held & set(docs))} documents of C2 are in the validation split (btval, btlong, own)")
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), d / "classifier.parquet")
    (d / "docs.txt").write_text("".join(f"{x}\n" for x in docs))
    report.update({"revisions": revisions, "checkpoint": args.checkpoint,
                   "validation_documents_checked": len(held),
                   "inputs": {k: g4arm.sha256_file(getattr(args, k)) for k in ("c1_data", "evalsets", "blind",
                                                                               "flan_eval")}})
    (d / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"C2DATA7 {json.dumps({'counts': report['counts'], 'tokens_max': report['tokens_max'], 'documents': len(docs)})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- cal7-texts

def read_glob(pattern, columns):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files for {pattern}")
    rows = {}
    for p in paths:
        for r in g4arm.read_rows(p, columns):
            if r["id"] in rows:
                raise SystemExit(f"{r['id']} twice in {pattern}")
            rows[r["id"]] = r
    return rows


def b_ids(evalsets):
    """The segments of rule (b): WMT24++ without is_bad_source (960) and FLAN (1,000)"""
    rows = g4arm.read_rows(evalsets, ["id", "set", "meta"])
    out = {"wmt24pp": sorted(r["id"] for r in rows if r["set"] == "wmt24pp" and not g4arm.bad_source(r["meta"])),
           "flan": sorted(r["id"] for r in rows if r["set"] == "flan")}
    if (len(out["wmt24pp"]), len(out["flan"])) != (g4arm.N_WMT_GOOD, g4arm.N_FLAN):
        raise SystemExit(f"{evalsets}: {len(out['wmt24pp'])} WMT24++ (good) and {len(out['flan'])} FLAN segments")
    return out


def sampling_models(directory):
    """{name: pairs pattern} of the sampling probe: <m>/t10s<k> and <m>/best8 for G4-0 and G4-4"""
    out = {}
    for m in ("G4-0", "G4-4"):
        for s in (*SAMPLES, "best8"):
            out[f"{m}/{s}"] = f"{directory}/{m}/{s}/pairs/pairs-w*.parquet"
    return out


def cal7_texts(args):
    """texts.parquet (model, set, id, text, tokens) for naturalness.py classify"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = []
    europarl = [r for r in g4arm.read_rows(args.evalsets, ["id", "set", "reference"]) if r["set"] == "europarl"]
    if len(europarl) != N_EUROPARL:
        raise SystemExit(f"{args.evalsets}: {len(europarl)} Europarl-direct segments, expected {N_EUROPARL}")
    out += [{"model": "human", "set": "europarl", "id": r["id"], "text": r["reference"]} for r in europarl]
    ids = b_ids(args.evalsets)
    wanted = {i: s for s, xs in ids.items() for i in xs}
    g40 = read_glob(args.g40, ["id", "set", "hypothesis"])
    for i, s in sorted(wanted.items()):
        if i not in g40:
            raise SystemExit(f"G4-0: no translation of {i} in {args.g40}")
        out.append({"model": "G4-0", "set": s, "id": i, "text": g40[i]["hypothesis"]})
    for name, pattern in sampling_models(args.sampling).items():
        rows = read_glob(pattern, ["id", "set", "hypothesis"])
        if set(rows) != set(wanted):
            raise SystemExit(f"{name}: {len(rows)} segments, not the {len(wanted)} of rule (b)")
        out += [{"model": name, "set": r["set"], "id": i, "text": r["hypothesis"]} for i, r in sorted(rows.items())]
    for r in g4arm.read_rows(args.c2_data, ["pid", "source", "split", "label", "text"]):
        if r["split"] == "test":
            out.append({"model": "C2-test", "set": r["source"], "id": f"{r['pid']}#{r['label']}", "text": r["text"]})
    for spec in args.calib or []:
        name, _, path = spec.partition("=")
        rows = g4arm.read_rows(path, ["model", "set", "id", "text"])
        out += [{"model": f"calib:{name}:{r['model']}", "set": r["set"], "id": r["id"], "text": r["text"]} for r in rows]
    keys = [(r["model"], r["set"], r["id"]) for r in out]
    if len(set(keys)) != len(keys):
        raise SystemExit("a (model, set, id) twice among the texts")
    for r in out:
        r["text"] = r["text"] if r["text"] is not None else ""
    counts = token_counter(tokenizer_of(args.checkpoint))([r["text"] for r in out])
    for r, n in zip(out, counts):
        r["tokens"] = n
    pq.write_table(pa.Table.from_pylist(out), args.out)
    print(f"CAL7_TEXTS {json.dumps(collections.Counter(r['model'].split(':')[0] if r['model'].startswith('calib:') else r['model'] for r in out))} "
          f"over_{C2_MAX_TOKENS} {sum(n > C2_MAX_TOKENS for n in counts)}", flush=True)


# ---------------------------------------------------------------------------------------------------- cal7

def logit_of(p):
    """clip(logit, -6, 6) from P(original): p 0 or 1 clip to the bound"""
    if p <= 0.0:
        return -LOGIT_CLIP
    if p >= 1.0:
        return LOGIT_CLIP
    return max(-LOGIT_CLIP, min(LOGIT_CLIP, math.log(p) - math.log1p(-p)))


def sd0(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs, ys):
    """Spearman's rho with average ranks for ties (None below 3 pairs or without variance)"""
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else sxy / (sx * sy)


def group_stats(groups, qe0, set_of):
    """Per set: the median within-group sd of the clipped logit and of the QE penalty, and the reported measures.
    groups: {id: [(logit, qe, tokens_out, overflow, empty), ...]} over the 8 samples"""
    out = {}
    for st in SETS:
        sd_logit, sd_pen, spread_gt1, all_clipped, used, left_out = [], [], 0, 0, 0, 0
        for i, samples in groups.items():
            if set_of[i] != st:
                continue
            kept = [x for x in samples if not x[3] and not x[4]]
            left_out += len(samples) - len(kept)
            if len(kept) < 2:
                continue
            used += 1
            logits = [x[0] for x in kept]
            sd_logit.append(sd0(logits))
            sd_pen.append(sd0([QE_WEIGHT * max(0.0, x[1] - qe0[i]) for x in kept]))
            spread_gt1 += max(logits) - min(logits) > 1
            all_clipped += all(abs(v) >= LOGIT_CLIP for v in logits)
        if not used:
            raise SystemExit(f"{st}: no group with two scored samples")
        out[st] = {"sources": used, "samples_left_out_empty_or_overflow": left_out,
                   "median_sd_logit": statistics.median(sd_logit), "median_sd_qe_penalty": statistics.median(sd_pen),
                   "share_logit_range_gt_1": spread_gt1 / used, "share_all_clipped": all_clipped / used}
    return out


def cal7(args):
    results = json.loads(Path(args.results).read_text())
    texts = {(r["model"], r["set"], r["id"]): r for r in g4arm.read_rows(args.texts)}
    preds = {(r["model"], r["set"], r["id"]): r["p_original"] for r in g4arm.read_rows(args.predictions)}
    if set(preds) != set(texts):
        raise SystemExit(f"{args.predictions}: {len(preds)} predictions for {len(texts)} texts")
    tokens = {k: r["tokens"] for k, r in texts.items()}

    def share(keys):
        keys = list(keys)
        return sum(preds[k] >= 0.5 for k in keys) / len(keys)

    europarl = [k for k in preds if k[0] == "human" and k[1] == "europarl"]
    g40 = [k for k in preds if k[0] == "G4-0"]
    if len(europarl) != N_EUROPARL or len(g40) != g4arm.N_WMT_GOOD + g4arm.N_FLAN:
        raise SystemExit(f"{len(europarl)} Europarl texts, {len(g40)} G4-0 texts")
    qe0 = {i: r["qe_source_hypothesis"] for i, r in read_glob(args.g40_scores, ["id", "qe_source_hypothesis"]).items()}
    set_of = {k[2]: k[1] for k in g40}
    probe = {}
    for m in ("G4-0", "G4-4"):
        groups = collections.defaultdict(list)
        for s in SAMPLES:
            pairs = read_glob(f"{args.sampling}/{m}/{s}/pairs/pairs-w*.parquet", ["id", "hypothesis", "tokens_out"])
            scores = read_glob(f"{args.sampling}/{m}/{s}/scores/metricx-w*.parquet", ["id", "qe_source_hypothesis"])
            for i, r in pairs.items():
                key = (f"{m}/{s}", set_of[i], i)
                empty = not r["hypothesis"]
                qe = scores[i]["qe_source_hypothesis"]
                if qe is None and not empty:
                    raise SystemExit(f"{m}/{s} {i}: no QE for a translation")
                if qe0.get(i) is None:
                    raise SystemExit(f"G4-0's greedy {i}: no QE0 in {args.g40_scores}")
                groups[i].append((logit_of(preds[key]), qe, r["tokens_out"], tokens[key] > C2_MAX_TOKENS, empty))
        if any(len(v) != len(SAMPLES) for v in groups.values()) or set(groups) != set(set_of):
            raise SystemExit(f"{m}: not 8 samples for every source of rule (b)")
        probe[m] = group_stats(groups, qe0, set_of)
        # t10s1 against best8: |delta logit| and the within-source Spearman with delta tokens and delta QE
        best = read_glob(f"{args.sampling}/{m}/best8/pairs/pairs-w*.parquet", ["id", "hypothesis", "tokens_out"])
        t1 = read_glob(f"{args.sampling}/{m}/t10s1/pairs/pairs-w*.parquet", ["id", "hypothesis", "tokens_out"])
        q1 = read_glob(f"{args.sampling}/{m}/t10s1/scores/metricx-w*.parquet", ["id", "qe_source_hypothesis"])
        qb = {i: min(x[1] for x in v if x[1] is not None) for i, v in groups.items()}
        for st in SETS:
            dl, dt, dq, absd = [], [], [], []
            for i in sorted(x for x in set_of if set_of[x] == st):
                a, b = logit_of(preds[(f"{m}/t10s1", st, i)]), logit_of(preds[(f"{m}/best8", st, i)])
                absd.append(abs(a - b))
                if t1[i]["hypothesis"] != best[i]["hypothesis"] and q1[i]["qe_source_hypothesis"] is not None:
                    dl.append(a - b)
                    dt.append(t1[i]["tokens_out"] - best[i]["tokens_out"])
                    dq.append(q1[i]["qe_source_hypothesis"] - qb[i])
            probe[m][st].update({"median_abs_dlogit_t10s1_best8": statistics.median(absd),
                                 "pairs_different": len(dl), "spearman_dlogit_dtokens": spearman(dl, dt),
                                 "spearman_dlogit_dqe": spearman(dl, dq)})
    # the test split by length
    data = {(r["source"], f"{r['pid']}#{r['label']}"): r for r in g4arm.read_rows(args.c2_data)}
    bins = collections.defaultdict(list)
    for k, p in preds.items():
        if k[0] != "C2-test":
            continue
        r = data[(k[1], k[2])]
        n = r["tokens"]
        b = "<=128" if n <= 128 else ("129-256" if n <= 256 else ">256")
        bins[b].append((p >= 0.5) == bool(r["label"]))
    by_length = {b: {"n": len(v), "accuracy": sum(v) / len(v)} for b, v in sorted(bins.items())}
    # calibration against C1 (from its existing classify runs)
    calib = {}
    for spec in args.c1 or []:
        name, _, path = spec.partition("=")
        c1 = {(r["model"], r["set"], r["id"]): r["p_original"] for r in g4arm.read_rows(path)}
        pairs = [(preds[(f"calib:{name}:{m}", s, i)], p1) for (m, s, i), p1 in c1.items()
                 if (f"calib:{name}:{m}", s, i) in preds]
        per_model = collections.defaultdict(list)
        for (m, s, i), p1 in c1.items():
            if (f"calib:{name}:{m}", s, i) in preds:
                per_model[m].append((preds[(f"calib:{name}:{m}", s, i)] >= 0.5, p1 >= 0.5))
        calib[name] = {"segments": len(pairs), "spearman_c2_c1": spearman([a for a, _ in pairs], [b for _, b in pairs]),
                       "share_original": {m: {"c2": sum(a for a, _ in v) / len(v), "c1": sum(b for _, b in v) / len(v),
                                              "n": len(v)} for m, v in sorted(per_model.items())}}
    acc = results["test_accuracy"]
    eu = share(europarl)
    g40_share = share(g40)
    conditions = {
        "1_test_accuracy": {"value": acc, "bound": f">= {ACCURACY_MIN}", "ok": acc >= ACCURACY_MIN},
        "2_europarl_original": {"value": eu, "bound": f">= {EUROPARL_MIN}", "ok": eu >= EUROPARL_MIN},
        "3_g40_greedy_share": {"value": g40_share, "bound": f"in [{SHARE_RANGE[0]}, {SHARE_RANGE[1]}]",
                               "ok": SHARE_RANGE[0] <= g40_share <= SHARE_RANGE[1]},
    }
    for st in SETS:
        x = probe["G4-0"][st]
        conditions[f"4_spread_{st}"] = {
            "value": x["median_sd_logit"], "qe_penalty": x["median_sd_qe_penalty"],
            "bound": f">= {SPREAD_MIN} and >= the QE penalty's",
            "ok": x["median_sd_logit"] >= SPREAD_MIN and x["median_sd_logit"] >= x["median_sd_qe_penalty"]}
    ok = all(c["ok"] for c in conditions.values())
    high = [n for n, c in calib.items() if c["spearman_c2_c1"] is not None and c["spearman_c2_c1"] >= SPEARMAN_HIGH]
    report = {"ok": ok, "conditions": conditions, "probe": probe, "test_by_length": by_length,
              "test_accuracy_by_source": results.get("test_accuracy_by_source"), "calibration": calib,
              "high_correlation_with_c1": high,
              "over_1022_tokens": {m: sum(1 for k in texts if k[0] == m and tokens[k] > C2_MAX_TOKENS)
                                   for m in sorted({k[0] for k in texts})},
              "tokens_max": {m: max(tokens[k] for k in texts if k[0] == m) for m in sorted({k[0] for k in texts})}}
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (d / "report.md").write_text(report_md(report))
    failed = [n for n, c in conditions.items() if not c["ok"]]
    if failed:
        print(f"C2CAL7_FAIL {failed} {json.dumps({n: conditions[n]['value'] for n in failed})}", flush=True)
        return 1
    print(f"C2CAL7_OK {json.dumps({n: round(c['value'], 4) for n, c in conditions.items()})}", flush=True)
    return 0


def report_md(r):
    def f(x):
        return "–" if x is None else (f"{x:.4f}".replace(".", ",") if isinstance(x, float) else str(x))

    L = ["# C2 bruikbaar als reward (`c2cal7`, protocol-g4.md G4-7.6)", "",
         f"Uitslag: **{'alle vier gehaald' if r['ok'] else 'niet gehaald: geen RL, aan Yeb'}**.", "",
         "| voorwaarde | waarde | grens | gehaald |", "|---|---:|---|---|"]
    for n, c in r["conditions"].items():
        extra = f" (QE-straf {f(c['qe_penalty'])})" if "qe_penalty" in c else ""
        L.append(f"| {n} | {f(c['value'])}{extra} | {c['bound']} | {'ja' if c['ok'] else 'nee'} |")
    L += ["", "Gerapporteerd, beslist niets:", "", "| model | set | bronnen | mediaan-sd logit | mediaan-sd QE-straf | "
          "max − min > 1 | alle 8 geknipt | mediaan \\|Δ logit\\| t10s1–best8 | Spearman Δlogit~Δtokens | "
          "Spearman Δlogit~ΔQE |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, sets in r["probe"].items():
        for st, x in sets.items():
            L.append(f"| {m} | {st} | {x['sources']} | {f(x['median_sd_logit'])} | {f(x['median_sd_qe_penalty'])} | "
                     f"{f(x['share_logit_range_gt_1'])} | {f(x['share_all_clipped'])} | "
                     f"{f(x['median_abs_dlogit_t10s1_best8'])} | {f(x['spearman_dlogit_dtokens'])} | "
                     f"{f(x['spearman_dlogit_dqe'])} |")
    L += ["", "Nauwkeurigheid op de testsplit per lengte (tokens van C2): " + ", ".join(
        f"{b}: {f(x['accuracy'])} (n {x['n']})" for b, x in r["test_by_length"].items()), ""]
    for n, c in r["calibration"].items():
        L.append(f"Kalibratie `{n}`: Spearman per segment C2~C1 {f(c['spearman_c2_c1'])} over {c['segments']} segmenten.")
    if r["high_correlation_with_c1"]:
        L += ["", "Bij deze hoge correlatie (Spearman ≥ 0,5) duwt wat C2 omhoog duwt waarschijnlijk ook C1 omhoog: een hack "
              "van C2 kan dan ook C1 misleiden; (b) zegt dan minder en de blinde beoordeling ((a2), (c)) weegt zwaarder."]
    return "\n".join(L) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data7")
    d.add_argument("--bt", required=True, help="g4arm.py fetch --arm G4-3 --split train's directory (revisions.json)")
    d.add_argument("--bt-validation", help="fetch --arm G4-3 --split validation's directory: no document of it in C2")
    d.add_argument("--c1-data", required=True, help="C1's classifier.parquet ($S/arms/data/classifier.parquet)")
    d.add_argument("--evalsets", required=True)
    d.add_argument("--blind", required=True, help="blind100.jsonl")
    d.add_argument("--flan-eval", required=True, help="mainset/flan-eval/eval-rows.parquet")
    d.add_argument("--checkpoint", required=True, help="v2 converted (config.json: its tokenizer)")
    d.add_argument("--out", required=True)
    t = sub.add_parser("cal7-texts")
    t.add_argument("--evalsets", required=True)
    t.add_argument("--g40", required=True, help="G4-0's greedy pairs: pattern of pairs-w*.parquet")
    t.add_argument("--sampling", required=True, help="$G/sampling (local copy): <m>/<sample>/pairs/, scores/")
    t.add_argument("--c2-data", required=True)
    t.add_argument("--checkpoint", required=True, help="C2's directory or v2 (config.json: its tokenizer)")
    t.add_argument("--calib", action="append", help="NAME=texts.parquet of an existing classify run of C1")
    t.add_argument("--out", required=True)
    c = sub.add_parser("cal7")
    c.add_argument("--results", required=True, help="C2's results.json")
    c.add_argument("--texts", required=True)
    c.add_argument("--predictions", required=True, help="naturalness.py classify with C2 on --texts")
    c.add_argument("--c2-data", required=True)
    c.add_argument("--sampling", required=True)
    c.add_argument("--g40-scores", required=True, help="G4-0's greedy MetricX: pattern of metricx-w*.parquet")
    c.add_argument("--c1", action="append", help="NAME=predictions.parquet of C1 on the texts of --calib NAME")
    c.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"data7": data7, "cal7-texts": cal7_texts, "cal7": cal7}[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
