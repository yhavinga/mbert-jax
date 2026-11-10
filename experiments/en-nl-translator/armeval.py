#!/usr/bin/env python3
"""The evaluation of the fine-tune's arms beyond the four sets of evalsets.py (protocol.md §4): the extra sets in
evalsets.py's data format, the classifier's data, the texts of the naturalness measures, the blind judge's package and
the report with the decision rule of §5.

    python armeval.py data --kind x --flan-eval eval-rows.parquet --blind blind100.jsonl --evalsets evalsets.parquet \
        --eval-meta B/eval-meta.parquet --eval B/eval.parquet --tokenizer TOK --out DIR
    python armeval.py data --kind instr --flan-eval eval-rows.parquet --evalsets evalsets.parquet --tokenizer TOK --out DIR
    python armeval.py classifier-data --eval-meta B/eval-meta.parquet --pairs run/pairs.parquet long/pairs.parquet --out c.parquet
    python armeval.py texts --evalsets evalsets.parquet --wmt base=P ... --flan base=P ... --out texts.parquet
    python armeval.py blind --blind blind100.jsonl --base DIR --arm B=DIR --arm N=DIR --out DIR
    python armeval.py report ... --out report.md --json report.json

data writes what run_baseline.sh translate reads ($E/data/ of evalsets.py fetch): evalsets.parquet (id, set, source,
reference, tokens, over_512, length_class, meta), segments-w<k>.parquet, classes-w<k>.txt and manifest.json.
- kind x (the base model and both arms): set flan, the 1,000 FLAN evaluation rows of set classifier (their whole
  instruction, `inputs`; §4.2.1); set blind, the 100 blind passages (§4.2.4; the WMT24++ ones keep their reference);
  set indomain, the 4,519 eval passages of the arms' sets (§4.2.2: the synthetic English as source, the Dutch original
  as reference, without context).
- kind instr (arm N with the conventional sentence, §4.1): set wmt24pp (all 998, the reference and meta of
  evalsets.parquet) and set flan (the same 1,000 rows).
classifier-data: §4.2.1's 2,000 passages (eval, without context, by sha256 of the pid), original against back.
texts: the translations the naturalness measures read, per model: WMT24++ (the 960 without is_bad_source) and FLAN.
blind: per arm judge-input.jsonl (blind_id, source, A, B) and key.jsonl (which of A and B is the base model).
report: rule (a) from the per-segment scores, rule (b) from the classifier, the in-domain table per source, the
check of N's mode, PLL and MTLD; rule (c) waits for the blind judge.
"""

import argparse
import collections
import glob
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE.parents[1] / "scripts", HERE):
    sys.path.insert(0, str(_path))

BLIND_SHA256 = "4dfe0675648a56caa872764a3208de7d22f6a6a6ded48130dd2e10567b1e0add"  # protocol §4.2.4
N_FLAN = 1000
N_INDOMAIN = 4519
N_BLIND = 100
N_WMT = 998
N_WMT_GOOD = 960  # WMT24++ without is_bad_source
N_CLASSIFIER, N_CLASSIFIER_TEST = 2000, 200
CLASSIFIER_SEED = "en-nl-sft/classifier"
AB_SEED = "en-nl-sft/blind/ab"
RULE_A = 0.20
RULE_B = 0.10
CLASSIFIER_MIN = 0.75
RULE_A_SETS = ["wmt24pp", "flores", "ntrex"]
SOURCES = ["handelingen", "raad", "dpc", "rechtspraak", "fineweb"]
SOURCE_NAMES = {"handelingen": "Handelingen", "raad": "raad", "dpc": "dpc", "rechtspraak": "rechtspraak",
                "fineweb": "FineWeb-2"}
MTLD_THRESHOLD = 0.72
# The rules compare means with their bounds; a difference that equals the bound up to rounding counts as on the bound
TOLERANCE = 1e-9
WORD = re.compile(r"\w+")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def bad_source(meta):
    return bool(json.loads(meta).get("is_bad_source", False)) if meta else False


# ---------------------------------------------------------------------------------------------------- data

def flan_rows(path):
    import pyarrow.parquet as pq

    rows = [r for r in pq.read_table(path).to_pylist() if r["set"] == "classifier"]
    if len(rows) != N_FLAN:
        raise SystemExit(f"{path}: {len(rows)} rows of set classifier, expected {N_FLAN}")
    return [{"id": f"flan:{r['host']}-{r['position']}", "set": "flan", "source": r["inputs"].strip(), "reference": None,
             "meta": {"task": r["source"]}} for r in rows]


def blind_rows(path, evalsets):
    digest = sha256_file(path)
    if digest != BLIND_SHA256:
        raise SystemExit(f"{path} has sha256 {digest}, the protocol fixed {BLIND_SHA256}")
    out = []
    for d in read_jsonl(path):
        ref = evalsets[d["id"]]["reference"] if d["id"] in evalsets else None
        if d["set"] == "wmt24pp" and ref is None:
            raise SystemExit(f"{d['blind_id']}: WMT24++ item {d['id']} is not in evalsets.parquet")
        out.append({"id": d["blind_id"], "set": "blind", "source": d["source"].strip(), "reference": ref,
                    "meta": {"blind_set": d["set"], "item": d["id"], **(d.get("meta") or {})}})
    if len(out) != N_BLIND:
        raise SystemExit(f"{path}: {len(out)} passages, expected {N_BLIND}")
    return out


def indomain_rows(meta_path, eval_path):
    import pyarrow.parquet as pq

    meta = pq.read_table(meta_path).to_pylist()
    messages = pq.read_table(eval_path, columns=["messages"]).column("messages").to_pylist()
    if len(meta) != len(messages) or len(meta) != N_INDOMAIN:
        raise SystemExit(f"{meta_path}: {len(meta)} rows, {eval_path}: {len(messages)}; expected {N_INDOMAIN} in both")
    out = []
    for m, msg in zip(meta, messages):
        if [x["role"] for x in msg] != ["user", "assistant"] or m["en"].strip() not in msg[0]["content"]:
            raise SystemExit(f"{m['id']}: eval.parquet's row is not the one of eval-meta.parquet")
        if m["source"] not in SOURCES:
            raise SystemExit(f"{m['id']}: unknown source {m['source']}")
        out.append({"id": f"indomain:{m['id']}", "set": "indomain", "source": m["en"].strip(),
                    "reference": msg[1]["content"].strip(),
                    "meta": {"source": m["source"], "kind": m["kind"], "length_class": m["length_class"]}})
    return out


def wmt_rows(evalsets):
    out = [{"id": r["id"], "set": "wmt24pp", "source": r["source"], "reference": r["reference"],
            "meta": json.loads(r["meta"])} for r in evalsets.values() if r["set"] == "wmt24pp"]
    if len(out) != N_WMT:
        raise SystemExit(f"evalsets.parquet has {len(out)} WMT24++ segments, expected {N_WMT}")
    return out


def data(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from evalsets import CLASS_ORDER, MARK_OVER, assign_workers, length_class
    from tokenizers import Tokenizer

    evalsets = {r["id"]: r for r in pq.read_table(args.evalsets).to_pylist()}
    rows = flan_rows(args.flan_eval)
    if args.kind == "x":
        rows += blind_rows(args.blind, evalsets) + indomain_rows(args.eval_meta, args.eval)
    else:
        rows = wmt_rows(evalsets) + rows
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids among the rows")
    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    for r in rows:
        if not r["source"]:
            raise SystemExit(f"{r['id']}: an empty source")
        r["tokens"] = len(tok.encode(r["source"], add_special_tokens=False).ids)
        r["over_512"] = r["tokens"] > MARK_OVER
        r["length_class"] = length_class(r["tokens"])  # stops above translate.py's longest class
        r["meta"] = json.dumps(r["meta"], ensure_ascii=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("id", pa.string()), ("set", pa.string()), ("source", pa.string()), ("reference", pa.string()),
                        ("tokens", pa.int64()), ("over_512", pa.bool_()), ("length_class", pa.string()),
                        ("meta", pa.string())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "evalsets.parquet")
    assignment = assign_workers(rows, args.workers)
    for k, part in assignment.items():
        if not part:
            raise SystemExit(f"worker {k} gets no length class ({len(rows)} rows): run_baseline.sh translate needs one")
        pq.write_table(pa.table({"id": pa.array([r["id"] for r in part], pa.string()),
                                 "text": pa.array([r["source"] for r in part], pa.string()),
                                 "tokens": pa.array([r["tokens"] for r in part], pa.int64())}),
                       out / f"segments-w{k}.parquet")
        classes = [c for c in CLASS_ORDER if any(r["length_class"] == c for r in part)]
        (out / f"classes-w{k}.txt").write_text(" ".join(classes) + "\n")
    inputs = [args.flan_eval, args.evalsets] + ([args.blind, args.eval_meta, args.eval] if args.kind == "x" else [])
    manifest = {
        "kind": args.kind,
        "sets": {name: {"segments": sum(r["set"] == name for r in rows),
                        "with_reference": sum(r["set"] == name and r["reference"] is not None for r in rows),
                        "source_tokens": sum(r["tokens"] for r in rows if r["set"] == name),
                        "max_tokens": max(r["tokens"] for r in rows if r["set"] == name),
                        "over_512": sum(r["over_512"] for r in rows if r["set"] == name),
                        "length_classes": {c: sum(r["length_class"] == c for r in rows if r["set"] == name)
                                           for c in CLASS_ORDER}}
                 for name in dict.fromkeys(r["set"] for r in rows)},
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "tokenizer": str(args.tokenizer),
        "workers": {f"w{k}": {"segments": len(part), "classes": sorted({r["length_class"] for r in part},
                                                                        key=CLASS_ORDER.index)}
                    for k, part in assignment.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for name, s in manifest["sets"].items():
        print(f"SET {name}: {s['segments']} segments ({s['with_reference']} with a reference), {s['source_tokens']:,} "
              f"source tokens, max {s['max_tokens']}, over 512: {s['over_512']}", flush=True)
    for w, s in manifest["workers"].items():
        print(f"WORKER {w}: {s['segments']} segments, classes {s['classes']}", flush=True)


# ---------------------------------------------------------------------------------------------------- classifier

def classifier_order(pid):
    return hashlib.sha256(f"{CLASSIFIER_SEED}/{pid}".encode()).hexdigest()


def classifier_data(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = pq.read_table(args.eval_meta, columns=["id", "kind", "source"]).to_pylist()
    wanted = {r["id"]: r for r in meta if r["kind"] == "bt"}
    pairs = {}
    for path in args.pairs:
        for r in pq.read_table(path, columns=["id", "pid", "kind", "split", "nl", "back"]).to_pylist():
            if r["id"] in wanted and r["kind"] == "passage":
                if r["id"] in pairs:
                    raise SystemExit(f"{r['id']} is in two pairs tables")
                pairs[r["id"]] = r
    missing = sorted(set(wanted) - set(pairs))
    if missing:
        raise SystemExit(f"{len(missing)} eval passages without a pair: {missing[:5]}")
    chosen = sorted(pairs.values(), key=lambda r: classifier_order(r["pid"]))[:N_CLASSIFIER]
    if len(chosen) != N_CLASSIFIER or len({r["pid"] for r in chosen}) != N_CLASSIFIER:
        raise SystemExit(f"{len(chosen)} passages ({len({r['pid'] for r in chosen})} pids), expected {N_CLASSIFIER}")
    rows = []
    for rank, r in enumerate(chosen):
        if r["split"] != "eval":
            raise SystemExit(f"{r['id']}: split {r['split']}")
        nl, back = (r["nl"] or "").strip(), (r["back"] or "").strip()
        if not nl or not back:
            raise SystemExit(f"{r['id']}: an empty original or back-translation")
        split = "train" if rank < N_CLASSIFIER - N_CLASSIFIER_TEST else "test"
        for label, text in ((1, nl), (0, back)):
            rows.append({"pid": r["pid"], "id": r["id"], "source": wanted[r["id"]]["source"], "rank": rank,
                         "split": split, "label": label, "text": text})
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    counts = collections.Counter((r["split"], r["label"]) for r in rows)
    print(f"CLASSIFIER_DATA {json.dumps({f'{s}/{l}': n for (s, l), n in sorted(counts.items())})} "
          f"sources {json.dumps(collections.Counter(r['source'] for r in rows if r['label'] == 1))}", flush=True)


# ---------------------------------------------------------------------------------------------------- texts

def parse_named(values):
    out = {}
    for v in values or []:
        name, _, path = v.partition("=")
        if not name or not path:
            raise SystemExit(f"{v!r}: NAME=PATH")
        out[name] = path
    return out


def read_rows(pattern, columns=None):
    """The rows of the parquet files of a glob (or one path), each id once."""
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no file matches {pattern}")
    out = {}
    for p in paths:
        for r in pq.read_table(p, columns=columns).to_pylist():
            if r["id"] in out:
                raise SystemExit(f"{r['id']} twice in {pattern}")
            out[r["id"]] = r
    return out


def wmt_ids(evalsets_path):
    import pyarrow.parquet as pq

    rows = pq.read_table(evalsets_path, columns=["id", "set", "meta"]).to_pylist()
    ids = [r["id"] for r in rows if r["set"] == "wmt24pp" and not bad_source(r["meta"])]
    if len(ids) != N_WMT_GOOD:
        raise SystemExit(f"{evalsets_path}: {len(ids)} WMT24++ segments without is_bad_source, expected {N_WMT_GOOD}")
    return ids


def texts(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    wmt = wmt_ids(args.evalsets)
    out = []
    for set_name, sources, expected in (("wmt24pp", parse_named(args.wmt), wmt), ("flan", parse_named(args.flan), None)):
        for model, pattern in sources.items():
            rows = read_rows(pattern, ["id", "set", "hypothesis"])
            ids = expected if expected is not None else sorted(i for i, r in rows.items() if r["set"] == "flan")
            if expected is None and len(ids) != N_FLAN:
                raise SystemExit(f"{model}: {len(ids)} FLAN translations in {pattern}, expected {N_FLAN}")
            for i in ids:
                if i not in rows or rows[i]["hypothesis"] is None:
                    raise SystemExit(f"{model}: no translation of {i} in {pattern}")
                out.append({"model": model, "set": set_name, "id": i, "text": rows[i]["hypothesis"]})
    pq.write_table(pa.Table.from_pylist(out), args.out)
    print(f"TEXTS {json.dumps(collections.Counter(r['model'] + '/' + r['set'] for r in out))}", flush=True)


# ---------------------------------------------------------------------------------------------------- blind

def base_is_a(arm, blind_id):
    return random.Random(f"{AB_SEED}/{arm}/{blind_id}").random() < 0.5


def blind(args):
    passages = read_jsonl(args.blind)
    digest = sha256_file(args.blind)
    if digest != BLIND_SHA256:
        raise SystemExit(f"{args.blind} has sha256 {digest}, the protocol fixed {BLIND_SHA256}")
    base = read_rows(args.base, ["id", "set", "source", "hypothesis"])
    for arm, pattern in parse_named(args.arm).items():
        mine = read_rows(pattern, ["id", "set", "source", "hypothesis"])
        judge, key = [], []
        for p in passages:
            b, a = base.get(p["blind_id"]), mine.get(p["blind_id"])
            if b is None or a is None or b["hypothesis"] is None or a["hypothesis"] is None:
                raise SystemExit(f"{arm}: {p['blind_id']} lacks the base model's or the arm's translation")
            if b["source"] != p["source"].strip() or a["source"] != p["source"].strip():
                raise SystemExit(f"{arm}: {p['blind_id']}: the translated source is not the blind passage")
            first = base_is_a(arm, p["blind_id"])
            judge.append({"blind_id": p["blind_id"], "source": p["source"].strip(),
                          "A": b["hypothesis"] if first else a["hypothesis"],
                          "B": a["hypothesis"] if first else b["hypothesis"]})
            key.append({"blind_id": p["blind_id"], "A": "base" if first else arm, "B": arm if first else "base",
                        "seed": f"{AB_SEED}/{arm}/{p['blind_id']}", "set": p["set"], "item": p["id"]})
        d = Path(args.out) / arm
        d.mkdir(parents=True, exist_ok=True)
        write_jsonl(d / "judge-input.jsonl", judge)
        write_jsonl(d / "key.jsonl", key)
        print(f"BLIND {arm}: {len(judge)} passages, base is A in {sum(k['A'] == 'base' for k in key)}", flush=True)


# ---------------------------------------------------------------------------------------------------- measures

def mtld_direction(words, threshold=MTLD_THRESHOLD):
    """McCarthy and Jarvis (2010): words per factor, a factor ending where the running type-token ratio falls to the
    threshold; the remainder counts as the fraction of a factor its ratio has fallen."""
    factors, types, count = 0.0, set(), 0
    for w in words:
        count += 1
        types.add(w)
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count:
        factors += (1 - len(types) / count) / (1 - threshold)
    return len(words) / factors if factors else float("nan")


def mtld(text):
    words = WORD.findall(text.lower())
    return (mtld_direction(words) + mtld_direction(words[::-1])) / 2


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.fmean(values) if values else None


def chrf(hyp, ref):
    import sacrebleu

    return sacrebleu.sentence_chrf(hyp, [ref]).score


def corpus_chrf(hyps, refs):
    import sacrebleu

    return sacrebleu.corpus_chrf(hyps, [refs]).score


def scored(directory):
    """{id: pairs row + metricx_ref, metricx_qe, comet22} of an output of run_arms.sh score (pairs/, scores/)."""
    pairs = read_rows(f"{directory}/pairs/pairs-w*.parquet")
    metricx = read_rows(f"{directory}/scores/metricx-w*.parquet")
    comet = read_rows(f"{directory}/scores/comet-w*.parquet") if glob.glob(f"{directory}/scores/comet-w*.parquet") else {}
    if set(metricx) != set(pairs):
        raise SystemExit(f"{directory}: MetricX scores for {len(metricx)} of {len(pairs)} pairs")
    out = {}
    for i, p in pairs.items():
        m = metricx[i]
        ref = m.get("ref_source_hypothesis_reference")
        if p["reference"] is not None and (ref is None or i not in comet):
            raise SystemExit(f"{directory}: {i} has a reference but no MetricX or COMET score with it")
        if m.get("qe_source_hypothesis") is None:
            raise SystemExit(f"{directory}: {i} has no MetricX-QE score")
        out[i] = {**p, "metricx_ref": ref, "metricx_qe": m["qe_source_hypothesis"],
                  "comet22": comet[i]["comet22"] if i in comet else None}
    return out


def counts(rows):
    return {"n": len(rows), "not_ended": sum(not r["ended"] for r in rows),
            "decode_loop": sum(bool(r["decode_loop"]) for r in rows),
            "english_left": sum(bool(r["english_left"]) for r in rows), "empty": sum(bool(r["empty"]) for r in rows)}


def metrics(rows, with_chrf=False):
    out = {**counts(rows), "metricx_qe": mean(r["metricx_qe"] for r in rows),
           "metricx_ref": mean(r["metricx_ref"] for r in rows), "comet22": mean(r["comet22"] for r in rows)}
    if with_chrf:
        out["chrf"] = mean(chrf(r["hypothesis"], r["reference"]) for r in rows)
        out["chrf_corpus"] = corpus_chrf([r["hypothesis"] for r in rows], [r["reference"] for r in rows]) if rows else None
    return out


def four_sets(scores_path, evalsets):
    """{set: rows} of a scores.parquet of evalsets.py summary; wmt24pp is the 960 without is_bad_source (rule (a)),
    wmt24pp_all the 998."""
    import pyarrow.parquet as pq

    rows = pq.read_table(scores_path).to_pylist()
    if {r["id"] for r in rows} != set(evalsets):
        raise SystemExit(f"{scores_path}: not the segments of evalsets.parquet")
    out = collections.defaultdict(list)
    for r in rows:
        out[f"{r['set']}_all" if r["set"] == "wmt24pp" else r["set"]].append(r)
        if r["set"] == "wmt24pp" and not bad_source(evalsets[r["id"]]["meta"]):
            out["wmt24pp"].append(r)
    return dict(out)


def rule_a(base, arm):
    out = {}
    for s in RULE_A_SETS:
        b, a = mean(r["metricx_qe"] for r in base[s]), mean(r["metricx_qe"] for r in arm[s])
        out[s] = {"n": len(arm[s]), "base": b, "arm": a, "delta": a - b, "limit": b + RULE_A,
                  "holds": a - b <= RULE_A + TOLERANCE}
    return {"sets": out, "holds": all(v["holds"] for v in out.values())}


def classifier_shares(predictions):
    """{model: {set: share, 'pooled': share, 'n': ...}} of the texts the classifier calls original (p >= 0.5)."""
    by = collections.defaultdict(list)
    for r in predictions:
        by[(r["model"], r["set"])].append(r["p_original"] >= 0.5)
    out = collections.defaultdict(dict)
    for (model, s), v in by.items():
        out[model][s] = sum(v) / len(v)
        out[model][f"n_{s}"] = len(v)
    for model, d in out.items():
        n = d.get("n_wmt24pp", 0) + d.get("n_flan", 0)
        d["pooled"] = (d.get("wmt24pp", 0) * d.get("n_wmt24pp", 0) + d.get("flan", 0) * d.get("n_flan", 0)) / n
        d["n"] = n
    return dict(out)


# ---------------------------------------------------------------------------------------------------- report

def f3(x, signed=False):
    if x is None:
        return "–"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


MODEL_NAMES = {"base": "basis (TG 27B)", "B": "B", "N": "N", "Ninstr": "N met instructiezin"}


def report(args):
    import pyarrow.parquet as pq

    evalsets = {r["id"]: r for r in pq.read_table(args.evalsets).to_pylist()}
    four = {"base": four_sets(args.baseline_scores, evalsets)}
    four.update({m: four_sets(p, evalsets) for m, p in parse_named(args.four).items()})
    x = {m: scored(d) for m, d in parse_named(args.x).items()}
    instr = {m: scored(d) for m, d in parse_named(args.instr).items()}
    arms = [m for m in ("B", "N") if m in four]
    result = {"created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "commit": args.commit, "rule_a": {},
              "four": {}, "rule_b": None, "classifier": None, "indomain": {}, "flan_qe": {}, "mode_check": {},
              "pll": {}, "mtld": {}, "counts": {}}
    for m in arms:
        result["rule_a"][m] = rule_a(four["base"], four[m])
    for m, sets in four.items():
        result["four"][m] = {s: metrics(rows) for s, rows in sets.items()}
    # in-domain, per source and together
    for m, rows in x.items():
        dom = [r for r in rows.values() if r["set"] == "indomain"]
        if len(dom) != N_INDOMAIN:
            raise SystemExit(f"{m}: {len(dom)} in-domain translations, expected {N_INDOMAIN}")
        by_source = collections.defaultdict(list)
        for r in dom:
            by_source[args.x_sources[r["id"]]].append(r)
        result["indomain"][m] = {s: metrics(by_source[s], with_chrf=True) for s in SOURCES if by_source[s]}
        result["indomain"][m]["all"] = metrics(dom, with_chrf=True)
        flan = [r for r in rows.values() if r["set"] == "flan"]
        result["flan_qe"][m] = metrics(flan)
        result["counts"][m] = {s: counts([r for r in rows.values() if r["set"] == s]) for s in ("flan", "blind", "indomain")}
    for m, rows in instr.items():
        wmt = [r for r in rows.values() if r["set"] == "wmt24pp" and not bad_source(evalsets[r["id"]]["meta"])]
        result["mode_check"][m] = {"wmt24pp": metrics(wmt), "flan": metrics([r for r in rows.values() if r["set"] == "flan"])}
    # the classifier
    if args.classifier:
        c = json.loads(Path(args.classifier).read_text())
        result["classifier"] = c
        if c["test_accuracy"] >= CLASSIFIER_MIN:
            shares = classifier_shares(pq.read_table(args.predictions).to_pylist())
            result["shares"] = shares
            result["rule_b"] = {m: {"delta": shares[m]["pooled"] - shares["base"]["pooled"],
                                    "holds": shares[m]["pooled"] - shares["base"]["pooled"] >= RULE_B - TOLERANCE}
                                for m in arms}
    # PLL and MTLD
    if args.pll:
        pll = pq.read_table(args.pll).to_pylist()
        by = collections.defaultdict(list)
        for r in pll:
            by[(r["model"], r["set"])].append(r)
        for (m, s), rows in sorted(by.items()):
            kept = [r for r in rows if r["pll"] is not None]
            result["pll"].setdefault(m, {})[s] = {
                "n": len(kept), "skipped": len(rows) - len(kept), "pll": mean(r["pll"] for r in kept),
                "pll_per_token": mean(r["pll"] / r["tokens"] for r in kept)}
    if args.texts:
        tx = pq.read_table(args.texts).to_pylist()
        by = collections.defaultdict(list)
        for r in tx:
            by[(r["model"], r["set"])].append(r)
        for (m, s), rows in sorted(by.items()):
            joined = "\n".join(r["text"] for r in sorted(rows, key=lambda r: r["id"]))
            result["mtld"].setdefault(m, {})[s] = mtld(joined)
    Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    Path(args.out).write_text(report_md(result, args), encoding="utf-8")
    for m, r in result["rule_a"].items():
        print(f"RULE_A {m}: {'holds' if r['holds'] else 'fails'} "
              f"{json.dumps({s: round(v['delta'], 4) for s, v in r['sets'].items()})}", flush=True)
    print(f"RULE_B {json.dumps(result['rule_b'])}", flush=True)


def report_md(r, args):
    L = [f"# Fine-tune van de en→nl-vertaler: armen B en N op TranslateGemma 27B", "",
         f"Gemaakt {r['created']} door `armeval.py report` (mbert-jax `en-nl-sft` @ `{r['commit']}`), volgens "
         "`experiments/en-nl-translator/protocol.md` (vastgelegd 10 nov). MetricX-24 hybrid XXL: lager is beter; "
         "COMET-22 en chrF: hoger is beter. Gemiddelden over de segmenten, onafgerond berekend.", ""]
    L += ["## Regel (a): betekenis, MetricX-24-QE tegen het uitgangspunt (≤ +0,20)", "",
          "| arm | set | n | basis | arm | verschil | grens | haalt |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for m, ra in r["rule_a"].items():
        for s, v in ra["sets"].items():
            L.append(f"| {m} | {s}{' (960)' if s == 'wmt24pp' else ''} | {v['n']} | {f3(v['base'])} | {f3(v['arm'])} | "
                     f"{f3(v['delta'], True)} | {f3(v['limit'])} | {'ja' if v['holds'] else 'nee'} |")
    L += [""] + [f"- **{m}**: regel (a) {'gehaald' if ra['holds'] else 'NIET gehaald'}." for m, ra in r["rule_a"].items()]
    L += ["", "## De vier sets (gerapporteerd; uitgangspunt = `eval/baseline-tg27b`)", "",
          "| set | model | n | MetricX ref | MetricX QE | COMET-22 | niet af | lus | Engels over | leeg |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sets = ["wmt24pp", "wmt24pp_all", "flores", "ntrex", "europarl"]
    for s in sets:
        for m, d in r["four"].items():
            if s in d:
                v = d[s]
                L.append(f"| {s} | {MODEL_NAMES.get(m, m)} | {v['n']} | {f3(v['metricx_ref'])} | {f3(v['metricx_qe'])} | "
                         f"{f3(v['comet22'])} | {v['not_ended']} | {v['decode_loop']} | {v['english_left']} | {v['empty']} |")
    L += ["", "`wmt24pp` = de 960 zonder `is_bad_source`, `wmt24pp_all` = alle 998.", ""]
    L += ["## Regel (b): classifier \"origineel of vertaald\" (S @ 653k)", ""]
    c = r["classifier"]
    if c is None:
        L += ["Geen classifier-uitslag.", ""]
    else:
        ok = c["test_accuracy"] >= CLASSIFIER_MIN
        L += [f"Nauwkeurigheid op de eigen testsplit: **{c['test_accuracy']:.4f}** ({c['test_examples']} teksten; "
              f"voorwaarde ≥ {CLASSIFIER_MIN}): {'gehaald' if ok else 'NIET gehaald: regel (b) is niet te meten, Yeb beslist (§4.2.1)'}.", ""]
        if ok:
            L += ["| model | WMT24++ (960) | FLAN (1.000) | samen | verschil met basis | haalt (≥ +0,10) |",
                  "|---|---:|---:|---:|---:|---|"]
            for m, d in r["shares"].items():
                rb = (r["rule_b"] or {}).get(m)
                delta = d["pooled"] - r["shares"]["base"]["pooled"]
                L.append(f"| {MODEL_NAMES.get(m, m)} | {pct(d.get('wmt24pp'))} | {pct(d.get('flan'))} | {pct(d['pooled'])} | "
                         f"{'–' if m == 'base' else f3(delta, True)} | {'' if rb is None else ('ja' if rb['holds'] else 'nee')} |")
            L += ["", "Aandeel = deel van de vertalingen dat de classifier \"origineel\" noemt (kans ≥ 0,5). "
                  "N met instructiezin beslist niet mee.", ""]
    L += ["## In-domein: de eval-split van de sets (4.519 passages), tegen het Nederlandse origineel", "",
          "Gerapporteerd, beslist niet mee (§4.2.2). Invoer: het synthetische Engels, zonder context.", "",
          "| bron | model | n | MetricX ref | MetricX QE | COMET-22 | chrF (gem.) | chrF (corpus) | niet af | lus |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SOURCES + ["all"]:
        for m, d in r["indomain"].items():
            if s in d:
                v = d[s]
                L.append(f"| {SOURCE_NAMES.get(s, 'samen')} | {MODEL_NAMES.get(m, m)} | {v['n']} | {f3(v['metricx_ref'])} | "
                         f"{f3(v['metricx_qe'])} | {f3(v['comet22'])} | {f2(v['chrf'])} | {f2(v['chrf_corpus'])} | "
                         f"{v['not_ended']} | {v['decode_loop']} |")
    L += ["", "## Controle van de modus: N met de instructiezin (beslist niet mee, §4.1)", "",
          "| set | model | n | MetricX ref | MetricX QE | COMET-22 | niet af | lus |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in ("base", "N"):
        if m in r["four"] and "wmt24pp" in r["four"][m]:
            v = r["four"][m]["wmt24pp"]
            L.append(f"| wmt24pp (960) | {MODEL_NAMES[m]} | {v['n']} | {f3(v['metricx_ref'])} | {f3(v['metricx_qe'])} | "
                     f"{f3(v['comet22'])} | {v['not_ended']} | {v['decode_loop']} |")
    for m, d in r["mode_check"].items():
        v = d["wmt24pp"]
        L.append(f"| wmt24pp (960) | {MODEL_NAMES.get(m, m)} | {v['n']} | {f3(v['metricx_ref'])} | {f3(v['metricx_qe'])} | "
                 f"{f3(v['comet22'])} | {v['not_ended']} | {v['decode_loop']} |")
    for m, v in list(r["flan_qe"].items()) + [(m, d["flan"]) for m, d in r["mode_check"].items()]:
        L.append(f"| flan (1.000) | {MODEL_NAMES.get(m, m)} | {v['n']} | – | {f3(v['metricx_qe'])} | – | "
                 f"{v['not_ended']} | {v['decode_loop']} |")
    L += ["", "## PLL (S @ 653k, PLL-word-l2r) en MTLD (gerapporteerd)", "",
          "| model | set | n | weggelaten (> 1.022 tokens) | PLL per vertaling | PLL per token | MTLD |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for m in dict.fromkeys(list(r["pll"]) + list(r["mtld"])):
        for s in ("wmt24pp", "flan"):
            p = r["pll"].get(m, {}).get(s)
            t = r["mtld"].get(m, {}).get(s)
            L.append(f"| {MODEL_NAMES.get(m, m)} | {s} | {p['n'] if p else '–'} | {p['skipped'] if p else '–'} | "
                     f"{f3(p['pll']) if p else '–'} | {f3(p['pll_per_token']) if p else '–'} | {f3(t)} |")
    L += ["", "PLL: hoger (minder negatief) = waarschijnlijker onder S. MTLD: hoger = meer woordenschatvariatie; over de "
          "aaneengesloten uitvoer per set.", "",
          "## Regel (c): blinde beoordeling", "",
          "Nog niet beoordeeld. Pakket per arm in `gs://t5-dutch-english-usc2/mbert/translator/sft/blind/<arm>/` "
          "(`judge-input.jsonl`; de sleutel `key.jsonl` alleen in de bucket). Een verse subagent beoordeelt met de "
          "rubriek van §4.2.4; twijfelgevallen naar Yeb; alleen armen die (a) en (b) halen.", "",
          "## Stand van de beslisregel", ""]
    for m in r["rule_a"]:
        a = r["rule_a"][m]["holds"]
        b = None if r["rule_b"] is None else r["rule_b"][m]["holds"]
        L.append(f"- **{m}**: (a) {'ja' if a else 'nee'}, (b) {'niet te meten' if b is None else ('ja' if b else 'nee')}, "
                 f"(c) open.")
    return "\n".join(L) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data")
    p.add_argument("--kind", choices=["x", "instr"], required=True)
    p.add_argument("--flan-eval", required=True)
    p.add_argument("--evalsets", required=True)
    p.add_argument("--blind")
    p.add_argument("--eval-meta")
    p.add_argument("--eval")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("classifier-data")
    p.add_argument("--eval-meta", required=True)
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("texts")
    p.add_argument("--evalsets", required=True, help="evalsets.parquet of the four sets (the 960 of WMT24++)")
    p.add_argument("--wmt", nargs="+", required=True, help="MODEL=parquet or glob with id, set, hypothesis")
    p.add_argument("--flan", nargs="+", required=True, help="MODEL=parquet or glob with id, set, hypothesis")
    p.add_argument("--out", required=True)
    p = sub.add_parser("blind")
    p.add_argument("--blind", required=True)
    p.add_argument("--base", required=True, help="glob of the base model's pairs of kind x")
    p.add_argument("--arm", nargs="+", required=True, help="ARM=glob of its pairs of kind x")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--evalsets", required=True, help="evalsets.parquet of the four sets")
    p.add_argument("--baseline-scores", required=True)
    p.add_argument("--four", nargs="+", required=True, help="ARM=scores.parquet of evalsets.py summary")
    p.add_argument("--x-data", required=True, help="evalsets.parquet of kind x (the in-domain sources)")
    p.add_argument("--x", nargs="+", required=True, help="MODEL=directory with pairs/ and scores/ of kind x")
    p.add_argument("--instr", nargs="*", default=[], help="MODEL=directory with pairs/ and scores/ of kind instr")
    p.add_argument("--classifier", default=None, help="results.json of naturalness.py classifier")
    p.add_argument("--predictions", default=None)
    p.add_argument("--pll", default=None)
    p.add_argument("--texts", default=None)
    p.add_argument("--commit", default="unknown")
    p.add_argument("--out", required=True)
    p.add_argument("--json", required=True)
    args = parser.parse_args(argv)
    if args.command == "data" and args.kind == "x" and not (args.blind and args.eval_meta and args.eval):
        parser.error("--kind x needs --blind, --eval-meta and --eval")
    if args.command == "report":
        import pyarrow.parquet as pq

        args.x_sources = {r["id"]: json.loads(r["meta"]).get("source")
                           for r in pq.read_table(args.x_data, columns=["id", "set", "meta"]).to_pylist()
                           if r["set"] == "indomain"}
    {"data": data, "classifier-data": classifier_data, "texts": texts, "blind": blind, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
