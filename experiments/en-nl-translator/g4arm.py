#!/usr/bin/env python3
"""Arm G4-1 of protocol-g4.md: the first translator arm on Gemma 4 31B-it (Yeb, 14 Nov ~00:40), LoRA rank 32 / alpha
16 on attention + MLP, trained on Gemma 4's back-translations resampled to the human length ratio and RL's Dutch books,
1 : 1, without FLAN replay; evaluated against Gemma 4 without fine-tune (G4-0) translated again in the same chain.
run_g4arm.sh runs these on a worker:

  fetch     the five Gemma 4 back-translation datasets at their pinned revisions (train split, data/ only) from Hugging
            Face, with the sha256 of every file (revisions.json)
  build     the training set (protocol-g4.md §2): the pool of back-translations (hard_fails empty, R2's length bound,
            no 13-gram with an evaluation set or a blind passage, an optional FineWeb-2 crawl bound, <= 2,048 Gemma 4
            tokens), RL's book pairs, the back-translations drawn to the word ratio of R2's real examples (RL's
            histogram matching, 5 bins) one per book pair, an optional FLAN replay share (default 0, never
            TranslateGemma's FLAN-nl), natural-v1 in Gemma 4's chat template; set, manifest and report
  check     every example of the set through MaxText's own SFT steps against the model's own chat template
  evaldata  the evaluation segments (the four sets, FLAN 1,000, blind 100) in Gemma 4 tokens, classes tiny-xlong, whole
            classes per worker (evalsets.assign_workers); the smoke test's 50 segments
  texts     the classifier's texts: WMT24++ (960) and FLAN of G4-1 and G4-0, and the control texts of TG and Gemma 4 (v6e)
  report    (a1) and (b) of protocol-g4.md §5, the state of (a2) and (c), TranslateGemma as a second reference
  tally     (a2) and (c) from the blind judge's lines and the key
  smoke     the smoke test's verdict (protocol-g4.md §6)
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
# translate.py (and translate_models.yml) of FLAN-nl in the repo, or beside this file on a worker
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

import g4lora  # noqa: E402
import select_passages as sp  # noqa: E402
import sft_build  # noqa: E402

ARM = "G4-1"
BASE = "G4-0"
SEED = "en-nl-g4arm/0"
PROMPT = g4lora.PROMPT              # natural-v1
MODEL = "gemma4-31b"
MAX_TOKENS = 2048                   # protocol-g4.md §2.1 point 4 and §3
LEN_FACTOR = 1.3333                 # R2's length bound (protocol.md §9, amendment R2)
MATCH_BINS = 5                      # RL's addendum of 12 Nov: the quintiles of the target
BT_PER_BOOK = 1                     # protocol-g4.md §2.4: back-translations : books = 1 : 1
BOOKS_EXPECTED = 12521              # RL's book pairs (protocol.md §9, amendment RL, the set with 5 bins)
PACK_FILL = 0.915                   # RL's packing fill at 2,048 (g4lora.batches_estimate)
EVAL_CLASSES = ("tiny", "small", "medium", "long", "xlong")  # decode limit <= 832 (protocol-g4.md §4.1)
SMOKE_N = 50
SMOKE_CLASS = "small"
SMOKE_QE_MAX_DELTA = 0.5
SMOKE_MIN_ENDED = 48
SMOKE_MODULES = 410
MEM_MAX_GIB = g4lora.MEM_MAX_GIB    # 29.5
# (a1): the human reference's mean MetricX-24-QE as a candidate, unrounded (refqe.json of 12 Nov, ref_mean)
HUMAN_QE = {"wmt24pp": 2.3917784375, "flores": 1.4938481225296443, "ntrex": 1.8419510766149223}
# protocol-g4.md §1: Gemma 4 31B-it without fine-tune on a v6e-8 (12/13 Nov, fork @ 2249474), reported beside G4-0 as a
# check (LEARNINGS, "Kleinere Gemma 4-modellen"); it decides nothing
G4_V6E_QE = {"wmt24pp": 2.230, "flores": 1.160, "ntrex": 1.554}
RULE_B = 0.10
RULE_C = 0.60
TOLERANCE = 1e-9
FOUR_SETS = {"wmt24pp": 998, "flores": 1012, "ntrex": 1997, "europarl": 640}
N_FLAN, N_BLIND, N_WMT_GOOD = 1000, 100, 960

# protocol-g4.md §2.1: the five datasets at the revisions of 14 Nov 00:17 CET (with the round trip's columns)
BT_REPOS = {
    "handelingen": ("yhavinga/handelingen-tweede-kamer-nl-en-gemma4", "3a129f586544d57eb7d6135b3c7b154153f22ee8"),
    "raad": ("yhavinga/openraadsinformatie-nl-en-gemma4", "6fb33363cf38a329beb15a1ed72e986ad220ccf5"),
    "dpc": ("yhavinga/rijksoverheid-websites-nl-en-gemma4", "0683859b68e6ee594b60e800d4d0110ff4c86276"),
    "rechtspraak": ("yhavinga/rechtspraak-nl-en-gemma4", "0eaca05493c9293f377ec3bf6b2e2c0d02963b82"),
    "fineweb": ("yhavinga/fineweb-2-nl-en-gemma4", "f7538a912cb3b24c810e2032d8dca108e216b2e0"),
}
BT_COLUMNS = ["id", "nl", "en", "hard_fails", "chrf", "model"]
# protocol-g4.md §2.5: never TranslateGemma's FLAN-nl as replay (the Gemma 4 pilot's directory is allowed)
REPLAY_REFUSED = ("yhavinga/flan-nl", "mbert/flan-nl/", "mainset/replay")
REPLAY_ALLOWED = ("mbert/flan-nl/g4-pilot",)
CRAWL = re.compile(r"^CC-MAIN-(\d{4})-(\d{2})$")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_parquet(rows, path, schema=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def read_rows(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def quantiles(values, qs=(5, 10, 25, 50, 75, 90, 95)):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    return {"n": len(values), **{f"p{q}": round(sft_build.quantile(values, q / 100), 4) for q in qs}}


# ---------------------------------------------------------------------------------------------------- fetch

def fetch(args):
    from huggingface_hub import snapshot_download

    out = Path(args.out)
    info = {}
    for source, (repo, revision) in BT_REPOS.items():
        d = out / source
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=str(d),
                          allow_patterns=["data/train-*.parquet", "README.md"])
        files = sorted(d.glob("data/train-*.parquet"))
        if not files:
            raise SystemExit(f"{repo}@{revision}: no data/train-*.parquet")
        info[source] = {"repo": repo, "revision": revision,
                        "files": {f.name: sha256_file(f) for f in files}}
        print(f"FETCH {source} {repo}@{revision[:9]}: {len(files)} files", flush=True)
    (out / "revisions.json").write_text(json.dumps(info, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------------- build

def crawl_key(value):
    """(year, week) of a Common Crawl snapshot name, or of a bare year (its last week)"""
    value = str(value).strip()
    if re.fullmatch(r"\d{4}", value):
        return int(value), 99
    m = CRAWL.match(value)
    if not m:
        raise SystemExit(f"not a crawl (CC-MAIN-YYYY-WW) or a year: {value!r}")
    return int(m.group(1)), int(m.group(2))


def read_bt(directory, revisions):
    """The train rows of the five datasets as {pid, source, nl, en, hard_fails, chrf, model, dump}; every file's sha256
    must be the one fetch recorded"""
    import pyarrow.parquet as pq

    rows = []
    for source in BT_REPOS:
        d = Path(directory) / source / "data"
        files = sorted(d.glob("train-*.parquet"))
        want = revisions[source]["files"]
        if sorted(f.name for f in files) != sorted(want):
            raise SystemExit(f"{d}: files {[f.name for f in files]}, revisions.json has {sorted(want)}")
        for f in files:
            if sha256_file(f) != want[f.name]:
                raise SystemExit(f"{f}: sha256 differs from revisions.json")
            names = pq.read_schema(f).names
            columns = BT_COLUMNS + (["dump"] if source == "fineweb" else [])
            missing = [c for c in columns if c not in names]
            if missing:
                raise SystemExit(f"{f}: no column(s) {missing}")
            for r in pq.read_table(f, columns=columns).to_pylist():
                rows.append({"pid": r["id"], "source": source, "nl": r["nl"], "en": r["en"],
                             "hard_fails": list(r["hard_fails"] or []), "chrf": r["chrf"], "model": r["model"],
                             "dump": r.get("dump")})
    ids = [r["pid"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids among the back-translation rows")
    return rows


def read_books(set_dir):
    """RL's book pairs: the examples of kind real of RL's set ($S/data/rl-b5/RL), English from the meta, Dutch from the
    assistant turn"""
    d = Path(set_dir)
    metas = read_rows(d / "train-meta.parquet", ["id", "kind", "source", "en"])
    msgs = read_rows(d / "train.parquet")
    if len(metas) != len(msgs):
        raise SystemExit(f"{d}: {len(msgs)} examples, {len(metas)} meta rows")
    out = []
    for m, x in zip(metas, msgs):
        if m["kind"] != "real":
            continue
        turns = x["messages"]
        if [t["role"] for t in turns] != ["user", "assistant"]:
            raise SystemExit(f"{m['id']}: not one user and one assistant turn")
        out.append({"pid": m["id"], "source": m["source"], "nl": turns[1]["content"].strip(), "en": m["en"].strip()})
    return out


class TokenCounter:
    """Gemma 4 tokens of an example as MaxText's SFT makes it: the prompt of g4lora.prompt_text (Gemma 4's turn around
    natural-v1 with the English) and the completion (the Dutch and <turn|>\\n); `check` verifies these counts against
    MaxText's own steps"""

    def __init__(self, tokenizer_dir):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def lengths(self, texts):
        out = []
        texts = list(texts)
        for k in range(0, len(texts), 4096):
            out += [len(e.ids) for e in self.tok.encode_batch(texts[k:k + 4096], add_special_tokens=False)]
        return out

    def examples(self, rows):
        prompts = self.lengths(g4lora.prompt_text(r["en"]) for r in rows)
        targets = self.lengths(g4lora.completion_text(r["nl"]) for r in rows)
        return prompts, targets


def blocked_grams(evalsets, blind):
    """The 13-grams no example may share: source and reference of the four evaluation sets, the blind passages' sources"""
    blocked = sft_build.blocked_ngrams(evalsets)
    for b in blind:
        blocked |= sp.ngrams(b["source"] or "")
    return blocked


def shares_gram(row, blocked):
    return bool((sp.ngrams(row["nl"]) | sp.ngrams(row["en"])) & blocked)


def bt_pool(rows, blocked, max_crawl, counter, report):
    """The eligible back-translations (protocol-g4.md §2.1) with their token counts; the report counts what each bound
    takes, per source, in the order of the protocol"""
    funnel = collections.defaultdict(collections.Counter)
    for r in rows:
        funnel[r["source"]]["rows"] += 1
    passed = [r for r in rows if not r["hard_fails"]]
    for r in rows:
        if r["hard_fails"]:
            funnel[r["source"]]["hard_fails"] += 1
    if max_crawl is not None:
        limit = crawl_key(max_crawl)
        keep = []
        for r in passed:
            if r["source"] == "fineweb":
                if not r["dump"]:
                    raise SystemExit(f"{r['pid']}: a FineWeb-2 row without dump")
                if crawl_key(r["dump"]) > limit:
                    funnel["fineweb"]["crawl"] += 1
                    continue
            keep.append(r)
        passed = keep
    bounds = sft_build.length_bounds(passed, LEN_FACTOR)
    keep = []
    for r in passed:
        if sft_build.length_fails(r, bounds):
            funnel[r["source"]]["length"] += 1
        elif shares_gram(r, blocked):
            funnel[r["source"]]["eval_13gram"] += 1
        else:
            keep.append(r)
    prompts, targets = counter.examples(keep)
    pool = []
    for r, p, t in zip(keep, prompts, targets):
        if p + t > MAX_TOKENS:
            funnel[r["source"]]["too_long"] += 1
            continue
        pool.append({**r, "tokens_prompt": p, "tokens_target": t})
    for r in pool:
        funnel[r["source"]]["eligible"] += 1
    report["bt"] = {"funnel": {s: dict(c) for s, c in sorted(funnel.items())}, "length_bounds": bounds,
                    "max_crawl": max_crawl, "eligible": len(pool)}
    return pool


def books_kept(books, blocked, counter, report):
    prompts, targets = counter.examples(books)
    out, stats = [], collections.Counter()
    for r, p, t in zip(books, prompts, targets):
        if shares_gram(r, blocked):
            stats["eval_13gram"] += 1
        elif p + t > MAX_TOKENS:
            stats["too_long"] += 1
        else:
            out.append({**r, "tokens_prompt": p, "tokens_target": t})
    report["books"] = {"rows": len(books), "dropped": dict(stats), "kept": len(out),
                       "sources": dict(collections.Counter(r["source"] for r in out))}
    return out


def check_replay_source(path, model):
    """protocol-g4.md §2.5: a replay source must be a Gemma 4 translation, never TranslateGemma's FLAN-nl"""
    text = str(path)
    if any(bad in text for bad in REPLAY_REFUSED) and not any(ok in text for ok in REPLAY_ALLOWED):
        raise SystemExit(f"--replay {path}: TranslateGemma's FLAN-nl is not a replay source (protocol-g4.md §2.5)")
    if not model or "gemma-4" not in model.lower():
        raise SystemExit(f"--replay-model {model!r}: the replay must be translated by Gemma 4 (protocol-g4.md §2.5)")


def replay_pool(path, model, flan_eval, blocked, counter, report):
    import pyarrow.parquet as pq

    check_replay_source(path, model)
    names = pq.read_schema(path).names
    missing = [c for c in ("id", "en", "nl") if c not in names]
    if missing:
        raise SystemExit(f"{path}: no column(s) {missing}")
    rows = read_rows(path, [c for c in ("id", "task", "en", "nl", "model") if c in names])
    bad = sorted({r["model"] for r in rows if "model" in r and "gemma-4" not in str(r["model"]).lower()})
    if bad:
        raise SystemExit(f"{path}: rows translated by {bad[:3]}, not Gemma 4 (protocol-g4.md §2.5)")
    eval_segments = {s for r in read_rows(flan_eval, ["segments"]) for s in r["segments"]}
    stats = collections.Counter()
    keep = []
    for r in rows:
        r = {"pid": r["id"], "source": "flan:" + str(r.get("task") or ""), "nl": r["nl"].strip(),
             "en": r["en"].strip()}
        if r["pid"] in eval_segments:
            stats["eval_segment"] += 1
        elif shares_gram(r, blocked):
            stats["eval_13gram"] += 1
        else:
            keep.append(r)
    prompts, targets = counter.examples(keep)
    pool = []
    for r, p, t in zip(keep, prompts, targets):
        if p + t > MAX_TOKENS:
            stats["too_long"] += 1
        else:
            pool.append({**r, "tokens_prompt": p, "tokens_target": t})
    report["replay"] = {"source": str(path), "model": model, "rows": len(rows), "dropped": dict(stats),
                        "candidates": len(pool)}
    return pool


def replay_count(n_rest, share):
    if not 0 <= share < 1:
        raise SystemExit(f"--replay-share {share}: between 0 and 1 (exclusive)")
    return round(n_rest * share / (1 - share))


def example(r, kind):
    user = g4lora.user_turn(r["en"])
    return ({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["nl"].strip()}]},
            {"id": f"{kind}:{r['pid']}", "kind": kind, "source": r["source"], "prompt": PROMPT, "en": r["en"].strip(),
             "tokens_prompt": r["tokens_prompt"], "tokens_target": r["tokens_target"],
             "word_ratio": round(sft_build.word_ratio(r["nl"], r["en"]), 4),
             "char_ratio": round(sft_build.char_ratio(r["nl"], r["en"]), 4), "chrf": r.get("chrf"),
             "dump": r.get("dump")})


META_SCHEMA_FIELDS = (("id", "string"), ("kind", "string"), ("source", "string"), ("prompt", "string"),
                      ("en", "string"), ("tokens_prompt", "int64"), ("tokens_target", "int64"),
                      ("word_ratio", "float64"), ("char_ratio", "float64"), ("chrf", "float64"), ("dump", "string"))


def build(args, counter=None):
    import pyarrow as pa

    out = Path(args.out)
    counter = counter or TokenCounter(args.tokenizer)
    report = {"arm": ARM, "seed": SEED, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "len_factor": LEN_FACTOR,
              "match_bins": MATCH_BINS, "bt_per_book": BT_PER_BOOK, "replay_share": args.replay_share,
              "fineweb_max_crawl": args.fineweb_max_crawl}
    revisions = json.loads((Path(args.bt) / "revisions.json").read_text())
    for source, (repo, revision) in BT_REPOS.items():
        got = revisions.get(source, {})
        if (got.get("repo"), got.get("revision")) != (repo, revision):
            raise SystemExit(f"{args.bt}/revisions.json: {source} is {got.get('repo')}@{got.get('revision')}, the "
                             f"protocol pins {repo}@{revision}")
    evalsets = read_rows(args.evalsets)
    if collections.Counter(r["set"] for r in evalsets) != collections.Counter(FOUR_SETS):
        raise SystemExit(f"{args.evalsets}: sets {dict(collections.Counter(r['set'] for r in evalsets))}, "
                         f"expected {FOUR_SETS}")
    blind = read_jsonl(args.blind)
    if sha256_file(args.blind) != blind_sha256():
        raise SystemExit(f"{args.blind}: not the blind passages of the TG protocol (§4.2.4)")
    blocked = blocked_grams(evalsets, blind)
    print("PROGRESS reading the back-translations", flush=True)
    bt_rows = read_bt(args.bt, revisions)
    pool = bt_pool(bt_rows, blocked, args.fineweb_max_crawl, counter, report)
    print(f"PROGRESS {len(pool):,} eligible back-translations", flush=True)
    books = books_kept(read_books(args.books), blocked, counter, report)
    if report["books"]["rows"] != BOOKS_EXPECTED:
        raise SystemExit(f"{args.books}: {report['books']['rows']} book pairs, RL had {BOOKS_EXPECTED}")
    target = sft_build.match_target(args.match_set)
    want = round(BT_PER_BOOK * len(books))
    drawn, info = sft_build.bt_match_draw(pool, target, want, MATCH_BINS, SEED)
    if info["limited"]:
        raise SystemExit(f"the match holds {info['most']} back-translations, {want} wanted (1 per book pair): the "
                         f"ratio of protocol-g4.md §2.4 needs an amendment")
    report["match"] = info
    replay = []
    if args.replay_share:
        if not args.replay:
            raise SystemExit("--replay-share > 0 needs --replay (a Gemma 4 FLAN-nl) and --replay-model")
        candidates = replay_pool(args.replay, args.replay_model, args.flan_eval, blocked, counter, report)
        n_replay = replay_count(len(drawn) + len(books), args.replay_share)
        if n_replay > len(candidates):
            raise SystemExit(f"{n_replay} replay examples wanted, {len(candidates)} candidates")
        replay = random.Random(f"{SEED}/replay").sample(candidates, n_replay)
        report["replay"]["drawn"] = n_replay
    elif args.replay:
        raise SystemExit("--replay without --replay-share: the share is 0 (protocol-g4.md §2.5)")
    pairs = [example(r, "bt") for r in drawn] + [example(r, "book") for r in books] + \
        [example(r, "replay") for r in replay]
    random.Random(f"{SEED}/shuffle").shuffle(pairs)
    d = out / ARM
    meta_schema = pa.schema([(n, getattr(pa, t)()) for n, t in META_SCHEMA_FIELDS])
    write_parquet([m for m, _ in pairs], d / "train.parquet")
    write_parquet([x for _, x in pairs], d / "train-meta.parquet", meta_schema)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    metas = [x for _, x in pairs]
    tokens = sum(x["tokens_prompt"] + x["tokens_target"] for x in metas)
    report["set"] = set_stats(metas, drawn)
    report["estimate"] = {"tokens": tokens, "steps_at_2048": round(tokens / (16 * 2048 * PACK_FILL)),
                          "pack_fill": PACK_FILL, "note": "16 sequences of 2,048 a step, RL's packing fill"}
    inputs = {"bt": revisions, "books": {f: sha256_file(Path(args.books) / f) for f in ("train.parquet",
                                                                                        "train-meta.parquet")},
              "match_set": {f: sha256_file(Path(args.match_set) / f) for f in ("train.parquet", "train-meta.parquet")},
              "evalsets": sha256_file(args.evalsets), "blind": sha256_file(args.blind)}
    if replay:
        inputs["replay"] = {"path": str(args.replay), "sha256": sha256_file(args.replay), "model": args.replay_model}
    report["inputs"] = inputs
    manifest = {"set": ARM, "examples": len(metas), "kinds": dict(collections.Counter(x["kind"] for x in metas)),
                "tokens": tokens, "prompt": PROMPT, "template": "g4lora.TEMPLATE (Gemma 4, thinking off)",
                "seed": SEED, "replay_share": args.replay_share, "fineweb_max_crawl": args.fineweb_max_crawl,
                "inputs": inputs}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(report_md(report), encoding="utf-8")
    print(f"BUILD {json.dumps({'examples': len(metas), 'kinds': manifest['kinds'], 'tokens': tokens, 'steps_est': report['estimate']['steps_at_2048']})}",
          flush=True)


def blind_sha256():
    """The sha256 of the blind passages fixed by the TG protocol (§4.2.4)"""
    import armeval

    return armeval.BLIND_SHA256


def set_stats(metas, drawn):
    by_kind = collections.defaultdict(list)
    for x in metas:
        by_kind[x["kind"]].append(x)
    out = {"examples": len(metas), "kinds": {}}
    for kind, xs in sorted(by_kind.items()):
        out["kinds"][kind] = {
            "examples": len(xs), "tokens": sum(x["tokens_prompt"] + x["tokens_target"] for x in xs),
            "target_tokens": sum(x["tokens_target"] for x in xs),
            "max_tokens": max(x["tokens_prompt"] + x["tokens_target"] for x in xs),
            "word_ratio": quantiles(x["word_ratio"] for x in xs),
            "by_source": dict(sorted(collections.Counter(x["source"] for x in xs).items()))}
    out["word_ratio_all"] = quantiles(x["word_ratio"] for x in metas)
    out["bt_chrf"] = quantiles(r["chrf"] for r in drawn)
    years = collections.Counter(crawl_key(r["dump"])[0] for r in drawn if r["source"] == "fineweb" and r.get("dump"))
    out["bt_fineweb_crawl_years"] = dict(sorted(years.items()))
    return out


def _n(x):
    return f"{x:,}".replace(",", ".") if isinstance(x, int) else str(x)


def report_md(r):
    L = [f"# Trainingsset {r['arm']} (protocol-g4.md §2)", "",
         f"Gemaakt door `g4arm.py build`. Prompt `{r['prompt']}` in het chat-template van Gemma 4 (thinking uit), "
         f"hoogstens {_n(r['max_tokens'])} tokens per voorbeeld, seed `{r['seed']}`, BT per boekpaar "
         f"{r['bt_per_book']}, {r['match_bins']} bakken, FLAN-replay-aandeel {r['replay_share']}, "
         f"FineWeb-2-crawlgrens {r['fineweb_max_crawl'] or 'geen'}.", "",
         "## Terugvertalingen: wat elke grens kost (train-split)", "",
         "| bron | rijen | hard_fails | crawljaar | lengtegrens | 13-gram | te lang | in de pool |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, c in r["bt"]["funnel"].items():
        L.append(f"| {s} | {_n(c.get('rows', 0))} | {_n(c.get('hard_fails', 0))} | {_n(c.get('crawl', 0))} | "
                 f"{_n(c.get('length', 0))} | {_n(c.get('eval_13gram', 0))} | {_n(c.get('too_long', 0))} | "
                 f"{_n(c.get('eligible', 0))} |")
    L += ["", "Lengtegrenzen (tekens NL / EN, mediaan / 1,3333 tot mediaan × 1,3333): " + "; ".join(
        f"{s} {b['median']} ({b['lo']}–{b['hi']})" for s, b in r["bt"]["length_bounds"].items()) + ".", "",
          "## Boeken (RL)", "", f"{json.dumps(r['books'], ensure_ascii=False)}", "",
          "## Hersampling (histogram matching naar de echte voorbeelden van R2)", "",
          "| bak | ondergrens | bovengrens | doel | aandeel | pool | getrokken |", "|---:|---:|---:|---:|---:|---:|---:|"]
    m = r["match"]
    for k, b in enumerate(m["per_bin"]):
        L.append(f"| {k} | {b['lo']} | {b['hi']} | {_n(b['target'])} | {b['target_share']} | {_n(b['pool'])} | "
                 f"{_n(b['drawn'])} |")
    L += ["", f"Gewenst {_n(m['want'])}, hoogstens {_n(m['most'])}, getrokken {_n(m['drawn'])}.", "",
          "| woorden NL / EN | n | p5 | p10 | p25 | p50 | p75 | p90 | p95 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, q in (("doel (R2 echt)", m["ratio"]["target"]), ("pool", m["ratio"]["pool"]),
                    ("getrokken BT", m["ratio"]["drawn"])):
        L.append(f"| {name} | {_n(q['n'])} | " + " | ".join(str(q.get(f"p{k}")) for k in (5, 10, 25, 50, 75, 90, 95))
                 + " |")
    L += ["", f"Bronnen: pool {json.dumps(m['pool_by_source'])}, getrokken {json.dumps(m['drawn_by_source'])}.", ""]
    if "replay" in r:
        L += ["## FLAN-replay", "", json.dumps(r["replay"], ensure_ascii=False), ""]
    s = r["set"]
    L += ["## De set", "", "| soort | voorbeelden | tokens | doeltokens | max tokens | woorden NL / EN p10 / p50 / p90 | bronnen |",
          "|---|---:|---:|---:|---:|---|---|"]
    for kind, k in s["kinds"].items():
        q = k["word_ratio"]
        L.append(f"| {kind} | {_n(k['examples'])} | {_n(k['tokens'])} | {_n(k['target_tokens'])} | {_n(k['max_tokens'])} | "
                 f"{q.get('p10')} / {q.get('p50')} / {q.get('p90')} | {json.dumps(k['by_source'])} |")
    e = r["estimate"]
    L += ["", f"Samen {_n(s['examples'])} voorbeelden, {_n(e['tokens'])} tokens; chrF van de rondgang in de getrokken BT "
          f"{json.dumps(s['bt_chrf'])}; FineWeb-2 per crawljaar {json.dumps(s['bt_fineweb_crawl_years'])}.", "",
          f"Verwachte epoch [schatting]: ~{_n(e['steps_at_2048'])} stappen (16 × 2.048 tokenplaatsen, vulling "
          f"{e['pack_fill']} zoals RL).", ""]
    return "\n".join(L) + "\n"


def check(args):
    d = Path(args.data)
    msgs = [x["messages"] for x in read_rows(d / "train.parquet")]
    metas = read_rows(d / "train-meta.parquet", ["id", "en", "tokens_prompt", "tokens_target"])
    if len(msgs) != len(metas):
        raise SystemExit(f"{d}: {len(msgs)} examples, {len(metas)} meta rows")
    if (d / "template.jinja").read_text(encoding="utf-8") != g4lora.TEMPLATE:
        raise SystemExit(f"{d}/template.jinja is not g4lora.TEMPLATE")
    lengths, loss_tokens, pad_id = g4lora.check_examples(msgs, [m["en"] for m in metas], args.tokenizer, MODEL,
                                                         [m["id"] for m in metas])
    differ = [m["id"] for m, n in zip(metas, lengths) if n != m["tokens_prompt"] + m["tokens_target"]]
    if differ:
        raise SystemExit(f"{len(differ)} examples whose MaxText length differs from the build's count: {differ[:3]}")
    if max(lengths) > MAX_TOKENS:
        raise SystemExit(f"an example of {max(lengths)} tokens, more than {MAX_TOKENS}")
    print(f"CHECK {json.dumps({'examples': len(lengths), 'tokens': sum(lengths), 'loss_tokens': loss_tokens, 'max': max(lengths), 'pad_id': pad_id})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- evaldata

def class_of(tokens):
    import translate

    classes = translate.MODELS[MODEL]["length_classes"]
    for name in EVAL_CLASSES:
        if tokens <= classes[name]["max_source"]:
            return name
    raise SystemExit(f"a segment of {tokens} Gemma 4 tokens: above class xlong (protocol-g4.md §4.1)")


def evaldata(args, count=None, check_prompts=None):
    """The segments of the evaluation in evalsets.py's data format, counted in Gemma 4 tokens"""
    import bt_natural
    import evalsets

    four = read_rows(args.four)
    if collections.Counter(r["set"] for r in four) != collections.Counter(FOUR_SETS):
        raise SystemExit(f"{args.four}: not the four sets {FOUR_SETS}")
    x = read_rows(args.x)
    extra = [r for r in x if r["set"] in ("flan", "blind")]
    got = collections.Counter(r["set"] for r in extra)
    if got != collections.Counter({"flan": N_FLAN, "blind": N_BLIND}):
        raise SystemExit(f"{args.x}: {dict(got)}, expected FLAN {N_FLAN} and blind {N_BLIND}")
    rows = [dict(r) for r in four + extra]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate ids among the evaluation segments")
    count = count or bt_natural.Counter(args.tokenizer).many
    for r, n in zip(rows, count([r["source"] for r in rows])):
        r["tokens"] = n
        r["length_class"] = class_of(n)
        r["over_512"] = n > 512
    segments = [{"id": r["id"], "text": r["source"], "tokens": r["tokens"]} for r in rows]
    if check_prompts is None:
        from transformers import AutoTokenizer

        overhead = bt_natural.check_prompts(segments, AutoTokenizer.from_pretrained(args.tokenizer), "en", "nl",
                                            prompts=(PROMPT,), model=MODEL)
    else:
        overhead = check_prompts(segments)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = ["id", "set", "source", "reference", "tokens", "over_512", "length_class", "meta"]
    write_parquet([{c: r[c] for c in columns} for r in rows], out / "evalsets.parquet")
    assignment = evalsets.assign_workers(rows, args.workers)
    for k, part in assignment.items():
        if not part:
            raise SystemExit(f"worker {k} gets no length class")
        write_parquet([{"id": r["id"], "text": r["source"], "tokens": r["tokens"]} for r in part],
                      out / f"segments-w{k}.parquet")
        (out / f"classes-w{k}.txt").write_text(
            " ".join(c for c in EVAL_CLASSES if any(r["length_class"] == c for r in part)) + "\n")
    good = [r for r in rows if r["set"] == "wmt24pp" and not bad_source(r["meta"])]
    if len(good) != N_WMT_GOOD:
        raise SystemExit(f"{len(good)} WMT24++ segments without is_bad_source, expected {N_WMT_GOOD}")
    smoke = sorted((r for r in good if r["length_class"] == SMOKE_CLASS), key=lambda r: r["id"])[:SMOKE_N]
    if len(smoke) != SMOKE_N:
        raise SystemExit(f"{len(smoke)} WMT24++ segments of class {SMOKE_CLASS}, the smoke test takes {SMOKE_N}")
    write_parquet([{"id": r["id"], "text": r["source"], "tokens": r["tokens"]} for r in smoke],
                  out / "segments-smoke.parquet")
    manifest = {
        "model": MODEL, "prompt": PROMPT, "tokenizer": str(args.tokenizer), "prompt_overhead": overhead,
        "inputs": {str(args.four): sha256_file(args.four), str(args.x): sha256_file(args.x)},
        "sets": {s: {"segments": sum(r["set"] == s for r in rows),
                     "classes": dict(collections.Counter(r["length_class"] for r in rows if r["set"] == s))}
                 for s in dict.fromkeys(r["set"] for r in rows)},
        "workers": {f"w{k}": {"segments": len(p), "classes": sorted({r["length_class"] for r in p},
                                                                    key=EVAL_CLASSES.index)}
                    for k, p in assignment.items()},
        "smoke": {"class": SMOKE_CLASS, "ids": [r["id"] for r in smoke]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for w, s in manifest["workers"].items():
        print(f"WORKER {w}: {s['segments']} segments, classes {s['classes']}", flush=True)
    print(f"EVALDATA {json.dumps(manifest['sets'])}", flush=True)


def bad_source(meta):
    return bool(json.loads(meta).get("is_bad_source", False)) if meta else False


# ---------------------------------------------------------------------------------------------------- texts

def texts(args):
    """The classifier's texts: WMT24++ (960) and FLAN (1,000) of each model's pairs, and the control texts of 13 Nov
    (g4-base-texts.parquet: base = TranslateGemma, G4 = Gemma 4 on the v6e) under the names TG and G4-v6e"""
    pairs = {}
    for spec in args.pairs:
        name, pattern = spec.split("=", 1)
        rows = {}
        for f in sorted(glob.glob(pattern)):
            for r in read_rows(f, ["id", "set", "hypothesis"]):
                rows[r["id"]] = r
        if not rows:
            raise SystemExit(f"{name}: no pairs in {pattern}")
        pairs[name] = rows
    evalsets = {r["id"]: r for r in read_rows(args.evalsets, ["id", "set", "meta"])}
    wmt = [i for i, r in evalsets.items() if r["set"] == "wmt24pp" and not bad_source(r["meta"])]
    flan = [i for i, r in evalsets.items() if r["set"] == "flan"]
    if len(wmt) != N_WMT_GOOD or len(flan) != N_FLAN:
        raise SystemExit(f"{args.evalsets}: {len(wmt)} WMT24++ and {len(flan)} FLAN segments")
    out = []
    for name, rows in pairs.items():
        for set_name, ids in (("wmt24pp", wmt), ("flan", flan)):
            for i in sorted(ids):
                h = rows.get(i, {}).get("hypothesis")
                if h is None:
                    raise SystemExit(f"{name}: no translation of {i}")
                out.append({"model": name, "set": set_name, "id": i, "text": h})
    control = read_rows(args.control)
    rename = {"base": "TG", "G4": "G4-v6e"}
    if collections.Counter((r["model"], r["set"]) for r in control) != collections.Counter(
            {(m, s): n for m in rename for s, n in (("wmt24pp", N_WMT_GOOD), ("flan", N_FLAN))}):
        raise SystemExit(f"{args.control}: not the control texts of 13 Nov (base and G4, 960 + 1,000 each)")
    out += [{**r, "model": rename[r["model"]]} for r in control]
    write_parquet(out, args.out)
    print(f"TEXTS {json.dumps(collections.Counter(r['model'] + '/' + r['set'] for r in out))}", flush=True)


# ---------------------------------------------------------------------------------------------------- report

def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.fmean(values) if values else None


def by_set(scored, evalsets):
    """{set: rows}: wmt24pp is the 960 without is_bad_source, wmt24pp_all the 998"""
    out = collections.defaultdict(list)
    for i, r in scored.items():
        s = evalsets[i]["set"]
        out[f"{s}_all" if s == "wmt24pp" else s].append(r)
        if s == "wmt24pp" and not bad_source(evalsets[i]["meta"]):
            out["wmt24pp"].append(r)
    return dict(out)


def rule_a1(sets):
    out = {s: {"n": len(sets[s]), "model": mean(r["metricx_qe"] for r in sets[s]), "human": HUMAN_QE[s]}
           for s in HUMAN_QE}
    for v in out.values():
        v["holds"] = v["model"] is not None and v["model"] <= v["human"] + TOLERANCE
    return {"sets": out, "holds": all(v["holds"] for v in out.values())}


def rule_b(shares, arm=ARM, base=BASE):
    a, b = shares[arm]["pooled"], shares[base]["pooled"]
    return {"arm": a, "base": b, "delta": a - b, "holds": a - b >= RULE_B - TOLERANCE}


def tally_rows(judgments, key, arm):
    """(a2) and (c) from the judge's lines and the key (key A/B name 'base' or the arm)"""
    keys = {k["blind_id"]: k for k in key}
    if len(judgments) != len(keys) or {j["blind_id"] for j in judgments} != set(keys):
        raise SystemExit(f"{len(judgments)} judgments for {len(keys)} keyed passages, or other ids")
    errors = collections.Counter()
    wins = collections.Counter()
    doubt = 0
    per_set = collections.defaultdict(collections.Counter)
    for j in judgments:
        k = keys[j["blind_id"]]
        for side in ("A", "B"):
            if j[f"fout_{side}"] not in ("ja", "nee"):
                raise SystemExit(f"{j['blind_id']}: fout_{side} {j[f'fout_{side}']!r}")
            if j[f"fout_{side}"] == "ja":
                errors[k[side]] += 1
                per_set[k.get("set", "?")][f"errors_{k[side]}"] += 1
        verdict = j["oordeel"]
        if verdict == "gelijk":
            winner = "equal"
        elif verdict in ("A beter", "B beter"):
            winner = k[verdict[0]]
        else:
            raise SystemExit(f"{j['blind_id']}: oordeel {verdict!r}")
        wins[winner] += 1
        per_set[k.get("set", "?")][winner] += 1
        doubt += j.get("zekerheid") == "twijfel"
    n = wins[arm] + wins["base"]
    share = wins[arm] / n if n else None
    return {"arm": arm, "passages": len(judgments), "arm_better": wins[arm], "base_better": wins["base"],
            "equal": wins["equal"], "share_of_unequal": share, "errors_arm": errors[arm], "errors_base": errors["base"],
            "doubt": doubt, "a2": errors[arm] <= errors["base"],
            "c": share is not None and share >= RULE_C - TOLERANCE,
            "per_set": {s: dict(c) for s, c in sorted(per_set.items())}}


def tally(args):
    result = tally_rows(read_jsonl(args.judgments), read_jsonl(args.key), args.arm)
    print("TALLY " + json.dumps(result), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


def scored_dir(directory):
    import armeval

    return armeval.scored(directory)


def report(args):
    import armeval

    evalsets = {r["id"]: r for r in read_rows(args.evalsets)}
    models = {ARM: scored_dir(args.arm), BASE: scored_dir(args.base)}
    for name, rows in models.items():
        if set(rows) != set(evalsets):
            raise SystemExit(f"{name}: scores for {len(rows)} of {len(evalsets)} segments")
    sets = {name: by_set(rows, evalsets) for name, rows in models.items()}
    four_tg = {r["id"]: r for r in read_rows(args.tg_scores)}
    tg_sets = collections.defaultdict(list)
    for i, r in four_tg.items():
        if i not in evalsets:
            raise SystemExit(f"{args.tg_scores}: {i} is not an evaluation segment")
        s = evalsets[i]["set"]
        tg_sets[f"{s}_all" if s == "wmt24pp" else s].append(r)
        if s == "wmt24pp" and not bad_source(evalsets[i]["meta"]):
            tg_sets["wmt24pp"].append(r)
    predictions = read_rows(args.predictions)
    shares = armeval.classifier_shares(predictions)
    for name in (ARM, BASE):
        if shares.get(name, {}).get("n") != N_WMT_GOOD + N_FLAN:
            raise SystemExit(f"{args.predictions}: {name} has {shares.get(name, {}).get('n')} texts")
    accuracy = json.loads(Path(args.classifier_results).read_text())["test_accuracy"] if args.classifier_results else None
    result = {"arm": ARM, "base": BASE, "a1": rule_a1(sets[ARM]), "a1_base": rule_a1(sets[BASE]),
              "b": rule_b(shares), "b_against_tg": rule_b(shares, ARM, "TG") if "TG" in shares else None,
              "b_base_against_tg": rule_b(shares, BASE, "TG") if "TG" in shares else None,
              "shares": shares, "classifier_accuracy": accuracy, "sets": {}}
    for s in ("wmt24pp", "wmt24pp_all", "flores", "ntrex", "europarl", "flan", "blind"):
        entry = {}
        for name in (ARM, BASE):
            rows = sets[name].get(s, [])
            entry[name] = {**armeval.counts(rows), "metricx_qe": mean(r["metricx_qe"] for r in rows),
                           "metricx_ref": mean(r["metricx_ref"] for r in rows),
                           "comet22": mean(r["comet22"] for r in rows), "length": armeval.length_ratios(rows)}
        if s in tg_sets:
            rows = tg_sets[s]
            entry["TG"] = {"n": len(rows), "metricx_qe": mean(r.get("metricx_qe") for r in rows),
                           "metricx_ref": mean(r.get("metricx_ref") for r in rows),
                           "comet22": mean(r.get("comet22") for r in rows)}
        result["sets"][s] = entry
    if args.lora_line:
        result["lora_merge"] = [json.loads(Path(p).read_text()) for p in args.lora_line]
    if args.train_summary:
        t = json.loads(Path(args.train_summary).read_text())
        result["training"] = {k: t.get(k) for k in ("steps", "steps_done", "throughput", "peak_gib_max", "times")}
        result["training"]["loss_first_last"] = [t["loss"][:3], t["loss"][-3:]] if t.get("loss") else None
    if args.tally:
        result["blind"] = json.loads(Path(args.tally).read_text())
        result["a2"] = result["blind"]["a2"]
        result["c"] = result["blind"]["c"]
    else:
        result["blind"] = None
    result["verdict"] = verdict(result)
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")
    Path(args.out).write_text(report_text(result), encoding="utf-8")
    print(f"REPORT {json.dumps({'a1': result['a1']['holds'], 'b': result['b'], 'verdict': result['verdict']})}",
          flush=True)


def verdict(r):
    if not r["a1"]["holds"] or not r["b"]["holds"]:
        return "helpt niet"
    if r.get("blind") is None:
        return "wacht op de blinde beoordeling ((a2), (c))"
    return "helpt" if r["a2"] and r["c"] else "helpt niet"


def f3(x, signed=False):
    if x is None:
        return "–"
    return (f"{x:+.3f}" if signed else f"{x:.3f}").replace(".", ",")


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%".replace(".", ",")


def report_text(r):
    L = [f"# Arm {r['arm']}: uitslag volgens protocol-g4.md §5", "",
         f"Gemaakt door `g4arm.py report`. Uitgangspunt {r['base']} (Gemma 4 31B-it zonder fine-tune, zelfde keten); "
         "TG = TranslateGemma 27B (tweede referentie, beslist niet).", "", f"**Uitslag: {r['verdict']}.**", "",
         "## (a1) MetricX-24-QE niet hoger dan de menselijke referentie", "",
         f"| set | n | {r['arm']} | {r['base']} | TG | mens | (a1) |", "|---|---:|---:|---:|---:|---:|---|"]
    for s, v in r["a1"]["sets"].items():
        L.append(f"| {s} | {v['n']} | {f3(v['model'])} | {f3(r['a1_base']['sets'][s]['model'])} | "
                 f"{f3(r['sets'][s].get('TG', {}).get('metricx_qe'))} | {f3(v['human'])} | "
                 f"{'ja' if v['holds'] else 'nee'} |")
    b = r["b"]
    L += ["", "Ter controle Gemma 4 zonder fine-tune op de v6e-8 (12/13 nov): " + " / ".join(
        f3(G4_V6E_QE[s]) for s in HUMAN_QE) + " (WMT24++ / FLORES / NTREX)."]
    L += ["", f"**(a1) {'gehaald' if r['a1']['holds'] else 'niet gehaald'}.**", "",
          "## (b) classifier: aandeel \"origineel\" (WMT24++ 960 + FLAN 1.000)", "",
          "| model | WMT24++ | FLAN | samen | n |", "|---|---:|---:|---:|---:|"]
    for name, sh in r["shares"].items():
        L.append(f"| {name} | {pct(sh.get('wmt24pp'))} | {pct(sh.get('flan'))} | {pct(sh.get('pooled'))} | {sh.get('n')} |")
    L += ["", f"{r['arm']} − {r['base']}: {f3(b['delta'], True)} (grens +0,10) → **(b) "
          f"{'gehaald' if b['holds'] else 'niet gehaald'}**."]
    if r.get("b_against_tg"):
        L.append(f"Ter informatie: {r['arm']} − TG {f3(r['b_against_tg']['delta'], True)}, {r['base']} − TG "
                 f"{f3(r['b_base_against_tg']['delta'], True)}.")
    L += ["", "## (a2) en (c): blinde beoordeling tegen " + r["base"], ""]
    if r.get("blind"):
        x = r["blind"]
        L.append(f"{x['arm']} beter {x['arm_better']}, {r['base']} beter {x['base_better']}, gelijk {x['equal']} → "
                 f"{pct(x['share_of_unequal'])} van de niet-gelijke (grens 60%): **(c) {'gehaald' if x['c'] else 'niet gehaald'}**; "
                 f"betekenisfouten {x['errors_arm']} tegen {x['errors_base']}: **(a2) {'gehaald' if x['a2'] else 'niet gehaald'}**; "
                 f"twijfel {x['doubt']}.")
    else:
        L.append("Nog niet beoordeeld (pakket `$S/blind/G4-1/`; alleen als (a1) en (b) gelden, protocol-g4.md §5).")
    L += ["", "## Per set (gerapporteerd, beslist niet)", "",
          "| set | model | n | QE | MetricX-ref | COMET-22 | uitvoer / referentie | < 0,7 × ref | niet af | lus | Engels over | leeg |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, entry in r["sets"].items():
        for name, v in entry.items():
            length = v.get("length") or {}
            L.append(f"| {s} | {name} | {v.get('n')} | {f3(v.get('metricx_qe'))} | {f3(v.get('metricx_ref'))} | "
                     f"{f3(v.get('comet22'))} | {f3(length.get('out_per_ref'))} | {pct(length.get('under_0_7_of_ref'))} | "
                     f"{v.get('not_ended', '–')} | {v.get('decode_loop', '–')} | {v.get('english_left', '–')} | "
                     f"{v.get('empty', '–')} |")
    if r.get("lora_merge"):
        L += ["", "LoRA-samenvoeging (`translate.py`, regel LORA): " + "; ".join(json.dumps(x) for x in r["lora_merge"][:4])]
    if r.get("training"):
        L += ["", "Training: " + json.dumps(r["training"], default=str)]
    if r.get("classifier_accuracy") is not None:
        L += ["", f"Nauwkeurigheid van de classifier: {r['classifier_accuracy']}."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- smoke

def lora_line(log_text):
    """The json of translate.py's LORA line (the merge's modules, update norm, changed elements)"""
    lines = [ln for ln in log_text.splitlines() if ln.startswith("LORA ")]
    return json.loads(lines[-1][len("LORA "):]) if lines else None


def smoke_verdict(summary, adapter, checkpoint_ok, lora, arm_rows, base_rows):
    """protocol-g4.md §6: every criterion with its value and whether it holds"""
    checks = {}
    loss = summary.get("loss") or []
    checks["steps"] = {"value": summary.get("steps_done"), "holds": summary.get("steps_done") == 8}
    checks["loss_finite"] = {"value": [v for _, v in loss],
                             "holds": len(loss) == 8 and all(math.isfinite(v) for _, v in loss)}
    checks["peak_gib"] = {"value": summary.get("peak_gib_max"),
                          "holds": summary.get("peak_gib_max") is not None and summary["peak_gib_max"] <= MEM_MAX_GIB}
    checks["adapter"] = {"value": {k: adapter.get(k) for k in ("rank", "alpha", "scale")},
                         "holds": (adapter.get("rank"), adapter.get("alpha"), adapter.get("scale")) == (32, 16.0, 0.5)}
    checks["checkpoint"] = {"value": checkpoint_ok, "holds": bool(checkpoint_ok)}
    checks["lora_merge"] = {"value": lora, "holds": bool(lora) and lora.get("modules") == SMOKE_MODULES
                            and (lora.get("relative_update_norm") or 0) > 0}
    for name, rows in (("arm", arm_rows), ("base", base_rows)):
        ended = sum(bool(r["ended"]) and not r["decode_loop"] and not r["empty"] for r in rows)
        checks[f"decode_{name}"] = {"value": {"segments": len(rows), "ended_ok": ended},
                                    "holds": len(rows) == SMOKE_N and ended >= SMOKE_MIN_ENDED}
        qe = [r.get("metricx_qe") for r in rows]
        checks[f"metricx_{name}"] = {"value": mean(qe),
                                     "holds": len(qe) == SMOKE_N and all(q is not None and math.isfinite(q) for q in qe)}
    delta = None
    if checks["metricx_arm"]["value"] is not None and checks["metricx_base"]["value"] is not None:
        delta = checks["metricx_arm"]["value"] - checks["metricx_base"]["value"]
    checks["qe_delta"] = {"value": delta, "holds": delta is not None and abs(delta) <= SMOKE_QE_MAX_DELTA}
    return {"checks": checks, "passed": all(c["holds"] for c in checks.values())}


def smoke(args):
    summary = json.loads(Path(args.summary).read_text())
    adapter = json.loads(Path(args.adapter).read_text())
    lora = lora_line(Path(args.translate_log).read_text(errors="replace"))
    out = {}
    for name, pairs, metricx in (("arm", args.arm_pairs, args.arm_metricx), ("base", args.base_pairs, args.base_metricx)):
        p = {r["id"]: r for r in read_rows(pairs)}
        m = {r["id"]: r for r in read_rows(metricx)}
        out[name] = [{**r, "metricx_qe": m.get(i, {}).get("qe_source_hypothesis"),
                      "metricx_ref": m.get(i, {}).get("ref_source_hypothesis_reference")} for i, r in p.items()]
    result = smoke_verdict(summary, adapter, args.checkpoint_ok, lora, out["arm"], out["base"])
    result["loss"] = summary.get("loss")
    result["throughput"] = summary.get("throughput")
    result["times"] = summary.get("times")
    result["epoch"] = summary.get("epoch")
    result["metricx_ref"] = {k: mean(r["metricx_ref"] for r in v) for k, v in out.items()}
    result["identical_translations"] = sum(
        a["hypothesis"] == b["hypothesis"] for a, b in zip(sorted(out["arm"], key=lambda r: r["id"]),
                                                           sorted(out["base"], key=lambda r: r["id"])))
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    L = ["# Smoke-test G4-1 (protocol-g4.md §6)", "", f"**{'Geslaagd' if result['passed'] else 'Niet geslaagd'}.**", "",
         "| controle | waarde | gehaald |", "|---|---|---|"]
    for k, c in result["checks"].items():
        L.append(f"| {k} | `{json.dumps(c['value'], default=str)[:160]}` | {'ja' if c['holds'] else 'nee'} |")
    L += ["", f"Gelijke vertalingen G4-1 (stap 8) en G4-0: {result['identical_translations']} van {SMOKE_N}. "
          f"MetricX-ref: {json.dumps(result['metricx_ref'])}.", "",
          f"Training: {json.dumps({k: result[k] for k in ('throughput', 'times', 'epoch')}, default=str)}", ""]
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"SMOKE {'PASSED' if result['passed'] else 'FAILED'} {json.dumps({k: c['holds'] for k, c in result['checks'].items()})}",
          flush=True)
    if not result["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--bt", required=True, help="fetch's directory (<source>/data/train-*.parquet, revisions.json)")
    p.add_argument("--books", required=True, help="RL's set ($S/data/rl-b5/RL: train.parquet, train-meta.parquet)")
    p.add_argument("--match-set", required=True, help="R2's set ($S/data/r2/R2), whose real examples are the target")
    p.add_argument("--evalsets", required=True, help="$E/data/evalsets.parquet (the four sets)")
    p.add_argument("--blind", required=True, help="blind100.jsonl")
    p.add_argument("--flan-eval", default=None, help="mainset/flan-eval/eval-rows.parquet (with a replay)")
    p.add_argument("--tokenizer", required=True, help="Gemma 4's tokenizer directory")
    p.add_argument("--replay-share", type=float, default=0.0)
    p.add_argument("--replay", default=None, help="a Gemma 4 FLAN-nl replay parquet (id, en, nl)")
    p.add_argument("--replay-model", default=None)
    p.add_argument("--fineweb-max-crawl", default=None, help="YYYY or CC-MAIN-YYYY-WW, inclusive (default: none)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p = sub.add_parser("evaldata")
    p.add_argument("--four", required=True, help="$E/data/evalsets.parquet")
    p.add_argument("--x", required=True, help="$S/arms/data/x/data/evalsets.parquet (FLAN, blind)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("texts")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--pairs", action="append", required=True, help="NAME=glob of pairs-w*.parquet")
    p.add_argument("--control", required=True, help="$S/re/classify/g4-base-texts.parquet")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--arm", required=True, help="local copy of G4-1's output (pairs/, scores/)")
    p.add_argument("--base", required=True, help="local copy of G4-0's output")
    p.add_argument("--tg-scores", required=True, help="$E/baseline-tg27b/scores.parquet")
    p.add_argument("--predictions", required=True)
    p.add_argument("--classifier-results", default=None)
    p.add_argument("--lora-line", action="append", default=[])
    p.add_argument("--train-summary", default=None)
    p.add_argument("--tally", default=None, help="tally's json after the blind review")
    p.add_argument("--out", required=True)
    p = sub.add_parser("tally")
    p.add_argument("--judgments", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--arm", default=ARM)
    p.add_argument("--out", default=None)
    p = sub.add_parser("smoke")
    for name in ("summary", "adapter", "translate-log", "arm-pairs", "arm-metricx", "base-pairs", "base-metricx", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--checkpoint-ok", action="store_true")
    args = parser.parse_args(argv)
    {"fetch": fetch, "build": build, "check": check, "evaldata": evaldata, "texts": texts, "report": report,
     "tally": tally, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
