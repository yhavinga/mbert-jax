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
Arm G4-2 (protocol-g4.md §9, amendment G4-2):
  fetch --split validation   the five datasets' validation split at the same revisions
  heldout   the held-out data: whole books out of G4-1's set, the BT validation, as evaluation segments (classes
            tiny-xlong, whole classes per worker) and as eval files of the training (messages)
  own       the eval file of G4-0's own greedy translations of the BT validation (the template and stack check)
  replay    the replay candidates from the Gemma 4 FLAN-nl after its post step (segments of kept examples, measures)
  build2    G4-2's set: G4-1's examples without the held-out books, the FLAN replay, three shuffled passes
  smoke2    the verdict of G4-2's smoke test (8 steps with the held-out evals; the own-output loss at step 0)
Arm G4-3 (protocol-g4.md §9, amendment G4-3):
  fetch --arm G4-3 [--split validation]   the five datasets at their revisions with the long rows (BT_REPOS3)
  heldout3  the long validation rows (more than 512 Dutch tokens) as evaluation segments of gemma4-31b-long and as an
            eval file; G4-2's held-out BT validation checked against the new revisions
  replay --pool N --pool-seed S   a larger replay pool, so that every pass gets replay rows of its own
  build3    G4-3's set: every BT train row without hard_fails (up to 2,048 Dutch tokens), the books of G4-2, 10% FLAN
            replay apart per pass, three shuffled passes; the batches of MaxText's packing per process, the smoke set
  check --arm G4-3   MaxText's own SFT steps on every distinct example, in processes
  smoke3    the verdict of G4-3's smoke test and the estimate of the full training's time (bound: 3 days)
Arm G4-4 (protocol-g4.md §9, amendment G4-4):
  build4    G4-4's set: G4-3's set pass by pass as it is, plus at most 20 pairs per English novel per pass (-b1-, group
            en_orig, arm RE's checks, one Dutch edition per book and one English id per work, near-duplicate
            passages out, no 13-gram of the evaluation sets) apart in every pass, and the replay that keeps 10% of
            every pass (the next rows of G4-3's replay order); three passes shuffled anew; packing, steps, smoke set
  check --arm G4-4   as G4-3's
  smoke4    the verdict of G4-4's smoke test (G4-3's checks, five eval sets) and the estimate of three passes (above
            120 h: exit 1)
  report --other G4-3=DIR   beside G4-0 the arm against G4-3 on the same measures (reported, deciding nothing)
  novels4   amendment of 16 Nov: eval-novels.parquet, unseen passages of seen novels (10 books of 10 authors x 50
            pairs that G4-4 does not train on, build4's checks; no key or example of G4-4's training set)
  report --novels-other SUMMARY   the novels' losses of G4-0, G4-4 and G4-3's final model (reported, deciding nothing)
Arm G4-5 (draft amendment G4-5, Yeb 17 Nov: one LoRA per source, branched from a shared warmup, combined by weights):
  build5    from G4-4's set: the shared warmup's set (pass 1 without the replay, as many as its steps take) and one set
            per branch (BT per source, the Dutch books, the novels with G4-4's passes as epochs, FLAN without flanval),
            every epoch freshly shuffled, as many rows as the branch's steps take; eval-flanval; the smoke sets
  split5    the search and decision parts of WMT24++, FLORES, NTREX and FLAN (seeded, per domain; FLORES per article;
            the blind passages' WMT24++ segments in the decision part), the search part's reference eval files
  smoke5    the verdict of G4-5's smoke test (warmup, a resumed branch, the search on G4-3's adapter) and its estimate
  report5   the training per branch, the search, the full evaluation of its top K on the search part, the selection by
            the fixed rule, (a1) and (b) of the selected one against G4-0 on the decision part
Arm G4-6 (draft amendment G4-6: preference training, DPO with LoRA, G4-0 as the reference; g4dpo.py):
  cands6    the candidates of the pairs from G4-4's set (books, novels, BT per source) and the novels of the eval set
            novels, as the rejected run's input (G4-0 translates them: translate SET=dpo)
  build6    the pairs (chosen = the human Dutch, rejected = G4-0's greedy translation; G4-0's failures, identical pairs,
            the chrF band, the length ratio, 13-grams, the token bound out), drawn per stratum, packed at 4,096 and
            2,048; the smoke sets; the preference eval sets pref_books, pref_btval, pref_novels; report
  check6    MaxText's own SFT steps on every side of every pair, and every packed row against MaxText's shift
  smoke6    the verdict of G4-6's smoke test (the policy is the reference at step 0, the base losses, memory, the
            translation of 50 segments) and the estimate of the full run
  pairtexts6, pairclass6   the classifier on chosen and rejected per kind (reported, decides nothing)
  report --train-summary of a --dpo run: the preference measures beside the rules
Arm G4-7 (amendment G4-7, protocol-g4.md §14: online RL, GRPO with LoRA from G4-0; g4rl.py), what comes after the RL:
  records7  the chain's records.json (gs:// paths of every iteration's report.json, stop.json and summary.json, the
            stop, smoke7, c2cal7) as the copies into report7's --records layout
  report7   the candidate chosen on the search part (G4-7.7: select5), the rule (a1), (a3), (b) against G4-0 on its
            decision part (G4-7.8), (a2) and (c) only after those three; reported: C2 beside C1 per set ("C2 rises,
            C1 not"), MetricX-ref, COMET-22, length, failures, MTLD, PLL (with --pll), every iteration of the training,
            the stop, c2cal7 and smoke7
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

import g4dpo  # noqa: E402
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


def write_parquet(rows, path, schema=None, row_group_size=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd", row_group_size=row_group_size)


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
    from huggingface_hub import HfApi, snapshot_download

    out = Path(args.out)
    split = getattr(args, "split", "train")
    arm = getattr(args, "arm", None) or ARM
    repos = BT_REPOS3 if arm == ARM3 else BT_REPOS
    info = {}
    for source, (repo, revision) in repos.items():
        d = out / source
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=str(d),
                          allow_patterns=[f"data/{split}-*.parquet", "README.md"])
        files = sorted(d.glob(f"data/{split}-*.parquet"))
        if not files:
            raise SystemExit(f"{repo}@{revision}: no data/{split}-*.parquet")
        info[source] = {"repo": repo, "revision": revision,
                        "files": {f.name: sha256_file(f) for f in files}}
        if arm == ARM3:
            # every file as the Hub's LFS record of that revision has it, and no file of the split left out
            lfs = {Path(x.rfilename).name: x.lfs.sha256 for x in HfApi().dataset_info(
                repo, revision=revision, files_metadata=True).siblings
                   if x.rfilename.startswith(f"data/{split}-") and x.lfs}
            if lfs != info[source]["files"]:
                raise SystemExit(f"{repo}@{revision}: the files of {split} differ from the Hub's LFS sha256: "
                                 f"{sorted(lfs)} against {sorted(info[source]['files'])}")
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


def read_bt(directory, revisions, split="train", extra=()):
    """The rows of a split of the five datasets as {pid, source, nl, en, hard_fails, chrf, model, dump} and the extra
    columns; every file's sha256 must be the one fetch recorded"""
    import pyarrow.parquet as pq

    rows = []
    for source in BT_REPOS:
        d = Path(directory) / source / "data"
        files = sorted(d.glob(f"{split}-*.parquet"))
        want = revisions[source]["files"]
        if sorted(f.name for f in files) != sorted(want):
            raise SystemExit(f"{d}: files {[f.name for f in files]}, revisions.json has {sorted(want)}")
        for f in files:
            if sha256_file(f) != want[f.name]:
                raise SystemExit(f"{f}: sha256 differs from revisions.json")
            names = pq.read_schema(f).names
            columns = BT_COLUMNS + (["dump"] if source == "fineweb" else []) + list(extra)
            missing = [c for c in columns if c not in names]
            if missing:
                raise SystemExit(f"{f}: no column(s) {missing}")
            for r in pq.read_table(f, columns=columns).to_pylist():
                rows.append({"pid": r["id"], "source": source, "nl": r["nl"], "en": r["en"],
                             "hard_fails": list(r["hard_fails"] or []), "chrf": r["chrf"], "model": r["model"],
                             "dump": r.get("dump"), **{c: r[c] for c in extra}})
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
        return [len(x) for x in self.ids(texts)]

    def ids(self, texts):
        """The token ids of each text, without special tokens (as MaxText's SFT tokenizes prompt and completion)"""
        out = []
        texts = list(texts)
        for k in range(0, len(texts), 4096):
            out += [list(e.ids) for e in self.tok.encode_batch(texts[k:k + 4096], add_special_tokens=False)]
        return out

    def special_id(self, token):
        return self.tok.token_to_id(token)

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
    if getattr(args, "arm", None) in (ARM3, ARM4):
        return check3(args)
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

    arm = getattr(args, "arm_name", None) or ARM
    evalsets = {r["id"]: r for r in read_rows(args.evalsets)}
    models = {arm: scored_dir(args.arm), BASE: scored_dir(args.base)}
    other = None
    if getattr(args, "other", None):
        other, other_dir = args.other.split("=", 1)
        models[other] = scored_dir(other_dir)
    # amendment G4-6b.7: further arms beside the arm on the same measures and in the same classify run, reported
    beside = []
    for spec in getattr(args, "beside", None) or []:
        name, beside_dir = spec.split("=", 1)
        if name in models:
            raise SystemExit(f"--beside {name}: already among {sorted(models)}")
        models[name] = scored_dir(beside_dir)
        beside.append(name)
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
    for name in models:
        if shares.get(name, {}).get("n") != N_WMT_GOOD + N_FLAN:
            raise SystemExit(f"{args.predictions}: {name} has {shares.get(name, {}).get('n')} texts")
    accuracy = json.loads(Path(args.classifier_results).read_text())["test_accuracy"] if args.classifier_results else None
    result = {"arm": arm, "base": BASE, "a1": rule_a1(sets[arm]), "a1_base": rule_a1(sets[BASE]),
              "b": rule_b(shares, arm), "b_against_tg": rule_b(shares, arm, "TG") if "TG" in shares else None,
              "b_base_against_tg": rule_b(shares, BASE, "TG") if "TG" in shares else None,
              "shares": shares, "classifier_accuracy": accuracy, "sets": {}}
    for s in ("wmt24pp", "wmt24pp_all", "flores", "ntrex", "europarl", "flan", "blind"):
        entry = {}
        for name in models:
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
        result["training"]["lr_at"] = t.get("lr_at")
        result["training"]["base_adapter"] = t.get("base_adapter")
        if t.get("dpo"):
            # amendment G4-6.7: the preference measures, reported
            result["pref"] = pref_measure(t)
    if getattr(args, "heldout", None):
        t = json.loads(Path(args.train_summary).read_text()) if args.train_summary else {}

        def many(value):
            return [value] if isinstance(value, str) else list(value)

        def pairs_of(patterns):
            rows = {}
            for pattern in many(patterns):
                files = sorted(glob.glob(pattern))
                if not files:
                    raise SystemExit(f"no held-out pairs in {pattern}")
                for f in files:
                    rows.update((r["id"], r) for r in read_rows(f))
            return rows

        held = [r for path in many(args.heldout) for r in read_rows(path)]
        if len({r["id"] for r in held}) != len(held):
            raise SystemExit("an id twice among the held-out files")
        result["heldout"] = heldout_measure(held, pairs_of(args.heldout_arm), pairs_of(args.heldout_base),
                                            t.get("eval"), arm, BASE)
        # amendment G4-6b: step 0 of the training is the start model with its adapter merged (G4-4), not G4-0
        result["heldout"]["start"] = (t.get("base_adapter") or {}).get("name") or BASE
        if other and getattr(args, "heldout_other", None):
            result["other_heldout"] = heldout_measure(held, pairs_of(args.heldout_arm), pairs_of(args.heldout_other),
                                                      t.get("eval"), arm, other)
    if other:
        # amendment G4-4.7: the arm beside another arm on the same measures, reported, deciding nothing
        o = {"name": other, "a1": rule_a1(sets[other]), "b": rule_b(shares, arm, other)}
        if getattr(args, "other_summary", None):
            evals = json.loads(Path(args.other_summary).read_text()).get("eval") or []
            curve = collections.defaultdict(dict)
            for e in evals:
                curve[e["set"]][e["step"]] = e["loss"]
            o["loss"] = {k: dict(sorted(v.items())) for k, v in sorted(curve.items())}
        if getattr(args, "other_tally", None):
            o["blind_vs_base"] = json.loads(Path(args.other_tally).read_text())
        result["other"] = o
    if getattr(args, "novels_other", None) or (other and args.train_summary):
        # amendment G4-4.11: the novels (unseen passages of seen novels), reported only
        t = json.loads(Path(args.train_summary).read_text()) if args.train_summary else {}
        o = json.loads(Path(args.novels_other).read_text()) if getattr(args, "novels_other", None) else None
        n = novels_measure(t, o, arm, BASE, other or ARM3)
        if n["loss"] or "other_loss" in n:
            result["novels"] = n
    if beside:
        result["beside"] = {name: {"a1": rule_a1(sets[name]), "b": rule_b(shares, arm, name)} for name in beside}
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
        L.append(f"Nog niet beoordeeld (pakket `$S/blind/{r['arm']}/`; alleen als (a1) en (b) gelden, protocol-g4.md §5).")
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
    if r.get("heldout"):
        L += heldout_text(r["heldout"], r["arm"], r["base"])
    if r.get("other"):
        L += other_text(r)
    if r.get("beside"):
        L += beside_text(r)
    if r.get("novels"):
        L += novels_text(r["novels"])
    if r.get("pref"):
        L += pref_text(r["pref"])
    if r.get("lora_merge"):
        L += ["", "LoRA-samenvoeging (`translate.py`, regel LORA): " + "; ".join(json.dumps(x) for x in r["lora_merge"][:4])]
    if r.get("training"):
        L += ["", "Training: " + json.dumps(r["training"], default=str)]
    if r.get("classifier_accuracy") is not None:
        L += ["", f"Nauwkeurigheid van de classifier: {r['classifier_accuracy']}."]
    return "\n".join(L) + "\n"


def heldout_text(h, arm, base):
    """The direct measure (amendment G4-2.6) as Markdown: reported, it decides nothing"""
    L = ["", "## De directe maat: dichter bij het originele Nederlands? (amendement G4-2.6, rapporteren, beslist niet)", "",
         *(["De lange validation-rijen (`btlong`, meer dan 512 Nederlandse tokens, elk heel vertaald; amendement "
            "G4-3.6) staan erbij en tellen niet mee in de lezing.", ""] if h.get("reported") else []),
         f"**Lezing (vooraf vastgelegd): {h['verdict']}.**", "",
         f"| set | verlies stap 0 ({h.get('start', BASE)}) | verlies laatste stap | lager | chrF " + arm + " | chrF " + base + " | hoger |",
         "|---|---:|---:|---|---:|---:|---|"]
    for name, v in list(h["reading"].items()) + list((h.get("reported") or {}).items()):
        sets = h["sets"][name]
        label = name if name in h["reading"] else f"{name} (rapporteren, buiten de lezing)"
        L.append(f"| {label} | {f3(v['loss_step0'])} | {f3(v['loss_last'])} (stap {v['last_step']}) | "
                 f"{'ja' if v['loss_down'] else 'nee'} | {f2(sets[arm]['chrf_corpus'])} | {f2(sets[base]['chrf_corpus'])} | "
                 f"{'ja' if v['chrf_up'] else 'nee'} |")
    L += ["", "| set | model | n | chrF (corpus) | chrF (gem.) | uitvoer / origineel (tekens) | niet af | lus | leeg |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in h["sets"].items():
        for model, v in entry.items():
            L.append(f"| {name} | {model} | {v['n']} | {f2(v['chrf_corpus'])} | {f2(v['chrf_mean'])} | "
                     f"{f3(v['chars_out_per_ref'])} | {v['not_ended']} | {v['decode_loop']} | {v['empty']} |")
    L += ["", "Verlies per stap (token-gewogen, het Nederlands en `<turn|>\\n`; stap 0 = het basismodel in de "
          "trainingsstack; `own` = G4-0's eigen greedy vertalingen als doel): " + "; ".join(
              f"{name} " + ", ".join(f"{k}: {f3(x)}" for k, x in curve.items()) for name, curve in h["loss"].items())]
    return L


def f2(x):
    return "–" if x is None else f"{x:.2f}".replace(".", ",")


def novels_measure(train_summary, other_summary, arm, base, other):
    """amendment G4-4.11: the novels' loss of the base (the arm's step 0), the arm per eval step and after its last, and
    the other arm's final model (step 0 of its run with --init-adapter); the same batches when the target tokens agree"""
    def curve(summary):
        return {e["step"]: e for e in (summary or {}).get("eval") or [] if e["set"] == "novels"}

    mine, theirs = curve(train_summary), curve(other_summary)
    out = {"name": "unseen passages of seen novels", "arm": arm, "base": base, "other": other,
           "loss": {k: v["loss"] for k, v in sorted(mine.items())}}
    if mine:
        last = max(mine)
        out.update({"base_loss": mine[0]["loss"] if 0 in mine else None, "arm_loss": mine[last]["loss"],
                    "arm_step": last, "tokens": mine[last]["tokens"]})
    if theirs:
        if 0 not in theirs:
            raise SystemExit(f"{other}'s novels run has no eval at step 0")
        out.update({"other_loss": theirs[0]["loss"], "other_tokens": theirs[0]["tokens"],
                    "other_adapter": (other_summary or {}).get("init_adapter")})
    if mine and theirs:
        out["same_batches"] = mine[min(mine)]["tokens"] == theirs[0]["tokens"]
    return out


def novels_text(n):
    """amendment G4-4.11 as Markdown: reported, it decides nothing"""
    arm, base, other = n["arm"], n["base"], n["other"]
    L = ["", "## Romanpassages: unseen passages of seen novels (amendement G4-4.9–G4-4.11, rapporteren, beslist niet)", "",
         "500 nieuwe passages uit 10 romans (10 auteurs) die in de training van G4-4 zitten (elk boek met ten hoogste 60 "
         "andere paren). Dit zijn geen held-out romans: boeken, personages en vertalers zijn gezien, dus het verlies is "
         "optimistischer dan op onbekende boeken. De set telt niet mee in (a1), (a2), (b), (c) of de directe maat.", "",
         "| model | verlies (token-gewogen) | stap | doeltokens |", "|---|---:|---|---:|",
         f"| {base} | {f3(n.get('base_loss'))} | stap 0 van {arm} | {n.get('tokens', '–')} |",
         f"| {arm} | {f3(n.get('arm_loss'))} | {n.get('arm_step', '–')} (laatste) | {n.get('tokens', '–')} |",
         f"| {other} | {f3(n.get('other_loss'))} | eindmodel (run `g4arm_{other}_novels`, stap 0) | "
         f"{n.get('other_tokens', '–')} |"]
    if "same_batches" in n:
        L += ["", f"Dezelfde eval-batches voor {arm} en {other} (gelijk aantal doeltokens): "
              f"{'ja' if n['same_batches'] else 'nee'}."]
    if n.get("loss"):
        L += ["", f"Verlies van {arm} per stap: " + ", ".join(f"{k}: {f3(x)}" for k, x in n["loss"].items()) + "."]
    return L


def beside_text(r):
    """amendment G4-6b.7: the arms beside the arm, (a1) and (b) from the same run, reported, deciding nothing"""
    arm = r["arm"]
    L = ["", "## Daarnaast (rapporteren, beslist niet)", "",
         "| model | " + " | ".join(f"QE {s}" for s in HUMAN_QE) + f" | (a1) | \"origineel\" | {arm} − model |",
         "|---|" + "---:|" * len(HUMAN_QE) + "---|---:|---:|"]
    for name, v in r["beside"].items():
        L.append(f"| {name} | " + " | ".join(f3(v["a1"]["sets"][s]["model"]) for s in HUMAN_QE) +
                 f" | {'gehaald' if v['a1']['holds'] else 'niet gehaald'} | {pct(v['b']['base'])} | "
                 f"{f3(v['b']['delta'], True)} |")
    return L


def other_text(r):
    """amendment G4-4.7: the arm against another arm (G4-3) on the same measures, reported, deciding nothing"""
    o, arm = r["other"], r["arm"]
    name = o["name"]
    L = ["", f"## Vergelijking met {name} (rapporteren, beslist niet)", "",
         f"| set | n | {arm} QE | {name} QE | verschil | mens |", "|---|---:|---:|---:|---:|---:|"]
    for s, v in r["a1"]["sets"].items():
        w = o["a1"]["sets"][s]
        diff = None if v["model"] is None or w["model"] is None else v["model"] - w["model"]
        L.append(f"| {s} | {v['n']} | {f3(v['model'])} | {f3(w['model'])} | {f3(diff, True)} | {f3(v['human'])} |")
    L += ["", f"(a1) voor {name}: {'gehaald' if o['a1']['holds'] else 'niet gehaald'}. (b) in dezelfde classify-run: "
          f"{arm} {pct(o['b']['arm'])}, {name} {pct(o['b']['base'])}, verschil {f3(o['b']['delta'], True)}."]
    h = r.get("other_heldout")
    if h:
        L += ["", f"| set | chrF {arm} | chrF {name} | {arm} hoger | verlies laatste stap {arm} | verlies laatste stap {name} |",
              "|---|---:|---:|---|---:|---:|"]
        own_loss = (r.get("heldout") or {}).get("loss") or {}
        for set_name, entry in h["sets"].items():
            mine, theirs = entry[arm], entry[name]
            base_set = set_name.split(":")[0]
            a = own_loss.get(base_set) or {}
            b = (o.get("loss") or {}).get(base_set) or {}
            last_a = a[max(a)] if a and ":" not in set_name else None
            last_b = b[max(b)] if b and ":" not in set_name else None
            L.append(f"| {set_name} | {f2(mine['chrf_corpus'])} | {f2(theirs['chrf_corpus'])} | "
                     f"{'ja' if mine['chrf_corpus'] > theirs['chrf_corpus'] else 'nee'} | {f3(last_a)} | {f3(last_b)} |")
    if o.get("blind_vs_base"):
        x = o["blind_vs_base"]
        L += ["", f"Blinde beoordeling van {name} tegen {r['base']} (zijn eigen pakket): {name} beter {x['arm_better']}, "
              f"{r['base']} beter {x['base_better']}, gelijk {x['equal']}; fouten {x['errors_arm']} tegen {x['errors_base']}."]
    return L


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


# ---------------------------------------------------------------------------------------------------- G4-2

ARM2 = "G4-2"
SEED2 = f"{SEED}/G4-2"
EPOCHS2 = 3
REPLAY_SHARE2 = 0.20                 # amendment G4-2.3 point 2: arm N's share
REPLAY_POOL2 = 30000
REPLAY_TOKENS2 = (48, 384)           # amendment G4-2.3 point 4 (iii): arm N's window, in Gemma 4 tokens of the Dutch
REPLAY_MODEL2 = "google/gemma-4-31B-it@842da37 (prompt flan-v1, MaxText fork f4d8615, flan-nl fd5e585)"
# the bucket places of the Gemma 4 FLAN-nl after its post step (flan_g4_full_run.sh); nothing else is a replay source
REPLAY_URIS2 = {"plan": "/mbert/flan-nl/plan2", "shards": "/mbert/flan-nl/translations/g4",
                "measures": "/mbert/flan-nl/g4-full/report/measures.parquet", "nl": "/mbert/flan-nl/nl-g4",
                "setup": "/mbert/flan-nl/g4-full/report/setup.json"}
HELDOUT_BOOK_MAX_PAIRS = 600         # amendment G4-2.5: books with at most this many pairs
HELDOUT_BOOKS_MIN_PAIRS = 500
HELDOUT_BOOKS_MIN_BOOKS = 2
HELDOUT_BOOKS_SAMPLE = 500
HELDOUT_BTVAL = 800
DECODE_FIT = 0.8                     # the Dutch at most this share of its class's decode limit
OWN_MAX_LOSS = 1.0                   # amendment G4-2.4: the base model's loss on its own greedy output, at step 0
EVAL_SETS2 = ("books", "btval", "own")
SMOKE2_EVAL_STEPS = [0, 4, 8]
LR2 = 1e-4
BOOK_ID = re.compile(r"^book:books:(\d+-\d+)-b1-\d+$")


def book_key(meta_id):
    """The book of a book pair of G4-1's set: the pair of book ids in book:books:<nl>-<en>-b1-<k>"""
    m = BOOK_ID.match(meta_id)
    if not m:
        raise SystemExit(f"{meta_id}: not a book pair's id (book:books:<nl>-<en>-b1-<k>)")
    return m.group(1)


def choose_heldout_books(counts, seed=f"{SEED2}/heldout-books"):
    """amendment G4-2.5: the books with at most HELDOUT_BOOK_MAX_PAIRS pairs in a seeded order, taken until there are
    HELDOUT_BOOKS_MIN_PAIRS pairs and HELDOUT_BOOKS_MIN_BOOKS books"""
    order = sorted(k for k, n in counts.items() if n <= HELDOUT_BOOK_MAX_PAIRS)
    random.Random(seed).shuffle(order)
    chosen, total = {}, 0
    for k in order:
        if total >= HELDOUT_BOOKS_MIN_PAIRS and len(chosen) >= HELDOUT_BOOKS_MIN_BOOKS:
            break
        chosen[k] = counts[k]
        total += counts[k]
    if total < HELDOUT_BOOKS_MIN_PAIRS or len(chosen) < HELDOUT_BOOKS_MIN_BOOKS:
        raise SystemExit(f"the books with <= {HELDOUT_BOOK_MAX_PAIRS} pairs hold only {total} pairs in {len(chosen)}")
    return chosen


def decode_class(src_tokens, nl_tokens):
    """The class of a held-out pair if the Dutch fits its decode limit (DECODE_FIT), else None; above xlong None"""
    import translate

    classes = translate.MODELS[MODEL]["length_classes"]
    for name in EVAL_CLASSES:
        if src_tokens <= classes[name]["max_source"]:
            return name if nl_tokens <= DECODE_FIT * classes[name]["max_decode"] else None
    return None


def g41_set(directory):
    """G4-1's set as [(messages row, meta row)], checked against its manifest"""
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text())
    msgs = read_rows(d / "train.parquet")
    metas = read_rows(d / "train-meta.parquet")
    if manifest.get("set") != ARM or len(msgs) != len(metas) or len(metas) != manifest.get("examples"):
        raise SystemExit(f"{d}: not G4-1's set ({manifest.get('set')}, {len(msgs)} examples, {len(metas)} meta rows, "
                         f"manifest {manifest.get('examples')})")
    if (d / "template.jinja").read_text(encoding="utf-8") != g4lora.TEMPLATE:
        raise SystemExit(f"{d}/template.jinja is not g4lora.TEMPLATE")
    return list(zip(msgs, metas)), manifest


def messages_row(en, nl):
    return {"messages": [{"role": "user", "content": g4lora.user_turn(en)}, {"role": "assistant", "content": nl.strip()}]}


EVAL_SCHEMA = (("id", "string"), ("set", "string"), ("source", "string"), ("reference", "string"),
               ("tokens", "int64"), ("over_512", "bool"), ("length_class", "string"), ("meta", "string"))
SEGMENT_SCHEMA = (("id", "string"), ("text", "string"), ("tokens", "int64"))


def schema(fields):
    import pyarrow as pa

    return pa.schema([(n, getattr(pa, "bool_" if t == "bool" else t)()) for n, t in fields])


def heldout(args, counter=None):
    """amendment G4-2.5: the held-out books and BT validation, as evaluation segments and as eval files"""
    import evalsets

    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, manifest = g41_set(args.g41)
    books = [{"pid": m["id"], "key": book_key(m["id"]), "en": m["en"].strip(), "nl": x["messages"][1]["content"].strip()}
             for x, m in pairs if m["kind"] == "book"]
    counts = collections.Counter(b["key"] for b in books)
    chosen = choose_heldout_books(counts)
    report = {"books_in_g41": len(counts), "book_pairs_in_g41": len(books), "heldout_books": chosen,
              "heldout_book_pairs": sum(chosen.values())}

    def fitted(rows, label):
        src = counter.lengths(r["en"] for r in rows)
        nl = counter.lengths(r["nl"] for r in rows)
        keep, lost = [], 0
        for r, a, b in zip(rows, src, nl):
            cls = decode_class(a, b)
            if cls is None:
                lost += 1
            else:
                keep.append({**r, "tokens": a, "tokens_nl": b, "length_class": cls})
        report[f"{label}_not_fitting_the_classes"] = lost
        return keep

    candidates = fitted([b for b in books if b["key"] in chosen], "books")
    candidates.sort(key=lambda r: r["pid"])
    sample = sorted(random.Random(f"{SEED2}/heldout-books/sample").sample(
        candidates, min(HELDOUT_BOOKS_SAMPLE, len(candidates))), key=lambda r: r["pid"])
    report["books_measured"] = len(sample)

    revisions = json.loads((Path(args.btval) / "revisions.json").read_text())
    for source, (repo, revision) in BT_REPOS.items():
        got = revisions.get(source, {})
        if (got.get("repo"), got.get("revision")) != (repo, revision):
            raise SystemExit(f"{args.btval}/revisions.json: {source} is {got.get('repo')}@{got.get('revision')}")
    rows = read_bt(args.btval, revisions, split="validation")
    train_ids = {m["id"][len("bt:"):] for _, m in pairs if m["kind"] == "bt"}   # bt:<the dataset's id>
    overlap = sorted(train_ids & {r["pid"] for r in rows})
    if overlap:
        raise SystemExit(f"{len(overlap)} validation rows are training rows of G4-1, e.g. {overlap[:3]}")
    bounds = json.loads(Path(args.data_report).read_text())["bt"]["length_bounds"]
    evalsets_rows = read_rows(args.evalsets)
    blind = read_jsonl(args.blind)
    if sha256_file(args.blind) != blind_sha256():
        raise SystemExit(f"{args.blind}: not the blind passages of the TG protocol (§4.2.4)")
    blocked = blocked_grams(evalsets_rows, blind)
    funnel = collections.Counter(rows=len(rows))
    pool = []
    for r in rows:
        if r["hard_fails"]:
            funnel["hard_fails"] += 1
        elif sft_build.length_fails(r, bounds):
            funnel["length"] += 1
        elif shares_gram(r, blocked):
            funnel["eval_13gram"] += 1
        else:
            pool.append(r)
    prompts, targets = counter.examples(pool)
    within = []
    for r, a, b in zip(pool, prompts, targets):
        if a + b > MAX_TOKENS:
            funnel["too_long"] += 1
        else:
            within.append(r)
    fit = fitted(within, "btval")
    fit.sort(key=lambda r: r["pid"])
    if len(fit) < HELDOUT_BTVAL:
        raise SystemExit(f"{len(fit)} BT validation rows pass the bounds, {HELDOUT_BTVAL} wanted")
    btval = sorted(random.Random(f"{SEED2}/heldout-btval").sample(fit, HELDOUT_BTVAL), key=lambda r: r["pid"])
    funnel["eligible"] = len(fit)
    report["btval_funnel"] = dict(funnel)
    report["btval_by_source"] = dict(sorted(collections.Counter(r["source"] for r in btval).items()))

    eval_rows = []
    for name, part in (("books", sample), ("btval", btval)):
        for r in part:
            src = f"books:{r['key']}" if name == "books" else r["source"]
            eval_rows.append({"id": f"heldout:{name}:{r['pid']}", "set": name, "source": r["en"], "reference": r["nl"],
                              "tokens": r["tokens"], "over_512": r["tokens"] > 512, "length_class": r["length_class"],
                              "meta": json.dumps({"source": src, "pid": r["pid"], "tokens_nl": r["tokens_nl"]})})
        prompts, targets = counter.examples([{"en": r["en"], "nl": r["nl"]} for r in part])
        if max(a + b for a, b in zip(prompts, targets)) > MAX_TOKENS:
            raise SystemExit(f"a held-out {name} example above {MAX_TOKENS} tokens")
        write_parquet([messages_row(r["en"], r["nl"]) for r in part], out / f"eval-{name}.parquet")
    write_parquet(eval_rows, out / "evalsets.parquet", schema(EVAL_SCHEMA))
    assignment = evalsets.assign_workers(eval_rows, args.workers)
    for k, part in assignment.items():
        write_parquet([{"id": r["id"], "text": r["source"], "tokens": r["tokens"]} for r in part],
                      out / f"segments-w{k}.parquet", schema(SEGMENT_SCHEMA))
        (out / f"classes-w{k}.txt").write_text(
            " ".join(c for c in EVAL_CLASSES if any(r["length_class"] == c for r in part)) + "\n")
    report["workers"] = {f"w{k}": {"segments": len(p), "classes": sorted({r["length_class"] for r in p},
                                                                         key=EVAL_CLASSES.index)}
                         for k, p in assignment.items()}
    report["classes"] = {name: dict(collections.Counter(r["length_class"] for r in eval_rows if r["set"] == name))
                         for name in ("books", "btval")}
    report["inputs"] = {"g41": {f: sha256_file(Path(args.g41) / f) for f in ("train.parquet", "train-meta.parquet")},
                        "btval": revisions, "evalsets": sha256_file(args.evalsets), "blind": sha256_file(args.blind),
                        "length_bounds": bounds}
    (out / "books.json").write_text(json.dumps({"books": chosen, "pairs": sum(chosen.values())}, indent=1) + "\n")
    (out / "manifest.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"HELDOUT {json.dumps({k: report[k] for k in ('heldout_books', 'heldout_book_pairs', 'books_measured', 'btval_funnel', 'btval_by_source', 'workers')})}",
          flush=True)


def own(args, counter=None):
    """amendment G4-2.5 "eigen": G4-0's greedy translations of the BT validation as the targets of an eval file"""
    counter = counter or TokenCounter(args.tokenizer)
    english = {r["id"]: r for r in read_rows(args.heldout, ["id", "set", "source"])}
    rows = {}
    for f in sorted(glob.glob(args.pairs)):
        for r in read_rows(f, ["id", "set", "hypothesis", "ended", "decode_loop", "empty"]):
            rows[r["id"]] = r
    btval = sorted(i for i, r in english.items() if r["set"] == "btval")
    missing = [i for i in btval if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} BT validation segments without G4-0's translation, e.g. {missing[:3]}")
    kept = [i for i in btval if rows[i]["ended"] and not rows[i]["decode_loop"] and not rows[i]["empty"]]
    pairs = [{"en": english[i]["source"], "nl": rows[i]["hypothesis"].strip()} for i in kept]
    prompts, targets = counter.examples(pairs)
    pairs = [p for p, a, b in zip(pairs, prompts, targets) if a + b <= MAX_TOKENS]
    if len(pairs) < len(btval) // 2:
        raise SystemExit(f"only {len(pairs)} of {len(btval)} of G4-0's translations are usable as targets")
    write_parquet([messages_row(p["en"], p["nl"]) for p in pairs], args.out)
    info = {"btval": len(btval), "ended_ok": len(kept), "within_tokens": len(pairs)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(info) + "\n")
    print(f"OWN {json.dumps(info)}", flush=True)


def check_replay_uris(uris):
    """amendment G4-2.3 point 3: the replay comes only from the Gemma 4 FLAN-nl's places in the bucket"""
    for key, want in REPLAY_URIS2.items():
        got = str(uris.get(key) or "").rstrip("/")
        if not got.startswith("gs://") or not got.endswith(want):
            raise SystemExit(f"replay {key}: {got!r} is not the Gemma 4 FLAN-nl's {want}")


def replay_candidates(args):
    """amendment G4-2.3 point 4 (i)-(iii): the candidates of the replay from the Gemma 4 FLAN-nl after its post step,
    a seeded uniform pool of REPLAY_POOL2 of them, as a parquet for build2 (id, en, nl, task, model)"""
    import pyarrow.parquet as pq

    uris = dict(u.split("=", 1) for u in args.uri)
    check_replay_uris(uris)
    setup = Path(args.setup).read_text(encoding="utf-8")
    if "Gemma 4 31B-it" not in setup or "flan-v1" not in setup:
        raise SystemExit(f"{args.setup}: not the setup of the Gemma 4 FLAN-nl (Gemma 4 31B-it, flan-v1)")
    report = {"uris": uris}
    kept = set()
    for path in sorted(Path(args.nl).glob("w*.parquet")):
        h = int(path.stem[1:])
        kept |= {(h, int(p)) for p in pq.read_table(path, columns=["position"]).column("position").to_pylist()}
    if not kept:
        raise SystemExit(f"{args.nl}: no example kept by assemble")
    plan = json.loads((Path(args.plan) / "plan.json").read_text())
    template_ids = {int(v) for v in plan.get("template_segments", {}).values()}
    ex = pq.read_table(Path(args.plan) / "examples.parquet",
                       columns=["host", "position", "source", "status", "instruction", "answer"]).to_pydict()
    task = {}
    for h, p, source, status, ins, ans in zip(*(ex[c] for c in ("host", "position", "source", "status", "instruction",
                                                                 "answer"))):
        if status != "translate" or (int(h), int(p)) not in kept:
            continue
        for kind, value in json.loads(ins) + json.loads(ans):
            if kind == "S" and int(value) not in template_ids:
                task.setdefault(int(value), source)
    report["kept_examples"] = len(kept)
    report["segments_of_kept_examples"] = len(task)
    m = pq.read_table(args.measures, columns=["id", "check", "af", "executed", "chatter", "english_left",
                                              "identical_to_source", "repairs", "tokens_out"]).to_pydict()
    funnel = collections.Counter()
    lo, hi = REPLAY_TOKENS2
    passing = []
    for k in range(len(m["id"])):
        i = int(m["id"][k])
        if i not in task:
            continue
        funnel["measured"] += 1
        why = ("check" if m["check"][k] else "not_af" if not m["af"][k] else "executed" if m["executed"][k] else
               "chatter" if m["chatter"][k] else "english_left" if m["english_left"][k] else
               "identical" if m["identical_to_source"][k] else "repaired" if m["repairs"][k] else
               "tokens" if not lo <= int(m["tokens_out"][k]) <= hi else None)
        if why:
            funnel[why] += 1
        else:
            passing.append(i)
    if funnel["measured"] != len(task):
        raise SystemExit(f"{args.measures}: {funnel['measured']} of the {len(task)} segments of kept examples measured")
    passing.sort()
    g43 = getattr(args, "arm", None) == ARM3
    size = getattr(args, "pool", None) or (REPLAY_POOL3 if g43 else REPLAY_POOL2)
    seed = getattr(args, "pool_seed", None) or (f"{SEED3}/replay-pool" if g43 else f"{SEED2}/replay-pool")
    pool = sorted(random.Random(seed).sample(passing, min(size, len(passing))))
    report["pool_seed"] = seed
    want = set(pool)
    report["funnel"] = dict(funnel)
    report["candidates"] = len(passing)
    report["pool"] = len(pool)
    # the inputs as read (the Gemma 4 FLAN-nl's places can change later)
    report["sha256"] = {**{f"nl/{f.name}": sha256_file(f) for f in sorted(Path(args.nl).glob("w*.parquet"))},
                        "measures": sha256_file(args.measures), "setup": sha256_file(args.setup),
                        "plan/examples.parquet": sha256_file(Path(args.plan) / "examples.parquet"),
                        "plan/plan.json": sha256_file(Path(args.plan) / "plan.json")}
    en, nl = {}, {}
    for path in sorted(Path(args.plan).glob("segments-w*.parquet")):
        t = pq.read_table(path, columns=["id", "text"]).to_pydict()
        en.update((int(i), x) for i, x in zip(t["id"], t["text"]) if int(i) in want)
    for path in sorted(Path(args.shards).rglob("*.parquet")):
        t = pq.read_table(path, columns=["id", "text_nl"]).to_pydict()
        for i, x in zip(t["id"], t["text_nl"]):
            if int(i) in want:
                if int(i) in nl:
                    raise SystemExit(f"segment {i} translated twice ({path})")
                nl[int(i)] = x
    lost = [i for i in pool if i not in en or i not in nl]
    if lost:
        raise SystemExit(f"{len(lost)} pool segments without English or Dutch, e.g. {lost[:3]}")
    import pyarrow as pa

    write_parquet([{"id": i, "en": en[i].strip(), "nl": nl[i].strip(), "task": task[i], "model": REPLAY_MODEL2}
                   for i in pool], args.out,
                  pa.schema([("id", pa.int64()), ("en", pa.string()), ("nl", pa.string()), ("task", pa.string()),
                             ("model", pa.string())]))
    report["by_task_top"] = collections.Counter(task[i] for i in pool).most_common(15)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"REPLAY {json.dumps({k: report[k] for k in ('kept_examples', 'segments_of_kept_examples', 'funnel', 'candidates', 'pool')})}",
          flush=True)


def build2(args, counter=None):
    """amendment G4-2.3: G4-1's examples without the held-out books, the FLAN replay (20%), three shuffled passes"""
    import pyarrow as pa

    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    pairs, base_manifest = g41_set(args.base_set)
    books = json.loads(Path(args.heldout_books).read_text())["books"]
    keep = [(x, m) for x, m in pairs if not (m["kind"] == "book" and book_key(m["id"]) in books)]
    if len(pairs) - len(keep) != sum(books.values()):
        raise SystemExit(f"{len(pairs) - len(keep)} book pairs left out, the held-out books have {sum(books.values())}")
    report = {"arm": ARM2, "seed": SEED2, "prompt": PROMPT, "epochs": args.epochs, "replay_share": args.replay_share,
              "base_set": {"manifest": base_manifest, "examples": len(pairs)},
              "heldout_books": books, "heldout_book_pairs": sum(books.values())}
    evalsets_rows = read_rows(args.evalsets)
    blind = read_jsonl(args.blind)
    if sha256_file(args.blind) != blind_sha256():
        raise SystemExit(f"{args.blind}: not the blind passages of the TG protocol (§4.2.4)")
    blocked = blocked_grams(evalsets_rows, blind)
    candidates = replay_pool(args.replay, args.replay_model, args.flan_eval, blocked, counter, report)
    n_replay = replay_count(len(keep), args.replay_share)
    if n_replay > len(candidates):
        raise SystemExit(f"{n_replay} replay examples wanted, {len(candidates)} candidates")
    replay = random.Random(f"{SEED}/replay").sample(candidates, n_replay)
    report["replay"]["drawn"] = n_replay
    one = keep + [example(r, "replay") for r in replay]
    rows = []
    for e in range(1, args.epochs + 1):
        order = list(one)
        random.Random(f"{SEED2}/shuffle/{e}").shuffle(order)
        rows += [(x, {**m, "epoch": e}) for x, m in order]
    d = out / ARM2
    meta_schema = pa.schema([(n, getattr(pa, t)()) for n, t in META_SCHEMA_FIELDS + (("epoch", "int64"),)])
    write_parquet([x for x, _ in rows], d / "train.parquet")
    write_parquet([{n: m.get(n) for n in meta_schema.names} for _, m in rows], d / "train-meta.parquet", meta_schema)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    metas = [m for _, m in one]
    tokens = sum(m["tokens_prompt"] + m["tokens_target"] for m in metas)
    report["set"] = set_stats(metas, [])
    report["estimate"] = {"tokens_per_epoch": tokens, "tokens": tokens * args.epochs,
                          "steps": round(tokens * args.epochs / (16 * 2048 * PACK_FILL)), "pack_fill": PACK_FILL}
    inputs = {"base_set": {f: sha256_file(Path(args.base_set) / f) for f in ("train.parquet", "train-meta.parquet")},
              "heldout_books": sha256_file(args.heldout_books), "evalsets": sha256_file(args.evalsets),
              "blind": sha256_file(args.blind),
              "replay": {"path": str(args.replay), "sha256": sha256_file(args.replay), "model": args.replay_model}}
    report["inputs"] = inputs
    manifest = {"set": ARM2, "examples": len(rows), "examples_per_epoch": len(one), "epochs": args.epochs,
                "kinds": dict(collections.Counter(m["kind"] for m in metas)), "tokens": tokens * args.epochs,
                "prompt": PROMPT, "template": "g4lora.TEMPLATE (Gemma 4, thinking off)", "seed": SEED2,
                "replay_share": args.replay_share, "inputs": inputs}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    (out / "report.md").write_text(report2_md(report), encoding="utf-8")
    print(f"BUILD2 {json.dumps({'examples': len(rows), 'per_epoch': len(one), 'kinds': manifest['kinds'], 'tokens': tokens * args.epochs, 'steps_est': report['estimate']['steps']})}",
          flush=True)


def report2_md(r):
    s, e = r["set"], r["estimate"]
    L = [f"# Trainingsset {r['arm']} (protocol-g4.md §9, amendement G4-2)", "",
         f"Gemaakt door `g4arm.py build2`. De voorbeelden van G4-1 ({_n(r['base_set']['examples'])}) zonder de "
         f"held-out boeken ({_n(r['heldout_book_pairs'])} paren: {json.dumps(r['heldout_books'])}), plus "
         f"FLAN-replay {r['replay_share']:.0%} van de voorbeelden, {r['epochs']} doorgangen, elk apart geschud "
         f"(seed `{r['seed']}/shuffle/<k>`); prompt `{r['prompt']}`.", "",
         "## FLAN-replay", "", json.dumps(r["replay"], ensure_ascii=False), "",
         "## Eén doorgang", "", "| soort | voorbeelden | tokens | doeltokens | max tokens | woorden NL / EN p10 / p50 / p90 | bronnen |",
         "|---|---:|---:|---:|---:|---|---|"]
    for kind, k in s["kinds"].items():
        q = k["word_ratio"]
        src = k["by_source"] if kind != "replay" else {"flan-taken": len(k["by_source"])}
        L.append(f"| {kind} | {_n(k['examples'])} | {_n(k['tokens'])} | {_n(k['target_tokens'])} | {_n(k['max_tokens'])} | "
                 f"{q.get('p10')} / {q.get('p50')} / {q.get('p90')} | {json.dumps(src)} |")
    L += ["", f"Samen {_n(s['examples'])} voorbeelden per doorgang, {_n(e['tokens_per_epoch'])} tokens; "
          f"{r['epochs']} doorgangen: {_n(e['tokens'])} tokens, ~{_n(e['steps'])} stappen [schatting, vulling "
          f"{e['pack_fill']} zoals RL].", ""]
    return "\n".join(L) + "\n"


def smoke2_verdict(summary, adapter):
    """amendment G4-2.4: the smoke test of G4-2, every criterion with its value and whether it holds"""
    checks = {}
    loss = summary.get("loss") or []
    checks["steps"] = {"value": summary.get("steps_done"), "holds": summary.get("steps_done") == 8}
    checks["loss_finite"] = {"value": [v for _, v in loss],
                             "holds": len(loss) == 8 and all(math.isfinite(v) for _, v in loss)}
    checks["peak_gib"] = {"value": summary.get("peak_gib_max"),
                          "holds": summary.get("peak_gib_max") is not None and summary["peak_gib_max"] <= MEM_MAX_GIB}
    checks["adapter"] = {"value": {k: adapter.get(k) for k in ("rank", "alpha", "scale")},
                         "holds": (adapter.get("rank"), adapter.get("alpha"), adapter.get("scale")) == (32, 16.0, 0.5)}
    lrs = list((summary.get("lr_at") or {}).values())
    checks["lr_peak"] = {"value": max(lrs) if lrs else None,
                         "holds": bool(lrs) and abs(max(lrs) - LR2) <= 1e-9}
    evals = summary.get("eval") or []
    for name in EVAL_SETS2:
        rows = [e for e in evals if e["set"] == name]
        checks[f"eval_{name}"] = {
            "value": [(e["step"], round(e["loss"], 4), e["tokens"]) for e in rows],
            "holds": [e["step"] for e in rows] == SMOKE2_EVAL_STEPS
            and all(math.isfinite(e["loss"]) and e["tokens"] >= 1 for e in rows)}
    own0 = next((e["loss"] for e in evals if e["set"] == "own" and e["step"] == 0), None)
    checks["own_loss_step0"] = {"value": own0, "holds": own0 is not None and own0 <= OWN_MAX_LOSS + TOLERANCE}
    return {"checks": checks, "passed": all(c["holds"] for c in checks.values())}


def smoke2(args):
    summary = json.loads(Path(args.summary).read_text())
    adapter = json.loads(Path(args.adapter).read_text())
    result = smoke2_verdict(summary, adapter)
    result.update({k: summary.get(k) for k in ("loss", "eval", "eval_plan", "lr_at", "throughput", "times", "epoch")})
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    L = ["# Smoke-test G4-2 (protocol-g4.md §9, amendement G4-2.4)", "",
         f"**{'Geslaagd' if result['passed'] else 'Niet geslaagd'}.**", "", "| controle | waarde | gehaald |", "|---|---|---|"]
    for k, c in result["checks"].items():
        L.append(f"| {k} | `{json.dumps(c['value'], default=str)[:200]}` | {'ja' if c['holds'] else 'nee'} |")
    L += ["", "Verlies per held-out set (stap, verlies, doeltokens): " + "; ".join(
        f"{e['set']} {e['step']} {e['loss']:.4f} {e['tokens']}" for e in summary.get("eval") or []), "",
          f"Training: {json.dumps({k: summary.get(k) for k in ('throughput', 'times')}, default=str)}", ""]
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"SMOKE2 {'PASSED' if result['passed'] else 'FAILED'} {json.dumps({k: c['holds'] for k, c in result['checks'].items()})} "
          f"own0 {result['checks']['own_loss_step0']['value']}", flush=True)
    if not result["passed"]:
        raise SystemExit(1)


def heldout_measure(heldout_rows, arm_pairs, base_pairs, evals, arm, base):
    """amendment G4-2.6: chrF of both models against the original Dutch and the loss at step 0 against the last, per
    held-out set (BT also per source), and the reading fixed beforehand"""
    import armeval

    meta = {r["id"]: json.loads(r["meta"]) for r in heldout_rows}
    groups = collections.defaultdict(list)
    for r in heldout_rows:
        groups[r["set"]].append(r["id"])
        if r["set"] in ("btval", "btlong"):
            groups[f"{r['set']}:{meta[r['id']]['source']}"].append(r["id"])
    out = {"sets": {}}
    for name, ids in sorted(groups.items()):
        entry = {}
        for model, pairs in ((arm, arm_pairs), (base, base_pairs)):
            missing = [i for i in ids if i not in pairs]
            if missing:
                raise SystemExit(f"{model}: no held-out translation of {missing[:3]}")
            rows = [pairs[i] for i in ids]
            hyps = [r["hypothesis"] for r in rows]
            refs = [r["reference"] for r in rows]
            entry[model] = {"n": len(rows), "chrf_corpus": armeval.corpus_chrf(hyps, refs),
                            "chrf_mean": mean(armeval.chrf(h, x) for h, x in zip(hyps, refs)),
                            "not_ended": sum(not r["ended"] for r in rows),
                            "decode_loop": sum(bool(r["decode_loop"]) for r in rows),
                            "empty": sum(bool(r["empty"]) for r in rows),
                            "chars_out_per_ref": sum(len(h) for h in hyps) / max(1, sum(len(x) for x in refs))}
        out["sets"][name] = entry
    losses = collections.defaultdict(dict)
    for e in evals or []:
        losses[e["set"]][e["step"]] = e["loss"]
    out["loss"] = {k: dict(sorted(v.items())) for k, v in losses.items()}
    def read(name):
        curve = out["loss"].get(name, {})
        steps = sorted(curve)
        loss_down = len(steps) >= 2 and steps[0] == 0 and curve[steps[-1]] < curve[0]
        chrf_up = out["sets"][name][arm]["chrf_corpus"] > out["sets"][name][base]["chrf_corpus"]
        return {"loss_step0": curve.get(0), "loss_last": curve[steps[-1]] if steps else None,
                "last_step": steps[-1] if steps else None, "loss_down": loss_down, "chrf_up": chrf_up}

    reading = {name: read(name) for name in ("books", "btval")}
    flags = [v["loss_down"] for v in reading.values()] + [v["chrf_up"] for v in reading.values()]
    out["reading"] = reading
    # amendment G4-3: the long validation rows, reported beside the reading fixed for G4-2, not part of it
    out["reported"] = {name: read(name) for name in ("btlong",) if name in out["sets"]}
    out["verdict"] = ("dichter bij het originele Nederlands" if all(flags) else
                      "niet dichter" if not any(flags) else "gemengd")
    return out


# ---------------------------------------------------------------------------------------------------- G4-3

ARM3 = "G4-3"
SEED3 = f"{SEED}/G4-3"
EPOCHS3 = 3
REPLAY_SHARE3 = 0.10                 # amendment G4-3.3 (Yeb, 15 Nov): 10% of the examples
REPLAY_POOL3 = 120000                # every pass its own replay rows: 3 x ~28,800 out of this pool
LR3 = 1e-5                           # amendment G4-3.4 (Yeb, 15 Nov)
SEQ3 = 4096                          # max_target_length: the longest example (3,972 tokens, measured 15 Nov) fits
LONG_TOKENS = 512                    # tokens_nl > 512 marks the long rows (the datasets' cards)
EVAL_SETS3 = ("books", "btval", "own", "btlong")
EVAL_EVERY3 = 750
SMOKE3_EVAL_STEPS = [0, 4, 8]
SMOKE3_LONGEST = 256                 # the smoke set: the longest examples first (every step of the 8 is one of them)
SMOKE3_RANDOM = 512                  # then others, so that the pipeline's count can be checked against pack_batches
HOSTS, ROWS_PER_HOST = 4, 4          # a v4-32: 4 processes of 4 chips, per_device_batch_size 1
LONG_MODEL = "gemma4-31b-long"       # translate_models.yml: the classes of the long validation rows (en->nl)
MAX_HOURS3 = 120                     # Yeb (15 Nov ~08:05): up to ~5 days three passes; above, the first two only
# sft_lora.py counts the training data before the first step; G4-2: python_to_setup_s 212.5 s for 12.1 M tokens (its
# smoke test, G4-1's set) and 404.2 s for 41.2 M (the full run), so ~6.6 s per million tokens beyond the fixed part
COUNT_S_PER_MTOKEN = 6.6
# amendment G4-3.2: the five datasets after the publication of the long rows (15 Nov ~07:40 CET, one commit each; the
# columns region, vm and time dropped; Handelingen's card changed once more at 08:05, its data files byte for byte
# those of the publication); the train split holds the rows of up to 512 and of 513-2,048 Dutch tokens
BT_REPOS3 = {
    "handelingen": ("yhavinga/handelingen-tweede-kamer-nl-en-gemma4", "e4010d9534789927f14f364d3ee6aa3850dd66b8"),
    "raad": ("yhavinga/openraadsinformatie-nl-en-gemma4", "926d94adc3c67a0aec59d13f7bcc78dd94d4601a"),
    "dpc": ("yhavinga/rijksoverheid-websites-nl-en-gemma4", "958b56a33b65c2a21d081ae5e4ef503b2c8913eb"),
    "rechtspraak": ("yhavinga/rechtspraak-nl-en-gemma4", "09a8f8c811c99ab34b311d3d03147eb638d6e28c"),
    "fineweb": ("yhavinga/fineweb-2-nl-en-gemma4", "15dc526d3359af2fa8f94b17320c91420b66f1fa"),
}
META_SCHEMA3_FIELDS = META_SCHEMA_FIELDS + (("long", "bool"), ("epoch", "int64"))


def revisions3(directory):
    """fetch --arm G4-3's revisions.json, checked against BT_REPOS3"""
    revisions = json.loads((Path(directory) / "revisions.json").read_text())
    for source, (repo, revision) in BT_REPOS3.items():
        got = revisions.get(source, {})
        if (got.get("repo"), got.get("revision")) != (repo, revision):
            raise SystemExit(f"{directory}/revisions.json: {source} is {got.get('repo')}@{got.get('revision')}, "
                             f"amendment G4-3 pins {repo}@{revision}")
    return revisions


def checked_blocked(evalsets_path, blind_path):
    blind = read_jsonl(blind_path)
    if sha256_file(blind_path) != blind_sha256():
        raise SystemExit(f"{blind_path}: not the blind passages of the TG protocol (§4.2.4)")
    return blocked_grams(read_rows(evalsets_path), blind)


def pack_batches(lengths, hosts=HOSTS, rows=ROWS_PER_HOST, seq=SEQ3):
    """The batches per process of MaxText 0.2.5's SFT pipeline on one parquet file: process k reads examples k,
    k + hosts, ... (HFDataSource: split_dataset_by_node over hosts on a file of one shard) and packs them with grain's
    PackAndBatchOperation (max_segments_per_seq -1): an example goes to the first of the batch's rows it fits; one that
    fits none closes the batch and opens the next; the last batch counts."""
    out = []
    for h in range(hosts):
        n, free = 0, None
        for length in lengths[h::hosts]:
            if length > seq:
                raise SystemExit(f"an example of {length} tokens, more than the sequence length {seq}")
            if free is None:
                free, n = [seq] * rows, n + 1
            for i in range(rows):
                if free[i] >= length:
                    free[i] -= length
                    break
            else:
                free, n = [seq] * rows, n + 1
                free[0] -= length
        out.append(n)
    return out


def steps_of(batches):
    """sft_lora.py's step count of a multi-host job (STEPS=auto): step_limit of the batches per process"""
    import sft_lora

    return sft_lora.step_limit(batches)


def long_class(src_tokens, nl_tokens):
    """The class of gemma4-31b-long for a long validation row: the first whose max_source holds the English, if the
    Dutch fits DECODE_FIT of its decode limit (else None, as decode_class)"""
    import translate

    for name, c in translate.MODELS[LONG_MODEL]["length_classes"].items():
        if src_tokens <= c["max_source"]:
            return name if nl_tokens <= DECODE_FIT * c["max_decode"] else None
    return None


def assign_long_workers(rows, workers):
    """{worker: [rows]}: whole classes of gemma4-31b-long, the largest estimated time first, each to the least loaded
    worker (evalsets.assign_workers with this model's classes)"""
    import evalsets
    import translate

    order = list(translate.MODELS[LONG_MODEL]["length_classes"])
    by_class = collections.defaultdict(list)
    for r in rows:
        by_class[r["length_class"]].append(r)
    load = {k: 0.0 for k in range(workers)}
    out = {k: [] for k in range(workers)}
    for name in sorted(by_class, key=lambda c: (-evalsets.class_seconds(by_class[c]), order.index(c))):
        k = min(load, key=lambda w: (load[w], w))
        load[k] += evalsets.class_seconds(by_class[name])
        out[k] += by_class[name]
    return {k: sorted(v, key=lambda r: (r["tokens"], r["id"])) for k, v in out.items()}


def heldout3(args, counter=None):
    """amendment G4-3.6: the long validation rows (tokens_nl > 512) of the five datasets at BT_REPOS3, without
    hard_fails and 13-grams of the evaluation sets, as evaluation segments of gemma4-31b-long (whole classes per worker)
    and as an eval file of the training; G4-2's 800 held-out BT validation rows must be the same rows there"""
    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    revisions = revisions3(args.btval)
    rows = read_bt(args.btval, revisions, split="validation", extra=("tokens_nl",))
    by_pid = {r["pid"]: r for r in rows}
    old = [r for r in read_rows(args.heldout) if r["set"] == "btval"]
    if len(old) != HELDOUT_BTVAL:
        raise SystemExit(f"{args.heldout}: {len(old)} BT validation rows, G4-2 held out {HELDOUT_BTVAL}")
    for r in old:
        pid = json.loads(r["meta"])["pid"]
        new = by_pid.get(pid)
        if new is None or (new["en"], new["nl"]) != (r["source"], r["reference"]):
            raise SystemExit(f"{pid}: G4-2's held-out BT validation row differs in the revisions of amendment G4-3")
    blocked = checked_blocked(args.evalsets, args.blind)
    funnel = collections.defaultdict(collections.Counter)
    pool = []
    for r in rows:
        if r["tokens_nl"] <= LONG_TOKENS:
            continue
        funnel[r["source"]]["long_rows"] += 1
        if r["hard_fails"]:
            funnel[r["source"]]["hard_fails"] += 1
        elif shares_gram(r, blocked):
            funnel[r["source"]]["eval_13gram"] += 1
        else:
            pool.append(r)
    prompts, targets = counter.examples(pool)
    src = counter.lengths(r["en"] for r in pool)
    nl = counter.lengths(r["nl"] for r in pool)
    keep = []
    for r, a, b, n_src, n_nl in zip(pool, prompts, targets, src, nl):
        cls = long_class(n_src, n_nl)
        if a + b > SEQ3:
            funnel[r["source"]]["too_long"] += 1
        elif cls is None:
            funnel[r["source"]]["not_fitting_the_classes"] += 1
        else:
            keep.append({**r, "tokens": n_src, "tokens_nl_counted": n_nl, "length_class": cls})
            funnel[r["source"]]["eligible"] += 1
    if not keep:
        raise SystemExit("no long validation row passes the bounds")
    keep.sort(key=lambda r: r["pid"])
    eval_rows = [{"id": f"heldout:btlong:{r['pid']}", "set": "btlong", "source": r["en"], "reference": r["nl"],
                  "tokens": r["tokens"], "over_512": r["tokens"] > 512, "length_class": r["length_class"],
                  "meta": json.dumps({"source": r["source"], "pid": r["pid"], "tokens_nl": r["tokens_nl_counted"]})}
                 for r in keep]
    write_parquet(eval_rows, out / "evalsets.parquet", schema(EVAL_SCHEMA))
    write_parquet([messages_row(r["en"], r["nl"]) for r in keep], out / "eval-btlong.parquet")
    assignment = assign_long_workers(eval_rows, args.workers)
    import translate

    order = list(translate.MODELS[LONG_MODEL]["length_classes"])
    for k, part in assignment.items():
        write_parquet([{"id": r["id"], "text": r["source"], "tokens": r["tokens"]} for r in part],
                      out / f"segments-w{k}.parquet", schema(SEGMENT_SCHEMA))
        (out / f"classes-w{k}.txt").write_text(
            " ".join(c for c in order if any(r["length_class"] == c for r in part)) + "\n")
    manifest = {"set": "btlong", "rows": len(eval_rows), "model": LONG_MODEL,
                "funnel": {s: dict(c) for s, c in sorted(funnel.items())},
                "by_source": dict(sorted(collections.Counter(r["source"] for r in keep).items())),
                "classes": dict(collections.Counter(r["length_class"] for r in eval_rows)),
                "workers": {f"w{k}": {"segments": len(p), "classes": sorted({r["length_class"] for r in p},
                                                                             key=order.index)}
                            for k, p in assignment.items()},
                "btval_checked": len(old), "seq": SEQ3, "max_example_tokens": max(a + b for a, b in zip(prompts, targets)),
                "inputs": {"btval": revisions, "g4_2_heldout": sha256_file(args.heldout),
                           "evalsets": sha256_file(args.evalsets), "blind": sha256_file(args.blind)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"HELDOUT3 {json.dumps({k: manifest[k] for k in ('rows', 'funnel', 'by_source', 'classes', 'workers', 'btval_checked')})}",
          flush=True)


def held_pids(paths):
    """The dataset ids (meta pid) of the held-out BT rows of evalsets files (G4-2's btval, G4-3's btlong)"""
    pids = set()
    for path in paths:
        for r in read_rows(path, ["set", "meta"]):
            if r["set"] in ("btval", "btlong"):
                pids.add(json.loads(r["meta"])["pid"])
    return pids


def build3(args, counter=None):
    """amendment G4-3.3: every BT train row of BT_REPOS3 without hard_fails and without a 13-gram of the evaluation
    sets, the books of G4-2 (G4-1's book pairs without the held-out books), FLAN replay as 10% of the examples with rows
    of its own in every pass where the pool allows, three passes shuffled apart; the batches of MaxText's packing at
    SEQ3 and the steps; the smoke test's set"""
    import pyarrow as pa

    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    revisions = revisions3(args.bt)
    pairs, base_manifest = g41_set(args.base_set)
    books = json.loads(Path(args.heldout_books).read_text())["books"]
    book_pairs = [(x, m) for x, m in pairs if m["kind"] == "book"]
    kept_books = [(x, {**m, "long": False}) for x, m in book_pairs if book_key(m["id"]) not in books]
    if len(book_pairs) - len(kept_books) != sum(books.values()):
        raise SystemExit(f"{len(book_pairs) - len(kept_books)} book pairs left out, the held-out books have "
                         f"{sum(books.values())}")
    blocked = checked_blocked(args.evalsets, args.blind)
    print("PROGRESS reading the back-translations", flush=True)
    rows = read_bt(args.bt, revisions, split="train", extra=("tokens_nl",))
    held = held_pids(args.heldout)
    overlap = sorted(held & {r["pid"] for r in rows})
    if overlap:
        raise SystemExit(f"{len(overlap)} held-out validation rows are train rows, e.g. {overlap[:3]}")
    funnel = collections.defaultdict(collections.Counter)
    for r in rows:
        funnel[r["source"]]["rows"] += 1
        funnel[r["source"]]["long_rows"] += r["tokens_nl"] > LONG_TOKENS
    passed = []
    for r in rows:
        if r["hard_fails"]:
            funnel[r["source"]]["hard_fails"] += 1
        else:
            passed.append(r)
    # R2's length bound is not applied (amendment G4-3.3); what it would take is reported
    bounds = sft_build.length_bounds(passed, LEN_FACTOR)
    pool = []
    for r in passed:
        funnel[r["source"]]["outside_r2_length_bound_kept"] += sft_build.length_fails(r, bounds)
        if shares_gram(r, blocked):
            funnel[r["source"]]["eval_13gram"] += 1
        else:
            pool.append(r)
    print(f"PROGRESS {len(pool):,} back-translations, counting tokens", flush=True)
    prompts, targets = counter.examples(pool)
    too_long = [r["pid"] for r, a, b in zip(pool, prompts, targets) if a + b > SEQ3]
    if too_long:
        raise SystemExit(f"{len(too_long)} back-translations above {SEQ3} tokens, e.g. {too_long[:3]}: the sequence "
                         f"length of amendment G4-3 does not hold them")
    bt = []
    for r, a, b in zip(pool, prompts, targets):
        x, m = example({**r, "tokens_prompt": a, "tokens_target": b}, "bt")
        bt.append((x, {**m, "long": r["tokens_nl"] > LONG_TOKENS}))
        funnel[r["source"]]["eligible"] += 1
        funnel[r["source"]]["eligible_long"] += r["tokens_nl"] > LONG_TOKENS
    report = {"arm": ARM3, "seed": SEED3, "prompt": PROMPT, "epochs": args.epochs, "replay_share": args.replay_share,
              "seq": SEQ3, "bt": {"funnel": {s: dict(c) for s, c in sorted(funnel.items())},
                                  "r2_length_bounds_not_applied": bounds},
              "base_set": {"manifest": base_manifest, "examples": len(pairs)}, "heldout_books": books,
              "heldout_book_pairs": sum(books.values()), "books": len(kept_books)}
    candidates = replay_pool(args.replay, args.replay_model, args.flan_eval, blocked, counter, report)
    n_replay = replay_count(len(bt) + len(kept_books), args.replay_share)
    order = random.Random(f"{SEED3}/replay").sample(candidates, len(candidates))
    if not order:
        raise SystemExit("no replay candidate")
    distinct = min(len(order), args.epochs * n_replay)
    report["replay"].update({"per_pass": n_replay, "distinct": distinct,
                             "apart_in_every_pass": distinct == args.epochs * n_replay})
    passes = []
    for e in range(1, args.epochs + 1):
        replay = [order[k % len(order)] for k in range((e - 1) * n_replay, e * n_replay)]
        one = bt + kept_books + [(x, {**m, "long": False}) for x, m in (example(r, "replay") for r in replay)]
        random.Random(f"{SEED3}/shuffle/{e}").shuffle(one)
        passes.append(one)
    rows_out = [(x, {**m, "epoch": e}) for e, one in enumerate(passes, 1) for x, m in one]
    d = out / ARM3
    meta_schema = pa.schema([(n, getattr(pa, "bool_" if t == "bool" else t)()) for n, t in META_SCHEMA3_FIELDS])
    write_parquet([x for x, _ in rows_out], d / "train.parquet")
    write_parquet([{n: m.get(n) for n in meta_schema.names} for _, m in rows_out], d / "train-meta.parquet", meta_schema)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    lengths = [m["tokens_prompt"] + m["tokens_target"] for _, m in rows_out]
    batches = pack_batches(lengths)
    steps = steps_of(batches)
    # amendment G4-3.5: above MAX_HOURS3 the training takes the first two passes (the first rows of train.parquet)
    two = sum(len(one) for one in passes[:2])
    batches2 = pack_batches(lengths[:two])
    per_pass = []
    for e, one in enumerate(passes, 1):
        kinds = collections.defaultdict(lambda: {"examples": 0, "tokens": 0, "target_tokens": 0})
        for _, m in one:
            k = "bt_long" if m["kind"] == "bt" and m["long"] else m["kind"]
            kinds[k]["examples"] += 1
            kinds[k]["tokens"] += m["tokens_prompt"] + m["tokens_target"]
            kinds[k]["target_tokens"] += m["tokens_target"]
        per_pass.append({"pass": e, "examples": len(one), "tokens": sum(v["tokens"] for v in kinds.values()),
                         "kinds": dict(sorted(kinds.items()))})
    report["set"] = set_stats([m for _, m in passes[0]], [])
    report["passes"] = per_pass
    report["estimate"] = {"tokens": sum(lengths), "examples": len(lengths), "seq": SEQ3,
                          "batches_per_process": batches, "steps": steps,
                          "fill": round(sum(lengths) / (sum(batches) * ROWS_PER_HOST * SEQ3), 4),
                          "max_tokens": max(lengths),
                          "two_passes": {"examples": two, "tokens": sum(lengths[:two]), "batches_per_process": batches2,
                                         "steps": steps_of(batches2)}}
    inputs = {"bt": revisions, "base_set": {f: sha256_file(Path(args.base_set) / f) for f in ("train.parquet",
                                                                                            "train-meta.parquet")},
              "heldout_books": sha256_file(args.heldout_books), "heldout": {p: sha256_file(p) for p in args.heldout},
              "evalsets": sha256_file(args.evalsets), "blind": sha256_file(args.blind),
              "replay": {"path": str(args.replay), "sha256": sha256_file(args.replay), "model": args.replay_model}}
    report["inputs"] = inputs
    manifest = {"set": ARM3, "examples": len(rows_out), "examples_per_epoch": [p["examples"] for p in per_pass],
                "epochs": args.epochs, "tokens": sum(lengths), "seq": SEQ3, "steps": steps, "prompt": PROMPT,
                "template": "g4lora.TEMPLATE (Gemma 4, thinking off)", "seed": SEED3,
                "replay_share": args.replay_share, "inputs": inputs}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    # the smoke set (amendment G4-3.5): pass 1's longest examples first, then a seeded sample of the others
    first = list(passes[0])
    longest = sorted(first, key=lambda p: (-(p[1]["tokens_prompt"] + p[1]["tokens_target"]), p[1]["id"]))
    longest = longest[:SMOKE3_LONGEST]
    taken = {m["id"] for _, m in longest}
    others = sorted((p for p in first if p[1]["id"] not in taken), key=lambda p: p[1]["id"])
    smoke_rows = [(x, {**m, "epoch": 1}) for x, m in longest +
                  random.Random(f"{SEED3}/smoke").sample(others, min(SMOKE3_RANDOM, len(others)))]
    ds = out / f"{ARM3}-smoke"
    write_parquet([x for x, _ in smoke_rows], ds / "train.parquet")
    write_parquet([{n: m.get(n) for n in meta_schema.names} for _, m in smoke_rows], ds / "train-meta.parquet",
                  meta_schema)
    (ds / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    smoke_lengths = [m["tokens_prompt"] + m["tokens_target"] for _, m in smoke_rows]
    smoke_batches = pack_batches(smoke_lengths)
    (ds / "manifest.json").write_text(json.dumps({
        "set": f"{ARM3}-smoke", "examples": len(smoke_rows), "longest": len(longest), "tokens": sum(smoke_lengths),
        "max_tokens": max(smoke_lengths), "min_of_longest": min(smoke_lengths[:len(longest)]),
        "batches_per_process": smoke_batches, "steps_limit": steps_of(smoke_batches), "seq": SEQ3}, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    (out / "report.md").write_text(report3_md(report), encoding="utf-8")
    print(f"BUILD3 {json.dumps({'examples': len(rows_out), 'per_pass': [p['examples'] for p in per_pass], 'tokens': sum(lengths), 'steps': steps, 'fill': report['estimate']['fill'], 'replay': report['replay']['per_pass'], 'smoke_batches': smoke_batches})}",
          flush=True)


def report3_md(r):
    e = r["estimate"]
    L = [f"# Trainingsset {r['arm']} (protocol-g4.md §9, amendement G4-3)", "",
         f"Gemaakt door `g4arm.py build3`. Alle BT-train-rijen van de vijf datasets op de revisies van amendement G4-3 "
         f"zonder `hard_fails` en zonder 13-gram van een evaluatieset of blinde passage (geen hersampling, geen "
         f"chrF-filter, geen lengtegrens van R2), de boeken van G4-2 ({_n(r['books'])} paren; de boeken van G4-1 zonder "
         f"de held-out boeken {json.dumps(r['heldout_books'])}), FLAN-replay {r['replay_share']:.0%} van de voorbeelden, "
         f"{r['epochs']} doorgangen, elk apart geschud (seed `{r['seed']}/shuffle/<k>`); prompt `{r['prompt']}`, "
         f"sequentielengte {_n(r['seq'])}.", "",
         "## Terugvertalingen (train-split)", "",
         "| bron | rijen | waarvan lang | hard_fails | 13-gram | in de set | waarvan lang | buiten R2's lengtegrens (niet toegepast) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, c in r["bt"]["funnel"].items():
        L.append(f"| {s} | {_n(c.get('rows', 0))} | {_n(c.get('long_rows', 0))} | {_n(c.get('hard_fails', 0))} | "
                 f"{_n(c.get('eval_13gram', 0))} | {_n(c.get('eligible', 0))} | {_n(c.get('eligible_long', 0))} | "
                 f"{_n(c.get('outside_r2_length_bound_kept', 0))} |")
    rep = r["replay"]
    L += ["", "## FLAN-replay", "", f"Per doorgang {_n(rep['per_pass'])}; verschillende rijen over de doorgangen "
          f"{_n(rep['distinct'])} ({'elke doorgang eigen rijen' if rep['apart_in_every_pass'] else 'de pool is te klein: rijen komen terug'}); "
          f"kandidaten {_n(rep['candidates'])} uit een pool van {_n(rep['rows'])} (weg: {json.dumps(rep['dropped'])}).", "",
          "## Per doorgang", "", "| doorgang | soort | voorbeelden | tokens | doeltokens |", "|---:|---|---:|---:|---:|"]
    for p in r["passes"]:
        for kind, k in p["kinds"].items():
            L.append(f"| {p['pass']} | {kind} | {_n(k['examples'])} | {_n(k['tokens'])} | {_n(k['target_tokens'])} |")
        L.append(f"| {p['pass']} | samen | {_n(p['examples'])} | {_n(p['tokens'])} | |")
    L += ["", f"Samen {_n(e['examples'])} rijen, {_n(e['tokens'])} tokens, langste voorbeeld {_n(e['max_tokens'])} tokens. "
          f"Packing zoals MaxText 0.2.5 (elk proces elk vierde voorbeeld, first-fit in 4 rijen van {_n(e['seq'])}): "
          f"batches per proces {e['batches_per_process']}, dus {_n(e['steps'])} stappen (`sft_lora.step_limit`), "
          f"vulling {e['fill']}. Alleen de eerste twee doorgangen (amendement G4-3.5, boven {MAX_HOURS3} uur): "
          f"{_n(e['two_passes']['examples'])} rijen, {_n(e['two_passes']['tokens'])} tokens, "
          f"{_n(e['two_passes']['steps'])} stappen.", ""]
    return "\n".join(L) + "\n"


def _check_chunk(job):
    """One task of check --arm G4-3: MaxText's own SFT steps on a chunk of distinct examples"""
    messages, ens, ids, tokenizer = job
    return g4lora.check_examples(messages, ens, tokenizer, MODEL, ids)


def check3(args):
    """amendment G4-3.3: MaxText's own SFT steps on every distinct example of G4-3's set (an example of the three
    passes once; the same id must be the same example in every pass), in processes; every row's length must be the
    build's count, and none above SEQ3"""
    import multiprocessing

    d = Path(args.data)
    msgs = [x["messages"] for x in read_rows(d / "train.parquet")]
    metas = read_rows(d / "train-meta.parquet", ["id", "en", "tokens_prompt", "tokens_target", "epoch"])
    if len(msgs) != len(metas):
        raise SystemExit(f"{d}: {len(msgs)} examples, {len(metas)} meta rows")
    if (d / "template.jinja").read_text(encoding="utf-8") != g4lora.TEMPLATE:
        raise SystemExit(f"{d}/template.jinja is not g4lora.TEMPLATE")
    distinct = {}
    for x, m in zip(msgs, metas):
        seen = distinct.setdefault(m["id"], (x, m))
        if seen[0] != x or (seen[1]["en"], seen[1]["tokens_prompt"], seen[1]["tokens_target"]) != (
                m["en"], m["tokens_prompt"], m["tokens_target"]):
            raise SystemExit(f"{m['id']}: not the same example in pass {seen[1]['epoch']} and pass {m['epoch']}")
    ids = sorted(distinct)
    chunks = [ids[k:k + args.chunk] for k in range(0, len(ids), args.chunk)]
    jobs = [([distinct[i][0] for i in c], [distinct[i][1]["en"] for i in c], c, args.tokenizer) for c in chunks]
    if args.processes > 1:
        with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
            results = pool.map(_check_chunk, jobs, chunksize=1)
    else:
        results = [_check_chunk(j) for j in jobs]
    lengths, loss_tokens, pads = {}, 0, set()
    for c, (ls, lt, pad) in zip(chunks, results):
        lengths.update(zip(c, ls))
        loss_tokens += lt
        pads.add(pad)
    differ = [i for i in ids if lengths[i] != distinct[i][1]["tokens_prompt"] + distinct[i][1]["tokens_target"]]
    if differ:
        raise SystemExit(f"{len(differ)} examples whose MaxText length differs from the build's count: {differ[:3]}")
    if max(lengths.values()) > SEQ3:
        raise SystemExit(f"an example of {max(lengths.values())} tokens, more than {SEQ3}")
    if len(pads) != 1:
        raise SystemExit(f"the chunks saw different pad ids: {pads}")
    print(f"CHECK {json.dumps({'examples': len(metas), 'distinct': len(ids), 'tokens': sum(lengths[m['id']] for m in metas), 'loss_tokens_distinct': loss_tokens, 'max': max(lengths.values()), 'pad_id': pads.pop(), 'seq': SEQ3})}",
          flush=True)


def training_time(summary, steps, tokens, smoke_tokens, every=EVAL_EVERY3):
    """The full training's time [estimate] from the smoke test: the count of the data before the first step, loading
    and compiling, the steps at the smoke test's median step, the evals at its longest round after step 0"""
    t = summary.get("times") or {}
    step_s = (summary.get("throughput") or {}).get("step_s_median")
    rounds = collections.defaultdict(float)
    for e in summary.get("eval") or []:
        rounds[e["step"]] += e["seconds"]
    later = [v for k, v in rounds.items() if k != 0]
    round_s = max(later) if later else rounds.get(0)
    if step_s is None or round_s is None:
        return None
    n_evals = steps // every + 1 + (1 if steps % every else 0)
    count_s = (t.get("python_to_setup_s") or 0) + COUNT_S_PER_MTOKEN * (tokens - smoke_tokens) / 1e6
    load_s = (t.get("setup_s") or 0) + (t.get("compile_s_est") or 0)
    seconds = count_s + load_s + steps * step_s + n_evals * round_s
    return {"steps": steps, "tokens": tokens, "step_s": step_s, "evals": n_evals, "eval_round_s": round(round_s, 1),
            "count_s": round(count_s), "load_s": round(load_s), "train_s": round(steps * step_s),
            "seconds": round(seconds), "hours": round(seconds / 3600, 2)}


def smoke3_estimate(summary, data_report, smoke_manifest):
    """amendment G4-3.5 (Yeb, 15 Nov ~08:05): three passes if their estimate stays within MAX_HOURS3 (~5 days), else
    the first two"""
    est = data_report["estimate"]
    three = training_time(summary, est["steps"], est["tokens"], smoke_manifest["tokens"])
    two = training_time(summary, est["two_passes"]["steps"], est["two_passes"]["tokens"], smoke_manifest["tokens"])
    if three is None:
        return {"passes": None, "hours": None, "three": None, "two": None, "max_hours": MAX_HOURS3}
    passes = 3 if three["hours"] <= MAX_HOURS3 else 2
    chosen = three if passes == 3 else two
    return {"passes": passes, "hours": chosen["hours"], "steps": chosen["steps"], "max_hours": MAX_HOURS3,
            "fill": est["fill"], "three": three, "two": two}


def smoke3_verdict(summary, adapter, smoke_manifest, eval_sets=EVAL_SETS3):
    """amendment G4-3.5: the smoke test of G4-3, every criterion with its value and whether it holds (G4-4: its five
    eval sets, amendment G4-4.11)"""
    checks = {}
    loss = summary.get("loss") or []
    checks["steps"] = {"value": summary.get("steps_done"), "holds": summary.get("steps_done") == 8}
    checks["loss_finite"] = {"value": [v for _, v in loss],
                             "holds": len(loss) == 8 and all(math.isfinite(v) for _, v in loss)}
    checks["peak_gib"] = {"value": summary.get("peak_gib_max"),
                          "holds": summary.get("peak_gib_max") is not None and summary["peak_gib_max"] <= MEM_MAX_GIB}
    checks["adapter"] = {"value": {k: adapter.get(k) for k in ("rank", "alpha", "scale")},
                         "holds": (adapter.get("rank"), adapter.get("alpha"), adapter.get("scale")) == (32, 16.0, 0.5)}
    seq = (summary.get("config") or {}).get("max_target_length")
    checks["seq"] = {"value": seq, "holds": seq == SEQ3}
    lrs = list((summary.get("lr_at") or {}).values())
    checks["lr_peak"] = {"value": max(lrs) if lrs else None, "holds": bool(lrs) and abs(max(lrs) - LR3) <= 1e-12}
    got = (summary.get("epoch") or {}).get("batches_per_process")
    checks["batches_as_build3"] = {"value": {"pipeline": got, "build3": smoke_manifest.get("batches_per_process")},
                                   "holds": got is not None and got == smoke_manifest.get("batches_per_process")}
    per_step = (summary.get("tokens") or {}).get("per_step") or []
    longest = smoke_manifest.get("min_of_longest") or 0
    # every step of the 8 holds 16 of the longest examples: each a row of its own
    checks["longest_rows"] = {"value": per_step,
                              "holds": len(per_step) == 8 and all(n >= 16 * longest for n in per_step)}
    evals = summary.get("eval") or []
    for name in eval_sets:
        rows = [e for e in evals if e["set"] == name]
        checks[f"eval_{name}"] = {
            "value": [(e["step"], round(e["loss"], 4), e["tokens"]) for e in rows],
            "holds": [e["step"] for e in rows] == SMOKE3_EVAL_STEPS
            and all(math.isfinite(e["loss"]) and e["tokens"] >= 1 for e in rows)}
    own0 = next((e["loss"] for e in evals if e["set"] == "own" and e["step"] == 0), None)
    checks["own_loss_step0"] = {"value": own0, "holds": own0 is not None and own0 <= OWN_MAX_LOSS + TOLERANCE}
    return {"checks": checks, "passed": all(c["holds"] for c in checks.values())}


def smoke3(args):
    """The smoke test's verdict, the estimate and the passes of the full training (PASSES); exit 1 if a check fails
    (the chain stops before the full training)"""
    summary = json.loads(Path(args.summary).read_text())
    adapter = json.loads(Path(args.adapter).read_text())
    smoke_manifest = json.loads(Path(args.smoke_manifest).read_text())
    data_report = json.loads(Path(args.data_report).read_text())
    result = smoke3_verdict(summary, adapter, smoke_manifest)
    result["estimate"] = smoke3_estimate(summary, data_report, smoke_manifest)
    result.update({k: summary.get(k) for k in ("loss", "eval", "eval_plan", "lr_at", "throughput", "times", "epoch",
                                               "memory", "peak_gib_max")})
    result["smoke_set"] = smoke_manifest
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    x = result["estimate"]
    L = ["# Smoke-test G4-3 (protocol-g4.md §9, amendement G4-3.5)", "",
         f"**{'Geslaagd' if result['passed'] else 'Niet geslaagd'}.** Schatting van de volle training met drie "
         f"doorgangen: {(x.get('three') or {}).get('hours')} uur (grens {MAX_HOURS3} uur); de training neemt "
         f"**{x.get('passes')} doorgangen** ({x.get('hours')} uur, {x.get('steps')} stappen).", "",
         "| controle | waarde | gehaald |", "|---|---|---|"]
    for k, c in result["checks"].items():
        L.append(f"| {k} | `{json.dumps(c['value'], default=str)[:240]}` | {'ja' if c['holds'] else 'nee'} |")
    L += ["", f"Schatting [schatting, uit de smoke-test en build3]: {json.dumps(x)}", "",
          "Verlies per held-out set (stap, verlies, doeltokens, seconden): " + "; ".join(
              f"{e['set']} {e['step']} {e['loss']:.4f} {e['tokens']} {e['seconds']}" for e in summary.get("eval") or []),
          "", f"Training: {json.dumps({k: summary.get(k) for k in ('throughput', 'times', 'peak_gib_max')}, default=str)}",
          ""]
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"SMOKE3 {'PASSED' if result['passed'] else 'FAILED'} {json.dumps({k: c['holds'] for k, c in result['checks'].items()})} "
          f"PASSES {x.get('passes')} HOURS {x.get('hours')} estimate {json.dumps(x)}", flush=True)
    if not result["passed"] or x.get("passes") not in (2, 3):
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------- G4-4

ARM4 = "G4-4"
SEED4 = f"{SEED}/G4-4"
EPOCHS4 = 3
REPLAY_SHARE4 = 0.10                 # amendment G4-4.3: 10% of every pass's examples, G4-3's round(n / 9) per pass
NOVELS_PER_PASS = 20                 # amendment G4-4.3 (Yeb, 15 Nov ~13:45): at most 20 pairs of a book in a pass
NOVEL_SOURCE = "books_en"            # arm RE's source name of the -b1- English novels with their Dutch translation
NOVEL_MAX_TOKENS = MAX_TOKENS        # arm RE's bound on a novel pair (2,048), here in Gemma 4's prompt
# arm RE's checks of a -b1- row (mbert-jax en-nl-sft-re @ 97c44b6, sft_build.py), copied unchanged: the inventory's
# passes (real_fails, its length bound and alignment check, no repeated id or English), no repeated Dutch, no Markdown
# trace, not cleaned on one side only; one Dutch edition per English book
B1_MARKDOWN = re.compile(r"(^|\s)_[^_\s][^_]*_($|[\s.,;:!?])|\*\*")
B1_COLUMNS = ["id", "book", "nl_book_id", "en_book_id", "group", "author", "en", "nl", "en_raw", "nl_raw", "passes",
              "dup_id", "dup_nl"]
# one book under two English ids (another English edition): passages whose first SAME_WORK_CHARS letters and digits
# (lower case) are equal on the English or on the Dutch side, at least SAME_WORK_MIN_SHARED between the two books
SAME_WORK_CHARS = 80
SAME_WORK_MIN_SHARED = 10
# rows per parquet row group of G4-4's files: pyarrow's default (1,048,576) put more than 2 GiB of nested text in one
# row group of train.parquet, which pyarrow 25 cannot read as one batch ("Nested data conversions not implemented for
# chunked array outputs"; check of 15 Nov); datasets 4.8.5 reads a parquet file in batches of its first row group and
# counts a file as one shard whatever its row groups, so the order every process reads is unchanged
ROW_GROUP4 = 100_000
# G4-4's eval sets: G4-3's and the novels (amendment G4-4.11)
EVAL_SETS4 = EVAL_SETS3 + ("novels",)


def read_b1(path):
    """(the rows of group en_orig, with B1_COLUMNS; {group: rows} of the whole file): b1-selected.parquet of the -b1-
    inventory (arm RE's read_b1)"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=B1_COLUMNS)
    groups = {g["values"]: g["counts"] for g in pc.value_counts(t["group"]).to_pylist()}
    return t.filter(pc.equal(t["group"], "en_orig")).to_pylist(), groups


def b1_fails(r):
    """The first of arm RE's row checks a -b1- row fails, or None (arm RE's b1_fails)"""
    if r["group"] != "en_orig":
        return "group"
    if r["dup_id"]:
        return "dup_id"
    if not r["passes"]:
        return "passes"
    if r["dup_nl"]:
        return "dup_nl"
    if B1_MARKDOWN.search(r["en"]) or B1_MARKDOWN.search(r["nl"]):
        return "markdown"
    # the loader's clean_line changed the text on one side only: a leading mark taken off there and not here
    if (r["en_raw"] is None) != (r["nl_raw"] is None):
        return "cleaned_one_side"
    return None


def _id(x):
    return int(str(x))


def b1_editions(rows):
    """({English book id: the rows of its chosen Dutch edition}, info): the rows without b1_fails; of an English book
    with several Dutch editions the one with the most such rows, a tie to the lowest Dutch id (arm RE's b1_editions)"""
    first, alone = collections.Counter(), collections.Counter()
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    english = set()
    for r in rows:
        english.add(r["en_book_id"])
        f = b1_fails(r)
        if f is None:
            by[r["en_book_id"]][r["book"]].append(r)
        else:
            first[f] += 1
        if r["group"] == "en_orig" and not r["dup_id"] and r["passes"]:
            alone["dup_nl"] += bool(r["dup_nl"])
            alone["markdown"] += bool(B1_MARKDOWN.search(r["en"]) or B1_MARKDOWN.search(r["nl"]))
            alone["cleaned_one_side"] += (r["en_raw"] is None) != (r["nl_raw"] is None)
    chosen, editions = {}, collections.Counter()
    for en, eds in by.items():
        book = min(eds, key=lambda b: (-len(eds[b]), _id(eds[b][0]["nl_book_id"])))
        chosen[en] = eds[book]
        editions[len(eds)] += 1
        first["other_edition"] += sum(len(v) for b, v in eds.items() if b != book)
    counts = sorted(len(v) for v in chosen.values())
    info = {"rows": len(rows), "first_failed": dict(sorted(first.items())),
            "alone_after_passes": dict(sorted(alone.items())), "english_books": len(english),
            "english_books_with_rows": len(chosen), "editions": dict(sorted(editions.items())),
            "rows_kept": sum(counts),
            "rows_per_book": {f"p{q}": quantile(counts, q / 100) for q in (0, 10, 50, 90, 100)} if counts else None}
    return chosen, info


def quantile(values, q):
    return sft_build.quantile(values, q)


def work_key(text):
    k = re.sub(r"[^a-z0-9]", "", text.lower())
    return k[:SAME_WORK_CHARS] if len(k) >= SAME_WORK_CHARS else None


def same_works(rows, chosen):
    """({English book id: rows}, info): of English books that are one work under several ids (at least
    SAME_WORK_MIN_SHARED passage keys in common on the English or on the Dutch side, over every en_orig row without a
    repeated id) the one with the most rows in chosen stays, a tie to the lowest id; the others go"""
    shared = {}
    for side in ("en", "nl"):
        books = collections.defaultdict(set)
        for r in rows:
            if r["group"] == "en_orig" and not r["dup_id"]:
                k = work_key(r[side])
                if k is not None:
                    books[k].add(r["en_book_id"])
        pair = collections.Counter()
        for ids in books.values():
            ids = sorted(ids, key=_id)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair[(ids[i], ids[j])] += 1
        shared[side] = pair
    edges = sorted({p for side in shared.values() for p, n in side.items() if n >= SAME_WORK_MIN_SHARED},
                   key=lambda p: (_id(p[0]), _id(p[1])))
    parent = {}

    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb, key=_id)] = min(ra, rb, key=_id)
    works = collections.defaultdict(list)
    for x in {x for e in edges for x in e}:
        works[root(x)].append(x)
    out = dict(chosen)
    dropped = []
    for members in works.values():
        keep = min(members, key=lambda x: (-len(chosen.get(x, [])), _id(x)))
        for x in sorted(members, key=_id):
            if x != keep and x in out:
                dropped.append({"en_book_id": str(x), "rows": len(out.pop(x)), "kept_instead": str(keep)})
    near = {side: sum(1 for n in pair.values() if 0 < n < SAME_WORK_MIN_SHARED) for side, pair in shared.items()}
    info = {"key_chars": SAME_WORK_CHARS, "min_shared": SAME_WORK_MIN_SHARED, "pairs_linked": len(edges),
            "works": len(works), "books_in_works": sum(len(m) for m in works.values()),
            "books_dropped": len(dropped), "rows_dropped": sum(d["rows"] for d in dropped), "dropped": dropped,
            "pairs_below_bound": near,
            "most_shared": {side: [[str(a), str(b), n] for (a, b), n in pair.most_common(12)]
                            for side, pair in shared.items()}}
    return out, info


def near_duplicates(chosen):
    """({English book id: rows}, info): a row whose English or Dutch passage key (work_key) is that of an earlier row
    (books by id, rows by id) goes, beyond the inventory's exact repeats: the same passage in another English book (a
    preview chapter, an omnibus) or twice in one book"""
    seen, out, dropped = {}, {}, collections.Counter()
    for en in sorted(chosen, key=_id):
        keep = []
        for r in sorted(chosen[en], key=lambda r: r["id"]):
            keys = [(side, k) for side in ("en", "nl") if (k := work_key(r[side])) is not None]
            hit = [seen[k] for k in keys if k in seen]
            if hit:
                dropped["same_book" if en in hit else "other_book"] += 1
            else:
                keep.append(r)
            for k in keys:
                seen.setdefault(k, en)
        if keep:
            out[en] = keep
    info = {"rows_dropped": sum(dropped.values()), **dict(sorted(dropped.items())),
            "books_left": len(out), "rows_left": sum(len(v) for v in out.values())}
    return out, info


def novel_draw(by_book, blocked, counter, seed, per_pass, passes):
    """({pass: rows}, info): per English book its rows (by id) in an order drawn with the seed, and of them the first
    per_pass x passes that share no 13-gram with blocked and fit NOVEL_MAX_TOKENS; the r-th of a book goes to pass
    r // per_pass + 1, so a pair is in one pass only, a pass has at most per_pass of a book, and a book with fewer
    pairs gives what it has to the first passes"""
    want = per_pass * passes
    out = {e: [] for e in range(1, passes + 1)}
    stats, kept_of = collections.Counter(), {}
    for en in sorted(by_book, key=_id):
        rows = sorted(by_book[en], key=lambda r: r["id"])
        random.Random(f"{seed}/novels/{en}").shuffle(rows)
        kept, i = 0, 0
        while kept < want and i < len(rows):
            part = rows[i:i + want - kept + 8]
            i += len(part)
            prompts, targets = counter.examples(part)
            for r, a, b in zip(part, prompts, targets):
                if kept == want:
                    break
                stats["looked_at"] += 1
                if shares_gram(r, blocked):
                    stats["eval_13gram"] += 1
                elif a + b > NOVEL_MAX_TOKENS:
                    stats["too_long"] += 1
                else:
                    out[kept // per_pass + 1].append({"pid": r["id"], "source": NOVEL_SOURCE, "en": r["en"],
                                                       "nl": r["nl"], "book": r["book"], "en_book_id": str(en),
                                                       "rank": kept, "tokens_prompt": a, "tokens_target": b})
                    kept += 1
        kept_of[str(en)] = kept
    counts = sorted(kept_of.values())
    info = {"per_pass": per_pass, "passes": passes, "books": len(by_book), **dict(sorted(stats.items())),
            "drawn": sum(counts), "per_pass_rows": {e: len(v) for e, v in out.items()},
            "per_pass_books": {e: len({r["en_book_id"] for r in v}) for e, v in out.items()},
            "books_below": {n: sum(c < n for c in counts) for n in range(per_pass, want + 1, per_pass)},
            "kept_per_book": {f"p{q}": quantile(counts, q / 100) for q in (0, 10, 50, 90, 100)} if counts else None}
    return out, info


def g43_set(directory, report_path):
    """G4-3's set as {pass: [(messages row, meta row)]}, checked: the manifest, report-G4-3.json's examples and tokens
    per pass and kind, the passes in order, every example outside the replay the same in each pass, the replay apart"""
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text())
    if (manifest.get("set"), manifest.get("seq"), manifest.get("seed"), manifest.get("epochs")) != (ARM3, SEQ3, SEED3,
                                                                                                    EPOCHS3):
        raise SystemExit(f"{d}: not G4-3's set ({manifest.get('set')}, seq {manifest.get('seq')})")
    msgs = read_rows(d / "train.parquet")
    metas = read_rows(d / "train-meta.parquet")
    if len(msgs) != len(metas) or len(metas) != manifest["examples"]:
        raise SystemExit(f"{d}: {len(msgs)} examples, {len(metas)} meta rows, manifest {manifest['examples']}")
    if (d / "template.jinja").read_text(encoding="utf-8") != g4lora.TEMPLATE:
        raise SystemExit(f"{d}/template.jinja is not g4lora.TEMPLATE")
    epochs = [m["epoch"] for m in metas]
    if epochs != sorted(epochs) or sorted(set(epochs)) != list(range(1, EPOCHS3 + 1)):
        raise SystemExit(f"{d}: the passes are not 1 ... {EPOCHS3} in order")
    passes = collections.defaultdict(list)
    for x, m in zip(msgs, metas):
        passes[m["epoch"]].append((x, m))
    report = json.loads(Path(report_path).read_text())
    for p in report["passes"]:
        got = pass_kinds(passes[p["pass"]])
        if (p["examples"], p["tokens"], p["kinds"]) != (len(passes[p["pass"]]), sum(v["tokens"] for v in got.values()),
                                                        got):
            raise SystemExit(f"{d}: pass {p['pass']} is not the pass of {report_path}")
    seen, replay = {}, []
    for e, one in passes.items():
        for x, m in one:
            if m["kind"] == "replay":
                continue
            s = seen.setdefault(m["id"], (x, m))
            if s[0] != x or {k: v for k, v in s[1].items() if k != "epoch"} != {k: v for k, v in m.items() if k != "epoch"}:
                raise SystemExit(f"{m['id']}: not the same example in pass {s[1]['epoch']} and pass {e}")
        replay.append({m["id"] for _, m in one if m["kind"] == "replay"})
    rest = [sorted(m["id"] for _, m in one if m["kind"] != "replay") for one in passes.values()]
    if any(r != rest[0] for r in rest):
        raise SystemExit(f"{d}: the examples outside the replay differ between the passes")
    if sum(map(len, replay)) != len(set().union(*replay)):
        raise SystemExit(f"{d}: a replay row in two passes")
    return dict(passes), manifest, report


def pass_kinds(one):
    kinds = collections.defaultdict(lambda: {"examples": 0, "tokens": 0, "target_tokens": 0})
    for _, m in one:
        k = "bt_long" if m["kind"] == "bt" and m["long"] else m["kind"]
        kinds[k]["examples"] += 1
        kinds[k]["tokens"] += m["tokens_prompt"] + m["tokens_target"]
        kinds[k]["target_tokens"] += m["tokens_target"]
    return dict(sorted(kinds.items()))


def build4(args, counter=None):
    """amendment G4-4.3: G4-3's set, pass by pass, as it is (its BT, its books, its replay rows of the pass), plus per
    pass at most NOVELS_PER_PASS pairs of every English novel (-b1-, group en_orig, arm RE's checks, one Dutch edition
    per book and one English id per work, no 13-gram of the evaluation sets or blind passages, apart in every pass) and
    the replay that keeps 10% of the pass (the next rows of G4-3's replay order, so every pass its own rows), every
    pass shuffled anew; MaxText's packing per process at SEQ3, the steps; the smoke test's set"""
    import pyarrow as pa

    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    print("PROGRESS reading G4-3's set", flush=True)
    g43, g43_manifest, g43_report = g43_set(args.g43, args.g43_report)
    inputs3 = g43_manifest["inputs"]
    for name, path, want in (("evalsets", args.evalsets, inputs3["evalsets"]), ("blind", args.blind, inputs3["blind"]),
                             ("replay", args.replay, inputs3["replay"]["sha256"])):
        if sha256_file(path) != want:
            raise SystemExit(f"{path}: not the {name} file G4-3's set was built from")
    if args.replay_model != inputs3["replay"]["model"]:
        raise SystemExit(f"--replay-model {args.replay_model!r}: G4-3's replay is {inputs3['replay']['model']!r}")
    blocked = checked_blocked(args.evalsets, args.blind)
    report = {"arm": ARM4, "seed": SEED4, "prompt": PROMPT, "epochs": args.epochs, "replay_share": args.replay_share,
              "seq": SEQ3, "novels_per_pass": NOVELS_PER_PASS, "novel_max_tokens": NOVEL_MAX_TOKENS,
              "g4_3": {"manifest": {k: g43_manifest[k] for k in ("set", "examples", "examples_per_epoch", "tokens",
                                                                 "steps", "seed")},
                       "files": {f: sha256_file(Path(args.g43) / f) for f in ("train.parquet", "train-meta.parquet")},
                       "report": sha256_file(args.g43_report)}}
    # G4-3's replay draw again (build3: the pool, its order with G4-3's seed, pass k the rows (k-1)r ... kr - 1): it
    # must give G4-3's replay rows of every pass; G4-4's added replay rows are the next ones in that order
    print("PROGRESS the replay order of G4-3", flush=True)
    candidates = replay_pool(args.replay, args.replay_model, args.flan_eval, blocked, counter, report)
    order = random.Random(f"{SEED3}/replay").sample(candidates, len(candidates))
    r3 = g43_report["replay"]["per_pass"]
    for e in range(1, EPOCHS3 + 1):
        theirs = {m["id"]: (x, m["tokens_prompt"], m["tokens_target"]) for x, m in g43[e] if m["kind"] == "replay"}
        again = {m["id"]: (x, m["tokens_prompt"], m["tokens_target"])
                 for x, m in (example(r, "replay") for r in order[(e - 1) * r3:e * r3])}
        if again != theirs:
            raise SystemExit(f"pass {e}: G4-3's replay draw is not reproduced "
                             f"({len(set(again) ^ set(theirs))} of {len(theirs)} ids differ)")
    print("PROGRESS reading the novels", flush=True)
    rows, groups = read_b1(args.novels)
    chosen, editions = b1_editions(rows)
    chosen, works = same_works(rows, chosen)
    del rows
    chosen, near = near_duplicates(chosen)
    drawn, draw = novel_draw(chosen, blocked, counter, SEED4, NOVELS_PER_PASS, args.epochs)
    report["novels"] = {"source": str(args.novels), "sha256": sha256_file(args.novels), "groups": groups,
                        "editions": editions, "same_works": works, "near_duplicates": near, "draw": draw}
    used = EPOCHS3 * r3
    passes, added = [], []
    for e in range(1, args.epochs + 1):
        base = [(x, {k: v for k, v in m.items() if k != "epoch"}) for x, m in g43[e]]
        n_rest = sum(m["kind"] != "replay" for _, m in base)
        novels = [(x, {**m, "long": False}) for x, m in (example(r, "novel") for r in drawn[e])]
        n_replay = replay_count(n_rest + len(novels), args.replay_share)
        extra = n_replay - r3
        if extra < 0 or used + extra > len(order):
            raise SystemExit(f"pass {e}: {n_replay} replay rows wanted, G4-3 gives {r3} and the pool has "
                             f"{len(order) - used} left: no replay apart in every pass")
        more = [(x, {**m, "long": False}) for x, m in (example(r, "replay") for r in order[used:used + extra])]
        used += extra
        added.append({"pass": e, "novels": len(novels), "replay_g4_3": r3, "replay_added": extra,
                      "replay": n_replay, "rest": n_rest})
        one = base + novels + more
        random.Random(f"{SEED4}/shuffle/{e}").shuffle(one)
        passes.append(one)
    novel_ids = [m["id"] for one in passes for _, m in one if m["kind"] == "novel"]
    replay_ids = [m["id"] for one in passes for _, m in one if m["kind"] == "replay"]
    if len(novel_ids) != len(set(novel_ids)) or len(replay_ids) != len(set(replay_ids)):
        raise SystemExit("a novel pair or a replay row in two passes")
    report["added"] = added
    report["replay"].update({"per_pass": [a["replay"] for a in added], "g4_3_per_pass": r3,
                             "distinct": len(replay_ids), "apart_in_every_pass": True, "order_used": used})
    rows_out = [(x, {**m, "epoch": e}) for e, one in enumerate(passes, 1) for x, m in one]
    d = out / ARM4
    meta_schema = pa.schema([(n, getattr(pa, "bool_" if t == "bool" else t)()) for n, t in META_SCHEMA3_FIELDS])
    write_parquet([x for x, _ in rows_out], d / "train.parquet", row_group_size=ROW_GROUP4)
    write_parquet([{n: m.get(n) for n in meta_schema.names} for _, m in rows_out], d / "train-meta.parquet", meta_schema,
                  row_group_size=ROW_GROUP4)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    lengths = [m["tokens_prompt"] + m["tokens_target"] for _, m in rows_out]
    if max(lengths) > SEQ3:
        raise SystemExit(f"an example of {max(lengths)} tokens, more than {SEQ3}")
    batches = pack_batches(lengths)
    steps = steps_of(batches)
    per_pass = []
    for e, one in enumerate(passes, 1):
        kinds = pass_kinds(one)
        tokens = sum(v["tokens"] for v in kinds.values())
        per_pass.append({"pass": e, "examples": len(one), "tokens": tokens, "kinds": kinds,
                         "novel_share_examples": kinds["novel"]["examples"] / len(one),
                         "novel_share_tokens": kinds["novel"]["tokens"] / tokens,
                         "replay_share_examples": kinds["replay"]["examples"] / len(one)})
    report["set"] = set_stats([m for _, m in passes[0]], [])
    report["passes"] = per_pass
    report["novels"]["distinct_pairs"] = len(novel_ids)
    report["estimate"] = {"tokens": sum(lengths), "examples": len(lengths), "seq": SEQ3,
                          "batches_per_process": batches, "steps": steps,
                          "fill": round(sum(lengths) / (sum(batches) * ROWS_PER_HOST * SEQ3), 4),
                          "max_tokens": max(lengths)}
    inputs = {"g4_3": report["g4_3"], "evalsets": sha256_file(args.evalsets), "blind": sha256_file(args.blind),
              "replay": {"path": str(args.replay), "sha256": sha256_file(args.replay), "model": args.replay_model},
              "novels": {"path": str(args.novels), "sha256": report["novels"]["sha256"]}}
    report["inputs"] = inputs
    manifest = {"set": ARM4, "examples": len(rows_out), "examples_per_epoch": [p["examples"] for p in per_pass],
                "epochs": args.epochs, "tokens": sum(lengths), "seq": SEQ3, "steps": steps, "prompt": PROMPT,
                "template": "g4lora.TEMPLATE (Gemma 4, thinking off)", "seed": SEED4,
                "replay_share": args.replay_share, "novels_per_pass": NOVELS_PER_PASS, "inputs": inputs}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    # the smoke test's set (amendment G4-4.5, as G4-3.5): pass 1's longest examples first, then a seeded sample
    first = list(passes[0])
    longest = sorted(first, key=lambda p: (-(p[1]["tokens_prompt"] + p[1]["tokens_target"]), p[1]["id"]))
    longest = longest[:SMOKE3_LONGEST]
    taken = {m["id"] for _, m in longest}
    others = sorted((p for p in first if p[1]["id"] not in taken), key=lambda p: p[1]["id"])
    smoke_rows = [(x, {**m, "epoch": 1}) for x, m in longest +
                  random.Random(f"{SEED4}/smoke").sample(others, min(SMOKE3_RANDOM, len(others)))]
    ds = out / f"{ARM4}-smoke"
    write_parquet([x for x, _ in smoke_rows], ds / "train.parquet", row_group_size=ROW_GROUP4)
    write_parquet([{n: m.get(n) for n in meta_schema.names} for _, m in smoke_rows], ds / "train-meta.parquet",
                  meta_schema, row_group_size=ROW_GROUP4)
    (ds / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    smoke_lengths = [m["tokens_prompt"] + m["tokens_target"] for _, m in smoke_rows]
    smoke_batches = pack_batches(smoke_lengths)
    (ds / "manifest.json").write_text(json.dumps({
        "set": f"{ARM4}-smoke", "examples": len(smoke_rows), "longest": len(longest), "tokens": sum(smoke_lengths),
        "max_tokens": max(smoke_lengths), "min_of_longest": min(smoke_lengths[:len(longest)]),
        "kinds": dict(collections.Counter(m["kind"] for _, m in smoke_rows)),
        "batches_per_process": smoke_batches, "steps_limit": steps_of(smoke_batches), "seq": SEQ3}, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    (out / "report.md").write_text(report4_md(report), encoding="utf-8")
    print(f"BUILD4 {json.dumps({'examples': len(rows_out), 'per_pass': [p['examples'] for p in per_pass], 'tokens': sum(lengths), 'steps': steps, 'fill': report['estimate']['fill'], 'novels': [a['novels'] for a in added], 'replay': [a['replay'] for a in added], 'smoke_batches': smoke_batches})}",
          flush=True)


KIND_NAMES4 = {"bt": "BT ≤ 512 tokens", "bt_long": "BT lang (513–2.048)", "book": "Nederlandse boeken",
               "novel": "romans (Engels origineel, menselijk Nederlands)", "replay": "FLAN-replay"}


def report4_md(r):
    e, n, x = r["estimate"], r["novels"], r["novels"]["draw"]
    w = n["same_works"]
    L = [f"# Trainingsset {r['arm']} (protocol-g4.md §9, amendement G4-4)", "",
         f"Gemaakt door `g4arm.py build4`. Per doorgang de set van G4-3 zoals hij is (BT ≤ 512 en lang, de boeken van "
         f"G4-2, de replayrijen van die doorgang; bestanden met sha256 {json.dumps(r['g4_3']['files'])}), plus "
         f"hoogstens {r['novels_per_pass']} romanparen per boek (`-b1-`, groep `en_orig`, de controles van arm RE "
         f"zonder QE, één Nederlandse editie per boek en één Engels id per werk, geen 13-gram van een evaluatieset of "
         f"blinde passage, ≤ {_n(r['novel_max_tokens'])} tokens; elke doorgang andere paren), plus de replay die de "
         f"doorgang op {r['replay_share']:.0%} houdt (de volgende rijen van G4-3's replayvolgorde); elke doorgang "
         f"opnieuw geschud (seed `{r['seed']}/shuffle/<k>`); prompt `{r['prompt']}`, sequentielengte {_n(r['seq'])}.", "",
         "## Per doorgang", "", "| doorgang | deel | voorbeelden | tokens | doeltokens |", "|---:|---|---:|---:|---:|"]
    for p in r["passes"]:
        for kind, k in p["kinds"].items():
            L.append(f"| {p['pass']} | {KIND_NAMES4.get(kind, kind)} | {_n(k['examples'])} | {_n(k['tokens'])} | "
                     f"{_n(k['target_tokens'])} |")
        L.append(f"| {p['pass']} | **samen** | **{_n(p['examples'])}** | **{_n(p['tokens'])}** | |")
    L += ["", "| doorgang | romans (voorbeelden) | romans (tokens) | replay (voorbeelden) | replay van G4-3 | replay erbij |",
          "|---:|---:|---:|---:|---:|---:|"]
    for p, a in zip(r["passes"], r["added"]):
        L.append(f"| {p['pass']} | {p['novel_share_examples']:.1%} | {p['novel_share_tokens']:.1%} | "
                 f"{p['replay_share_examples']:.2%} | {_n(a['replay_g4_3'])} | {_n(a['replay_added'])} |")
    L += ["", f"Samen {_n(e['examples'])} rijen, {_n(e['tokens'])} tokens, langste voorbeeld {_n(e['max_tokens'])} tokens. "
          f"Packing zoals MaxText 0.2.5 (elk proces elk vierde voorbeeld, first-fit in 4 rijen van {_n(e['seq'])}): "
          f"batches per proces {e['batches_per_process']}, dus **{_n(e['steps'])} stappen** (`sft_lora.step_limit`), "
          f"vulling {e['fill']}.", "",
          "## Romans", "",
          f"Bron `{n['source']}` (sha256 `{n['sha256']}`), groepen {json.dumps(n['groups'])}.", "",
          f"- Controles van arm RE (eerste die faalt): {json.dumps(n['editions']['first_failed'])}; Engelse boeken "
          f"{_n(n['editions']['english_books'])}, met rijen {_n(n['editions']['english_books_with_rows'])}; edities per "
          f"boek {json.dumps(n['editions']['editions'])}; rijen over {_n(n['editions']['rows_kept'])}.",
          f"- Eén werk onder meer Engelse id's (≥ {w['min_shared']} gedeelde passagebegins van {w['key_chars']} tekens): "
          f"{w['works']} werken met {w['books_in_works']} id's; weg {w['books_dropped']} id's met {_n(w['rows_dropped'])} "
          f"rijen; paren met 1–{w['min_shared'] - 1} gedeelde begins (niet samengevoegd): {json.dumps(w['pairs_below_bound'])}.",
          f"- Bijna-dubbele passages (zelfde begin van {w['key_chars']} tekens als een eerdere rij, Engels of "
          f"Nederlands): weg {json.dumps({k: v for k, v in n['near_duplicates'].items() if k not in ('books_left', 'rows_left')})}; "
          f"over {_n(n['near_duplicates']['rows_left'])} rijen in {_n(n['near_duplicates']['books_left'])} boeken.",
          f"- Trekking (seed `{r['seed']}/novels/<boek>`): {_n(x['books'])} boeken, bekeken {_n(x.get('looked_at', 0))}, "
          f"13-gram {_n(x.get('eval_13gram', 0))}, te lang {_n(x.get('too_long', 0))}; getrokken {_n(x['drawn'])}; per "
          f"doorgang {json.dumps(x['per_pass_rows'])} paren uit {json.dumps(x['per_pass_books'])} boeken; boeken met "
          f"minder dan 20 / 40 / 60 paren: {json.dumps(x['books_below'])}; per boek {json.dumps(x['kept_per_book'])}.",
          f"- **Verschillende romanparen over de drie doorgangen: {_n(n['distinct_pairs'])}** (geen paar in twee "
          f"doorgangen).", "",
          "## FLAN-replay", "",
          f"Kandidaten {_n(r['replay']['candidates'])} uit een pool van {_n(r['replay']['rows'])} (weg: "
          f"{json.dumps(r['replay']['dropped'])}); G4-3's trekking opnieuw gemaakt en gelijk aan zijn replayrijen; per "
          f"doorgang {json.dumps(r['replay']['per_pass'])}; verschillende rijen {_n(r['replay']['distinct'])} (elke "
          f"doorgang eigen rijen; gebruikt {_n(r['replay']['order_used'])} van de volgorde).", ""]
    return "\n".join(L) + "\n"


def smoke4_estimate(summary, data_report, smoke_manifest):
    """amendment G4-4.5: the three passes' time; above MAX_HOURS3 the chain stops before the training (the main
    session decides)"""
    est = data_report["estimate"]
    three = training_time(summary, est["steps"], est["tokens"], smoke_manifest["tokens"])
    if three is None:
        return {"passes": None, "hours": None, "three": None, "max_hours": MAX_HOURS3}
    return {"passes": 3 if three["hours"] <= MAX_HOURS3 else None, "hours": three["hours"], "steps": three["steps"],
            "max_hours": MAX_HOURS3, "fill": est["fill"], "three": three}


def smoke4(args):
    """The smoke test's verdict of G4-4 (G4-3.5's checks on G4-4's smoke set) and the estimate; exit 1 if a check
    fails or the estimate is above MAX_HOURS3 (the chain stops before the full training)"""
    summary = json.loads(Path(args.summary).read_text())
    adapter = json.loads(Path(args.adapter).read_text())
    smoke_manifest = json.loads(Path(args.smoke_manifest).read_text())
    data_report = json.loads(Path(args.data_report).read_text())
    if data_report.get("arm") != ARM4:
        raise SystemExit(f"{args.data_report}: not G4-4's data report")
    result = smoke3_verdict(summary, adapter, smoke_manifest, EVAL_SETS4)
    result["estimate"] = smoke4_estimate(summary, data_report, smoke_manifest)
    result.update({k: summary.get(k) for k in ("loss", "eval", "eval_plan", "lr_at", "throughput", "times", "epoch",
                                               "memory", "peak_gib_max")})
    result["smoke_set"] = smoke_manifest
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    x = result["estimate"]
    L = ["# Smoke-test G4-4 (protocol-g4.md §9, amendement G4-4.5)", "",
         f"**{'Geslaagd' if result['passed'] else 'Niet geslaagd'}.** Schatting van de volle training (drie "
         f"doorgangen): {x.get('hours')} uur, {x.get('steps')} stappen (grens {MAX_HOURS3} uur; daarboven stopt de "
         f"keten vóór de training).", "", "| controle | waarde | gehaald |", "|---|---|---|"]
    for k, c in result["checks"].items():
        L.append(f"| {k} | `{json.dumps(c['value'], default=str)[:240]}` | {'ja' if c['holds'] else 'nee'} |")
    L += ["", f"Schatting [schatting, uit de smoke-test en build4]: {json.dumps(x)}", "",
          "Verlies per held-out set (stap, verlies, doeltokens, seconden): " + "; ".join(
              f"{e['set']} {e['step']} {e['loss']:.4f} {e['tokens']} {e['seconds']}" for e in summary.get("eval") or []),
          "", f"Smoke-set: {json.dumps(smoke_manifest)}", "",
          f"Training: {json.dumps({k: summary.get(k) for k in ('throughput', 'times', 'peak_gib_max')}, default=str)}",
          ""]
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"SMOKE4 {'PASSED' if result['passed'] else 'FAILED'} {json.dumps({k: c['holds'] for k, c in result['checks'].items()})} "
          f"PASSES {x.get('passes')} HOURS {x.get('hours')} estimate {json.dumps(x)}", flush=True)
    if not result["passed"] or x.get("passes") != 3:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------- G4-4: novels
# amendment of 16 Nov (G4-4.9-G4-4.11): a fifth eval set, unseen passages of seen novels (no en_orig work that passes
# the checks is outside G4-4's training), reported only
NOVELS_EVAL_SEED = f"{SEED4}/novels-eval"
NOVELS_EVAL_BOOKS = 10
NOVELS_EVAL_PAIRS = 50
NOVELS_EVAL_MIN_ROWS = 100           # candidate rows a book needs: room for 50 after the filters


def author_key(name):
    """An author as a key: lower case letters only, the name's parts sorted ("Rand, Ayn" and "Ayn Rand" are one)"""
    return " ".join(sorted(re.findall(r"[^\W\d_]+", (name or "").lower())))


def example_digest(user, assistant):
    return hashlib.sha256(f"{user}\x00{assistant}".encode()).digest()


def g44_trained(train_path, meta_path):
    """(novel pids of G4-4's set, the 80-character keys of every training example's English and Dutch, the digests of
    every training example's user and assistant turns)"""
    import pyarrow.parquet as pq

    pids, keys, digests = set(), set(), set()
    meta = pq.ParquetFile(meta_path)
    for batch in meta.iter_batches(columns=["id", "kind", "en"], batch_size=100_000):
        for i, kind, en in zip(*(batch.column(c).to_pylist() for c in ("id", "kind", "en"))):
            if kind == "novel":
                if not i.startswith("novel:"):
                    raise SystemExit(f"{meta_path}: a novel example with id {i}")
                pids.add(i[len("novel:"):])
            if (k := work_key(en or "")) is not None:
                keys.add(k)
    n = 0
    for batch in pq.ParquetFile(train_path).iter_batches(columns=["messages"], batch_size=100_000):
        for turns in batch.column("messages").to_pylist():
            if [t["role"] for t in turns] != ["user", "assistant"]:
                raise SystemExit(f"{train_path}: an example without one user and one assistant turn")
            digests.add(example_digest(turns[0]["content"], turns[1]["content"]))
            if (k := work_key(turns[1]["content"])) is not None:
                keys.add(k)
            n += 1
    if n != meta.metadata.num_rows:
        raise SystemExit(f"{train_path}: {n} examples, {meta_path} {meta.metadata.num_rows}")
    return pids, keys, digests, n


def novels_eval_draw(by_book, trained_pids, train_keys, train_digests, blocked, counter, seed=NOVELS_EVAL_SEED,
                     books=NOVELS_EVAL_BOOKS, pairs=NOVELS_EVAL_PAIRS, min_rows=NOVELS_EVAL_MIN_ROWS):
    """(rows, info), amendment G4-4.10: of every English book its rows that are no novel pair of G4-4; books with at
    least min_rows of them and an author; `books` authors drawn, one book each; per book its rows in a seeded order and
    the first `pairs` without a 13-gram of blocked, within NOVEL_MAX_TOKENS, without an 80-character key of a training
    example and not a training example"""
    candidates = {en: [r for r in rows if str(r["id"]) not in trained_pids] for en, rows in by_book.items()}
    by_author = collections.defaultdict(list)
    for en, rows in candidates.items():
        first = min(by_book[en], key=lambda r: r["id"])
        a = author_key(first["author"])
        if len(rows) >= min_rows and a:
            by_author[a].append(en)
    authors = sorted(by_author)
    if len(authors) < books:
        raise SystemExit(f"{len(authors)} authors with a book of >= {min_rows} candidate rows, {books} wanted")
    chosen_authors = random.Random(f"{seed}/authors").sample(authors, books)
    out, info_books, stats = [], [], collections.Counter()
    for a in chosen_authors:
        en = random.Random(f"{seed}/book/{a}").choice(sorted(by_author[a], key=_id))
        rows = sorted(candidates[en], key=lambda r: r["id"])
        random.Random(f"{seed}/rows/{en}").shuffle(rows)
        kept, i, seen = [], 0, collections.Counter()
        while len(kept) < pairs and i < len(rows):
            part = rows[i:i + pairs - len(kept) + 8]
            i += len(part)
            prompts, targets = counter.examples(part)
            for r, p, t in zip(part, prompts, targets):
                if len(kept) == pairs:
                    break
                seen["looked_at"] += 1
                x = messages_row(r["en"], r["nl"])
                if shares_gram(r, blocked):
                    seen["eval_13gram"] += 1
                elif p + t > NOVEL_MAX_TOKENS:
                    seen["too_long"] += 1
                elif any(k in train_keys for k in (work_key(r["en"]), work_key(r["nl"])) if k is not None):
                    seen["train_key"] += 1
                elif example_digest(x["messages"][0]["content"], x["messages"][1]["content"]) in train_digests:
                    seen["train_example"] += 1
                else:
                    kept.append({"pid": str(r["id"]), "en_book_id": str(en), "book": r["book"], "author": r["author"],
                                 "author_key": a, "en": r["en"], "nl": r["nl"], "tokens_prompt": p, "tokens_target": t})
        if len(kept) < pairs:
            raise SystemExit(f"book {en} ({a}): {len(kept)} pairs, {pairs} wanted")
        stats.update(seen)
        info_books.append({"en_book_id": str(en), "book": kept[0]["book"], "author": kept[0]["author"],
                           "author_key": a, "books_of_author": len(by_author[a]), "candidate_rows": len(rows),
                           "trained_pairs": len(by_book[en]) - len(rows), "pairs": len(kept), **dict(seen)})
        out += kept
    out = random.Random(f"{seed}/order").sample(out, len(out))
    info = {"seed": seed, "books_wanted": books, "pairs_per_book": pairs, "min_candidate_rows": min_rows,
            "books_with_rows": len(by_book), "candidate_rows": sum(map(len, candidates.values())),
            "eligible_books": sum(map(len, by_author.values())), "eligible_authors": len(authors),
            "authors": chosen_authors, "books": info_books, **dict(sorted(stats.items())), "pairs": len(out)}
    return out, info


def novels4(args, counter=None):
    """amendment G4-4.10: eval-novels.parquet, unseen passages of seen novels: the -b1- rows through build4's steps
    (checked against G4-4's data report), the rows G4-4 does not train on, NOVELS_EVAL_BOOKS books of as many authors,
    NOVELS_EVAL_PAIRS pairs each; no pair, 80-character key or example of G4-4's training set among them"""
    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    report4 = json.loads(Path(args.data_report).read_text())
    if report4.get("arm") != ARM4:
        raise SystemExit(f"{args.data_report}: not G4-4's data report")
    manifest4 = json.loads((Path(args.g44) / "manifest.json").read_text())
    train, meta = Path(args.g44) / "train.parquet", Path(args.g44) / "train-meta.parquet"
    inputs = {"novels": sha256_file(args.novels), "train": sha256_file(train), "train_meta": sha256_file(meta),
              "evalsets": sha256_file(args.evalsets), "blind": sha256_file(args.blind),
              "data_report": sha256_file(args.data_report), "manifest": sha256_file(Path(args.g44) / "manifest.json")}
    if inputs["novels"] != report4["novels"]["sha256"] or inputs["novels"] != manifest4["inputs"]["novels"]["sha256"]:
        raise SystemExit(f"{args.novels}: not the -b1- file of G4-4's set")
    if inputs["evalsets"] != manifest4["inputs"]["evalsets"] or inputs["blind"] != manifest4["inputs"]["blind"]:
        raise SystemExit("not the evaluation or blind files of G4-4's set")
    blocked = checked_blocked(args.evalsets, args.blind)
    print("PROGRESS reading G4-4's training set", flush=True)
    trained_pids, train_keys, train_digests, n_train = g44_trained(train, meta)
    if n_train != manifest4["examples"] or len(trained_pids) != report4["novels"]["distinct_pairs"]:
        raise SystemExit(f"G4-4's set: {n_train} examples, {len(trained_pids)} novel pairs; the manifest says "
                         f"{manifest4['examples']}, the report {report4['novels']['distinct_pairs']}")
    print("PROGRESS the novels through build4's steps", flush=True)
    rows, groups = read_b1(args.novels)
    chosen, editions = b1_editions(rows)
    chosen, works = same_works(rows, chosen)
    del rows
    chosen, near = near_duplicates(chosen)
    if (near["books_left"], near["rows_left"]) != (report4["novels"]["near_duplicates"]["books_left"],
                                                   report4["novels"]["near_duplicates"]["rows_left"]):
        raise SystemExit(f"build4's steps give {near['books_left']} books, {near['rows_left']} rows; G4-4's report "
                         f"{report4['novels']['near_duplicates']}")
    found = sum(str(r["id"]) in trained_pids for v in chosen.values() for r in v)
    if found != len(trained_pids):
        raise SystemExit(f"{found} of G4-4's {len(trained_pids)} novel pairs among the rows of build4's steps")
    drawn, info = novels_eval_draw(chosen, trained_pids, train_keys, train_digests, blocked, counter,
                                   books=NOVELS_EVAL_BOOKS, pairs=NOVELS_EVAL_PAIRS, min_rows=NOVELS_EVAL_MIN_ROWS)
    # the checks again on the output: no pair of the training, no key, no example, no 13-gram, the token bound
    msgs = [messages_row(r["en"], r["nl"]) for r in drawn]
    again = {"trained_pid": sum(r["pid"] in trained_pids for r in drawn),
             "train_key": sum(any(k in train_keys for k in (work_key(r["en"]), work_key(r["nl"])) if k is not None)
                              for r in drawn),
             "train_example": sum(example_digest(x["messages"][0]["content"], x["messages"][1]["content"])
                                  in train_digests for x in msgs),
             "eval_13gram": sum(shares_gram(r, blocked) for r in drawn),
             "too_long": sum(r["tokens_prompt"] + r["tokens_target"] > NOVEL_MAX_TOKENS for r in drawn),
             "pids_twice": len(drawn) - len({r["pid"] for r in drawn})}
    if any(again.values()):
        raise SystemExit(f"the drawn pairs fail a check: {again}")
    write_parquet(msgs, out / "eval-novels.parquet")
    meta_rows = [{k: r[k] for k in ("pid", "en_book_id", "book", "author", "tokens_prompt", "tokens_target")}
                 for r in drawn]
    write_parquet(meta_rows, out / "eval-novels-meta.parquet")
    lengths = [r["tokens_prompt"] + r["tokens_target"] for r in drawn]
    # the eval batches per process as MaxText packs the file at SEQ3 on the v4-32; sft_lora.py runs the smallest count
    # on every process (a3c9e62), which must be at least one
    eval_batches = pack_batches(lengths)
    if min(eval_batches) < 1:
        raise SystemExit(f"a process gets no eval batch of the novels: {eval_batches}")
    manifest = {"set": "novels", "name": "unseen passages of seen novels", "arm": ARM4,
                "eval_batches_per_process": eval_batches, "eval_batches_run": min(eval_batches),
                "note": "the books are in G4-4's training (up to 60 pairs each): the loss is more optimistic than on "
                        "unseen books (amendment G4-4.9)",
                "prompt": PROMPT, "format": "messages_row (natural-v1, Gemma 4's template; the Dutch as the answer)",
                "pairs": len(drawn), "tokens": sum(lengths), "target_tokens": sum(r["tokens_target"] for r in drawn),
                "max_tokens": max(lengths), "tokens_q": quantiles(lengths), "draw": info, "checks_on_output": again,
                "build4_steps": {"editions": {k: editions[k] for k in ("english_books", "english_books_with_rows",
                                                                       "rows_kept")},
                                 "same_works_dropped": works["books_dropped"], "near_duplicates": near,
                                 "groups": groups},
                "g4_4": {"examples": n_train, "novel_pairs": len(trained_pids), "keys": len(train_keys),
                         "distinct_examples": len(train_digests)},
                "inputs": {"paths": {"novels": str(args.novels), "g44": str(args.g44), "evalsets": str(args.evalsets),
                                     "blind": str(args.blind), "data_report": str(args.data_report)},
                           "sha256": inputs},
                "files": {"eval-novels.parquet": sha256_file(out / "eval-novels.parquet"),
                          "eval-novels-meta.parquet": sha256_file(out / "eval-novels-meta.parquet")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"NOVELS4 {json.dumps({'pairs': len(drawn), 'tokens': sum(lengths), 'max_tokens': max(lengths), 'books': [(b['en_book_id'], b['author'], b['pairs'], b['candidate_rows']) for b in info['books']], 'checks': again}, ensure_ascii=False)}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- G4-5
# draft amendment G4-5 (Yeb, 17 Nov): one LoRA per source of G4-4's mix, each branched from a shared warmup with its
# whole training state (LoRA parameters and AdamW's state), combined as W0 + Delta_s + sum_i w_i (Delta_i - Delta_s)
# with weights searched on the search part of the rules' evaluation sets; the rules decide on the decision part

ARM5 = "G4-5"
SEED5 = f"{SEED}/G4-5"
BT_BRANCHES5 = {"bt_handelingen": "handelingen", "bt_raad": "raad", "bt_fineweb": "fineweb",
                "bt_rechtspraak": "rechtspraak", "bt_dpc": "dpc"}
BRANCHES5 = (*BT_BRANCHES5, "books", "novels", "flan")
WARMUP5 = "warmup"
WARMUP_STEPS5 = 1000                 # the shared warmup: past the jump of G4-4's curve (~96% of its drop by step 750)
LR_WARMUP_STEPS5 = 279               # G4-4's linear LR warmup, int(0.03 x 9,316): the warmup follows G4-4's start
BRANCH_STEPS5 = 1000                 # every branch: ~10 epochs of the smallest source (the Dutch books, ~103 steps)
DECAY_FRACTION5 = 0.1                # every branch's cooldown: the last 10% of its own steps, linear to LR / 10
LR5 = 1e-5                           # the warmup's constant LR, continued by the branches (Yeb, 17 Nov, point 4)
RANK5, ALPHA5 = 32, 16.0             # with the warmup's state carried over, every branch has the warmup's rank
EVAL_EVERY5 = 500
FLANVAL5 = 500                       # G4-4's replay rows held out of the FLAN branch: the eval set flanval
MAX_EPOCHS5 = 60
SPLIT_SHARE5 = 0.5
SPLIT_SETS5 = ("wmt24pp", "flores", "ntrex", "flan")
SEARCH_LOSS_SETS5 = ("wmt24pp", "flores", "ntrex")   # the search part's sets with a human reference: their losses
EVAL_SETS5 = EVAL_SETS4 + ("flanval", "wmt24pp_s", "flores_s", "ntrex_s")
OBJECTIVE5 = {"wmt24pp_s": 1.0, "flores_s": 1.0, "ntrex_s": 1.0}
REFQE_MAX_DIFF5 = 0.05               # the human references' QE again: its mean over a whole set within this of HUMAN_QE
SEARCH_BUDGET5, SEARCH_TOP_K5 = 80, 4  # the search's points (anchors included) and the candidates of the full evaluation
SMOKE5_STEPS = 8                     # the warmup's smoke run, then the branch's from its step 8: 8 steps each
SMOKE5_LR_WARMUP = 2
SMOKE5_DECAY = 0.5                   # the branch's smoke test: 4 of its 8 steps in the cooldown
SMOKE5_EVAL_STEPS = [0, 4, 8]
SMOKE5_BRANCH_EVAL_STEPS = [8, 12, 16]
EQUAL5 = 1e-6                        # the same model on the same eval batches: the same loss
MERGE_MAX_DIFF5 = 0.02               # a merged kernel (bf16 at use) against LoRA in the forward pass, same adapter
# addendum G4-5.7b (Yeb, 18 Nov): before the search, the merged warmup alone (w = 0) on these sets against the warmup's
# own eval at its step 1,000 (MERGE_MAX_DIFF5); g4mix.REPEAT_MAX_DIFF for the repeated first point
WARMUP_CHECK_SETS5 = ("books", "novels")
# addendum G4-5.7b: the temperature probe (reported, decides nothing): G4-0 and the warmup, eval only, the loss at
# logits / T for T on this grid; T chosen on the mean loss of TEMP_FIT_SETS5; T = 1 against the stack within TEMP_EQUAL5
TEMPERATURES5 = "0.5:3.0:0.05"
TEMP_FIT_SETS5 = ("books", "btval")
TEMP_EQUAL5 = 1e-4
# addendum G4-5.7b A (Yeb 18 Nov ~09:15): the search follows the calibrated objective if one temperature on G4-0 gives
# at least this share of the loss drop from G4-0 to the warmup on the objective's sets (mean of the per-set shares)
SEARCH_RULE_SHARE5 = 0.5


def g44_set(directory, report_path):
    """G4-4's set as {pass: [(messages row, meta row)]}, checked: the manifest, report-G4-4.json's examples and tokens
    per pass and kind, the passes in order, the examples outside the novels and the replay the same ids in every pass"""
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text())
    if (manifest.get("set"), manifest.get("seq"), manifest.get("seed"), manifest.get("epochs")) != (ARM4, SEQ3, SEED4,
                                                                                                    EPOCHS4):
        raise SystemExit(f"{d}: not G4-4's set ({manifest.get('set')}, seq {manifest.get('seq')})")
    if (d / "template.jinja").read_text(encoding="utf-8") != g4lora.TEMPLATE:
        raise SystemExit(f"{d}/template.jinja is not g4lora.TEMPLATE")
    msgs = read_rows(d / "train.parquet")
    metas = read_rows(d / "train-meta.parquet")
    if len(msgs) != len(metas) or len(metas) != manifest["examples"]:
        raise SystemExit(f"{d}: {len(msgs)} examples, {len(metas)} meta rows, manifest {manifest['examples']}")
    epochs = [m["epoch"] for m in metas]
    if epochs != sorted(epochs) or sorted(set(epochs)) != list(range(1, EPOCHS4 + 1)):
        raise SystemExit(f"{d}: the passes are not 1 ... {EPOCHS4} in order")
    passes = collections.defaultdict(list)
    for x, m in zip(msgs, metas):
        passes[m["epoch"]].append((x, m))
    report = json.loads(Path(report_path).read_text())
    if report.get("arm") != ARM4:
        raise SystemExit(f"{report_path}: not G4-4's data report")
    for p in report["passes"]:
        got = pass_kinds(passes[p["pass"]])
        if (p["examples"], p["tokens"], p["kinds"]) != (len(passes[p["pass"]]), sum(v["tokens"] for v in got.values()),
                                                        got):
            raise SystemExit(f"{d}: pass {p['pass']} is not the pass of {report_path}")
    rest = [sorted(m["id"] for _, m in one if m["kind"] not in ("novel", "replay")) for one in passes.values()]
    if any(r != rest[0] for r in rest):
        raise SystemExit(f"{d}: the examples outside the novels and the replay differ between the passes")
    return dict(passes), manifest, report


def sources5(passes, report):
    """{branch: [(x, meta)]} of G4-4's set (the BT per source, the books: pass 1; the FLAN replay: every pass's rows),
    and the novels per pass {pass: [(x, meta)]}; checked against report-G4-4.json"""
    out = {b: [] for b in BRANCHES5 if b != "novels"}
    novels = {e: [] for e in passes}
    by_source = {s: b for b, s in BT_BRANCHES5.items()}
    replay_ids = set()
    for e, one in sorted(passes.items()):
        for x, m in one:
            m = {k: v for k, v in m.items() if k != "epoch"}
            if m["kind"] == "novel":
                novels[e].append((x, m))
            elif m["kind"] == "replay":
                if m["id"] in replay_ids:
                    raise SystemExit(f"{m['id']}: a replay row in two passes")
                replay_ids.add(m["id"])
                out["flan"].append((x, m))
            elif e == 1 and m["kind"] == "bt":
                if m["source"] not in by_source:
                    raise SystemExit(f"{m['id']}: BT of an unknown source {m['source']}")
                out[by_source[m["source"]]].append((x, m))
            elif e == 1 and m["kind"] == "book":
                out["books"].append((x, m))
            elif m["kind"] not in ("bt", "book"):
                raise SystemExit(f"{m['id']}: an example of kind {m['kind']}")
    kinds = report["set"]["kinds"]
    want = kinds["bt"]["by_source"]
    got = {s: len(out[b]) for b, s in BT_BRANCHES5.items()}
    if got != want or len(out["books"]) != kinds["book"]["examples"]:
        raise SystemExit(f"BT per source {got} and books {len(out['books'])}: report-G4-4.json has {want} and "
                         f"{kinds['book']['examples']}")
    n_novels = [p["kinds"]["novel"]["examples"] for p in report["passes"]]
    if [len(novels[e]) for e in sorted(novels)] != n_novels or report["novels"]["distinct_pairs"] != sum(n_novels):
        raise SystemExit(f"novels per pass {[len(v) for v in novels.values()]}, report {n_novels}")
    if len(out["flan"]) != report["replay"]["distinct"]:
        raise SystemExit(f"{len(out['flan'])} replay rows, report-G4-4.json {report['replay']['distinct']}")
    return out, novels


def tokens_of(rows):
    return [m["tokens_prompt"] + m["tokens_target"] for _, m in rows]


def shortest_prefix(lengths, steps):
    """The fewest first examples whose packing (pack_batches) gives MaxText at least `steps` steps (steps_of)"""
    if steps_of(pack_batches(lengths)) < steps:
        raise SystemExit(f"{len(lengths)} examples give {steps_of(pack_batches(lengths))} steps, {steps} wanted")
    lo, hi = 1, len(lengths)
    while lo < hi:
        mid = (lo + hi) // 2
        if steps_of(pack_batches(lengths[:mid])) >= steps:
            hi = mid
        else:
            lo = mid + 1
    return lo


def epochs_rows(per_epoch, steps, seed):
    """Epoch after epoch (epoch e: per_epoch(e) in a fresh order, seed <seed>/epoch/<e>) until MaxText's packing gives
    `steps` steps; then the shortest prefix that does. Every row's meta carries its epoch."""
    rows = []
    for e in range(1, MAX_EPOCHS5 + 1):
        one = sorted(per_epoch(e), key=lambda p: p[1]["id"])
        if not one:
            raise SystemExit(f"{seed}: epoch {e} has no example")
        random.Random(f"{seed}/epoch/{e}").shuffle(one)
        rows += [(x, {**m, "epoch": e}) for x, m in one]
        if steps_of(pack_batches(tokens_of(rows))) >= steps:
            return rows[:shortest_prefix(tokens_of(rows), steps)]
    raise SystemExit(f"{seed}: {MAX_EPOCHS5} epochs give fewer than {steps} steps")


def write_set5(d, rows, name, steps, extra):
    """train.parquet, train-meta.parquet, template.jinja and manifest.json of one set of G4-5"""
    import pyarrow as pa

    meta_schema = pa.schema([(n, getattr(pa, "bool_" if t == "bool" else t)()) for n, t in META_SCHEMA3_FIELDS])
    write_parquet([x for x, _ in rows], d / "train.parquet", row_group_size=ROW_GROUP4)
    write_parquet([{n: m.get(n) for n in meta_schema.names} for _, m in rows], d / "train-meta.parquet", meta_schema,
                  row_group_size=ROW_GROUP4)
    (d / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    lengths = tokens_of(rows)
    batches = pack_batches(lengths)
    manifest = {"set": f"{ARM5}-{name}", "examples": len(rows), "tokens": sum(lengths), "max_tokens": max(lengths),
                "steps": steps, "batches_per_process": batches, "step_limit": steps_of(batches), "seq": SEQ3,
                "prompt": PROMPT, "template": "g4lora.TEMPLATE (Gemma 4, thinking off)",
                "kinds": dict(collections.Counter(m["kind"] for _, m in rows)), **extra}
    if manifest["step_limit"] < steps:
        raise SystemExit(f"{name}: {manifest['step_limit']} steps in the set, {steps} wanted")
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def smoke_set5(rows, seed):
    """As G4-3.5's smoke set: the SMOKE3_LONGEST longest examples first, then SMOKE3_RANDOM others (seeded)"""
    longest = sorted(rows, key=lambda p: (-(p[1]["tokens_prompt"] + p[1]["tokens_target"]), p[1]["id"]))
    longest = longest[:SMOKE3_LONGEST]
    taken = {m["id"] for _, m in longest}
    others = sorted({m["id"]: (x, m) for x, m in rows if m["id"] not in taken}.values(), key=lambda p: p[1]["id"])
    out = [(x, {**m, "epoch": 1}) for x, m in longest + random.Random(seed).sample(others, min(SMOKE3_RANDOM,
                                                                                               len(others)))]
    return out, len(longest)


def build5(args, counter=None):
    """draft amendment G4-5: from G4-4's set (its examples byte for byte) the shared warmup's set (pass 1 without the
    replay: BT, books, novels in G4-4's proportions, freshly shuffled, as many as WARMUP steps take) and one set per
    branch (its source's examples epoch after epoch, each epoch freshly shuffled, as many as the branch's steps take;
    the novels' epochs are G4-4's passes, so its pairs come before any repeat); FLAN without the FLANVAL5 rows of
    its eval set; the smoke sets of the warmup and of the branch with the longest example"""
    out = Path(args.out)
    steps = {b: args.branch_steps for b in BRANCHES5}
    for spec in args.steps or []:
        name, _, n = spec.partition("=")
        if name not in steps or not n.isdigit() or int(n) < 1:
            raise SystemExit(f"--steps {spec!r}: BRANCH=N with a branch of {BRANCHES5}")
        steps[name] = int(n)
    print("PROGRESS reading G4-4's set", flush=True)
    passes, manifest4, report4 = g44_set(args.g44, args.g44_report)
    sources, novels = sources5(passes, report4)
    held = {str(r["pid"]) for r in read_rows(args.novels_eval, ["pid"])}
    trained = {m["id"][len("novel:"):] for one in novels.values() for _, m in one}
    if held & trained:
        raise SystemExit(f"{len(held & trained)} rows of the novels' eval set are novel pairs of G4-4")
    flan = sorted(sources["flan"], key=lambda p: p[1]["id"])
    flanval = random.Random(f"{SEED5}/flanval").sample(flan, args.flanval)
    val_ids = {m["id"] for _, m in flanval}
    sources["flan"] = [p for p in flan if p[1]["id"] not in val_ids]
    report = {"arm": ARM5, "seed": SEED5, "seq": SEQ3, "prompt": PROMPT, "warmup_steps": args.warmup_steps,
              "branch_steps": steps, "g4_4": {"manifest": {k: manifest4[k] for k in ("set", "examples", "tokens",
                                                                                      "steps", "seed")},
                                              "files": {f: sha256_file(Path(args.g44) / f)
                                                        for f in ("train.parquet", "train-meta.parquet")},
                                              "report": sha256_file(args.g44_report)},
              "novels_eval": {"path": str(args.novels_eval), "sha256": sha256_file(args.novels_eval),
                              "rows": len(held)}, "sets": {}}
    # the shared warmup: pass 1 of G4-4 without the replay (FLAN not in the warmup: Yeb, 17 Nov), a fresh order
    first = [(x, {k: v for k, v in m.items() if k != "epoch"}) for x, m in passes[1] if m["kind"] != "replay"]
    first.sort(key=lambda p: p[1]["id"])
    random.Random(f"{SEED5}/warmup").shuffle(first)
    n = shortest_prefix(tokens_of(first), args.warmup_steps)
    warm = [(x, {**m, "epoch": 1}) for x, m in first[:n]]
    d = out / f"{ARM5}-{WARMUP5}"
    report["sets"][WARMUP5] = write_set5(d, warm, WARMUP5, args.warmup_steps,
                                         {"seed": f"{SEED5}/warmup", "pool": len(first),
                                          "pool_kinds": dict(collections.Counter(m["kind"] for _, m in first))})
    print(f"PROGRESS warmup {json.dumps(report['sets'][WARMUP5])}", flush=True)
    longest_branch, longest = None, -1
    branch_rows = {}
    def per_epoch(branch):
        if branch == "novels":
            return lambda e: novels[(e - 1) % len(novels) + 1]
        return lambda e: sources[branch]

    for b in BRANCHES5:
        unit = sum(len(v) for v in novels.values()) if b == "novels" else len(sources[b])
        rows = epochs_rows(per_epoch(b), steps[b], f"{SEED5}/{b}")
        branch_rows[b] = rows
        if b == "novels":
            epochs = len(rows) / unit
            distinct = len({m["id"] for _, m in rows})
        else:
            epochs = len(rows) / unit
            distinct = min(len(rows), unit)
        report["sets"][b] = write_set5(out / f"{ARM5}-{b}", rows, b, steps[b],
                                       {"seed": f"{SEED5}/{b}", "rows_per_epoch": unit if b != "novels" else
                                        [len(v) for _, v in sorted(novels.items())], "epochs": round(epochs, 3),
                                        "distinct": distinct, "source_tokens": sum(tokens_of(
                                            [p for v in novels.values() for p in v] if b == "novels" else sources[b]))})
        top = max(tokens_of(rows))
        if top > longest:
            longest_branch, longest = b, top
        print(f"PROGRESS {b} {json.dumps(report['sets'][b])}", flush=True)
    # the FLAN branch's eval set: G4-4's replay rows it does not train on
    flan_ids = {m["id"] for _, m in branch_rows["flan"]}
    if flan_ids & val_ids:
        raise SystemExit("a flanval row in the FLAN branch")
    write_parquet([x for x, _ in flanval], out / "eval-flanval.parquet")
    write_parquet([{"id": m["id"], "source": m["source"], "tokens": m["tokens_prompt"] + m["tokens_target"]}
                   for _, m in flanval], out / "eval-flanval-meta.parquet")
    report["flanval"] = {"rows": len(flanval), "seed": f"{SEED5}/flanval", "tokens": sum(tokens_of(flanval)),
                         "batches_per_process": pack_batches(tokens_of(flanval))}
    # the smoke sets: the warmup's, and the branch's with the longest example (the memory of the longest rows)
    smoke = {}
    for name, rows in ((WARMUP5, warm), (longest_branch, branch_rows[longest_branch])):
        srows, n_long = smoke_set5(rows, f"{SEED5}/smoke/{name}")
        tag = "smoke-warmup" if name == WARMUP5 else "smoke-branch"
        smoke[tag] = write_set5(out / f"{ARM5}-{tag}", srows, tag, SMOKE5_STEPS,
                                {"from": name, "longest": n_long,
                                 "min_of_longest": min(tokens_of(srows)[:n_long])})
    report["smoke"] = smoke
    total = args.warmup_steps + sum(steps.values())
    report["estimate"] = {"steps": total, "tokens": sum(s["tokens"] for s in report["sets"].values())}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    (out / "report.md").write_text(report5_data_md(report), encoding="utf-8")
    print(f"BUILD5 {json.dumps({k: (v['examples'], v['tokens'], v['steps'], v.get('epochs')) for k, v in report['sets'].items()})} "
          f"smoke-branch {longest_branch}", flush=True)


BRANCH_NAMES5 = {"warmup": "warmup (BT, boeken, romans; zonder FLAN)", "bt_handelingen": "BT Handelingen",
                 "bt_raad": "BT Openraadsinformatie", "bt_fineweb": "BT FineWeb-2", "bt_rechtspraak": "BT Rechtspraak",
                 "bt_dpc": "BT Rijksoverheid (DPC)", "books": "Nederlandse boeken",
                 "novels": "romans (Engels origineel)", "flan": "FLAN-replay"}


def report5_data_md(r):
    L = [f"# Trainingssets {r['arm']} (concept-amendement G4-5)", "",
         "Gemaakt door `g4arm.py build5` uit de set van G4-4 (voorbeelden byte voor byte; bestanden met sha256 "
         f"{json.dumps(r['g4_4']['files'])}). Elke epoch van een tak is opnieuw geschud (seed `{r['seed']}/<tak>/epoch/<e>`); "
         "de romantak neemt de doorgangen van G4-4 als epochs; FLAN zonder de rijen van `flanval`.", "",
         "| set | voorbeelden | tokens | stappen | epochs | verschillende | rijen per epoch | batches per proces |",
         "|---|---:|---:|---:|---:|---:|---|---|"]
    for k, s in r["sets"].items():
        L.append(f"| {BRANCH_NAMES5.get(k, k)} | {_n(s['examples'])} | {_n(s['tokens'])} | {_n(s['steps'])} | "
                 f"{s.get('epochs', '–')} | {_n(s.get('distinct', s['examples']))} | {s.get('rows_per_epoch', s.get('pool'))} | "
                 f"{s['batches_per_process']} |")
    L += ["", f"Samen {_n(r['estimate']['steps'])} stappen, {_n(r['estimate']['tokens'])} tokens. `flanval`: "
          f"{r['flanval']['rows']} rijen (seed `{r['flanval']['seed']}`). Smoke-sets: {json.dumps(r['smoke'])}", ""]
    return "\n".join(L) + "\n"


def split5(args, counter=None):
    """draft amendment G4-5: the search and decision parts of the rules' sets (WMT24++, FLORES, NTREX, FLAN), fixed
    before any result of G4-5: per set and stratum (WMT24++, FLORES: the domain) its units (FLORES: articles, else
    segments) in a seeded order, half of the segments to the search part; the blind passages' WMT24++ segments to the
    decision part. The search part's sets with a human reference as eval files (natural-v1, the reference as the
    answer)."""
    import g4mix

    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.evalsets)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["set"]].append(r)
    want = {**FOUR_SETS, "flan": N_FLAN, "blind": N_BLIND}
    if {s: len(v) for s, v in by.items()} != want:
        raise SystemExit(f"{args.evalsets}: {dict((s, len(v)) for s, v in by.items())}, expected {want}")
    ids = {r["id"] for r in rows}
    forced = {json.loads(r["meta"]).get("item") for r in by["blind"]} & ids
    assign, sets = {}, {}
    for s in SPLIT_SETS5:
        part = g4mix.split_set(by[s], f"{SEED5}/split/{s}", args.share, forced)
        assign.update(part)
        units, strata = collections.defaultdict(set), collections.defaultdict(collections.Counter)
        for r in by[s]:
            units[part[r["id"]]].add(g4mix.unit_of(r))
            strata[part[r["id"]]][g4mix.stratum_of(r)] += 1
        sets[s] = {p: {"segments": sum(v == p for v in part.values()), "units": len(units[p]),
                       "strata": dict(sorted(strata[p].items()))} for p in ("search", "decision")}
        sets[s]["forced_decision"] = len(forced & {r["id"] for r in by[s]})
        if s == "wmt24pp":
            good = [r for r in by[s] if not bad_source(r["meta"])]
            for p in ("search", "decision"):
                sets[s][p]["good"] = sum(part[r["id"]] == p for r in good)
    files = {}
    for s in SEARCH_LOSS_SETS5:
        keep = [r for r in sorted(by[s], key=lambda r: r["id"])
                if assign[r["id"]] == "search" and not (s == "wmt24pp" and bad_source(r["meta"]))]
        pairs = [{"en": r["source"], "nl": r["reference"]} for r in keep]
        prompts, targets = counter.examples(pairs)
        lengths = [a + b for a, b in zip(prompts, targets)]
        if max(lengths) > SEQ3:
            raise SystemExit(f"{s}: an example of {max(lengths)} tokens, more than {SEQ3}")
        name = f"eval-{s}-s.parquet"
        write_parquet([messages_row(p["en"], p["nl"]) for p in pairs], out / name)
        write_parquet([{"id": r["id"], "tokens": n} for r, n in zip(keep, lengths)], out / f"eval-{s}-s-meta.parquet")
        files[name] = {"rows": len(keep), "tokens": sum(lengths), "target_tokens": sum(targets),
                       "batches_per_process": pack_batches(lengths), "sha256": sha256_file(out / name)}
    split = {"arm": ARM5, "seed": f"{SEED5}/split/<set>/<stratum>", "share": args.share,
             "unit": "per stratum (WMT24++ and FLORES: the domain): FLORES its article (URL), else the segment; the "
                     "blind passages' WMT24++ segments in the decision part", "sets": sets,
             "files": files, "inputs": {"evalsets": sha256_file(args.evalsets)}, "assign": dict(sorted(assign.items()))}
    split["digest"] = g4mix.digest(split["assign"])
    (out / "split.json").write_text(json.dumps(split, indent=1) + "\n")
    print(f"SPLIT5 {json.dumps({'sets': sets, 'files': {k: (v['rows'], v['batches_per_process']) for k, v in files.items()}, 'digest': split['digest']})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- G4-5 smoke

def eval_at(summary, step):
    return {e["set"]: e["loss"] for e in summary.get("eval") or [] if e["step"] == step}


def run_checks(summary, adapter, manifest, steps, eval_steps, prefix):
    """The checks of G4-3.5 that hold for every training run of G4-5's smoke test"""
    checks = {}
    loss = summary.get("loss") or []
    checks[f"{prefix}steps"] = {"value": summary.get("steps_done"), "holds": summary.get("steps_done") == steps}
    checks[f"{prefix}loss_finite"] = {"value": [v for _, v in loss],
                                      "holds": len(loss) == steps and all(math.isfinite(v) for _, v in loss)}
    checks[f"{prefix}peak_gib"] = {"value": summary.get("peak_gib_max"),
                                   "holds": summary.get("peak_gib_max") is not None
                                   and summary["peak_gib_max"] <= MEM_MAX_GIB}
    checks[f"{prefix}adapter"] = {"value": {k: adapter.get(k) for k in ("rank", "alpha", "scale")},
                                  "holds": (adapter.get("rank"), adapter.get("alpha"), adapter.get("scale")) ==
                                  (RANK5, ALPHA5, ALPHA5 / RANK5)}
    seq = (summary.get("config") or {}).get("max_target_length")
    checks[f"{prefix}seq"] = {"value": seq, "holds": seq == SEQ3}
    got = (summary.get("epoch") or {}).get("batches_per_process")
    checks[f"{prefix}batches_as_build5"] = {"value": {"pipeline": got, "build5": manifest.get("batches_per_process")},
                                            "holds": got is not None and got == manifest.get("batches_per_process")}
    evals = summary.get("eval") or []
    for name in EVAL_SETS5:
        rows = [e for e in evals if e["set"] == name]
        checks[f"{prefix}eval_{name}"] = {
            "value": [(e["step"], round(e["loss"], 4), e["tokens"]) for e in rows],
            "holds": [e["step"] for e in rows] == eval_steps
            and all(math.isfinite(e["loss"]) and e["tokens"] >= 1 for e in rows)}
    return checks


def lr_is(summary, step, value):
    got = (summary.get("lr_at") or {}).get(str(step))
    return {"value": {"step": step, "lr": got, "want": value}, "holds": got is not None and abs(got - value) <= 1e-12}


def smoke5_verdict(w, wa, b, ba, c, wm, bm, g43, g43_novels):
    """The smoke test of G4-5: the warmup (8 steps), a branch resumed at its step 8 (8 more, with a cooldown), the
    search run on the warmup, that branch and G4-3's final adapter"""
    checks = run_checks(w, wa, wm, 8, SMOKE5_EVAL_STEPS, "warmup_")
    checks["warmup_lr_constant_at_end"] = lr_is(w, 7, LR5)
    own0 = eval_at(w, 0).get("own")
    checks["warmup_own_loss_step0"] = {"value": own0, "holds": own0 is not None and own0 <= OWN_MAX_LOSS + TOLERANCE}
    checks.update(run_checks(b, ba, bm, 8, SMOKE5_BRANCH_EVAL_STEPS, "branch_"))
    r = b.get("resume") or {}
    checks["branch_resumed"] = {"value": r, "holds": r.get("step") == 8 and b.get("steps") == 16
                                and b.get("own_steps") == 8 and bool(r.get("counters"))}
    checks["branch_lr_continues"] = lr_is(b, 8, LR5)
    checks["branch_lr_cooldown_end"] = lr_is(b, 15, LR5 * 0.1)
    w8, b8 = eval_at(w, 8), eval_at(b, 8)
    checks["branch_start_is_warmup_end"] = {
        "value": {s: (w8.get(s), b8.get(s)) for s in EVAL_SETS5},
        "holds": all(s in w8 and s in b8 and abs(w8[s] - b8[s]) <= EQUAL5 for s in EVAL_SETS5)}
    x = c.get("combine") or {}
    checks["search_ran"] = {"value": {k: (x.get("search") or {}).get(k) for k in ("evaluations", "stopped", "best")},
                            "holds": bool(x) and (x.get("search") or {}).get("evaluations", 0) >= 4}
    checks["search_repeat_equal"] = {"value": x.get("repeat"), "holds": bool((x.get("repeat") or {}).get("equal"))}
    c0, w0 = eval_at(c, 0), eval_at(w, 0)
    checks["search_base_is_base"] = {"value": {s: (w0.get(s), c0.get(s)) for s in EVAL_SETS5},
                                     "holds": all(s in c0 and s in w0 and abs(c0[s] - w0[s]) <= EQUAL5
                                                  for s in EVAL_SETS5)}
    points = {p["phase"]: p for p in (x.get("search") or {}).get("points", [])}
    names = [comp["name"] for comp in (x.get("spec") or {}).get("components", [])]
    warm = (points.get("anchor/warmup") or {}).get("losses") or {}
    checks["merge_warmup_is_lora"] = {"value": {s: (warm.get(s), w8.get(s)) for s in warm},
                                      "holds": bool(warm) and all(abs(v - w8[s]) <= MERGE_MAX_DIFF5
                                                                  for s, v in warm.items())}
    g = (points.get(f"anchor/alone/{names.index('g43') - 1}") if "g43" in names else None) or {}
    want = {"books": [e["loss"] for e in g43.get("eval") or [] if e["set"] == "books"][-1:],
            "novels": [e["loss"] for e in g43_novels.get("eval") or [] if e["set"] == "novels" and e["step"] == 0]}
    got = g.get("losses") or {}
    checks["merge_g43_is_g43"] = {"value": {s: (got.get(s), (want[s] or [None])[0]) for s in want},
                                  "holds": all(s in got and want[s] and abs(got[s] - want[s][0]) <= MERGE_MAX_DIFF5
                                               for s in want)}
    checks["search_peak_gib"] = {"value": c.get("peak_gib_max"),
                                 "holds": c.get("peak_gib_max") is not None and c["peak_gib_max"] <= MEM_MAX_GIB}
    return {"checks": checks, "passed": all(v["holds"] for v in checks.values())}


def smoke5_estimate(w, b, c, data_report, wm, bm):
    """[estimate] the warmup and every branch at the smoke test's step and eval times (G4-3.5's method, evals every
    EVAL_EVERY5), the search at the smoke search's seconds per point, from build5's report"""
    sets = data_report["sets"]
    out = {"runs": {}}
    out["runs"][WARMUP5] = training_time(w, sets[WARMUP5]["steps"], sets[WARMUP5]["tokens"], wm["tokens"],
                                         EVAL_EVERY5)
    for name in BRANCHES5:
        out["runs"][name] = training_time(b, sets[name]["steps"], sets[name]["tokens"], bm["tokens"], EVAL_EVERY5)
    x = c.get("combine") or {}
    per_point = [e["seconds"] for e in x.get("evals") or [] if e["tag"].startswith("search/")]
    top = [e["seconds"] for e in x.get("evals") or [] if e["tag"].startswith("top/")]
    if per_point and all(v is not None for v in out["runs"].values()):
        t = c.get("times") or {}
        points = SEARCH_BUDGET5
        seconds = ((t.get("python_to_setup_s") or 0) + (t.get("setup_s") or 0) + (t.get("compile_s_est") or 0)
                   + points * statistics.fmean(per_point) + SEARCH_TOP_K5 * (max(top) if top else 0))
        out["search"] = {"points": points, "point_s": round(statistics.fmean(per_point), 1),
                         "hours": round(seconds / 3600, 2)}
        out["hours"] = round(sum(v["hours"] for v in out["runs"].values()) + out["search"]["hours"], 1)
    else:
        out["hours"] = None
    return out


def smoke5(args):
    """The smoke test's verdict of G4-5 and the estimate of the warmup, the branches and the search; exit 1 if a check
    fails or the estimate is above MAX_HOURS3"""
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    w, wa, b, ba, c = (load(p) for p in (args.warmup_summary, args.warmup_adapter, args.branch_summary,
                                         args.branch_adapter, args.combine_summary))
    wm, bm, data_report = load(args.warmup_manifest), load(args.branch_manifest), load(args.data_report)
    if data_report.get("arm") != ARM5:
        raise SystemExit(f"{args.data_report}: not G4-5's data report")
    result = smoke5_verdict(w, wa, b, ba, c, wm, bm, load(args.g43_summary), load(args.g43_novels_summary))
    result["estimate"] = smoke5_estimate(w, b, c, data_report, wm, bm)
    result["combine"] = c.get("combine")
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    x = result["estimate"]
    L = ["# Smoke-test G4-5 (concept-amendement G4-5)", "",
         f"**{'Geslaagd' if result['passed'] else 'Niet geslaagd'}.** Schatting van warmup, takken en zoektocht: "
         f"{x.get('hours')} uur (grens {MAX_HOURS3} uur; daarboven stopt de keten vóór de training).", "",
         "| controle | waarde | gehaald |", "|---|---|---|"]
    for k, v in result["checks"].items():
        L.append(f"| {k} | `{json.dumps(v['value'], default=str)[:240]}` | {'ja' if v['holds'] else 'nee'} |")
    L += ["", f"Schatting [schatting, uit de smoke-test en build5]: {json.dumps(x)}", ""]
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"SMOKE5 {'PASSED' if result['passed'] else 'FAILED'} {json.dumps({k: v['holds'] for k, v in result['checks'].items()})} "
          f"HOURS {x.get('hours')}", flush=True)
    if not result["passed"] or x.get("hours") is None or x["hours"] > MAX_HOURS3:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------- G4-5 report

def part_ids5(evalsets, assign, part):
    """{set: [ids]} of a part: wmt24pp the segments without is_bad_source, flores, ntrex, flan"""
    out = {s: [] for s in SPLIT_SETS5}
    for i, r in sorted(evalsets.items()):
        if r["set"] in out and assign.get(i) == part and not (r["set"] == "wmt24pp" and bad_source(r["meta"])):
            out[r["set"]].append(i)
    return out


def part_measures5(scored, predictions, refqe, ids, base):
    """Per model on one part: the mean QE per (a1) set and the human reference's on the same segments, (a1); the share
    "original" on WMT24++ + FLAN pooled and its difference to the base, (b)"""
    human = {s: mean(refqe[i] for i in ids[s]) for s in HUMAN_QE}
    wanted = {("wmt24pp", i) for i in ids["wmt24pp"]} | {("flan", i) for i in ids["flan"]}
    shares = {}
    for name in scored:
        got = [r["p_original"] >= 0.5 for r in predictions if r["model"] == name and (r["set"], r["id"]) in wanted]
        if len(got) != len(wanted):
            raise SystemExit(f"{name}: {len(got)} classifier texts of the {len(wanted)} of the part")
        shares[name] = sum(got) / len(got)
    out = {}
    for name, rows in scored.items():
        qe = {s: mean(rows[i]["metricx_qe"] for i in ids[s]) for s in HUMAN_QE}
        a1 = {s: qe[s] is not None and qe[s] <= human[s] + TOLERANCE for s in HUMAN_QE}
        out[name] = {"qe": qe, "a1_sets": a1, "a1": all(a1.values()),
                     "excess": max(qe[s] - human[s] for s in HUMAN_QE), "share": shares[name],
                     "b_delta": shares[name] - shares[base], "b": shares[name] - shares[base] >= RULE_B - TOLERANCE}
    return {"human": human, "n": {s: len(v) for s, v in ids.items()}, "models": out}


def select5(candidates, search_part):
    """The fixed rule: of the candidates that hold (a1) on the search part, the largest (b) difference there (equal:
    the earlier one of the top K, i.e. the lower search loss); if none holds (a1), the smallest largest QE excess"""
    m = search_part["models"]
    eligible = [c for c in candidates if m[c]["a1"]]
    if eligible:
        return min(eligible, key=lambda c: (-m[c]["b_delta"], candidates.index(c))), "a1 on the search part, max (b)"
    return min(candidates, key=lambda c: (m[c]["excess"], -m[c]["b_delta"], candidates.index(c))), \
        "no candidate holds (a1) on the search part: the smallest QE excess"


def training_table5(runs):
    """Per run the eval loss at its first and last step per set, the branches also against the warmup's last"""
    out = {}
    for name, s in runs:
        evals = s.get("eval") or []
        steps = sorted({e["step"] for e in evals})
        if not steps:
            raise SystemExit(f"{name}: no eval")
        out[name] = {"first_step": steps[0], "last_step": steps[-1], "first": eval_at(s, steps[0]),
                     "last": eval_at(s, steps[-1]), "steps_done": s.get("steps_done"),
                     "resume": (s.get("resume") or {}).get("step")}
    warm = out.get(WARMUP5)
    for name, x in out.items():
        if warm and name != WARMUP5:
            x["vs_warmup"] = {k: v - warm["last"][k] for k, v in x["last"].items() if k in warm["last"]}
            x["start_is_warmup_end"] = all(abs(x["first"].get(k, math.inf) - v) <= EQUAL5
                                           for k, v in warm["last"].items())
    return out


def report5(args):
    import g4mix

    evalsets = {r["id"]: r for r in read_rows(args.evalsets)}
    split = json.loads(Path(args.split).read_text())
    assign = split["assign"]
    if g4mix.digest(assign) != split["digest"]:
        raise SystemExit(f"{args.split}: the assignment is not the one of its digest")
    cands = [c.split("=", 1) for c in args.candidate]
    refs = [c.split("=", 1) for c in args.reference or []]
    scored = {name: scored_dir(d) for name, d in cands + [(BASE, args.base)] + refs}
    for name, rows in scored.items():
        if set(rows) != set(evalsets):
            raise SystemExit(f"{name}: scores for {len(rows)} of {len(evalsets)} segments")
    refqe = {r["id"]: r["qe_source_reference"] for r in read_rows(args.refqe)}
    missing = [i for i, r in evalsets.items() if r["set"] in HUMAN_QE and refqe.get(i) is None]
    if missing:
        raise SystemExit(f"{args.refqe}: no QE of the human reference of {len(missing)} segments, e.g. {missing[:3]}")
    check = {}
    for s in HUMAN_QE:
        ids = [i for i, r in evalsets.items() if r["set"] == s and not (s == "wmt24pp" and bad_source(r["meta"]))]
        got = mean(refqe[i] for i in ids)
        check[s] = {"again": got, "protocol": HUMAN_QE[s], "diff": got - HUMAN_QE[s]}
        if abs(got - HUMAN_QE[s]) > REFQE_MAX_DIFF5:
            raise SystemExit(f"the human references' QE of {s} again: {got:.4f}, the protocol's {HUMAN_QE[s]:.4f}")
    predictions = read_rows(args.predictions)
    parts = {p: part_measures5(scored, predictions, refqe, part_ids5(evalsets, assign, p), BASE)
             for p in ("search", "decision")}
    names = [n for n, _ in cands]
    chosen, why = select5(names, parts["search"])
    d = parts["decision"]["models"][chosen]
    result = {"arm": ARM5, "base": BASE, "candidates": names, "selected": chosen, "selection": why,
              "refqe_check": check, "split": {k: split[k] for k in ("seed", "share", "sets", "digest")},
              "parts": parts, "a1": {"holds": d["a1"], "sets": d["a1_sets"]}, "b": {"holds": d["b"],
                                                                                    "delta": d["b_delta"]}}
    search = json.loads(Path(args.search).read_text())
    combine = search.get("combine") or {}
    if not (combine.get("repeat") or {}).get("equal") or not (combine.get("warmup_check") or {}).get("holds"):
        raise SystemExit(f"{args.search}: the search's checks of addendum G4-5.7b do not hold: repeat "
                         f"{combine.get('repeat')}, warmup check {combine.get('warmup_check')}")
    result["search"] = {k: combine.get(k) for k in ("spec", "search", "repeat", "warmup_check", "seconds")}
    if args.smoke:
        # addendum G4-5.7b: the smoke test's anchor "G4-3 alone" (merged as translated) stays the evidence that a
        # merge of an update of G4-3's size keeps its loss
        result["smoke_g43"] = json.loads(Path(args.smoke).read_text())["checks"]["merge_g43_is_g43"]
    result["search"]["base"] = eval_at(search, 0)
    stacked = {}
    for name, dpath in cands:
        meta = Path(dpath) / "adapter.json"
        if meta.exists():
            stacked[name] = json.loads(meta.read_text())
    result["stacked"] = stacked
    result["training"] = training_table5([(n, json.loads(Path(p).read_text())) for n, p in
                                          (x.split("=", 1) for x in args.run)])
    if args.tally:
        t = json.loads(Path(args.tally).read_text())
        if t.get("arm") != chosen:
            raise SystemExit(f"{args.tally}: a tally of {t.get('arm')}, the selected candidate is {chosen}")
        result["blind"], result["a2"], result["c"] = t, t["a2"], t["c"]
    else:
        result["blind"] = None
    result["verdict"] = verdict(result)
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")
    Path(args.out).write_text(report5_text(result), encoding="utf-8")
    print(f"REPORT5 {json.dumps({'selected': chosen, 'a1': d['a1'], 'b': d['b_delta'], 'verdict': result['verdict']})}",
          flush=True)


def report5_text(r):
    sets = list(HUMAN_QE)
    L = [f"# Arm {r['arm']}: uitslag volgens het concept-amendement G4-5", "",
         "Gemaakt door `g4arm.py report5`. Per bron een LoRA vanaf een gezamenlijke warmup, gecombineerd met gewichten "
         "die op het zoekdeel gezocht zijn; de regels (a1) en (b) tegen G4-0 op het beslisdeel.", "",
         f"**Uitslag: {r['verdict']}.** Gekozen combinatie: **{r['selected']}** ({r['selection']}).", "",
         "## Besluit op het beslisdeel", "",
         "| set | n | " + r["selected"] + " | G4-0 | mens (zelfde segmenten) | (a1) |", "|---|---:|---:|---:|---:|---|"]
    dec = r["parts"]["decision"]
    m, base = dec["models"][r["selected"]], dec["models"][r["base"]]
    for s in sets:
        L.append(f"| {s} | {dec['n'][s]} | {f3(m['qe'][s])} | {f3(base['qe'][s])} | {f3(dec['human'][s])} | "
                 f"{'ja' if m['a1_sets'][s] else 'nee'} |")
    L += ["", f"(a1) **{'gehaald' if r['a1']['holds'] else 'niet gehaald'}**. (b): aandeel \"origineel\" "
          f"{pct(m['share'])} tegen G4-0 {pct(base['share'])} (WMT24++ {dec['n']['wmt24pp']} + FLAN {dec['n']['flan']}), "
          f"verschil {f3(m['b_delta'], True)} (grens +0,10) → **{'gehaald' if r['b']['holds'] else 'niet gehaald'}**.", ""]
    if r.get("blind"):
        b = r["blind"]
        L += [f"Blind tegen G4-0: {b['arm_better']} beter, {b['base_better']} slechter, {b['equal']} gelijk; fouten "
              f"{b['errors_arm']} tegen {b['errors_base']} → (a2) {'ja' if r['a2'] else 'nee'}, (c) "
              f"{'ja' if r['c'] else 'nee'}.", ""]
    else:
        L += ["(a2) en (c): nog niet beoordeeld (alleen als (a1) en (b) gelden).", ""]
    for part, title in (("search", "Zoekdeel (de keuze)"), ("decision", "Beslisdeel, alle kandidaten (na de keuze, "
                                                                          "rapporteren)")):
        p = r["parts"][part]
        L += [f"## {title}", "", "| model | " + " | ".join(f"QE {s}" for s in sets) + " | (a1) | origineel | "
              "verschil met G4-0 |", "|---|" + "---:|" * len(sets) + "---|---:|---:|"]
        L.append("| mens | " + " | ".join(f3(p["human"][s]) for s in sets) + " | | | |")
        for name, x in p["models"].items():
            L.append(f"| {name} | " + " | ".join(f3(x["qe"][s]) for s in sets) +
                     f" | {'ja' if x['a1'] else 'nee'} | {pct(x['share'])} | {f3(x['b_delta'], True)} |")
        L.append("")
    s = r["search"]
    x = s.get("search") or {}
    spec = s.get("spec") or {}
    rule = spec.get("objective_rule") or {}
    L += ["## Zoektocht naar de gewichten (verlies op het zoekdeel, forward-only)", "",
          f"Zoekdoel (aanvulling G4-5.7b A): **{spec.get('objective_mode')}** "
          f"({'het verlies bij de beste T van elk punt' if spec.get('objective_mode') == 'calibrated' else 'het verlies bij T = 1'}); "
          f"regel: aandeel van de daling dat één T op G4-0 verklaart {pct(rule.get('mean_share'))} (grens "
          f"{pct(rule.get('threshold'))}, T = {rule.get('t')}). Beide doelen staan per punt in de tabel.", "",
          f"Doel {json.dumps((s.get('spec') or {}).get('objective'))}; {x.get('evaluations')} punten, gestopt: "
          f"{x.get('stopped')}; beste {json.dumps(x.get('best'))}; herhaling van het eerste punt: "
          f"{json.dumps(s.get('repeat'))}. Basismodel (stap 0): {json.dumps(s.get('base'))}.", ""]
    w = s.get("warmup_check") or {}
    L += [f"Controle vóór de zoektocht (aanvulling G4-5.7b): de samengevoegde warmup alleen (w = 0) tegen het eigen "
          f"eval-verlies van de warmup op stap {w.get('step')}: "
          f"{json.dumps({k: [round(w['got'][k], 4), round(w['want'][k], 4)] for k in w.get('want', {})})}, grootste "
          f"verschil {f3(max(w.get('diff', {}).values(), default=None))} (grens {w.get('max_diff')}), zelfde doeltokens "
          f"{'ja' if w.get('tokens_equal') else 'nee'}: {'gehaald' if w.get('holds') else 'niet gehaald'}.", ""]
    if r.get("smoke_g43"):
        g = r["smoke_g43"]
        L += [f"Smoke-test (aanvulling G4-5.7a/b): het anker \"G4-3 alleen\", samengevoegd in de bf16-kernels, tegen "
              f"G4-3's eigen verlies (samengevoegd, eigen): {json.dumps(g['value'])}: "
              f"{'gehaald' if g['holds'] else 'niet gehaald'}.", ""]
    L += ["| fase | gewichten | doel (gekozen) | doel T = 1 | gekalibreerd (T) | verliezen |",
          "|---|---|---:|---:|---:|---|"]
    for p in x.get("points") or []:
        if p["phase"].startswith("anchor") or p in (x.get("top") or []):
            cal = (f"{p['objective_calibrated']:.4f} ({p['t_calibrated']:.2f})"
                   if p.get("objective_calibrated") is not None else "–")
            t1 = f"{p['objective_t1']:.4f}" if p.get("objective_t1") is not None else "–"
            L.append(f"| {p['phase']} | {p['w']} | {p['value']:.4f} | {t1} | {cal} | "
                     f"{json.dumps({k: round(v, 4) for k, v in (p.get('losses') or {}).items()})} |")
    L += ["", "## Per tak: welke bron beweegt welke maat (verlies, laatste stap; tak − warmup)", ""]
    t = r["training"]
    names = sorted({k for v in t.values() for k in v["last"]})
    L += ["| run | stap | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
    for run, v in t.items():
        cells = [f"{v['last'][k]:.3f}" + (f" ({v['vs_warmup'][k]:+.3f})" if 'vs_warmup' in v and k in v['vs_warmup']
                                          else "") if k in v["last"] else "–" for k in names]
        L.append(f"| {run} | {v['last_step']} | " + " | ".join(cells) + " |")
    L += ["", f"Controle menselijke referenties (QE opnieuw, hele set): {json.dumps(r['refqe_check'])}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- G4-5 temperature

def temp_curves(summary, name):
    """{set: {"t", "loss", "stack", "tokens"}} of an eval-only temperature run (sft_lora.py --temperatures), its eval
    at step 0; "stack" is the same round's own loss, which T = 1 must give within TEMP_EQUAL5"""
    out = {e["set"]: {"t": e["temperatures"]["t"], "loss": e["temperatures"]["loss"], "stack": e["loss"],
                      "tokens": e["tokens"]}
           for e in summary.get("eval") or [] if e["step"] == 0 and e.get("temperatures")}
    if not out or summary.get("steps_done") != 0:
        raise SystemExit(f"{name}: no eval-only temperature round at step 0 ({summary.get('steps_done')} steps)")
    for s, c in out.items():
        if 1.0 not in c["t"] or abs(c["loss"][c["t"].index(1.0)] - c["stack"]) > TEMP_EQUAL5:
            raise SystemExit(f"{name}: {s} at T = 1 is not the round's own loss {c['stack']} within {TEMP_EQUAL5}")
    return out


def best_t(t, values):
    """(T, value, at an end of the grid) of the lowest value; equal values: the T nearest to 1"""
    k = min(range(len(t)), key=lambda i: (values[i], abs(t[i] - 1.0)))
    return t[k], values[k], k in (0, len(t) - 1)


def temp_fit(curves, sets):
    """T of the lowest mean loss over the fit sets (each set's token-weighted loss, the sets weighted equally)"""
    t = curves[sets[0]]["t"]
    if any(curves[s]["t"] != t for s in sets):
        raise SystemExit(f"the fit sets {sets} have different grids")
    mean_loss = [statistics.fmean(curves[s]["loss"][k] for s in sets) for k in range(len(t))]
    T, value, edge = best_t(t, mean_loss)
    return {"sets": list(sets), "t": T, "value": value, "edge": edge, "value_at_1": mean_loss[t.index(1.0)]}


def drop_share(l1, lt, lft):
    """(L1 - L_T) / (L1 - L_fine-tune): the share of the fine-tune's change that one temperature gives; None when the
    fine-tune changes the loss by less than 0.001"""
    if l1 is None or lt is None or lft is None or abs(l1 - lft) < 1e-3:
        return None
    return (l1 - lt) / (l1 - lft)


def g44_final(report):
    """G4-4's loss per set at its last step and at step 0 (g4arm.py report's json: heldout.loss, novels)"""
    final, start = {}, {}
    for s, by_step in ((report.get("heldout") or {}).get("loss") or {}).items():
        last = max(by_step, key=int)
        final[s], start[s] = by_step[last], by_step.get("0")
    nov = report.get("novels") or {}
    if nov.get("arm_loss") is not None:
        final["novels"], start["novels"] = nov["arm_loss"], nov.get("base_loss")
    return final, start


def temp5(args):
    """Addendum G4-5.7b: how much of the drop of the teacher-forced loss on human Dutch one temperature of G4-0 gives
    (reported, decides nothing)"""
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    base = temp_curves(load(args.base), args.base)
    warm = load(args.warmup)
    w0, wend = eval_at(warm, 0), eval_at(warm, warm["steps"])
    wtok = {e["set"]: e["tokens"] for e in warm.get("eval") or [] if e["step"] == warm["steps"]}
    other = [s for s in base if base[s]["tokens"] != wtok.get(s)]
    if other or set(base) != set(wtok):
        raise SystemExit(f"the probe's sets or target tokens are not the warmup run's: {other or sorted(set(base) ^ set(wtok))}")
    g44, g44_start = g44_final(load(args.g44_report))
    off = {s: (g44_start[s], w0.get(s)) for s in g44 if g44_start.get(s) is None or w0.get(s) is None
           or abs(g44_start[s] - w0[s]) > TEMP_EQUAL5}
    if off:
        raise SystemExit(f"G4-4's loss at step 0 is not G4-0's of the warmup run (other batches?): {off}")
    run_w = temp_curves(load(args.warmup_run), args.warmup_run) if args.warmup_run else None
    fit = temp_fit(base, TEMP_FIT_SETS5)
    rule = search_rule5(base, wend)
    rows = {}
    for s, c in base.items():
        l1 = c["loss"][c["t"].index(1.0)]
        lfit = c["loss"][c["t"].index(fit["t"])]
        tb, lb, edge = best_t(c["t"], c["loss"])
        row = {"tokens": c["tokens"], "g40_t1": l1, "g40_stack_step0": w0.get(s), "g40_t1_vs_stack": l1 - w0[s],
               "g40_tfit": lfit, "best_t": tb, "best_loss": lb, "best_edge": edge, "warmup": wend.get(s),
               "g44": g44.get(s)}
        for ft in ("warmup", "g44"):
            row[f"share_fit_{ft}"] = drop_share(l1, lfit, row[ft])
            row[f"share_best_{ft}"] = drop_share(l1, lb, row[ft])
        if run_w and s in run_w:
            cw = run_w[s]
            tw, lw, ew = best_t(cw["t"], cw["loss"])
            row |= {"warmup_t1": cw["loss"][cw["t"].index(1.0)], "warmup_t1_vs_run": cw["loss"][cw["t"].index(1.0)] - wend[s],
                    "warmup_best_t": tw, "warmup_best_loss": lw, "warmup_best_edge": ew}
        rows[s] = row
    result = {"arm": ARM5, "grid": {"t": base[next(iter(base))]["t"]}, "fit": fit,
              "fit_warmup": temp_fit(run_w, TEMP_FIT_SETS5) if run_w else None, "search_rule": rule, "sets": rows,
              "curves": {"G4-0": base, "warmup": run_w}, "warmup_step": warm["steps"],
              "inputs": {"base": args.base, "warmup_run": args.warmup_run, "warmup": args.warmup,
                         "g44_report": args.g44_report}}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    Path(args.out).write_text(temp5_text(result), encoding="utf-8")
    print(f"TEMP5 {json.dumps({'fit': fit, 'fit_warmup': result['fit_warmup'], 'search_rule': rule, 'share_fit_warmup': {s: r['share_fit_warmup'] for s, r in rows.items()}})}",
          flush=True)


def search_rule5(base, warmup_end):
    """Addendum G4-5.7b A, fixed before the search: on the objective's sets (OBJECTIVE5) the one T of G4-0 with the
    lowest objective (as the calibrated objective of a point), per set the share (L1 - L_T) / (L1 - L1 of the warmup at
    its last step), their mean; at least SEARCH_RULE_SHARE5: the search follows the calibrated objective, else T = 1"""
    import g4mix

    sets = list(OBJECTIVE5)
    value, t = g4mix.calibrated_objective({s: base[s] for s in sets}, OBJECTIVE5)
    shares = {}
    for s in sets:
        c = base[s]
        l1, lt = c["loss"][c["t"].index(1.0)], c["loss"][c["t"].index(t)]
        shares[s] = drop_share(l1, lt, warmup_end.get(s))
    if any(v is None for v in shares.values()):
        raise SystemExit(f"the search rule needs the warmup's drop on {sets}: shares {shares}")
    mean_share = statistics.fmean(shares.values())
    return {"sets": sets, "t": t, "objective_at_t": value, "shares": shares, "mean_share": mean_share,
            "threshold": SEARCH_RULE_SHARE5,
            "objective": "calibrated" if mean_share >= SEARCH_RULE_SHARE5 else "t1"}


def temp5_text(r):
    t = r["grid"]["t"]
    lo, hi = f3(t[0]).replace(",000", ",00"), f3(t[-1]).replace(",000", ",00")

    def tt(x, edge):
        if edge:
            return f"≤ {lo}" if x == t[0] else f"≥ {hi}"
        return f"{x:.2f}".replace(".", ",")

    fit, fw = r["fit"], r["fit_warmup"]
    sets = list(r["sets"])
    L = [f"# Arm {r['arm']}: temperatuurproef (aanvulling G4-5.7b; rapporteren, beslist niets)", "",
         "Gemaakt door `g4arm.py temp5`. Vraag: welk deel van de daling van het teacher-forced verlies op menselijk "
         "Nederlands bij de fine-tunes geeft één temperatuur van G4-0 al (kalibratie)? Verlies van softmax(logits / T), "
         "token-gewogen, op dezelfde eval-sets en doeltokens als de trainingsstack, één forward-pass per batch voor alle T "
         f"(rooster {lo}–{hi}, stap {t[1] - t[0]:.2f}".replace(".", ",") + "). T verandert de volgorde van de logits "
         "niet: de greedy-uitvoer is bij elke T gelijk.", "",
         f"**T gekozen op {' + '.join(fit['sets'])}** (gemiddelde van de twee verliezen): **T\\* = "
         f"{tt(fit['t'], fit['edge'])}** (gemiddeld verlies {f3(fit['value_at_1'])} bij T = 1, {f3(fit['value'])} bij "
         "T\\*).", ""]
    if fw:
        L += [f"Warmup-adapter (stap {r['warmup_step']}), zelfde keuze: T\\* = {tt(fw['t'], fw['edge'])} "
              f"({f3(fw['value_at_1'])} bij T = 1, {f3(fw['value'])} bij T\\*).", ""]
    sr = r["search_rule"]
    L += [f"**Regel voor de zoektocht (aanvulling G4-5.7b A)**: één T voor G4-0 op {', '.join(sr['sets'])} "
          f"(laagste gemiddelde): T = {tt(sr['t'], sr['t'] in (t[0], t[-1]))}; aandeel van de daling G4-0 → warmup "
          f"per set {', '.join(f'{k} {pct(v)}' for k, v in sr['shares'].items())}, gemiddeld **{pct(sr['mean_share'])}** "
          f"(grens {pct(sr['threshold'])}) → de zoektocht kiest op het **"
          f"{'gekalibreerde doel (verlies bij de beste T per punt)' if sr['objective'] == 'calibrated' else 'doel bij T = 1'}**.", ""]
    L += ["Lezing: aandeel = (L₁ − L_T) / (L₁ − L_fine-tune) per set, L₁ en L_T van G4-0 in deze proef; "
          "1 = één temperatuur geeft de hele verandering, 0 = niets. Bij `own` (G4-0's eigen greedy uitvoer als doel) "
          "stijgt het verlies bij de fine-tunes: het aandeel is dan dat van de stijging.", "",
          "## Per set", "",
          "| set | doeltokens | G4-0 T=1 | G4-0 T\\* | beste T (set) | G4-0 bij beste T | warmup stap "
          f"{r['warmup_step']} | G4-4 eind | aandeel T\\* (warmup) | aandeel T\\* (G4-4) | aandeel beste T (warmup) | "
          "aandeel beste T (G4-4) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in sets:
        x = r["sets"][s]
        L.append(f"| {s} | {x['tokens']} | {f3(x['g40_t1'])} | {f3(x['g40_tfit'])} | {tt(x['best_t'], x['best_edge'])} | "
                 f"{f3(x['best_loss'])} | {f3(x['warmup'])} | {f3(x['g44'])} | {pct(x['share_fit_warmup'])} | "
                 f"{pct(x['share_fit_g44'])} | {pct(x['share_best_warmup'])} | {pct(x['share_best_g44'])} |")
    if fw:
        L += ["", f"## Warmup-adapter (stap {r['warmup_step']}): ligt zijn optimum bij T = 1?", "",
              "| set | T=1 (deze proef) | T=1 − eigen eval | beste T | verlies bij beste T |", "|---|---:|---:|---:|---:|"]
        for s in sets:
            x = r["sets"][s]
            if "warmup_t1" in x:
                L.append(f"| {s} | {f3(x['warmup_t1'])} | {x['warmup_t1_vs_run']:+.1e} | "
                         f"{tt(x['warmup_best_t'], x['warmup_best_edge'])} | {f3(x['warmup_best_loss'])} |")
    L += ["", "## Controles", "",
          f"- T = 1 is per set het eigen verlies van dezelfde ronde (binnen {TEMP_EQUAL5}); de doeltokens zijn die van "
          "de warmup-run; G4-4's verlies op stap 0 is dat van de warmup-run (zelfde batches).",
          "- G4-0 T = 1 in deze proef (num_vocab_tiling 1) tegen stap 0 van de warmup-run (trainingsstack, "
          "num_vocab_tiling 8): " + ", ".join(f"{s} {r['sets'][s]['g40_t1_vs_stack']:+.1e}" for s in sets) + ".", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- sampling probe

# addendum G4-5.7b B (Yeb 18 Nov ~09:15; reported, decides nothing): G4-0 and G4-4 sampled at T = 0.7, T = 1.0 and T*
# of the temperature probe (only if T* <= 1.5), and best-of-8 at T = 1.0 (seeds 1-8, seed 1 the T = 1.0 setting, per
# segment the sample with the lowest MetricX-24-QE), on WMT24++ (960) and FLAN (1,000)
SAMPLE_MODELS5 = ("G4-0", "G4-4")
SAMPLE_TSTAR_MAX5 = 1.5
SAMPLE_BEST_OF5 = 8
SAMPLE_SEED5 = 1


def samples5(t_star):
    """([(name, T, seed)], note): t07, t10s1 ... t10s8 (t10s1 is the T = 1.0 setting), tstar if T* <= 1.5 and not
    0.7 or 1.0 (then the same setting as one already there)"""
    out = [("t07", 0.7, SAMPLE_SEED5)] + [(f"t10s{k}", 1.0, k) for k in range(1, SAMPLE_BEST_OF5 + 1)]
    if t_star > SAMPLE_TSTAR_MAX5:
        return out, f"T* = {t_star:g} > {SAMPLE_TSTAR_MAX5:g}: not sampled"
    if t_star in (0.7, 1.0):
        return out, f"T* = {t_star:g}: the setting {'t07' if t_star == 0.7 else 't10s1'}"
    return out + [("tstar", t_star, SAMPLE_SEED5)], f"T* = {t_star:g}: sampled (tstar)"


def samples5_cli(args):
    t_star = json.loads(Path(args.temperature).read_text())["fit"]["t"]
    settings, note = samples5(t_star)
    print("SAMPLES " + " ".join(f"{n}={t:g}:{seed}" for n, t, seed in settings))
    print(f"NOTE {note}")


def sampledata5(args):
    """The sampling probe's segments: WMT24++ without is_bad_source (960) and FLAN (1,000) of the evaluation's
    segments-w<k>.parquet, each on its worker as there (the same length classes; classes-w<k>.txt copied)"""
    evalsets = {r["id"]: r for r in read_rows(args.evalsets, ["id", "set", "meta"])}
    keep = {i for i, r in evalsets.items() if (r["set"] == "wmt24pp" and not bad_source(r["meta"])) or r["set"] == "flan"}
    n = collections.Counter(evalsets[i]["set"] for i in keep)
    if (n["wmt24pp"], n["flan"]) != (N_WMT_GOOD, N_FLAN):
        raise SystemExit(f"{args.evalsets}: {dict(n)}, not {N_WMT_GOOD} WMT24++ and {N_FLAN} FLAN")
    src, out = Path(args.segments), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen = collections.Counter()
    for k in range(args.workers):
        mine = [r for r in read_rows(src / f"segments-w{k}.parquet") if r["id"] in keep]
        write_parquet(mine, out / f"segments-w{k}.parquet")
        (out / f"classes-w{k}.txt").write_text((src / f"classes-w{k}.txt").read_text())
        seen.update(r["id"] for r in mine)
    if set(seen) != keep or max(seen.values()) != 1:
        raise SystemExit(f"the workers' segments hold {len(seen)} of the {len(keep)} probe segments (or one twice)")
    print(f"SAMPLEDATA {json.dumps({'wmt24pp': n['wmt24pp'], 'flan': n['flan'], 'workers': args.workers})}", flush=True)


def samplejoin(args):
    """The pairs of several samples as one file, so that MetricX and COMET load once: ids NAME|ID"""
    out = []
    for name, path in (x.split("=", 1) for x in args.pairs):
        if "|" in name:
            raise SystemExit(f"--pairs {name}: no | in a sample's name")
        out += [{**r, "id": f"{name}|{r['id']}"} for r in read_rows(path)]
    write_parquet(out, args.out)
    print(f"SAMPLEJOIN {len(out)} pairs of {len(args.pairs)} samples", flush=True)


def samplesplit(args):
    """samplejoin's scores back per sample: --out holds {sample}; a sample without a row gets no file"""
    by = collections.defaultdict(list)
    for r in read_rows(args.scores):
        name, sep, i = r["id"].partition("|")
        if not sep:
            raise SystemExit(f"{args.scores}: the id {r['id']} is not NAME|ID")
        by[name].append({**r, "id": i})
    for name, rows in by.items():
        path = Path(args.out.replace("{sample}", name))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(rows, path)
    print(f"SAMPLESPLIT {json.dumps({k: len(v) for k, v in by.items()})}", flush=True)


def rows_by_id(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for r in read_rows(f):
            out[r["id"]] = r
    return out


def best8(args):
    """Best-of-N: per segment the sample with the lowest MetricX-24-QE (equal: the earlier run), its pairs and scores
    as an output of its own (pairs/pairs-wall.parquet, scores/metricx-wall.parquet, comet-wall.parquet) and the choice"""
    runs = [x.split("=", 1) for x in args.run]
    if len(runs) != SAMPLE_BEST_OF5:
        raise SystemExit(f"best-of-{SAMPLE_BEST_OF5}: {len(runs)} runs")
    data = {name: {k: rows_by_id(f"{d}/{sub}/{k}-w*.parquet") for k, sub in
                   (("pairs", "pairs"), ("metricx", "scores"), ("comet", "scores"))} for name, d in runs}
    ids = set(data[runs[0][0]]["pairs"])
    for name, x in data.items():
        if set(x["pairs"]) != ids or set(x["metricx"]) != ids:
            raise SystemExit(f"{name}: {len(x['pairs'])} pairs, {len(x['metricx'])} MetricX rows, not the {len(ids)} of {runs[0][0]}")
    pairs, metricx, comet, choice = [], [], [], []
    for i in sorted(ids):
        qe = [data[n]["metricx"][i].get("qe_source_hypothesis") for n, _ in runs]
        if any(q is None for q in qe):
            raise SystemExit(f"{i}: a sample without MetricX-QE {qe}")
        k = min(range(len(runs)), key=lambda j: (qe[j], j))
        name = runs[k][0]
        pairs.append(data[name]["pairs"][i])
        metricx.append(data[name]["metricx"][i])
        if i in data[name]["comet"]:
            comet.append(data[name]["comet"][i])
        choice.append({"id": i, "chosen": name, "qe": qe})
    out = Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    write_parquet(pairs, out / "pairs" / "pairs-wall.parquet")
    write_parquet(metricx, out / "scores" / "metricx-wall.parquet")
    if comet:
        write_parquet(comet, out / "scores" / "comet-wall.parquet")
    write_parquet(choice, out / "choice.parquet")
    print(f"BEST8 {json.dumps({'segments': len(ids), 'chosen': collections.Counter(c['chosen'] for c in choice)})}",
          flush=True)


def sample_measures5(rows, ids, origin):
    """The probe's measures of one output on one set: QE, MetricX-ref and COMET (with a reference), not ended, loop,
    English left, length against the source and the reference, share "original" (origin: {id: p_original})"""
    rs = [rows[i] for i in ids]
    with_ref = [r for r in rs if r["reference"]]
    n = len(rs)
    return {"n": n, "metricx_qe": mean(r["metricx_qe"] for r in rs),
            "metricx_ref": mean(r["metricx_ref"] for r in with_ref) if with_ref else None,
            "comet22": mean(r["comet22"] for r in with_ref) if with_ref else None,
            "not_ended": sum(not r["ended"] for r in rs) / n, "decode_loop": sum(bool(r["decode_loop"]) for r in rs) / n,
            "english_left": sum(bool(r["english_left"]) for r in rs) / n,
            "length_source": sum(len(r["hypothesis"]) for r in rs) / sum(len(r["source"]) for r in rs),
            "length_reference": (sum(len(r["hypothesis"]) for r in with_ref) / sum(len(r["reference"]) for r in with_ref)
                                 if with_ref else None),
            "original": sum(origin[i] >= 0.5 for i in ids) / n}


def sample5(args):
    """The sampling probe's report (addendum G4-5.7b B): per model and setting the measures on WMT24++ (960) and FLAN
    (1,000), the share "original" of one classify run with the greedy outputs of G4-0 and G4-4 as anchors"""
    evalsets = {r["id"]: r for r in read_rows(args.evalsets, ["id", "set", "meta"])}
    ids = {"wmt24pp": sorted(i for i, r in evalsets.items() if r["set"] == "wmt24pp" and not bad_source(r["meta"])),
           "flan": sorted(i for i, r in evalsets.items() if r["set"] == "flan")}
    settings, note = samples5(json.loads(Path(args.temperature).read_text())["fit"]["t"])
    meta = {"settings": settings, "note": note}
    predictions = read_rows(args.predictions)
    result = {"arm": ARM5, "settings": meta, "human_qe_wmt24pp": HUMAN_QE["wmt24pp"], "outputs": {}}
    for name, d in (x.split("=", 1) for x in args.output):
        rows = scored_dir(d)
        origin = {r["id"]: r["p_original"] for r in predictions if r["model"] == name}
        out = {}
        for s, want in ids.items():
            missing = [i for i in want if i not in rows or i not in origin]
            if missing:
                raise SystemExit(f"{name}: {len(missing)} of the {len(want)} {s} segments without a score or a "
                                 f"classifier text, e.g. {missing[:3]}")
            out[s] = sample_measures5(rows, want, origin)
        both = ids["wmt24pp"] + ids["flan"]
        out["original_pooled"] = sum(origin[i] >= 0.5 for i in both) / len(both)
        result["outputs"][name] = out
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    Path(args.out).write_text(sample5_text(result), encoding="utf-8")
    print(f"SAMPLE5 {json.dumps({k: round(v['original_pooled'], 4) for k, v in result['outputs'].items()})}", flush=True)


def sample5_text(r):
    by_name = {n: (t, seed) for n, t, seed in r["settings"]["settings"]}
    label = {"best8": f"best-of-{SAMPLE_BEST_OF5} bij T = 1,0 (laagste QE)"}

    def name_of(key):
        model, _, setting = key.partition("/")
        if not setting:
            return f"{model} greedy"
        if setting in label:
            return f"{model} {label[setting]}"
        t, seed = by_name[setting]
        return f"{model} T = {t:g} (seed {seed})".replace(".", ",")

    L = ["# Sampling-proef (aanvulling G4-5.7b B; rapporteren, beslist niets)", "",
         "Gemaakt door `g4arm.py sample5`. G4-0 en G4-4 (eindadapter, ingebakken zoals `translate.py --lora`) met "
         "MaxText's `weighted`-sampling (categorisch over logits / T) in de fork `f4d8615`, prompt `natural-v1`, de "
         "klassen en lengtes van de evaluatie van G4-4, op WMT24++ (960 goede segmenten) en FLAN (1.000). Het aandeel "
         "\"origineel\" komt uit één classify-run met de greedy uitvoer van G4-0 en G4-4 als ankers.", "",
         f"Instellingen: {', '.join(f'{n} T = {t:g} seed {seed}' for n, t, seed in r['settings']['settings'])}; "
         f"{r['settings']['note']}.", "",
         "## Aandeel \"origineel\" (regel (b): WMT24++ + FLAN samen)", "",
         "| uitvoer | origineel (samen) | WMT24++ | FLAN |", "|---|---:|---:|---:|"]
    for key, x in r["outputs"].items():
        L.append(f"| {name_of(key)} | {pct(x['original_pooled'])} | {pct(x['wmt24pp']['original'])} | "
                 f"{pct(x['flan']['original'])} |")
    L += ["", "## Kwaliteit en vorm", "",
          "| uitvoer | set | QE | MetricX-ref | COMET-22 | niet af | lus | Engels over | tekens / bron | tekens / ref |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
          f"| mens | wmt24pp | {f3(r['human_qe_wmt24pp'])} | | | | | | | |"]
    for key, x in r["outputs"].items():
        for s in ("wmt24pp", "flan"):
            m = x[s]
            L.append(f"| {name_of(key)} | {s} | {f3(m['metricx_qe'])} | {f3(m['metricx_ref'])} | {f3(m['comet22'])} | "
                     f"{pct(m['not_ended'])} | {pct(m['decode_loop'])} | {pct(m['english_left'])} | "
                     f"{f3(m['length_source'])} | {f3(m['length_reference'])} |")
    L += ["", "QE en MetricX-ref: lager is beter; COMET-22: hoger is beter. Best-of-8 kiest op QE, dus zijn QE is per "
          "constructie de laagste van de acht en geen onafhankelijke maat.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- main

# ---------------------------------------------------------------------------------------------------- G4-6

ARM6 = "G4-6"
SEED6 = f"{SEED}/G4-6"
BT_SOURCES6 = ("handelingen", "raad", "fineweb", "rechtspraak", "dpc")
# Pairs per stratum (draft amendment G4-6.2): every Dutch book (human English, original Dutch), the novels (original
# English, human Dutch) and the back-translations (Gemma 4's English, original Dutch) about as many, the BT equal per
# source
WANT6 = {"book": 11000, "novel": 16000, **{f"bt:{s}": 3200 for s in BT_SOURCES6}}
CANDIDATES6 = 1.4         # candidates translated per wanted pair, for what build6's filters take (books: every one);
                          # DPC loses ~18% to the chrF ceiling alone (its round trip's p75-p95, spec §3.1)
PAIR_MAX6 = 2048          # a pair's tokens (the prompt twice, both sides): every pair fits a row of the fallback
SEQ6 = SEQ3               # 4,096: G4-4's rows
SEQ6_FALLBACK = 2048      # amendment G4-6.4: the rows if 4,096 does not fit
ROWS6 = HOSTS * ROWS_PER_HOST
CHRF6 = {"novel": (25.0, 80.0), "book": (50.0, 80.0), "bt": (50.0, 80.0)}   # kept: low <= chrF(rejected, chosen) < high
LENGTH6 = (0.67, 1.5)     # kept: characters of the rejected over the chosen
PREF_SETS6 = ("pref_books", "pref_btval", "pref_novels")
EVAL_EVERY6 = 200
LR6 = 1e-5
DPO6 = {"loss": "sigmoid", "beta": 0.1, "label_smoothing": 0.0, "nll_alpha": 1.0, "length_norm": False}
SMOKE6_BATCHES = 12       # the smoke set: the training file's first batches (8 steps and room)
SMOKE6_STEPS = 8
SMOKE6_EVAL_STEPS = [0, 4, 8]
STEP0_MAX_DIFF6 = 1e-4    # step 0: the policy is the reference (loss log 2, margin 0, each NLL the reference's)
BASE_MAX_DIFF6 = 2e-3     # step 0: an SFT eval file's loss against G4-4's step 0 (the same base, the loss computed apart)
MAX_HOURS6 = 24
FAILURES6 = ("not_ended", "decode_loop", "empty", "english_left", "identical")
# Arm G4-6b (amendment G4-6b): G4-6 from G4-4 (its final adapter merged into the base kernels: the start model and the
# reference), G4-4's greedy translation of G4-6's candidates as the rejected side
ARM6B = "G4-6b"
START6B = "G4-4"
BASE_MAX_DIFF6B = 0.05    # step 0: an SFT eval file's loss against G4-4's own eval at its last step (merged into bf16
                          # kernels against LoRA in the forward pass; G4-5.7b measured 0.015 and 0.012 for its warmup)
OWN6B = "own4"            # G4-4's greedy held-out translations of the BT validation as targets (as own is G4-0's)
COMPARE6B = ("candidates", "identical", "chrf_low", "chrf_high", "length_ratio", "kept", "drawn", "short_of_want")
BANDS6 = ("chrf_low", "chrf_high", "length_ratio", "eval_13gram")
# Arm G4-6c (amendment G4-6c): G4-6b's start model, reference and pairs with a milder loss: beta 0.5, NLL weight 2.0, the
# turn's end (<turn|>\n, the last two target tokens of every side) outside the log-ratio; evals and kept adapters at
# the checkpoints; the checkpoint chosen on the search part of G4-5's split (G4-5.6's rule), decided on its decision part
ARM6C = "G4-6c"
DPO6C = {"loss": "sigmoid", "beta": 0.5, "label_smoothing": 0.0, "nll_alpha": 2.0, "length_norm": False, "end_tokens": 2}
END6C = g4lora.END
EVAL_EVERY6C = 240
SAVE_STEPS6C = (240, 480)
SAVE_ARG6C = ",".join(map(str, SAVE_STEPS6C))
SMOKE_SAVE6C = 4
LOSS_MAX_DIFF6C = 1e-4    # step 1's loss against its DPO term + nll_alpha x the chosen NLL over every target token
# Addendum G4-6c.6e: the reference of the training batches from a run of its own (sft_lora.py --ref-only), read in by
# the training run (--ref-in). The CPU proof: the reference read in against the one computed in the training's own
# process, and the training with it against the training without it, each value within this relative bound
# (max(1, |x|)); whether bitwise is reported. The smoke test (and the full run, in the chain): the run read its reference
# and its step 1, where the policy is still the reference, has |DPO term - log 2| and |margin| within STEP1_MAX_DIFF6E
# (G4-6c's smoke test with the reference in its own process, chain g4h1120e: 7.5e-5 and 1.5e-4)
REF_MAX_DIFF6E = 1e-6
STEP1_MAX_DIFF6E = 1e-3
LONG6C = 1.5              # reported: the share of outputs longer than this times their reference's words
CKPT6C = re.compile(r"^G4-6c-s([1-9][0-9]*)$")
SPLIT_DIGEST6C = "f953c026597eedcb20ab56dab7d0b4389057697c01550017ba954ba6963440f3"   # G4-5.4's search and decision parts
SFT_SETS6C = ("books", "btval", "own", "own4", "btlong", "novels")


def english_of(user):
    """The English of a natural-v1 user turn (g4lora.user_turn), checked by making the turn again"""
    marker = "\u0000"
    pre, post = g4lora.user_turn(marker).split(marker)
    if not user.startswith(pre) or not user.endswith(post) or len(user) < len(pre) + len(post):
        raise SystemExit(f"not a natural-v1 user turn: {user[:80]!r}")
    en = user[len(pre):len(user) - len(post)]
    if g4lora.user_turn(en) != user:
        raise SystemExit(f"the English of a user turn does not make the turn again: {user[:80]!r}")
    return en


def stratum6(meta):
    """novel, book, or bt:<source> of a G4-4 example"""
    if meta["kind"] == "bt":
        if meta["source"] not in BT_SOURCES6:
            raise SystemExit(f"{meta['id']}: BT of an unknown source {meta['source']}")
        return f"bt:{meta['source']}"
    if meta["kind"] not in ("novel", "book"):
        raise SystemExit(f"{meta['id']}: an example of kind {meta['kind']}")
    return meta["kind"]


def kind6(stratum):
    return stratum.split(":")[0]


def assign_balanced(rows, workers):
    """{worker: [rows]}: every length class over every worker (a class's rows by tokens, in turn), so that each worker
    decodes about the same tokens; a worker loads each of its classes once (~3 min)"""
    out = {k: [] for k in range(workers)}
    for cls in EVAL_CLASSES:
        part = sorted((r for r in rows if r["length_class"] == cls), key=lambda r: (r["tokens"], r["id"]))
        for i, r in enumerate(part):
            out[i % workers].append(r)
    if sum(len(v) for v in out.values()) != len(rows):
        raise SystemExit("a row of a class beyond tiny-xlong")
    return {k: sorted(v, key=lambda r: (r["tokens"], r["id"])) for k, v in out.items()}


CANDS6_SCHEMA = (("id", "string"), ("example", "string"), ("set", "string"), ("stratum", "string"), ("kind", "string"),
                 ("source", "string"), ("en", "string"), ("nl", "string"), ("tokens_en", "int64"),
                 ("tokens_nl", "int64"), ("tokens_prompt", "int64"), ("tokens_target", "int64"),
                 ("length_class", "string"))


def cands6(args, counter=None):
    """amendment G4-6.2, before the rejected run: the candidates of G4-6's pairs from G4-4's set (its distinct examples:
    the Dutch books, the novels of the three passes, the BT of up to 512 Dutch tokens per source; no long BT, no FLAN
    replay), each whose Dutch fits 0.8 of its class's decode limit (decode_class, as the held-out sets) and whose pair,
    estimated with the rejected as long as the chosen, fits PAIR_MAX6; per stratum a seeded draw of CANDIDATES6 x WANT6
    (books: every one); and the novels of the eval set novels (pref_novels). Writes the rejected run's input in
    evalsets.py's format (evalsets.parquet, segments-w<k>.parquet, classes-w<k>.txt, every class over every worker),
    cands.parquet and manifest.json"""
    counter = counter or TokenCounter(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    passes, _, _ = g44_set(args.g44, args.g44_report)
    distinct = {}
    for e in sorted(passes):
        for x, m in passes[e]:
            if m["kind"] not in ("bt", "book", "novel"):
                continue
            nl = x["messages"][1]["content"]
            seen = distinct.setdefault(m["id"], (m, nl))
            if seen[1] != nl or seen[0]["en"] != m["en"]:
                raise SystemExit(f"{m['id']}: two examples under one id")
    ids = sorted(distinct)
    metas = [distinct[i][0] for i in ids]
    src = counter.lengths(m["en"].strip() for m in metas)
    tgt = counter.lengths(distinct[i][1].strip() for i in ids)
    funnel = collections.defaultdict(collections.Counter)
    eligible = collections.defaultdict(list)
    for i, m, a, b in zip(ids, metas, src, tgt):
        s = stratum6(m)
        funnel[s]["examples"] += 1
        cls = decode_class(a, b)
        if cls is None:
            funnel[s]["class_none"] += 1
            continue
        if 2 * (m["tokens_prompt"] + m["tokens_target"]) > PAIR_MAX6:
            funnel[s]["pair_too_long_estimate"] += 1
            continue
        funnel[s]["eligible"] += 1
        eligible[s].append({"id": f"g46:{i}", "example": i, "set": "train", "stratum": s, "kind": m["kind"],
                            "source": m["source"], "en": m["en"].strip(), "nl": distinct[i][1].strip(),
                            "tokens_en": a, "tokens_nl": b, "tokens_prompt": m["tokens_prompt"],
                            "tokens_target": m["tokens_target"], "length_class": cls})
    missing = sorted(s for s in WANT6 if not eligible.get(s))
    if missing:
        raise SystemExit(f"no eligible example of {missing}")
    cands = []
    for s in sorted(WANT6):
        rows = sorted(eligible[s], key=lambda r: r["id"])
        n = len(rows) if s == "book" else min(len(rows), math.ceil(WANT6[s] * args.factor))
        random.Random(f"{SEED6}/candidates/{s}").shuffle(rows)
        cands += rows[:n]
        funnel[s]["drawn"] = n
    novels = read_rows(args.novels_eval)
    novels_meta = read_rows(args.novels_eval_meta)
    if len(novels) != len(novels_meta):
        raise SystemExit(f"{args.novels_eval}: {len(novels)} rows, its meta {len(novels_meta)}")
    ens = [english_of(x["messages"][0]["content"]) for x in novels]
    nls = [x["messages"][1]["content"].strip() for x in novels]
    funnel["pref_novels"]["examples"] = len(novels)
    for m, en, nl, a, b in zip(novels_meta, ens, nls, counter.lengths(ens), counter.lengths(nls)):
        cls = decode_class(a, b)
        if cls is None:
            funnel["pref_novels"]["class_none"] += 1
            continue
        cands.append({"id": f"g46:pref_novels:{m['pid']}", "example": m["pid"], "set": "pref_novels",
                      "stratum": "pref_novels", "kind": "novel", "source": "books_en", "en": en.strip(), "nl": nl,
                      "tokens_en": a, "tokens_nl": b, "tokens_prompt": m["tokens_prompt"],
                      "tokens_target": m["tokens_target"], "length_class": cls})
    funnel["pref_novels"]["drawn"] = sum(r["set"] == "pref_novels" for r in cands)
    if len({r["id"] for r in cands}) != len(cands):
        raise SystemExit("an id twice among the candidates")
    write_parquet(cands, out / "cands.parquet", schema(CANDS6_SCHEMA))
    eval_rows = [{"id": r["id"], "set": r["stratum"], "source": r["en"], "reference": r["nl"], "tokens": r["tokens_en"],
                  "over_512": r["tokens_en"] > 512, "length_class": r["length_class"],
                  "meta": json.dumps({"kind": r["kind"], "source": r["source"], "example": r["example"],
                                      "tokens_nl": r["tokens_nl"]})} for r in cands]
    write_parquet(eval_rows, out / "evalsets.parquet", schema(EVAL_SCHEMA))
    tokens_nl = {r["id"]: r["tokens_nl"] for r in cands}
    workers = {}
    for k, part in assign_balanced(eval_rows, args.workers).items():
        write_parquet([{"id": r["id"], "text": r["source"], "tokens": r["tokens"]} for r in part],
                      out / f"segments-w{k}.parquet", schema(SEGMENT_SCHEMA))
        classes = [c for c in EVAL_CLASSES if any(r["length_class"] == c for r in part)]
        (out / f"classes-w{k}.txt").write_text(" ".join(classes) + "\n")
        workers[f"w{k}"] = {"segments": len(part), "classes": classes, "dutch_tokens": sum(tokens_nl[r["id"]] for r in part)}
    g44 = Path(args.g44)
    manifest = {"arm": ARM6, "seed": SEED6, "prompt": PROMPT, "model": MODEL, "want": WANT6, "factor": args.factor,
                "pair_max": PAIR_MAX6, "funnel": {k: dict(v) for k, v in sorted(funnel.items())},
                "segments": len(cands), "dutch_tokens": sum(tokens_nl.values()), "workers": workers,
                "classes": dict(collections.Counter(r["length_class"] for r in cands)),
                "inputs": {"g44_train": sha256_file(g44 / "train.parquet"),
                           "g44_meta": sha256_file(g44 / "train-meta.parquet"),
                           "g44_report": sha256_file(args.g44_report), "novels_eval": sha256_file(args.novels_eval),
                           "novels_eval_meta": sha256_file(args.novels_eval_meta)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for w, s in workers.items():
        print(f"WORKER {w}: {s['segments']} segments, {s['dutch_tokens']} Dutch tokens, classes {s['classes']}",
              flush=True)
    print(f"CANDS6 {json.dumps({'segments': len(cands), 'funnel': manifest['funnel']})}", flush=True)


def g40_rows(patterns):
    """{id: row} of collected pairs (evalsets.py collect: hypothesis and its checks), every id once"""
    rows = {}
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"no pairs in {pattern}")
        for f in files:
            for r in read_rows(f, ["id", "hypothesis", "ended", "decode_loop", "english_left", "empty"]):
                if r["id"] in rows:
                    raise SystemExit(f"{r['id']} twice in {pattern}")
                rows[r["id"]] = r
    return rows


def failure6(chosen, g40):
    """G4-0's translation as the rejected side: the first failure of FAILURES6 (None: usable), and the text"""
    hyp = (g40["hypothesis"] or "").strip()
    if not g40["ended"]:
        return "not_ended", hyp
    if g40["decode_loop"]:
        return "decode_loop", hyp
    if g40["empty"] or not hyp:
        return "empty", hyp
    if g40["english_left"]:
        return "english_left", hyp
    if hyp == chosen:
        return "identical", hyp
    return None, hyp


def band6(kind, chosen, rejected, blocked, chrf):
    """The first of BANDS6 a usable pair fails (None: kept), its chrF (rejected against chosen) and length ratio"""
    c = chrf(rejected, chosen)
    ratio = len(rejected) / max(1, len(chosen))
    low, high = CHRF6[kind]
    if c < low:
        return "chrf_low", c, ratio
    if c >= high:
        return "chrf_high", c, ratio
    if not LENGTH6[0] <= ratio <= LENGTH6[1]:
        return "length_ratio", c, ratio
    if sp.ngrams(rejected) & blocked:
        return "eval_13gram", c, ratio
    return None, c, ratio


def tokenized6(pairs, counter):
    """Each pair with its token ids: prompt (g4lora.prompt_text of its English), chosen and rejected
    (g4lora.completion_text), as MaxText's SFT makes an example"""
    prompts = counter.ids(g4lora.prompt_text(p["en"]) for p in pairs)
    chosen = counter.ids(g4lora.completion_text(p["chosen"]) for p in pairs)
    rejected = counter.ids(g4lora.completion_text(p["rejected"]) for p in pairs)
    return [{**p, "prompt": a, "chosen_ids": b, "rejected_ids": c, "tokens_pair": 2 * len(a) + len(b) + len(c)}
            for p, a, b, c in zip(pairs, prompts, chosen, rejected)]


def packed6(pairs, seq, pad, bos):
    return g4dpo.pack_rows([{"id": p["id"], "prompt": p["prompt"], "chosen": p["chosen_ids"],
                             "rejected": p["rejected_ids"]} for p in pairs], seq, ROWS6, pad_id=pad, bos_id=bos)


def pack_stats(rows, seq):
    batches = 1 + max(r["batch"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)
    pairs = sum(len(r["pairs"]) for r in rows)
    return {"rows": len(rows), "batches": batches, "steps": batches, "pairs": pairs, "tokens": tokens,
            "fill": round(tokens / (len(rows) * seq), 4), "pairs_per_step": round(pairs / batches, 2),
            "max_pairs_in_row": max(len(r["pairs"]) for r in rows)}


PAIRS6_SCHEMA = (("id", "string"), ("set", "string"), ("stratum", "string"), ("kind", "string"), ("source", "string"),
                 ("en", "string"), ("chosen", "string"), ("rejected", "string"), ("chrf", "float64"),
                 ("length_ratio", "float64"), ("tokens_prompt", "int64"), ("tokens_chosen", "int64"),
                 ("tokens_rejected", "int64"), ("order", "int64"))


def build6(args, counter=None, chrf=None):
    """amendment G4-6.2-G4-6.3, after the rejected run: the pairs (the candidates of cands6 with G4-0's translation as
    the rejected side; out: G4-0's failures and identical pairs, then the chrF band of the kind, the length ratio, a
    13-gram of the evaluation sets in the rejected text, more than PAIR_MAX6 tokens), per stratum a seeded draw of
    WANT6, one seeded order over all; packed (g4dpo.pack_rows) at SEQ6 and SEQ6_FALLBACK; the smoke sets (the first
    SMOKE6_BATCHES batches); the preference eval sets (held-out books and BT validation with G4-0's held-out
    translations, the novels of the eval set novels; out only G4-0's failures, identical pairs and too long ones);
    train-chosen.parquet (the chosen sides as SFT examples), pairs.parquet, report.json and report.md"""
    import armeval

    counter = counter or TokenCounter(args.tokenizer)
    chrf = chrf or armeval.chrf
    arm, rejected_model = getattr(args, "arm", None) or ARM6, getattr(args, "rejected_model", None) or BASE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pad, bos = counter.special_id("<pad>"), counter.special_id("<bos>")
    if pad is None or bos is None:
        raise SystemExit(f"{args.tokenizer}: no <pad> or <bos>")
    cands = read_rows(args.cands)
    g40 = g40_rows(args.rejected)
    missing = [r["id"] for r in cands if r["id"] not in g40]
    if missing:
        raise SystemExit(f"{len(missing)} candidates without {rejected_model}'s translation, e.g. {missing[:3]}")
    blind = read_jsonl(args.blind)
    if sha256_file(args.blind) != blind_sha256():
        raise SystemExit(f"{args.blind}: not the blind passages of the TG protocol (§4.2.4)")
    blocked = blocked_grams(read_rows(args.evalsets), blind)
    funnel = collections.defaultdict(collections.Counter)
    measured = collections.defaultdict(lambda: {"chrf": [], "ratio": []})
    survivors = collections.defaultdict(list)
    pref_novels = []
    for r in cands:
        s = r["stratum"]
        funnel[s]["candidates"] += 1
        why, hyp = failure6(r["nl"], g40[r["id"]])
        if why:
            funnel[s][why] += 1
            continue
        pair = {"id": r["id"], "set": r["set"], "stratum": s, "kind": r["kind"], "source": r["source"], "en": r["en"],
                "chosen": r["nl"], "rejected": hyp}
        if r["set"] == "pref_novels":
            pref_novels.append({**pair, "chrf": chrf(hyp, r["nl"]), "length_ratio": len(hyp) / max(1, len(r["nl"]))})
            continue
        why, c, ratio = band6(r["kind"], r["nl"], hyp, blocked, chrf)
        measured[r["kind"]]["chrf"].append(c)
        measured[r["kind"]]["ratio"].append(ratio)
        if why:
            funnel[s][why] += 1
            continue
        survivors[s].append({**pair, "chrf": c, "length_ratio": ratio})
    drawn = []
    for s in sorted(WANT6):
        rows = tokenized6(sorted(survivors.get(s, []), key=lambda p: p["id"]), counter)
        fit = [p for p in rows if p["tokens_pair"] <= PAIR_MAX6]
        funnel[s]["too_long"] = len(rows) - len(fit)
        funnel[s]["kept"] = len(fit)
        random.Random(f"{SEED6}/draw/{s}").shuffle(fit)
        drawn += fit[:WANT6[s]]
        funnel[s]["drawn"] = min(len(fit), WANT6[s])
        funnel[s]["short_of_want"] = max(0, WANT6[s] - len(fit))
    drawn.sort(key=lambda p: p["id"])
    random.Random(f"{SEED6}/order").shuffle(drawn)
    # the preference eval sets: G4-0's held-out translations of the held-out books and BT validation
    held = [r for r in read_rows(args.heldout) if r["set"] in ("books", "btval")]
    held_g40 = g40_rows(args.heldout_g40)
    prefs = {"pref_novels": pref_novels, "pref_books": [], "pref_btval": []}
    for r in held:
        name = f"pref_{r['set']}"
        funnel[name]["candidates"] += 1
        if r["id"] not in held_g40:
            raise SystemExit(f"{r['id']}: no held-out translation of {rejected_model}")
        why, hyp = failure6(r["reference"].strip(), held_g40[r["id"]])
        if why:
            funnel[name][why] += 1
            continue
        chosen = r["reference"].strip()
        prefs[name].append({"id": f"{name}:{r['id']}", "set": name, "stratum": name, "kind": r["set"],
                            "source": json.loads(r["meta"]).get("source"), "en": r["source"].strip(),
                            "chosen": chosen, "rejected": hyp, "chrf": chrf(hyp, chosen),
                            "length_ratio": len(hyp) / max(1, len(chosen))})
    for name in PREF_SETS6:
        rows = tokenized6(sorted(prefs[name], key=lambda p: p["id"]), counter)
        prefs[name] = [p for p in rows if p["tokens_pair"] <= PAIR_MAX6]
        funnel[name]["too_long"] = len(rows) - len(prefs[name])
        funnel[name]["kept"] = len(prefs[name])
        if not prefs[name]:
            raise SystemExit(f"{name}: no pair")
    packing = {}
    for seq in (SEQ6, SEQ6_FALLBACK):
        rows = packed6(drawn, seq, pad, bos)
        g4dpo.write_rows(rows, out / f"train-{seq}.parquet")
        smoke_rows = [r for r in rows if r["batch"] < SMOKE6_BATCHES]
        if 1 + max(r["batch"] for r in smoke_rows) < SMOKE6_BATCHES:
            raise SystemExit(f"fewer than {SMOKE6_BATCHES} batches at {seq}")
        g4dpo.write_rows(smoke_rows, out / f"smoke-{seq}.parquet")
        packing[str(seq)] = {"train": pack_stats(rows, seq), "smoke": pack_stats(smoke_rows, seq)}
        for name in PREF_SETS6:
            pref_rows = packed6(prefs[name], seq, pad, bos)
            g4dpo.write_rows(pref_rows, out / f"eval-{name}-{seq}.parquet")
            packing[str(seq)][name] = pack_stats(pref_rows, seq)
    write_parquet([messages_row(p["en"], p["chosen"]) for p in drawn], out / "train-chosen.parquet",
                  row_group_size=ROW_GROUP4)
    (out / "template.jinja").write_text(g4lora.TEMPLATE, encoding="utf-8")
    every = [(k, p) for k, p in enumerate(drawn)] + [(-1, p) for name in PREF_SETS6 for p in prefs[name]]
    write_parquet([{"id": p["id"], "set": p["set"], "stratum": p["stratum"], "kind": p["kind"], "source": p["source"],
                    "en": p["en"], "chosen": p["chosen"], "rejected": p["rejected"], "chrf": p["chrf"],
                    "length_ratio": p["length_ratio"], "tokens_prompt": len(p["prompt"]),
                    "tokens_chosen": len(p["chosen_ids"]), "tokens_rejected": len(p["rejected_ids"]), "order": k}
                   for k, p in every], out / "pairs.parquet", schema(PAIRS6_SCHEMA))
    kept = collections.defaultdict(list)
    for p in drawn:
        kept[p["kind"]].append(p)
    report = {
        "arm": arm, "rejected_model": rejected_model, "seed": SEED6, "prompt": PROMPT, "want": WANT6, "chrf_bands": CHRF6, "length_band": LENGTH6,
        "pair_max": PAIR_MAX6, "pad_id": pad, "bos_id": bos, "rows_per_batch": ROWS6,
        "funnel": {k: dict(v) for k, v in sorted(funnel.items())},
        "pairs": {"train": len(drawn), **{name: len(prefs[name]) for name in PREF_SETS6}},
        "by_stratum": dict(sorted(collections.Counter(p["stratum"] for p in drawn).items())),
        "measured": {k: {"chrf": quantiles(v["chrf"]), "length_ratio": quantiles(v["ratio"])}
                     for k, v in sorted(measured.items())},
        "kept": {k: {"pairs": len(v), "chrf": quantiles([p["chrf"] for p in v]),
                     "length_ratio": quantiles([p["length_ratio"] for p in v]),
                     "tokens_chosen": quantiles([len(p["chosen_ids"]) for p in v]),
                     "tokens_rejected": quantiles([len(p["rejected_ids"]) for p in v]),
                     "tokens_pair": quantiles([p["tokens_pair"] for p in v])} for k, v in sorted(kept.items())},
        "prefs": {name: {"chrf": quantiles([p["chrf"] for p in prefs[name]]),
                         "length_ratio": quantiles([p["length_ratio"] for p in prefs[name]])} for name in PREF_SETS6},
        "packing": packing,
        "inputs": {"cands": sha256_file(args.cands), "heldout": sha256_file(args.heldout),
                   "evalsets": sha256_file(args.evalsets), "blind": sha256_file(args.blind),
                   "rejected": sorted(sha256_file(f) for p in args.rejected for f in glob.glob(p)),
                   "heldout_g40": sorted(sha256_file(f) for p in args.heldout_g40 for f in glob.glob(p))},
        "files": {f.name: sha256_file(f) for f in sorted(out.glob("*.parquet"))},
    }
    if getattr(args, "compare", None):
        report["compare"] = compare6(report, [p["id"] for p in drawn], Path(args.compare))
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "report.md").write_text(report6_md(report), encoding="utf-8")
    print(f"BUILT6 {json.dumps({'pairs': report['pairs'], 'steps': {k: v['train']['steps'] for k, v in packing.items()}})}",
          flush=True)


def compare6(report, drawn_ids, other_dir):
    """amendment G4-6b.4 (reported, decides nothing): this build against another one (G4-6's, rejected = G4-0) on the
    same candidates: the funnel per stratum, the chrF of the kept pairs, and the drawn training pairs both have"""
    other = json.loads((other_dir / "report.json").read_text())
    other_ids = {p["id"] for p in read_rows(other_dir / "pairs.parquet", ["id", "set"]) if p["set"] == "train"}
    mine = set(drawn_ids)
    return {"arm": other.get("arm", ARM6), "rejected_model": other.get("rejected_model", BASE),
            "funnel": {s: {k: [other["funnel"].get(s, {}).get(k, 0), f.get(k, 0)] for k in COMPARE6B}
                       for s, f in report["funnel"].items()},
            "chrf_measured_p50": {k: [other["measured"].get(k, {}).get("chrf", {}).get("p50"), v["chrf"].get("p50")]
                                  for k, v in report["measured"].items()},
            "train_pairs": [len(other_ids), len(mine)], "train_pairs_in_both": len(mine & other_ids)}


def compare6_md(c, r):
    o, me = c["arm"], r["arm"]
    L = ["", f"## Tegen {o} (rejected {c['rejected_model']}, dezelfde kandidaten; rapporteren, beslist niet)", "",
         f"Per stratum: {o} / {me}.", "", "| stratum | " + " | ".join(COMPARE6B) + " |",
         "|---|" + "---:|" * len(COMPARE6B)]
    for s, f in c["funnel"].items():
        L.append(f"| {s} | " + " | ".join(f"{a} / {b}" for a, b in (f[k] for k in COMPARE6B)) + " |")
    L += ["", "Mediaan chrF(rejected, chosen) na de fouten, per soort: " + ", ".join(
        f"{k} {a} / {b}" for k, (a, b) in c["chrf_measured_p50"].items()) + ".",
          f"Getrokken trainingsparen: {o} {c['train_pairs'][0]}, {me} {c['train_pairs'][1]}, in beide "
          f"{c['train_pairs_in_both']}."]
    return L


def report6_md(r):
    rejected = r.get("rejected_model", BASE)
    L = [f"# Arm {r.get('arm', ARM6)}: de paren (`g4arm.py build6`, rejected = {rejected})", "",
         f"Seed `{r['seed']}`, prompt `{r['prompt']}`; chrF-banden {json.dumps(r['chrf_bands'])}, lengteband "
         f"{json.dumps(r['length_band'])}, paar ≤ {r['pair_max']} tokens.", "",
         "## Trechter per stratum", "", "| stratum | " + " | ".join(
             ("candidates", *FAILURES6, *BANDS6, "too_long", "kept", "drawn", "short_of_want")) + " |",
         "|---|" + "---:|" * (len(FAILURES6) + len(BANDS6) + 5)]
    for s, f in r["funnel"].items():
        L.append(f"| {s} | " + " | ".join(str(f.get(k, 0)) for k in (
            "candidates", *FAILURES6, *BANDS6, "too_long", "kept", "drawn", "short_of_want")) + " |")
    L += ["", "## chrF (rejected tegen chosen) en lengteverhouding", "",
          "| soort | wat | n | p5 | p25 | p50 | p75 | p95 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in r["measured"].items():
        for what in ("chrf", "length_ratio"):
            q = v[what]
            L.append(f"| {k} | {what} (na de fouten van {rejected}) | {q.get('n')} | {q.get('p5')} | {q.get('p25')} | "
                     f"{q.get('p50')} | {q.get('p75')} | {q.get('p95')} |")
    for k, v in r["kept"].items():
        q = v["chrf"]
        L.append(f"| {k} | chrf (getrokken) | {q.get('n')} | {q.get('p5')} | {q.get('p25')} | {q.get('p50')} | "
                 f"{q.get('p75')} | {q.get('p95')} |")
    L += ["", "## Packing", "", "| rijen van | set | rijen | batches (stappen) | paren | tokens | vulling | paren per stap |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for seq, sets in r["packing"].items():
        for name, s in sets.items():
            L.append(f"| {seq} | {name} | {s['rows']} | {s['batches']} | {s['pairs']} | {s['tokens']} | {s['fill']} | "
                     f"{s['pairs_per_step']} |")
    L += ["", f"Paren: {json.dumps(r['pairs'])}; per stratum {json.dumps(r['by_stratum'])}.", ""]
    if r.get("compare"):
        L += compare6_md(r["compare"], r) + [""]
    return "\n".join(L)


def check6(args):
    """amendment G4-6.3: every side of every pair through MaxText's own SFT steps (g4lora.check_examples: the ids of
    MaxText's apply_chat_template, tokenization and prompt masking against the model's own tokenizer and template),
    MaxText's pad id the build's; and every packed row equal to those ids packed as build6 packs them and shifted by
    MaxText's own shift_and_refine"""
    import multiprocessing

    import transformers
    from maxtext.input_pipeline import input_pipeline_utils as ipu

    d = Path(args.data)
    report = json.loads((d / "report.json").read_text())
    pairs = {p["id"]: p for p in read_rows(d / "pairs.parquet")}
    sides = [(f"{i}/{side}", p["en"], p[side]) for i, p in sorted(pairs.items()) for side in ("chosen", "rejected")]
    jobs = []
    for k in range(0, len(sides), args.chunk):
        part = sides[k:k + args.chunk]
        jobs.append(([[{"role": "user", "content": g4lora.user_turn(en)}, {"role": "assistant", "content": nl}]
                      for _, en, nl in part], [en for _, en, _ in part], [label for label, _, _ in part],
                     args.tokenizer))
    if args.processes > 1:
        with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
            results = pool.map(_check_chunk6, jobs, chunksize=1)
    else:
        results = [_check_chunk6(j) for j in jobs]
    pads = {pad for _, _, pad in results}
    if pads != {report["pad_id"]}:
        raise SystemExit(f"MaxText's pad ids {pads}, the build's {report['pad_id']}")
    own = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    if own.bos_token_id != report["bos_id"]:
        raise SystemExit(f"the tokenizer's bos id {own.bos_token_id}, the build's {report['bos_id']}")

    def ids(text):
        return own(text, add_special_tokens=False)["input_ids"]

    cache, rows_checked = {}, 0
    for f in sorted(d.glob("*.parquet")):
        # eval-own4 is an SFT eval file (G4-4's held-out translations as targets), not packed pairs
        if f.name in ("pairs.parquet", "train-chosen.parquet", f"eval-{OWN6B}.parquet"):
            continue
        seq = int(f.stem.rsplit("-", 1)[1])
        for row in read_rows(f):
            parts = []
            for pid in row["pairs"]:
                if pid not in cache:
                    p = pairs[pid]
                    prompt = ids(g4lora.prompt_text(p["en"]))
                    cache[pid] = [(prompt, ids(g4lora.completion_text(p["chosen"]))),
                                  (prompt, ids(g4lora.completion_text(p["rejected"])))]
                parts += cache[pid]
            want = g4dpo.row_unshifted(parts, seq, report["pad_id"])
            batch = {k: v[None] for k, v in want.items()}
            want = {k: v[0] for k, v in ipu.shift_and_refine(batch, ignored_ids=[report["pad_id"], report["bos_id"]],
                                                              axis=1).items()}
            differ = [k for k in g4dpo.KEYS if list(want[k]) != list(row[k])]
            if differ:
                raise SystemExit(f"{f.name} row of batch {row['batch']} with pairs {row['pairs'][:2]}: {differ} differ "
                                 "from MaxText's packing and shift")
            rows_checked += 1
    print(f"CHECK6 {json.dumps({'sides': len(sides), 'pad_id': pads.pop(), 'rows': rows_checked})}", flush=True)


def _check_chunk6(job):
    messages, ens, labels, tokenizer = job
    return g4lora.check_examples(messages, ens, tokenizer, MODEL, labels)


def eval_at6(summary, name, step):
    return next((e for e in summary.get("eval") or [] if e["set"] == name and e["step"] == step), None)


def last_evals(summary):
    """{set: the eval at the set's last step} of a training summary"""
    last = {}
    for e in summary.get("eval") or []:
        if e["set"] not in last or e["step"] > last[e["set"]]["step"]:
            last[e["set"]] = e
    return last


def smoke6_verdict(summary, adapter, base_summary, seq, lora, arm_rows, base_rows, build_report, start=None, arm=None,
                   saved=None):
    """amendment G4-6.6: every check with its value and whether it holds. start (amendment G4-6b.6): the arm that the
    run starts from with its adapter merged (G4-4): step 0's SFT losses are then that arm's own eval at its last step
    (within BASE_MAX_DIFF6B), the run's base_adapter is that arm's, own4 is evaluated too, and the smoke translations'
    base is that arm. arm G4-6c (amendment G4-6c.6): its spec and eval interval, and saved (the adapter.json the run
    kept after step SMOKE_SAVE6C), the log-ratio's tokens and the loss's composition"""
    spec = g4dpo.spec(**(DPO6C if arm == ARM6C else DPO6))
    checks = {}
    loss = summary.get("loss") or []
    checks["steps"] = {"value": summary.get("steps_done"), "holds": summary.get("steps_done") == SMOKE6_STEPS}
    checks["loss_finite"] = {"value": [v for _, v in loss],
                             "holds": len(loss) == SMOKE6_STEPS and all(math.isfinite(v) for _, v in loss)}
    checks["peak_gib"] = {"value": summary.get("peak_gib_max"),
                          "holds": summary.get("peak_gib_max") is not None and summary["peak_gib_max"] <= MEM_MAX_GIB}
    checks["adapter"] = {"value": {k: adapter.get(k) for k in ("rank", "alpha", "scale")},
                         "holds": (adapter.get("rank"), adapter.get("alpha"), adapter.get("scale")) == (32, 16.0, 0.5)}
    dpo = summary.get("dpo") or {}
    steps = dpo.get("steps") or []
    checks["dpo_steps"] = {"value": [(s["pairs"], round(s["loss"], 5)) for s in steps],
                           "holds": len(steps) == SMOKE6_STEPS and all(s["pairs"] > 0 and math.isfinite(s["loss"])
                                                                        for s in steps)}
    checks["dpo_spec"] = {"value": dpo.get("spec"), "holds": dpo.get("spec") == spec}
    sft_sets = [s for s in EVAL_SETS4 if not (s == "btlong" and seq == SEQ6_FALLBACK)]
    names = sft_sets + list(PREF_SETS6) + ([OWN6B] if start else [])
    have = {(e["set"], e["step"]) for e in summary.get("eval") or []}
    checks["evals"] = {"value": sorted(have), "holds": all((n, s) in have for n in names for s in SMOKE6_EVAL_STEPS)}
    expected = g4dpo.expected_step0(spec)
    step0 = {}
    for name in PREF_SETS6:
        e = eval_at6(summary, name, 0) or {}
        diffs = [abs(e.get("loss", math.inf) - expected["loss"]), abs(e.get("margin", math.inf)),
                 abs(e.get("chosen_nll", math.inf) - e.get("ref_chosen_nll", -math.inf)),
                 abs(e.get("rejected_nll", math.inf) - e.get("ref_rejected_nll", -math.inf))]
        step0[name] = {"loss": e.get("loss"), "margin": e.get("margin"), "max_diff": max(diffs)}
    checks["policy_is_reference_at_step0"] = {
        "value": step0, "holds": all(v["max_diff"] <= STEP0_MAX_DIFF6 for v in step0.values())}
    base = {}
    start_last = last_evals(base_summary) if start else {}
    for name in sft_sets:
        e = eval_at6(summary, name, 0)
        b = start_last.get(name) if start else eval_at6(base_summary, name, 0)
        base[name] = {"run": e and e["loss"], ("start_last" if start else "g4_4_step0"): b and b["loss"],
                      "diff": abs(e["loss"] - b["loss"]) if e and b else None}
    bound = BASE_MAX_DIFF6B if start else BASE_MAX_DIFF6
    checks["base_losses_at_step0"] = {"value": base, "holds": all(v["diff"] is not None and v["diff"] <= bound
                                                                  for v in base.values())}
    if start:
        b = summary.get("base_adapter") or {}
        checks["base_adapter"] = {"value": b, "holds": b.get("name") == start and b.get("modules") == SMOKE_MODULES
                                  and (b.get("rank"), b.get("alpha"), b.get("scale")) == (32, 16.0, 0.5)}
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
    if arm == ARM6C:
        # amendment G4-6c.6: every side's log-ratio n_end tokens short of its NLL; step 1's loss = its DPO term +
        # nll_alpha x the chosen NLL over every target token; the adapter of step SMOKE_SAVE6C kept, not the last one
        n_end, ends = spec["end_tokens"], {}
        for name in PREF_SETS6:
            e = eval_at6(summary, name, 0) or {}
            ends[name] = {"pairs": e.get("pairs"), "chosen": e.get("chosen_tokens_all", 0) - e.get("chosen_tokens", 0),
                          "rejected": e.get("rejected_tokens_all", 0) - e.get("rejected_tokens", 0)}
        checks["end_tokens"] = {"value": ends, "holds": all(
            v["pairs"] and v["chosen"] == n_end * v["pairs"] and v["rejected"] == n_end * v["pairs"] for v in ends.values())}
        first = steps[0] if steps else {}
        got = loss[0][1] if loss else None
        want = first.get("loss", math.nan) + spec["nll_alpha"] * first.get("chosen_nll_all", math.nan)
        checks["loss_composition"] = {"value": {"loss": got, "dpo_plus_nll": want}, "holds": got is not None and
                                      math.isfinite(want) and abs(got - want) <= LOSS_MAX_DIFF6C * max(1.0, abs(got))}
        kept = (summary.get("saved_adapters") or {}).get(str(SMOKE_SAVE6C)) or {}
        meta = {k: (saved or {}).get(k) for k in ("rank", "alpha", "scale", "steps")}
        last_abs = summary.get("adapter_sum_abs")
        checks["saved_adapter"] = {"value": {"saved": kept, "meta": meta, "last_sum_abs": last_abs},
                                   "holds": (meta["rank"], meta["alpha"], meta["scale"], meta["steps"]) ==
                                   (32, 16.0, 0.5, SMOKE_SAVE6C) and kept.get("sum_abs", 0) > 0 and
                                   last_abs is not None and kept["sum_abs"] != last_abs}
        # addendum G4-6c.6e: the run read the reference of its training batches (--ref-in) and that reference is the
        # policy's at step 1
        checks["ref_in"] = ref_in_check(summary, spec)
    estimate = smoke6_estimate(summary, build_report, seq, EVAL_EVERY6C if arm == ARM6C else EVAL_EVERY6)
    checks["estimate_hours"] = {"value": estimate["hours"], "holds": estimate["hours"] <= MAX_HOURS6}
    # reported: the rejected side of pref_btval is the start model's own translation of the BT validation, as the eval
    # set own (G4-0) or own4 (G4-4)
    own_set = OWN6B if start else "own"
    own0, rej0 = eval_at6(summary, own_set, 0), eval_at6(summary, "pref_btval", 0)
    return {"checks": checks, "passed": all(c["holds"] for c in checks.values()), "estimate": estimate,
            "own_against_pref_btval_rejected": {own_set: own0 and own0["loss"], "pref_btval_rejected_nll":
                                                rej0 and rej0.get("rejected_nll")}}


def ref_in_check(summary, spec):
    """Addendum G4-6c.6e: {value, holds} of a G4-6c run that read its training batches' reference (--ref-in): the
    record of it (no pass of its own over them), and step 1's DPO term and margin, where the policy is still the
    reference, within STEP1_MAX_DIFF6E of log 2 and 0"""
    train = ((summary.get("dpo") or {}).get("ref") or {}).get("train") or {}
    steps = (summary.get("dpo") or {}).get("steps") or []
    first = steps[0] if steps else {}
    expected = g4dpo.expected_step0(spec)
    diffs = {"loss": abs(first.get("loss", math.inf) - expected["loss"]), "margin": abs(first.get("margin", math.inf))}
    return {"value": {"ref_in": train.get("ref_in"), "sha256": train.get("sha256"), "step": first.get("step"), **diffs},
            "holds": bool(train.get("ref_in")) and bool(train.get("sha256")) and "seconds" not in train
            and first.get("step") == 1 and all(v <= STEP1_MAX_DIFF6E for v in diffs.values())}


def smoke6_estimate(summary, build_report, seq, every=EVAL_EVERY6):
    """The full run's hours: every batch of the training file at the smoke run's median step time, an eval round
    (every set, as long as the smoke run's at step 0) at step 0, every `every` steps and after the last, the
    smoke run's time to its first step, and the reference pass before it (amendment G4-6.4a: every training batch at
    the smoke run's median time per batch, and the preference eval files as in the smoke run)"""
    steps = build_report["packing"][str(seq)]["train"]["steps"]
    step_s = (summary.get("throughput") or {}).get("step_s_median")
    round_s = sum(e.get("seconds", 0.0) for e in summary.get("eval") or [] if e["step"] == 0)
    rounds = 1 + steps // every + (0 if steps % every == 0 else 1)
    times = summary.get("times") or {}
    before = (times.get("python_to_setup_s") or 0) + (times.get("setup_s") or 0) + (times.get("compile_s_est") or 0)
    ref = (summary.get("dpo") or {}).get("ref") or {}
    ref_s = steps * (ref.get("train") or {}).get("batch_s_median", 0.0) + sum(
        r.get("seconds", 0.0) for name, r in ref.items() if name != "train")
    if step_s is None:
        return {"steps": steps, "hours": math.inf}
    total = steps * step_s + rounds * round_s + before + ref_s
    return {"steps": steps, "step_s": step_s, "eval_round_s": round_s, "rounds": rounds, "before_s": before,
            "ref_s": round(ref_s, 1), "hours": round(total / 3600, 2)}


def smoke6(args):
    summary = json.loads(Path(args.summary).read_text())
    adapter = json.loads(Path(args.adapter).read_text())
    base_summary = json.loads(Path(args.base_summary).read_text())
    build_report = json.loads(Path(args.build_report).read_text())
    lora = lora_line(Path(args.translate_log).read_text(errors="replace"))
    rows = {}
    for name, pairs, metricx in (("arm", args.arm_pairs, args.arm_metricx), ("base", args.base_pairs, args.base_metricx)):
        p = {r["id"]: r for r in read_rows(pairs)}
        m = {r["id"]: r for r in read_rows(metricx)}
        rows[name] = [{**r, "metricx_qe": m.get(i, {}).get("qe_source_hypothesis")} for i, r in p.items()]
    start = getattr(args, "start", None)
    arm = getattr(args, "arm", None) or ARM6
    saved = json.loads(Path(args.saved_adapter).read_text()) if getattr(args, "saved_adapter", None) else None
    if arm == ARM6C and saved is None:
        raise SystemExit("smoke6 --arm G4-6c needs --saved-adapter (the adapter the smoke run kept, amendment G4-6c.6)")
    result = smoke6_verdict(summary, adapter, base_summary, args.seq, lora, rows["arm"], rows["base"], build_report,
                            start=start, arm=arm, saved=saved)
    result["seq"] = args.seq
    result["eval"] = summary.get("eval")
    result["throughput"] = summary.get("throughput")
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    amendment = "G4-6c.6" if arm == ARM6C else "G4-6b.6" if start else "G4-6.6"
    L = [f"# Smoke-test {arm} (amendement {amendment}), rijen van {args.seq}"
         + (f"; startmodel {start} (ingebakken), basis van de vertalingen {start}" if start else ""), "",
         f"**{'Geslaagd' if result['passed'] else 'Niet geslaagd'}.** Schatting van de volle run: "
         f"{result['estimate']['hours']} u.", "", "| controle | waarde | gehaald |", "|---|---|---|"]
    for k, c in result["checks"].items():
        L.append(f"| {k} | `{json.dumps(c['value'], default=str)[:200]}` | {'ja' if c['holds'] else 'nee'} |")
    L += ["", f"Ter controle (beslist niet): {json.dumps(result['own_against_pref_btval_rejected'])}.", ""]
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"SMOKE6 {'PASSED' if result['passed'] else 'FAILED'} {json.dumps({k: c['holds'] for k, c in result['checks'].items()})}",
          flush=True)
    if not result["passed"]:
        raise SystemExit(1)


def pref_measure(summary):
    """The preference measures of G4-6's training (reported, deciding nothing): per preference eval set and step the
    pairs' loss, accuracy, margin, the NLL of the chosen and the rejected sides; the training steps per window of 50;
    the loss on own (G4-0's greedy translations of the BT validation) per eval step"""
    out = {"sets": collections.defaultdict(dict), "own": {}, OWN6B: {}, "train": []}
    for e in summary.get("eval") or []:
        if e.get("kind") == "pref":
            out["sets"][e["set"]][e["step"]] = {k: e[k] for k in ("pairs", "loss", "accuracy", "margin", "chosen_nll",
                                                                  "rejected_nll", "ref_chosen_nll", "ref_rejected_nll")}
            # amendment G4-6c: the NLL over every target token beside the log-ratio's (the same without end_tokens)
            out["sets"][e["set"]][e["step"]].update({k: e[k] for k in ("chosen_nll_all", "rejected_nll_all") if k in e})
        elif e["set"] in ("own", OWN6B):
            out[e["set"]][e["step"]] = e["loss"]
    steps = (summary.get("dpo") or {}).get("steps") or []
    for k in range(0, len(steps), 50):
        w = steps[k:k + 50]
        n = sum(s["pairs"] for s in w) or 1
        out["train"].append({"steps": [w[0]["step"], w[-1]["step"]],
                             **{m: sum(s[m] * s["pairs"] for s in w) / n for m in ("loss", "accuracy", "margin",
                                                                                 "chosen_nll", "rejected_nll")}})
    out["sets"] = {k: dict(sorted(v.items())) for k, v in sorted(out["sets"].items())}
    out["own"] = dict(sorted(out["own"].items()))
    out[OWN6B] = dict(sorted(out[OWN6B].items()))
    out["spec"] = (summary.get("dpo") or {}).get("spec")
    # amendment G4-6b: the reference is the start model with its adapter merged (G4-4), else G4-0
    out["reference"] = (summary.get("base_adapter") or {}).get("name") or BASE
    # the reading fixed beforehand (draft amendment G4-6.7; reported, deciding nothing): the held-out accuracy of the
    # held-out books and BT validation at the last eval step
    last = {k: out["sets"][k][max(out["sets"][k])]["accuracy"] for k in READING_SETS6 if out["sets"].get(k)}
    if len(last) == len(READING_SETS6):
        learned = [v >= READING_LEARNED6 for v in last.values()]
        out["reading"] = {"accuracy_last": last, "verdict": "voorkeur geleerd" if all(learned) else
                          "niet geleerd" if all(v < READING_NOT6 for v in last.values()) else "gemengd"}
    return out


def pref_text(p):
    ref = p.get("reference", BASE)
    L = ["", "## Voorkeuren (rapporteren, beslist niet)", "",
         f"Verlies `{json.dumps(p['spec'])}`. Per held-out voorkeursset en eval-stap: paren, verlies, nauwkeurigheid "
         "(beloning chosen > rejected), marge (beta × verschil van de log-ratio's), NLL per token van chosen en "
         f"rejected onder het model (en onder de referentie {ref}).", "",
         "| set | stap | paren | verlies | nauwkeurigheid | marge | NLL chosen | NLL rejected | ref chosen | ref rejected |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, steps in p["sets"].items():
        for step, v in steps.items():
            L.append(f"| {name} | {step} | {v['pairs']} | {f3(v['loss'])} | {f3(v['accuracy'])} | {f3(v['margin'])} | "
                     f"{f3(v['chosen_nll'])} | {f3(v['rejected_nll'])} | {f3(v['ref_chosen_nll'])} | "
                     f"{f3(v['ref_rejected_nll'])} |")
    if p.get("reading"):
        L += ["", f"**Lezing (vooraf vastgelegd, beslist niet): {p['reading']['verdict']}** (nauwkeurigheid na de laatste "
              f"stap: {json.dumps(p['reading']['accuracy_last'])}; geleerd: beide ≥ {READING_LEARNED6}, niet geleerd: "
              f"beide < {READING_NOT6})."]
    L += ["", "Verlies op `own` (G4-0's eigen greedy vertalingen van de BT-validatie; G4-4: 0,075 → 0,339): " + ", ".join(
        f"{k}: {f3(v)}" for k, v in p["own"].items()) + "."]
    if p.get(OWN6B):
        L += ["", f"Verlies op `{OWN6B}` (G4-4's eigen greedy vertalingen van de BT-validatie): " + ", ".join(
            f"{k}: {f3(v)}" for k, v in p[OWN6B].items()) + "."]
    L += ["",
        "Training per 50 stappen (gewogen naar paren): " + "; ".join(
            f"{w['steps'][0]}–{w['steps'][1]}: verlies {f3(w['loss'])}, nauwk. {f3(w['accuracy'])}, marge "
            f"{f3(w['margin'])}" for w in p["train"])]
    return L


READING_SETS6 = ("pref_books", "pref_btval")
READING_LEARNED6 = 0.70   # the reading's bound: accuracy on both held-out preference sets at the last step
READING_NOT6 = 0.60
PAIRTEXTS6 = 1000         # the classifier on chosen and rejected: at most this many pairs per kind (reported only)


def pairtexts6(args):
    """amendment G4-6.7 (reported, decides nothing): the classifier's texts of chosen and rejected, a seeded sample of
    PAIRTEXTS6 training pairs per kind, model chosen / rejected, set = the kind"""
    pairs = [p for p in read_rows(args.pairs) if p["set"] == "train"]
    out = []
    for kind in sorted({p["kind"] for p in pairs}):
        part = sorted((p for p in pairs if p["kind"] == kind), key=lambda p: p["id"])
        part = random.Random(f"{SEED6}/pairtexts/{kind}").sample(part, min(PAIRTEXTS6, len(part)))
        for p in sorted(part, key=lambda p: p["id"]):
            out += [{"model": side, "set": kind, "id": p["id"], "text": p[side]} for side in ("chosen", "rejected")]
    write_parquet(out, args.out)
    print(f"PAIRTEXTS6 {json.dumps(collections.Counter(r['model'] + '/' + r['set'] for r in out))}", flush=True)


def pairclass6(args):
    """The share the classifier calls original, per kind, of the chosen and the rejected sides (reported only)"""
    arm, rejected = getattr(args, "arm", None) or ARM6, getattr(args, "rejected_model", None) or BASE
    by = collections.defaultdict(list)
    for r in read_rows(args.predictions):
        by[(r["set"], r["model"])].append(r["p_original"] >= 0.5)
    result = {kind: {side: {"n": len(by[(kind, side)]), "original": sum(by[(kind, side)]) / len(by[(kind, side)])}
                     for side in ("chosen", "rejected") if by[(kind, side)]}
              for kind in sorted({k for k, _ in by})}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    L = [f"# {arm}: de classifier op chosen en rejected (rapporteren, beslist niet)", "",
         f"| soort | n | \"origineel\" chosen | \"origineel\" rejected ({rejected}) | verschil |", "|---|---:|---:|---:|---:|"]
    for kind, v in result.items():
        c, r = v.get("chosen", {}), v.get("rejected", {})
        diff = c["original"] - r["original"] if c and r else None
        L.append(f"| {kind} | {c.get('n')} | {pct(c.get('original'))} | {pct(r.get('original'))} | {f3(diff, True)} |")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"PAIRCLASS6 {json.dumps(result)}", flush=True)


# ---------------------------------------------------------------------------------------------------- G4-6c

def ends6c(args):
    """amendment G4-6c.3: every packed file of G4-6b's pairs (the training files, the smoke sets and the preference eval
    files at both row lengths): each segment's target tokens one run of more than end_tokens tokens whose last
    end_tokens are the ids of <turn|>\\n by Gemma 4's tokenizer, so the log-ratio leaves out exactly the turn's end"""
    import transformers

    n = DPO6C["end_tokens"]
    ids = transformers.AutoTokenizer.from_pretrained(args.tokenizer)(END6C, add_special_tokens=False)["input_ids"]
    if len(ids) != n:
        raise SystemExit(f"{END6C!r} is {ids}, not {n} tokens")
    d = Path(args.data)
    want = [f"{k}-{seq}.parquet" for k in ("train", "smoke", *(f"eval-{x}" for x in PREF_SETS6))
            for seq in (SEQ6, SEQ6_FALLBACK)]
    missing = [f for f in want if not (d / f).is_file()]
    if missing:
        raise SystemExit(f"{d}: no {missing}")
    files = {f: g4dpo.check_ends(d / f, n, ids) for f in sorted(want)}
    print(f"ENDS6C {json.dumps({'end': END6C, 'end_ids': ids, 'end_tokens': n, 'files': files})}", flush=True)


def ckpt_step6c(name):
    m = CKPT6C.match(name)
    if not m:
        raise SystemExit(f"{name} is no checkpoint of G4-6c (G4-6c-s<step>)")
    return int(m.group(1))


def part_counts6c(scored, ids):
    """Per model and set of a part: n, not finished, loop, English left, empty (armeval.counts), the mean QE, and with a
    reference the median of output words per reference word and the share above LONG6C x the reference"""
    import armeval

    out = {}
    for name, rows_by_id in scored.items():
        out[name] = {}
        for s, part in ids.items():
            rows = [rows_by_id[i] for i in part]
            lengths = armeval.length_ratios(rows)
            with_ref = [r for r in rows if r.get("reference") and armeval.words(r["reference"])]
            longer = (sum(armeval.words(r["hypothesis"]) > LONG6C * armeval.words(r["reference"]) for r in with_ref)
                      / len(with_ref)) if with_ref else None
            out[name][s] = {**armeval.counts(rows), "metricx_qe": mean(r["metricx_qe"] for r in rows),
                            "out_per_ref": lengths and lengths["out_per_ref"], "over_1_5_ref": longer}
    return out


def training6c(summary, steps):
    """The training's evals at step 0 and at the checkpoints: the SFT files' loss, and per preference file its pairs'
    loss, accuracy, margin and the NLLs (the log-ratio's tokens and every target token) with the reference's"""
    out = {}
    for step in [0, *steps]:
        sft = {n: (eval_at6(summary, n, step) or {}).get("loss") for n in SFT_SETS6C}
        pref = {}
        for n in PREF_SETS6:
            e = eval_at6(summary, n, step)
            if e is None:
                raise SystemExit(f"the training has no eval of {n} at step {step}")
            pref[n] = {k: e.get(k) for k in ("pairs", "loss", "accuracy", "margin", "chosen_nll", "chosen_nll_all",
                                             "rejected_nll", "rejected_nll_all", "ref_chosen_nll", "ref_rejected_nll")}
        missing = [n for n, v in sft.items() if v is None]
        if missing:
            raise SystemExit(f"the training has no eval of {missing} at step {step}")
        out[step] = {"sft": sft, "pref": pref}
    return out


def report6c(args):
    """amendment G4-6c.7-G4-6c.9: the checkpoint chosen on the search part (G4-5.6's rule, select5), the rule (a1) and
    (b) against G4-0 on its decision part, and per checkpoint the failures, the length and the training's measures"""
    import g4mix

    evalsets = {r["id"]: r for r in read_rows(args.evalsets)}
    split = json.loads(Path(args.split).read_text())
    assign = split["assign"]
    if g4mix.digest(assign) != split["digest"] or split["digest"] != SPLIT_DIGEST6C:
        raise SystemExit(f"{args.split}: not the split of G4-5.4 (digest {split['digest']}, want {SPLIT_DIGEST6C})")
    cands = [c.split("=", 1) for c in args.candidate]
    names = [n for n, _ in cands]
    steps = [ckpt_step6c(n) for n in names]
    if steps != sorted(set(steps)):
        raise SystemExit(f"the checkpoints {names}: not in the order of their steps")
    refs = [c.split("=", 1) for c in args.reference or []]
    scored = {name: scored_dir(d) for name, d in cands + [(BASE, args.base)] + refs}
    for name, rows in scored.items():
        if set(rows) != set(evalsets):
            raise SystemExit(f"{name}: scores for {len(rows)} of {len(evalsets)} segments")
    refqe = {r["id"]: r["qe_source_reference"] for r in read_rows(args.refqe)}
    missing = [i for i, r in evalsets.items() if r["set"] in HUMAN_QE and refqe.get(i) is None]
    if missing:
        raise SystemExit(f"{args.refqe}: no QE of the human reference of {len(missing)} segments, e.g. {missing[:3]}")
    check = {}
    for s in HUMAN_QE:
        ids = [i for i, r in evalsets.items() if r["set"] == s and not (s == "wmt24pp" and bad_source(r["meta"]))]
        got = mean(refqe[i] for i in ids)
        check[s] = {"again": got, "protocol": HUMAN_QE[s], "diff": got - HUMAN_QE[s]}
        if abs(got - HUMAN_QE[s]) > REFQE_MAX_DIFF5:
            raise SystemExit(f"the human references' QE of {s} again: {got:.4f}, the protocol's {HUMAN_QE[s]:.4f}")
    predictions = read_rows(args.predictions)
    part_ids = {p: part_ids5(evalsets, assign, p) for p in ("search", "decision")}
    parts = {p: part_measures5(scored, predictions, refqe, part_ids[p], BASE) for p in ("search", "decision")}
    # the choice reads the search part alone (amendment G4-6c.7)
    chosen, why = select5(names, parts["search"])
    d = parts["decision"]["models"][chosen]
    summary = json.loads(Path(args.train_summary).read_text())
    last = summary.get("steps_done")
    if steps[-1] != last or steps[:-1] != list(SAVE_STEPS6C):
        raise SystemExit(f"the checkpoints {steps}: not the kept steps {list(SAVE_STEPS6C)} and the last {last}")
    result = {"arm": ARM6C, "base": BASE, "candidates": names, "steps": dict(zip(names, steps)), "selected": chosen,
              "selection": why, "refqe_check": check,
              "split": {k: split[k] for k in ("seed", "share", "sets", "digest")}, "parts": parts,
              "a1": {"holds": d["a1"], "sets": d["a1_sets"]}, "b": {"holds": d["b"], "delta": d["b_delta"]},
              "counts": {p: part_counts6c(scored, part_ids[p]) for p in ("search", "decision")},
              "training": training6c(summary, steps), "spec": (summary.get("dpo") or {}).get("spec"),
              "reference": (summary.get("base_adapter") or {}).get("name") or BASE,
              "saved_adapters": summary.get("saved_adapters"), "references": [n for n, _ in refs]}
    beside = {}
    for spec in args.beside_summary or []:
        name, path = spec.split("=", 1)
        b = json.loads(Path(path).read_text())
        bl = last_evals(b)
        beside[name] = {"step": max((e["step"] for e in b.get("eval") or []), default=None),
                        "sft": {n: bl[n]["loss"] for n in SFT_SETS6C if n in bl},
                        "pref": {n: {k: bl[n].get(k) for k in ("loss", "accuracy", "margin", "chosen_nll", "rejected_nll")}
                                 for n in PREF_SETS6 if n in bl},
                        "spec": (b.get("dpo") or {}).get("spec")}
    result["beside"] = beside
    if args.tally:
        t = json.loads(Path(args.tally).read_text())
        if t.get("arm") != chosen:
            raise SystemExit(f"{args.tally}: a tally of {t.get('arm')}, the selected checkpoint is {chosen}")
        result["blind"], result["a2"], result["c"] = t, t["a2"], t["c"]
    else:
        result["blind"] = None
    result["verdict"] = verdict(result)
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")
    Path(args.out).write_text(report6c_text(result), encoding="utf-8")
    print(f"REPORT6C {json.dumps({'selected': chosen, 'a1': d['a1'], 'b': d['b_delta'], 'b_holds': d['b'], 'verdict': result['verdict']})}",
          flush=True)


def report6c_text(r):
    sets = list(HUMAN_QE)
    L = [f"# Arm {r['arm']}: uitslag volgens amendement G4-6c (protocol-g4.md §13)", "",
         "Gemaakt door `g4arm.py report6c`. Mildere DPO vanaf G4-4 (referentie G4-4) op de paren van G4-6b; drie "
         "checkpoints, de keuze op het zoekdeel van G4-5.4 (regel van G4-5.6), het besluit met (a1) en (b) tegen G4-0 "
         "op het beslisdeel.", "",
         f"**Uitslag: {r['verdict']}.** Gekozen checkpoint: **{r['selected']}** ({r['selection']}).", "",
         f"Verlies `{json.dumps(r['spec'])}`.", "",
         "## Besluit op het beslisdeel", "",
         "| set | n | " + r["selected"] + " | G4-0 | mens (zelfde segmenten) | (a1) |", "|---|---:|---:|---:|---:|---|"]
    dec = r["parts"]["decision"]
    m, base = dec["models"][r["selected"]], dec["models"][r["base"]]
    for s in sets:
        L.append(f"| {s} | {dec['n'][s]} | {f3(m['qe'][s])} | {f3(base['qe'][s])} | {f3(dec['human'][s])} | "
                 f"{'ja' if m['a1_sets'][s] else 'nee'} |")
    L += ["", f"(a1) **{'gehaald' if r['a1']['holds'] else 'niet gehaald'}**. (b): aandeel \"origineel\" "
          f"{pct(m['share'])} tegen G4-0 {pct(base['share'])} (WMT24++ {dec['n']['wmt24pp']} + FLAN {dec['n']['flan']}), "
          f"verschil {f3(m['b_delta'], True)} (grens +0,10) → **{'gehaald' if r['b']['holds'] else 'niet gehaald'}**.", ""]
    if r.get("blind"):
        b = r["blind"]
        L += [f"Blind tegen G4-0: {b['arm_better']} beter, {b['base_better']} slechter, {b['equal']} gelijk; fouten "
              f"{b['errors_arm']} tegen {b['errors_base']} → (a2) {'ja' if r['a2'] else 'nee'}, (c) "
              f"{'ja' if r['c'] else 'nee'}.", ""]
    else:
        L += ["(a2) en (c): nog niet beoordeeld (alleen als (a1) en (b) gelden).", ""]
    for part, title in (("search", "Zoekdeel (de keuze)"),
                        ("decision", "Beslisdeel, alle modellen (na de keuze, rapporteren)")):
        p = r["parts"][part]
        L += [f"## {title}", "", "| model | " + " | ".join(f"QE {s}" for s in sets) + " | (a1) | max. QE-overschrijding | "
              "origineel | verschil met G4-0 |", "|---|" + "---:|" * len(sets) + "---|---:|---:|---:|"]
        L.append("| mens | " + " | ".join(f3(p["human"][s]) for s in sets) + " | | | | |")
        for name, x in p["models"].items():
            L.append(f"| {name} | " + " | ".join(f3(x["qe"][s]) for s in sets) +
                     f" | {'ja' if x['a1'] else 'nee'} | {f3(x['excess'], True)} | {pct(x['share'])} | "
                     f"{f3(x['b_delta'], True)} |")
        L.append("")
    L += ["## Per model: niet af, lussen, lengte (rapporteren, beslist niet)", "",
          "Uitvoer / referentie: mediaan van de woorden van de uitvoer per woord van de referentie; > 1,5 × ref: aandeel "
          "uitvoer met meer dan 1,5 keer de woorden van de referentie. FLAN heeft geen referentie.", "",
          "| deel | model | set | n | QE | niet af | lus | Engels over | leeg | uitvoer / ref | > 1,5 × ref |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for part, label in (("search", "zoek"), ("decision", "beslis")):
        for name, per in r["counts"][part].items():
            for s, c in per.items():
                L.append(f"| {label} | {name} | {s} | {c['n']} | {f3(c['metricx_qe'])} | {c['not_ended']} | "
                         f"{c['decode_loop']} | {c['english_left']} | {c['empty']} | {f3(c['out_per_ref'])} | "
                         f"{pct(c['over_1_5_ref'])} |")
    L += ["", f"## Per checkpoint: de training (referentie {r['reference']}; rapporteren, beslist niet)", "",
          "Verlies per token (het Nederlands en `<turn|>\\n`) op de SFT-evalsets, op stap 0 (G4-4 ingebakken) en op de "
          "checkpoints.", "", "| stap | " + " | ".join(SFT_SETS6C) + " |", "|---:|" + "---:|" * len(SFT_SETS6C)]
    for step, t in r["training"].items():
        L.append(f"| {step} | " + " | ".join(f3(t["sft"][n]) for n in SFT_SETS6C) + " |")
    for name, b in (r.get("beside") or {}).items():
        L.append(f"| {name} ({b['step']}) | " + " | ".join(f3(b["sft"].get(n)) for n in SFT_SETS6C) + " |")
    L += ["", "Voorkeurssets: NLL per token van chosen en rejected over de tokens van de log-ratio (zonder `<turn|>\\n`) "
          "en over alle doeltokens (de maat van G4-6b); ref = de referentie G4-4 over de log-ratio-tokens.", "",
          "| set | stap | paren | verlies | nauwkeurigheid | marge | NLL chosen | NLL chosen (alle) | NLL rejected | "
          "NLL rejected (alle) | ref chosen | ref rejected |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in PREF_SETS6:
        for step, t in r["training"].items():
            v = t["pref"][n]
            L.append(f"| {n} | {step} | {v['pairs']} | {f3(v['loss'])} | {f3(v['accuracy'])} | {f3(v['margin'])} | "
                     f"{f3(v['chosen_nll'])} | {f3(v['chosen_nll_all'])} | {f3(v['rejected_nll'])} | "
                     f"{f3(v['rejected_nll_all'])} | {f3(v['ref_chosen_nll'])} | {f3(v['ref_rejected_nll'])} |")
        for name, b in (r.get("beside") or {}).items():
            v = b["pref"].get(n) or {}
            L.append(f"| {n} | {name} ({b['step']}) | | {f3(v.get('loss'))} | {f3(v.get('accuracy'))} | "
                     f"{f3(v.get('margin'))} | – | {f3(v.get('chosen_nll'))} | – | {f3(v.get('rejected_nll'))} | – | – |")
    L += ["", f"Bewaarde adapters: {json.dumps(r.get('saved_adapters'))}.", "",
          f"Controle menselijke referenties (QE opnieuw, hele set): {json.dumps(r['refqe_check'])}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- G4-7

ARM7 = "G4-7"
CAND7 = re.compile(r"^G4-7-i([1-9][0-9]*)$")
EVERY7 = 5                # amendment G4-7.7: the adapters after iteration 5, 10, ..., 30 and the last before a stop
RULE_A3 = 0.005           # amendment G4-7.8 (a3): per set COMET-22 >= G4-0's - 0.005 on the same segments
A3_SETS = ("wmt24pp", "flores", "ntrex")
B_SETS7 = ("wmt24pp", "flan")             # (b)'s texts: WMT24++ (good) + FLAN
SFT_SETS7 = ("books", "btval", "own")     # spec §6.3: the SFT evals after every 5th iteration (reported)
SPEARMAN_HIGH7 = 0.5      # amendment G4-7.10 (spec §5.3.5): C2 ~ C1 per segment at or above this, the sentence below
RULES7 = ("S1", "S2", "S3", "S4", "S6")   # amendment G4-7.5 (no S5)
HIGH_SPEARMAN_TEXT7 = ("Bij deze hoge correlatie (Spearman per segment ≥ 0,5) duwt wat C2 omhoog duwt waarschijnlijk "
                       "ook C1 omhoog. Een hack van C2 kan dan ook C1 misleiden; (b) zegt dan minder, en de blinde "
                       "beoordeling ((a2), (c)) weegt zwaarder bij het lezen van de uitslag (de lezing van spec "
                       "§5.3.5 en G4-7.10; beslist niets).")


def iterations_max7():
    import g4rl

    return g4rl.ITERATIONS7


def cand_n7(name):
    """n of a candidate G4-7-i<n>: the adapter after n iterations (1..30)"""
    m = CAND7.match(name)
    if not m or not 1 <= int(m.group(1)) <= iterations_max7():
        raise SystemExit(f"{name} is no candidate of G4-7 (G4-7-i<n>, n = 1..{iterations_max7()})")
    return int(m.group(1))


def check_candidates7(names, last):
    """amendment G4-7.7: the candidates in the order of n, at most the iterations done (last + 1); every adapter after
    a multiple of 5 iterations among them; one more only as the largest (the last before a stop)"""
    ns_ = [cand_n7(n) for n in names]
    if ns_ != sorted(set(ns_)):
        raise SystemExit(f"the candidates {names}: not in the order of their iterations")
    done = last + 1
    want = list(range(EVERY7, done + 1, EVERY7))
    missing = [k for k in want if k not in ns_]
    extra = [k for k in ns_ if k not in want]
    if missing or ns_[-1] > done or len(extra) > 1 or (extra and extra[0] != ns_[-1]):
        raise SystemExit(f"the candidates {names} after {done} iterations: G4-7.7 takes the adapters after "
                         f"{', '.join(map(str, want)) or 'no multiple of 5'} iterations and the last before a stop")
    return ns_


def records7_list(listing, out_dir):
    """records.json of the chain -> [(gs:// source, local destination)] in report7's --records layout
    (DIR/iter-<i>/report.json, stop.json, summary.json; DIR/cal7.json; DIR/smoke7.json) and DIR/stopped.json; the
    iterations 0, 1, ... in order, the stop's last iteration the last of them"""
    its = listing.get("iterations")
    if not isinstance(its, list) or not its:
        raise SystemExit("records: no iterations")
    got = [x.get("i") if isinstance(x, dict) else None for x in its]
    if got != list(range(len(its))) or len(its) > iterations_max7():
        raise SystemExit(f"records: the iterations {got}, not 0, 1, ... (at most {iterations_max7()})")
    stopped = listing.get("stopped")
    if (not isinstance(stopped, dict) or stopped.get("last_iteration") != len(its) - 1
            or not isinstance(stopped.get("stop"), list) or not set(stopped["stop"]) <= set(RULES7)
            or "reason" not in stopped):
        raise SystemExit(f"records: stopped {stopped!r}: not {{'last_iteration': {len(its) - 1}, 'stop': [rules of "
                         f"{', '.join(RULES7)}], 'reason': ...}}")

    def uri(value, what):
        if not isinstance(value, str) or not value.startswith("gs://") or any(c.isspace() for c in value):
            raise SystemExit(f"records: {what} is {value!r}, not a gs:// path")
        return value

    out = Path(out_dir)
    pairs = []
    for x in its:
        d = out / f"iter-{x['i']}"
        d.mkdir(parents=True, exist_ok=True)
        for k in ("report", "stop", "summary"):
            pairs.append((uri(x.get(k), f"iteration {x['i']}'s {k}"), d / f"{k}.json"))
    pairs.append((uri(listing.get("cal7"), "cal7"), out / "cal7.json"))
    if listing.get("smoke") is not None:
        pairs.append((uri(listing["smoke"], "smoke"), out / "smoke7.json"))
    (out / "stopped.json").write_text(json.dumps(stopped, indent=1) + "\n")
    return pairs


def records7(args):
    for src, dst in records7_list(json.loads(Path(args.list).read_text()), args.out):
        print(f"{src} {dst}")


def iterations7(records, stopped):
    """amendment G4-7.10, per iteration (reported, decides nothing): from g4rl.py iter7's report the reward and its
    terms, the failure shares per kind (c2_overflow too), QE, C2, l / l0, the groups without signal; from the run's
    summary (sft_lora.py --grpo) the k1 KL old - ref, the sampler/learner gap, TIS, the clip fraction and the KL term
    (token-weighted over the steps [choice of the subagent]), the SFT evals (at the last eval step of the run); from
    stop7 RL-val and the rules with their values. Every iteration before the last stopped nothing; the last one's
    rules are the chain's stop."""
    last = stopped["last_iteration"]
    out = []
    for i in range(last + 1):
        d = Path(records) / f"iter-{i}"
        rep, stop, summary = (json.loads((d / f"{k}.json").read_text()) for k in ("report", "stop", "summary"))
        if rep.get("iteration") != i or stop.get("iteration") != i:
            raise SystemExit(f"{d}: the report of iteration {rep.get('iteration')} and the stop of "
                             f"{stop.get('iteration')}, not {i}")
        grpo = summary.get("grpo") or {}
        if not grpo.get("passes") or not grpo.get("steps") or not grpo.get("spec"):
            raise SystemExit(f"{d}/summary.json: no GRPO spec, passes or steps (sft_lora.py --grpo)")
        if set(stop.get("rules") or {}) != set(RULES7):
            raise SystemExit(f"{d}/stop.json: the rules {sorted(stop.get('rules') or {})}, not {list(RULES7)}")
        hits = [k for k in RULES7 if stop["rules"][k]["hit"]]
        if hits != stop["stop"]:
            raise SystemExit(f"{d}/stop.json: stop {stop['stop']}, its rules hit {hits}")
        if i < last and hits:
            raise SystemExit(f"{d}/stop.json: iteration {i} stopped ({hits}), the chain went on to iteration {last}")
        if i == last and hits != sorted(stopped["stop"], key=RULES7.index):
            raise SystemExit(f"{d}/stop.json: the last iteration stopped on {hits}, the chain says {stopped['stop']}")
        steps = grpo["steps"]
        tokens = sum(s["tokens"] for s in steps)

        def weighted(key, steps=steps, tokens=tokens):
            return sum(s[key] * s["tokens"] for s in steps) / tokens if tokens else None

        evals = last_evals(summary)
        n = rep["samples"]
        out.append({
            "i": i, "n": i + 1, "samples": n, "sources": rep["sources"],
            "groups_without_signal": rep["groups_without_signal"],
            **{k: rep[k] for k in ("reward_mean", "reward_mean_ok", "logit_mean_ok", "qe_pen_mean_ok", "len_pen_mean_ok",
                                   "qe_mean", "p_c2_mean", "origineel_share", "ratio_median", "fail_share")},
            "fails": {k: v / n for k, v in rep["fails"].items()},
            "train_samples": (rep.get("train") or {}).get("samples"),
            "k1_old_ref": grpo["passes"]["k1_old_ref"], "sampler_gap": grpo["passes"]["sampler_gap"],
            "tis": bool(grpo["spec"]["tis"]), "steps": len(steps), "step_first": steps[0]["step"],
            "step_last": steps[-1]["step"], "clip_frac": weighted("clip_frac"), "kl": weighted("kl"),
            "loss": weighted("loss"),
            "val": {k: stop["validation"].get(k) for k in ("sources", "fail_share", "qe_delta", "ratio_median",
                                                           "c2_overflow", "p_c2_mean", "c2_origineel_share")},
            "rules": {k: {"value": stop["rules"][k]["value"], "hit": stop["rules"][k]["hit"]} for k in RULES7},
            "stop": stop["stop"], "tis_next": stop.get("tis_next"),
            "sft": {k: evals[k]["loss"] for k in SFT_SETS7 if k in evals},
            "sft_step": max((evals[k]["step"] for k in SFT_SETS7 if k in evals), default=None)})
    return out


def rule_a3(scored, ids, name, base):
    """amendment G4-7.8 (a3): per set (WMT24++ without is_bad_source as (a1), FLORES, NTREX) the mean COMET-22 with
    reference of the model at least the base's on the same segments - RULE_A3"""
    sets = {}
    for s in A3_SETS:
        m = [scored[name][i]["comet22"] for i in ids[s]]
        b = [scored[base][i]["comet22"] for i in ids[s]]
        if not ids[s] or any(v is None for v in m + b):
            raise SystemExit(f"(a3) {s}: {name} or {base} has no COMET-22 score of every one of {len(ids[s])} segments")
        mm, bm = statistics.fmean(m), statistics.fmean(b)
        sets[s] = {"n": len(ids[s]), "model": mm, "base": bm, "delta": mm - bm,
                   "holds": mm - bm >= -RULE_A3 - TOLERANCE}
    return {"sets": sets, "holds": all(v["holds"] for v in sets.values())}


def c2_part7(c1, c2, ids, models, base):
    """C1 and C2 side by side on one part (amendment G4-7.8, G4-7.10; C2 decides nothing): per model the share
    "original" (p >= 0.5) of each on WMT24++ (good), FLAN and both pooled, the differences to the base, and the flag
    "C2 rises, C1 not" [choice of the subagent: C2's pooled difference reaches (b)'s bound of 0.10, C1's does not]"""
    out = {}
    for name in models:
        x = {}
        for label, preds in (("c1", c1), ("c2", c2)):
            per = {}
            for s in B_SETS7:
                missing = [i for i in ids[s] if (name, s, i) not in preds]
                if missing:
                    raise SystemExit(f"{label.upper()}: no prediction of {name} on {len(missing)} {s} texts of the part, "
                                     f"e.g. {missing[:3]}")
                per[s] = sum(preds[(name, s, i)] >= 0.5 for i in ids[s]) / len(ids[s])
            per["pooled"] = sum(per[s] * len(ids[s]) for s in B_SETS7) / sum(len(ids[s]) for s in B_SETS7)
            x[label] = per
        out[name] = x
    for x in out.values():
        for label in ("c1", "c2"):
            x[f"{label}_delta"] = {k: x[label][k] - out[base][label][k] for k in (*B_SETS7, "pooled")}
        x["c2_up_c1_not"] = (x["c2_delta"]["pooled"] >= RULE_B - TOLERANCE
                             and x["c1_delta"]["pooled"] < RULE_B - TOLERANCE)
    return out


def part_counts7(scored, ids):
    """part_counts6c with the mean MetricX-24 with reference, COMET-22 and MTLD (armeval.mtld per output) per model and
    set of a part"""
    import armeval

    out = part_counts6c(scored, ids)
    for name, per in out.items():
        for s, c in per.items():
            rows = [scored[name][i] for i in ids[s]]
            c["metricx_ref"] = mean(r["metricx_ref"] for r in rows)
            c["comet22"] = mean(r["comet22"] for r in rows)
            c["mtld"] = mean(armeval.mtld(r["hypothesis"] or "") for r in rows)
    return out


def pll_part7(pll, ids, models):
    """naturalness.py pll (S @ 653k, PLL-word-l2r) on the classify run's texts, per model and set of a part: the texts
    scored and left out (longer than its row), the mean PLL per text and per token, as armeval.py report reads it"""
    out = {}
    for name in models:
        out[name] = {}
        for s in B_SETS7:
            missing = [i for i in ids[s] if (name, s, i) not in pll]
            if missing:
                raise SystemExit(f"--pll: no row of {name} on {len(missing)} {s} texts of the part, e.g. {missing[:3]}")
            rows = [pll[(name, s, i)] for i in ids[s]]
            kept = [r for r in rows if r["pll"] is not None]
            out[name][s] = {"n": len(kept), "skipped": len(rows) - len(kept), "pll": mean(r["pll"] for r in kept),
                            "pll_per_token": mean(r["pll"] / r["tokens"] for r in kept)}
    return out


def cal7_short(c):
    """c2cal7's report (g4c2.py cal7): passed (else no RL should have run, G4-7.6), its four conditions, and per
    calibration run the Spearman per segment of C2 with C1"""
    if not c.get("ok"):
        failed = [n for n, v in (c.get("conditions") or {}).items() if not v.get("ok")]
        raise SystemExit(f"c2cal7 did not pass ({failed}): amendment G4-7.6 runs no RL then")
    calib = {n: {"segments": v.get("segments"), "spearman_c2_c1": v.get("spearman_c2_c1")}
             for n, v in (c.get("calibration") or {}).items()}
    high = [n for n, v in calib.items() if v["spearman_c2_c1"] is not None and v["spearman_c2_c1"] >= SPEARMAN_HIGH7]
    return {"ok": True, "conditions": c["conditions"], "calibration": calib, "high": high,
            "over_1022_tokens": c.get("over_1022_tokens")}


def verdict7(r):
    """amendment G4-7.8: (a1), (a3) and (b) first; (a2) and (c) only after them; helps if all five hold"""
    if not (r["a1"]["holds"] and r["a3"]["holds"] and r["b"]["holds"]):
        return "helpt niet"
    if r.get("blind") is None:
        return "wacht op de blinde beoordeling ((a2), (c))"
    return "helpt" if r["a2"] and r["c"] else "helpt niet"


def report7(args):
    """amendment G4-7.7-G4-7.10: the candidate chosen on the search part (G4-5.6's rule, select5, unchanged), the rule
    (a1), (a3), (b) against G4-0 on its decision part, (a2) and (c) from the blind tally only after those three; per
    candidate on both parts C2 beside C1 and the other measures; every iteration of the training, the stop, c2cal7,
    smoke7 (reported, deciding nothing)"""
    import g4mix

    evalsets = {r["id"]: r for r in read_rows(args.evalsets)}
    split = json.loads(Path(args.split).read_text())
    assign = split["assign"]
    if g4mix.digest(assign) != split["digest"] or split["digest"] != SPLIT_DIGEST6C:
        raise SystemExit(f"{args.split}: not the split of G4-5.4 (digest {split['digest']}, want {SPLIT_DIGEST6C})")
    stopped = json.loads(Path(args.stopped).read_text())
    last = stopped.get("last_iteration")
    if (not isinstance(last, int) or not 0 <= last < iterations_max7() or not isinstance(stopped.get("stop"), list)
            or not set(stopped["stop"]) <= set(RULES7)):
        raise SystemExit(f"{args.stopped}: last_iteration {last!r}, stop {stopped.get('stop')!r} (rules of "
                         f"{', '.join(RULES7)})")
    cands = [c.split("=", 1) for c in args.candidate]
    names = [n for n, _ in cands]
    ns_ = check_candidates7(names, last)
    refs = [c.split("=", 1) for c in args.reference or []]
    scored = {name: scored_dir(d) for name, d in cands + [(BASE, args.base)] + refs}
    for name, rows in scored.items():
        if set(rows) != set(evalsets):
            raise SystemExit(f"{name}: scores for {len(rows)} of {len(evalsets)} segments")
    refqe = {r["id"]: r["qe_source_reference"] for r in read_rows(args.refqe)}
    missing = [i for i, r in evalsets.items() if r["set"] in HUMAN_QE and refqe.get(i) is None]
    if missing:
        raise SystemExit(f"{args.refqe}: no QE of the human reference of {len(missing)} segments, e.g. {missing[:3]}")
    check = {}
    for s in HUMAN_QE:
        ids = [i for i, r in evalsets.items() if r["set"] == s and not (s == "wmt24pp" and bad_source(r["meta"]))]
        got = mean(refqe[i] for i in ids)
        check[s] = {"again": got, "protocol": HUMAN_QE[s], "diff": got - HUMAN_QE[s]}
        if abs(got - HUMAN_QE[s]) > REFQE_MAX_DIFF5:
            raise SystemExit(f"the human references' QE of {s} again: {got:.4f}, the protocol's {HUMAN_QE[s]:.4f}")
    predictions = read_rows(args.predictions)
    predictions_c2 = read_rows(args.predictions_c2)
    keys = [(r["model"], r["set"], r["id"]) for r in predictions]
    keys_c2 = [(r["model"], r["set"], r["id"]) for r in predictions_c2]
    if len(set(keys_c2)) != len(keys_c2) or set(keys_c2) != set(keys):
        raise SystemExit(f"{args.predictions_c2}: not C2 on the texts of {args.predictions} (each once)")
    part_ids = {p: part_ids5(evalsets, assign, p) for p in ("search", "decision")}
    parts = {p: part_measures5(scored, predictions, refqe, part_ids[p], BASE) for p in ("search", "decision")}
    # the choice reads the search part alone (amendment G4-7.7)
    chosen, why = select5(names, parts["search"])
    d = parts["decision"]["models"][chosen]
    a3 = {p: {name: rule_a3(scored, part_ids[p], name, BASE) for name in scored} for p in parts}
    c1 = {k: r["p_original"] for k, r in zip(keys, predictions)}
    c2 = {k: r["p_original"] for k, r in zip(keys_c2, predictions_c2)}
    truncated = {label: {name: sum(bool(r.get("truncated")) for r in rows if r["model"] == name) for name in scored}
                 for label, rows in (("c1", predictions), ("c2", predictions_c2))}
    iterations = iterations7(args.records, stopped)
    result = {"arm": ARM7, "base": BASE, "candidates": names, "iterations_of": dict(zip(names, ns_)),
              "selected": chosen, "selection": why, "refqe_check": check,
              "split": {k: split[k] for k in ("seed", "share", "sets", "digest")}, "parts": parts,
              "a1": {"holds": d["a1"], "sets": d["a1_sets"]},
              "a3": a3["decision"][chosen], "b": {"holds": d["b"], "delta": d["b_delta"]},
              "a3_all": a3, "c2": {p: c2_part7(c1, c2, part_ids[p], list(scored), BASE) for p in parts},
              "truncated": truncated,
              "counts": {p: part_counts7(scored, part_ids[p]) for p in parts},
              "pll": ({p: pll_part7({(r["model"], r["set"], r["id"]): r for r in read_rows(args.pll)}, part_ids[p],
                                    list(scored)) for p in parts} if args.pll else None),
              "references": [n for n, _ in refs], "iterations": iterations, "stopped": stopped,
              "cal7": cal7_short(json.loads(Path(args.cal7).read_text())),
              "smoke7": json.loads(Path(args.smoke).read_text()) if args.smoke else None}
    result["blind_due"] = result["a1"]["holds"] and result["a3"]["holds"] and result["b"]["holds"]
    if args.tally:
        t = json.loads(Path(args.tally).read_text())
        if not result["blind_due"]:
            raise SystemExit(f"{args.tally}: a blind tally, but (a1), (a3) and (b) do not all hold for {chosen} "
                             "(amendment G4-7.8: the blind review only after them)")
        if t.get("arm") != chosen:
            raise SystemExit(f"{args.tally}: a tally of {t.get('arm')}, the selected candidate is {chosen}")
        result["blind"], result["a2"], result["c"] = t, t["a2"], t["c"]
    else:
        result["blind"] = None
    result["verdict"] = verdict7(result)
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")
    Path(args.out).write_text(report7_text(result), encoding="utf-8")
    print(f"REPORT7 {json.dumps({'selected': chosen, 'a1': d['a1'], 'a3': result['a3']['holds'], 'b': d['b_delta'], 'b_holds': d['b'], 'blind_due': result['blind_due'], 'verdict': result['verdict']})}",
          flush=True)


def f4(x, signed=False):
    if x is None:
        return "–"
    return (f"{x:+.4f}" if signed else f"{x:.4f}").replace(".", ",")


def report7_text(r):
    sets = list(HUMAN_QE)
    dec = r["parts"]["decision"]
    sel, base = r["selected"], r["base"]
    m, b0 = dec["models"][sel], dec["models"][base]
    L = [f"# Arm {r['arm']}: uitslag volgens amendement G4-7 (protocol-g4.md §14)", "",
         "Gemaakt door `g4arm.py report7`. Online RL (GRPO) met LoRA vanaf G4-0, C2 als reward-classifier; de "
         "kandidaten (adapters na iteratie 5, 10, … en de laatste vóór een stop) vertaald en gescoord zoals G4-5.5; de "
         "keuze op het zoekdeel van G4-5.4 (regel van G4-5.6, `select5`, G4-7.7), het besluit met (a1), (a3) en (b) tegen "
         "G4-0 op het beslisdeel (G4-7.8); (a2) en (c) alleen als die drie gelden. Alle getallen hieronder zijn gemeten "
         "op de genoemde bestanden; C2 en alles na \"rapporteren\" beslist niets.", "",
         f"**Uitslag: {r['verdict']}.** Gekozen kandidaat: **{sel}** ({r['selection']}); kandidaten "
         f"{', '.join(r['candidates'])}.", "",
         "## Besluit op het beslisdeel (G4-7.8)", "",
         "(a1) MetricX-24-QE niet hoger dan de menselijke referentie op dezelfde segmenten.", "",
         f"| set | n | {sel} | G4-0 | mens (zelfde segmenten) | (a1) |", "|---|---:|---:|---:|---:|---|"]
    for s in sets:
        L.append(f"| {s} | {dec['n'][s]} | {f3(m['qe'][s])} | {f3(b0['qe'][s])} | {f3(dec['human'][s])} | "
                 f"{'ja' if m['a1_sets'][s] else 'nee'} |")
    L += ["", f"(a1) **{'gehaald' if r['a1']['holds'] else 'niet gehaald'}**.", "",
          "(a3) COMET-22 met referentie niet lager dan G4-0 − 0,005 op dezelfde segmenten (WMT24++ zonder "
          "`is_bad_source`, als (a1) [keuze subagent]).", "",
          f"| set | n | {sel} | G4-0 | verschil | (a3) |", "|---|---:|---:|---:|---:|---|"]
    for s, v in r["a3"]["sets"].items():
        L.append(f"| {s} | {v['n']} | {f4(v['model'])} | {f4(v['base'])} | {f4(v['delta'], True)} | "
                 f"{'ja' if v['holds'] else 'nee'} |")
    L += ["", f"(a3) **{'gehaald' if r['a3']['holds'] else 'niet gehaald'}**. (b) C1: aandeel \"origineel\" "
          f"{pct(m['share'])} tegen G4-0 {pct(b0['share'])} (WMT24++ {dec['n']['wmt24pp']} + FLAN {dec['n']['flan']}), "
          f"verschil {f3(m['b_delta'], True)} (grens +0,10) → **{'gehaald' if r['b']['holds'] else 'niet gehaald'}**.", ""]
    if r.get("blind"):
        b = r["blind"]
        L += [f"Blind tegen G4-0: {b['arm_better']} beter, {b['base_better']} slechter, {b['equal']} gelijk; fouten "
              f"{b['errors_arm']} tegen {b['errors_base']} → (a2) {'ja' if r['a2'] else 'nee'}, (c) "
              f"{'ja' if r['c'] else 'nee'}.", ""]
    elif r["blind_due"]:
        L += ["(a2) en (c): (a1), (a3) en (b) gelden; de blinde beoordeling (pakket `$S/blind/G4-7/`) is nog niet "
              "geteld.", ""]
    else:
        L += ["(a2) en (c): niet beoordeeld; de blinde beoordeling gebeurt alleen als (a1), (a3) en (b) gelden.", ""]
    for part, title in (("search", "Zoekdeel (de keuze, G4-7.7)"),
                        ("decision", "Beslisdeel, alle modellen (na de keuze, rapporteren)")):
        p = r["parts"][part]
        L += [f"## {title}", "", "(a3) staat erbij om te rapporteren; de keuze gebruikt alleen (a1) en (b).", "",
              "| model | " + " | ".join(f"QE {s}" for s in sets) + " | (a1) | max. QE-overschrijding | (a3) | "
              "C1 origineel | verschil met G4-0 |", "|---|" + "---:|" * len(sets) + "---|---:|---|---:|---:|"]
        L.append("| mens | " + " | ".join(f3(p["human"][s]) for s in sets) + " | | | | | |")
        for name, x in p["models"].items():
            L.append(f"| {name} | " + " | ".join(f3(x["qe"][s]) for s in sets) +
                     f" | {'ja' if x['a1'] else 'nee'} | {f3(x['excess'], True)} | "
                     f"{'ja' if r['a3_all'][part][name]['holds'] else 'nee'} | {pct(x['share'])} | "
                     f"{f3(x['b_delta'], True)} |")
        L.append("")
    L += ["## C1 en C2 per set (rapporteren, beslist niets)", "",
          "Aandeel \"origineel\" (p ≥ 0,5) van C1 (de classifier van regel (b), S @ 653k) en C2 (de reward-classifier, "
          "dmbert v2) op dezelfde teksten. Signaal van reward hacking: \"C2 stijgt, C1 niet\" = C2 − G4-0 ≥ +0,10 "
          "(de grens van (b)) en C1 − G4-0 < +0,10, op WMT24++ (goed) + FLAN samen [keuze subagent].", "",
          "| deel | model | C1 WMT24++ | C1 FLAN | C1 samen | C1 − G4-0 | C2 WMT24++ | C2 FLAN | C2 samen | C2 − G4-0 | "
          "C2 stijgt, C1 niet |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for part, label in (("search", "zoek"), ("decision", "beslis")):
        for name, x in r["c2"][part].items():
            L.append(f"| {label} | {name} | {pct(x['c1']['wmt24pp'])} | {pct(x['c1']['flan'])} | {pct(x['c1']['pooled'])} | "
                     f"{f3(x['c1_delta']['pooled'], True)} | {pct(x['c2']['wmt24pp'])} | {pct(x['c2']['flan'])} | "
                     f"{pct(x['c2']['pooled'])} | {f3(x['c2_delta']['pooled'], True)} | "
                     f"{'**ja**' if x['c2_up_c1_not'] else 'nee'} |")
    L += ["", "Teksten afgekapt op 1.024 tokens (alle teksten van het model in de classify-run): " + "; ".join(
        f"{name} C1 {r['truncated']['c1'][name]}, C2 {r['truncated']['c2'][name]}" for name in r["truncated"]["c2"]) + ".",
        "", "## Per model: MetricX, COMET, niet af, lussen, lengte, MTLD (rapporteren, beslist niet)", "",
        "Uitvoer / ref: mediaan van de woorden van de uitvoer per woord van de referentie; > 1,5 × ref: aandeel uitvoer "
        "met meer dan 1,5 keer de woorden van de referentie; MTLD: gemiddelde per uitvoer (`armeval.mtld`). FLAN heeft "
        "geen referentie.", "",
        "| deel | model | set | n | QE | MetricX-ref | COMET-22 | niet af | lus | Engels over | leeg | uitvoer / ref | "
        "> 1,5 × ref | MTLD |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for part, label in (("search", "zoek"), ("decision", "beslis")):
        for name, per in r["counts"][part].items():
            for s, c in per.items():
                L.append(f"| {label} | {name} | {s} | {c['n']} | {f3(c['metricx_qe'])} | {f3(c['metricx_ref'])} | "
                         f"{f4(c['comet22'])} | {c['not_ended']} | {c['decode_loop']} | {c['english_left']} | "
                         f"{c['empty']} | {f3(c['out_per_ref'])} | {pct(c['over_1_5_ref'])} | {f3(c['mtld'])} |")
    L += ["", "## PLL (rapporteren, beslist niet)", ""]
    if r["pll"] is None:
        L += ["PLL: **niet gemeten** (geen `naturalness.py pll`-uitvoer bij dit verslag).", ""]
    else:
        L += ["PLL-word-l2r onder S @ 653k (`naturalness.py pll`) op de teksten van de classify-run; hoger (minder "
              "negatief) = waarschijnlijker onder S.", "",
              "| deel | model | set | n | weggelaten | PLL per vertaling | PLL per token |", "|---|---|---|---:|---:|---:|---:|"]
        for part, label in (("search", "zoek"), ("decision", "beslis")):
            for name, per in r["pll"][part].items():
                for s, v in per.items():
                    L.append(f"| {label} | {name} | {s} | {v['n']} | {v['skipped']} | {f3(v['pll'])} | "
                             f"{f3(v['pll_per_token'])} |")
        L.append("")
    its = r["iterations"]
    kinds = list(its[0]["fails"]) if its else []
    L += ["## De training per iteratie (rapporteren, beslist niets)", "",
          "Iteratie i (0, 1, …) traint de adapter G4-7-i<i+1>. Reward en termen: gemiddelde over alle samples en over "
          "de samples zonder fail; logit = clip(logit C2, −6, 6), QE-straf 2·max(0, QE − QE₀), lengtestraf "
          "2·max(0, |ln ℓ/ℓ₀| − ln 1,15). Fail-aandelen per soort over alle samples (een sample kan meer soorten "
          "hebben).", "",
          "| i | samples | reward | reward (geen fail) | logit | QE-straf | lengtestraf | fail | " +
          " | ".join(kinds) + " | QE | C2 p | C2 origineel | ℓ/ℓ₀ mediaan |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(kinds) + "---:|---:|---:|---:|"]
    for x in its:
        L.append(f"| {x['i']} | {x['samples']} | {f3(x['reward_mean'])} | {f3(x['reward_mean_ok'])} | "
                 f"{f3(x['logit_mean_ok'])} | {f3(x['qe_pen_mean_ok'])} | {f3(x['len_pen_mean_ok'])} | "
                 f"{pct(x['fail_share'])} | " + " | ".join(pct(x["fails"][k]) for k in kinds) +
                 f" | {f3(x['qe_mean'])} | {f3(x['p_c2_mean'])} | {pct(x['origineel_share'])} | {f3(x['ratio_median'])} |")
    L += ["", "Leerder: k1-KL old − ref per token (S3), verschil sampler/leerder (gemiddelde |log π_old − log π_sampler| "
          "per token), TIS in deze iteratie en voor de volgende, clip-fractie en KL-term van het verlies "
          "(token-gewogen over de stappen [keuze subagent]), groepen zonder signaal (max − min < 0,05).", "",
          "| i | stappen | k1 old − ref | sampler/leerder | TIS | TIS volgende | clip-fractie | KL-term | verlies | "
          "groepen zonder signaal | samples getraind |", "|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for x in its:
        L.append(f"| {x['i']} | {x['step_first']}–{x['step_last']} ({x['steps']}) | {f4(x['k1_old_ref'])} | "
                 f"{f4(x['sampler_gap'])} | {'aan' if x['tis'] else 'uit'} | {'aan' if x['tis_next'] else 'uit'} | "
                 f"{f4(x['clip_frac'])} | {f4(x['kl'])} | {f4(x['loss'])} | {x['groups_without_signal']} van "
                 f"{x['sources']} | {x['train_samples']} |")
    L += ["", "RL-val (greedy, de nieuwe adapter) en de stopregels (G4-7.5) met hun waarden; * = geraakt.", "",
          "| i | RL-val fail | QE − anker | ℓ/ℓ₀ mediaan | C2 p | C2 origineel | c2_overflow | S1 | S2 | S3 | S4 | S6 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in its:
        v = x["val"]
        L.append(f"| {x['i']} | {pct(v['fail_share'])} | {f3(v['qe_delta'], True)} | {f3(v['ratio_median'])} | "
                 f"{f3(v['p_c2_mean'])} | {pct(v['c2_origineel_share'])} | {v['c2_overflow']} | " +
                 " | ".join(f"{f3(x['rules'][k]['value'])}{'*' if x['rules'][k]['hit'] else ''}" for k in RULES7) + " |")
    with_sft = [x for x in its if x["sft"]]
    L += ["", "SFT-evaluaties (verlies per token, op de laatste eval-stap van de run; na elke 5e iteratie).", ""]
    if with_sft:
        L += ["| i | stap | " + " | ".join(SFT_SETS7) + " |", "|---:|---:|" + "---:|" * len(SFT_SETS7)]
        for x in with_sft:
            L.append(f"| {x['i']} | {x['sft_step']} | " + " | ".join(f3(x["sft"].get(k)) for k in SFT_SETS7) + " |")
    else:
        L.append("Geen SFT-evaluatie in de samenvattingen van de runs.")
    st = r["stopped"]
    L += ["", "## De stop", "",
          (f"Gestopt na iteratie {st['last_iteration']} (adapter G4-7-i{st['last_iteration'] + 1}) op "
           f"{', '.join(st['stop'])}: " if st["stop"] else
           f"Geen stopregel geraakt; de training eindigde na iteratie {st['last_iteration']} "
           f"(adapter G4-7-i{st['last_iteration'] + 1}): ") + f"{st['reason']}.", ""]
    c = r["cal7"]
    L += ["## C2 bruikbaar als reward (`c2cal7`, vóór de RL)", "",
          "| voorwaarde | waarde | grens | gehaald |", "|---|---:|---|---|"]
    for n, v in c["conditions"].items():
        extra = f" (QE-straf {f4(v['qe_penalty'])})" if "qe_penalty" in v else ""
        L.append(f"| {n} | {f4(v['value'])}{extra} | {v['bound']} | {'ja' if v['ok'] else 'nee'} |")
    L.append("")
    if c["calibration"]:
        for n, v in c["calibration"].items():
            L.append(f"Kalibratie `{n}`: Spearman per segment C2~C1 {f4(v['spearman_c2_c1'])} over {v['segments']} "
                     "segmenten.")
        L.append("")
        if c["high"]:
            L += [HIGH_SPEARMAN_TEXT7, ""]
    else:
        L += ["Geen kalibratie tegen C1 in het verslag van `c2cal7` (Spearman niet gemeten).", ""]
    L += ["## Smoke-test (`smoke7`)", ""]
    s = r["smoke7"]
    if s is None:
        L += ["Geen verslag van `smoke7` bij dit verslag.", ""]
    else:
        passed = s.get("passed", s.get("ok"))
        checks = s.get("checks") if isinstance(s.get("checks"), dict) else {}
        failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get("holds", v.get("ok", True))]
        L += [f"Geslaagd: {'ja' if passed else 'nee' if passed is not None else '–'}; controles {len(checks)}, niet "
              f"gehaald: {', '.join(failed) or 'geen'}" +
              (f"; schatting {json.dumps(s['estimate'], ensure_ascii=False)}" if s.get("estimate") is not None else "") +
              ".", ""]
    L += [f"Controle menselijke referenties (QE opnieuw, hele set): {json.dumps(r['refqe_check'])}", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train", choices=("train", "validation"))
    p.add_argument("--arm", default=ARM, choices=(ARM, ARM2, ARM3), help="G4-3: the revisions with the long rows")
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
    p.add_argument("--arm", default=None, choices=(ARM, ARM2, ARM3, ARM4))
    p.add_argument("--processes", type=int, default=8, help="G4-3, G4-4: processes of the check")
    p.add_argument("--chunk", type=int, default=5000, help="G4-3: distinct examples per task")
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
    p.add_argument("--arm-name", default=ARM, help="the arm's name (G4-1, G4-2)")
    p.add_argument("--heldout", action="append", default=None,
                   help="G4-2, G4-3: a held-out evalsets.parquet (the direct measure; G4-3 also heldout3's)")
    p.add_argument("--heldout-arm", action="append", default=None, help="glob of the arm's held-out pairs-w*.parquet")
    p.add_argument("--heldout-base", action="append", default=None, help="glob of G4-0's held-out pairs-w*.parquet")
    p.add_argument("--other", default=None, help="G4-4: NAME=DIR of another arm's output (pairs/, scores/), reported")
    p.add_argument("--heldout-other", action="append", default=None, help="glob of the other arm's held-out pairs")
    p.add_argument("--other-summary", default=None, help="the other arm's training summary.json (its eval losses)")
    p.add_argument("--other-tally", default=None, help="the other arm's tally.json against G4-0, if there is one")
    p.add_argument("--novels-other", default=None,
                   help="G4-4: summary.json of the other arm's novels run (g4arm_G4-3_novels), amendment G4-4.11")
    p.add_argument("--beside", action="append", default=[],
                   help="G4-6b: NAME=DIR of a further arm's output (pairs/, scores/), in the same classify run, reported")
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
    p = sub.add_parser("heldout")
    p.add_argument("--g41", required=True, help="G4-1's set ($G/data/G4-1)")
    p.add_argument("--btval", required=True, help="fetch --split validation's directory (revisions.json)")
    p.add_argument("--data-report", required=True, help="G4-1's data report.json (the length bounds)")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("own")
    p.add_argument("--heldout", required=True, help="heldout's evalsets.parquet")
    p.add_argument("--pairs", required=True, help="glob of G4-0's held-out pairs-w*.parquet")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("replay")
    for name in ("plan", "shards", "measures", "nl", "setup", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--uri", action="append", default=[], help="KEY=gs://... where each input came from")
    p.add_argument("--arm", default=None, choices=(ARM2, ARM3),
                   help="G4-3: a pool of REPLAY_POOL3 with G4-3's seed (default: G4-2's 30,000)")
    p.add_argument("--pool", type=int, default=None, help="the pool's size instead of the arm's")
    p.add_argument("--pool-seed", default=None, help="the pool's seed instead of the arm's")
    p = sub.add_parser("build2")
    p.add_argument("--base-set", required=True, help="G4-1's set ($G/data/G4-1)")
    p.add_argument("--heldout-books", required=True, help="heldout's books.json")
    p.add_argument("--replay", required=True, help="replay's parquet")
    p.add_argument("--replay-model", default=REPLAY_MODEL2)
    p.add_argument("--replay-share", type=float, default=REPLAY_SHARE2)
    p.add_argument("--epochs", type=int, default=EPOCHS2)
    p.add_argument("--flan-eval", required=True)
    p.add_argument("--evalsets", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke2")
    for name in ("summary", "adapter", "out"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("heldout3")
    p.add_argument("--btval", required=True, help="fetch --arm G4-3 --split validation's directory")
    p.add_argument("--heldout", required=True, help="G4-2's held-out evalsets.parquet ($G/data/heldout/)")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build3")
    p.add_argument("--bt", required=True, help="fetch --arm G4-3's directory (train split)")
    p.add_argument("--base-set", required=True, help="G4-1's set ($G/data/G4-1): the books")
    p.add_argument("--heldout-books", required=True, help="heldout's books.json")
    p.add_argument("--heldout", action="append", required=True,
                   help="the held-out evalsets.parquet of heldout and heldout3 (no training row among them)")
    p.add_argument("--replay", required=True, help="replay's parquet (the pool of G4-3)")
    p.add_argument("--replay-model", default=REPLAY_MODEL2)
    p.add_argument("--replay-share", type=float, default=REPLAY_SHARE3)
    p.add_argument("--epochs", type=int, default=EPOCHS3)
    p.add_argument("--flan-eval", required=True)
    p.add_argument("--evalsets", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke3")
    for name in ("summary", "adapter", "smoke-manifest", "data-report", "out"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("build4")
    p.add_argument("--g43", required=True, help="G4-3's set ($G/data/G4-3)")
    p.add_argument("--g43-report", required=True, help="G4-3's data report ($G/data/report-G4-3.json)")
    p.add_argument("--novels", required=True, help="b1-selected.parquet of the -b1- inventory")
    p.add_argument("--replay", required=True, help="G4-3's replay pool ($G/data/replay-G4-3/replay.parquet)")
    p.add_argument("--replay-model", default=REPLAY_MODEL2)
    p.add_argument("--replay-share", type=float, default=REPLAY_SHARE4)
    p.add_argument("--epochs", type=int, default=EPOCHS4)
    p.add_argument("--flan-eval", required=True)
    p.add_argument("--evalsets", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke4")
    for name in ("summary", "adapter", "smoke-manifest", "data-report", "out"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("novels4")
    p.add_argument("--g44", required=True, help="G4-4's set ($G/data/G4-4: train.parquet, train-meta.parquet, manifest)")
    p.add_argument("--data-report", required=True, help="G4-4's data report ($G/data/report-G4-4.json)")
    p.add_argument("--novels", required=True, help="b1-selected.parquet of the -b1- inventory")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--blind", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build5")
    p.add_argument("--g44", required=True, help="G4-4's set ($G/data/G4-4)")
    p.add_argument("--g44-report", required=True, help="G4-4's data report ($G/data/report-G4-4.json)")
    p.add_argument("--novels-eval", required=True, help="eval-novels-meta.parquet of novels4 ($G/data/heldout4)")
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS5)
    p.add_argument("--branch-steps", type=int, default=BRANCH_STEPS5)
    p.add_argument("--steps", action="append", default=[], help="BRANCH=N: another number of steps for one branch")
    p.add_argument("--flanval", type=int, default=FLANVAL5, help="G4-4's replay rows held out as the eval set flanval")
    p.add_argument("--out", required=True)
    p = sub.add_parser("split5")
    p.add_argument("--evalsets", required=True, help="$G/data/eval/evalsets.parquet (the four sets, FLAN, blind)")
    p.add_argument("--share", type=float, default=SPLIT_SHARE5)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke5")
    for name in ("warmup-summary", "warmup-adapter", "branch-summary", "branch-adapter", "combine-summary",
                 "warmup-manifest", "branch-manifest", "data-report", "g43-summary", "g43-novels-summary", "out"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("report5")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--split", required=True, help="split5's split.json")
    p.add_argument("--refqe", required=True, help="MetricX-QE of the human references (id, qe_source_reference)")
    p.add_argument("--candidate", action="append", required=True, help="NAME=DIR (pairs/, scores/) in the top's order")
    p.add_argument("--base", required=True, help="G4-0's output (pairs/, scores/)")
    p.add_argument("--reference", action="append", default=[], help="NAME=DIR of another model, reported")
    p.add_argument("--predictions", required=True)
    p.add_argument("--search", required=True, help="summary.json of the search run")
    p.add_argument("--run", action="append", required=True, help="NAME=summary.json of the warmup and every branch")
    p.add_argument("--tally", default=None, help="tally's json of the selected candidate against G4-0")
    p.add_argument("--smoke", default=None, help="smoke5's json (its anchor G4-3 alone, reported)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("samples5")
    p.add_argument("--temperature", required=True, help="temp5's json (its T* on books + btval)")
    p = sub.add_parser("sampledata5")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--segments", required=True, help="a directory with the evaluation's segments-w<k>.parquet, classes-w<k>.txt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("best8")
    p.add_argument("--run", action="append", required=True, help="NAME=DIR (pairs/, scores/) of a sample, in seed order")
    p.add_argument("--out", required=True)
    p = sub.add_parser("sample5")
    p.add_argument("--evalsets", required=True)
    p.add_argument("--output", action="append", required=True,
                   help="NAME=DIR (pairs/, scores/): G4-0 and G4-4 greedy, MODEL/SETTING sampled, MODEL/best8")
    p.add_argument("--predictions", required=True, help="the probe's classify run (predictions.parquet)")
    p.add_argument("--temperature", required=True, help="temp5's json (T*: the settings, as samples5)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("samplejoin")
    p.add_argument("--pairs", action="append", required=True, help="NAME=pairs parquet of a sample")
    p.add_argument("--out", required=True)
    p = sub.add_parser("samplesplit")
    p.add_argument("--scores", required=True, help="scores of samplejoin's pairs (ids NAME|ID)")
    p.add_argument("--out", required=True, help="a path with {sample}")
    p = sub.add_parser("temp5")
    p.add_argument("--base", required=True, help="summary.json of G4-0's temperature run (sft_lora.py --temperatures)")
    p.add_argument("--warmup-run", default=None, help="summary.json of the warmup adapter's temperature run")
    p.add_argument("--warmup", required=True, help="summary.json of the warmup run (its evals at step 0 and 1,000)")
    p.add_argument("--g44-report", required=True, help="G4-4's report.json (its losses at the last step)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("cands6")
    p.add_argument("--g44", required=True, help="G4-4's set ($G/data/G4-4/)")
    p.add_argument("--g44-report", required=True, help="report-G4-4.json")
    p.add_argument("--novels-eval", required=True, help="$H4/eval-novels.parquet")
    p.add_argument("--novels-eval-meta", required=True, help="$H4/eval-novels-meta.parquet")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--factor", type=float, default=CANDIDATES6, help="candidates per wanted pair")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build6")
    p.add_argument("--cands", required=True, help="cands6's cands.parquet")
    p.add_argument("--rejected", action="append", required=True, help="glob of G4-0's collected pairs of the candidates")
    p.add_argument("--heldout", required=True, help="$H/evalsets.parquet (held-out books and BT validation)")
    p.add_argument("--heldout-g40", action="append", required=True, help="glob of G4-0's held-out pairs")
    p.add_argument("--evalsets", required=True, help="$E/data/evalsets.parquet (13-grams)")
    p.add_argument("--blind", required=True, help="blind100.jsonl (13-grams)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default=ARM6, choices=(ARM6, ARM6B))
    p.add_argument("--rejected-model", default=BASE, choices=(BASE, START6B),
                   help="whose translations --rejected and --heldout-g40 are (G4-6b: G4-4)")
    p.add_argument("--compare", default=None, help="G4-6b: another build6 directory (report.json, pairs.parquet)")
    p = sub.add_parser("check6")
    p.add_argument("--data", required=True, help="build6's directory")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--chunk", type=int, default=5000)
    p = sub.add_parser("smoke6")
    for name in ("summary", "adapter", "base-summary", "build-report", "translate-log", "arm-pairs", "arm-metricx",
                 "base-pairs", "base-metricx", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--seq", type=int, default=SEQ6, choices=(SEQ6, SEQ6_FALLBACK))
    p.add_argument("--arm", default=ARM6, choices=(ARM6, ARM6B, ARM6C))
    p.add_argument("--start", default=None, choices=(START6B,),
                   help="G4-6b, G4-6c: the arm the run starts from (its adapter merged; --base-summary is its training)")
    p.add_argument("--saved-adapter", default=None,
                   help="G4-6c: the adapter.json the smoke run kept after step SMOKE_SAVE6C (amendment G4-6c.6)")
    p = sub.add_parser("pairtexts6")
    p.add_argument("--pairs", required=True, help="build6's pairs.parquet")
    p.add_argument("--out", required=True)
    p = sub.add_parser("pairclass6")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default=ARM6, choices=(ARM6, ARM6B))
    p.add_argument("--rejected-model", default=BASE, choices=(BASE, START6B))
    p = sub.add_parser("ends6c")
    p.add_argument("--data", required=True, help="G4-6b's build6 directory (its packed files)")
    p.add_argument("--tokenizer", required=True)
    p = sub.add_parser("report6c")
    for name in ("evalsets", "split", "refqe", "base", "predictions", "train-summary", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--candidate", action="append", required=True, help="G4-6c-s<step>=DIR, in the order of the steps")
    p.add_argument("--reference", action="append", default=[], help="NAME=DIR, reported beside")
    p.add_argument("--beside-summary", action="append", default=[], help="NAME=summary.json of another run, reported")
    p.add_argument("--tally", default=None)
    p = sub.add_parser("records7")
    p.add_argument("--list", required=True, help="the chain's records.json (gs:// paths)")
    p.add_argument("--out", required=True, help="report7's --records directory (made here, with stopped.json)")
    p = sub.add_parser("report7")
    for name in ("evalsets", "split", "refqe", "base", "predictions", "predictions-c2", "records", "stopped", "cal7",
                 "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--candidate", action="append", required=True, help="G4-7-i<n>=DIR (pairs/, scores/), in the order of n")
    p.add_argument("--reference", action="append", default=[], help="NAME=DIR, reported beside (G4-4)")
    p.add_argument("--smoke", default=None, help="smoke7's json (reported)")
    p.add_argument("--pll", default=None, help="naturalness.py pll of the classify run's texts (reported)")
    p.add_argument("--tally", default=None, help="tally's json of the selected candidate against G4-0")
    args = parser.parse_args(argv)
    {"fetch": fetch, "build": build, "check": check, "evaldata": evaldata, "texts": texts, "report": report,
     "tally": tally, "smoke": smoke, "heldout": heldout, "own": own, "replay": replay_candidates, "build2": build2,
     "smoke2": smoke2, "heldout3": heldout3, "build3": build3, "smoke3": smoke3, "build4": build4,
     "smoke4": smoke4, "novels4": novels4, "build5": build5, "split5": split5, "smoke5": smoke5,
     "report5": report5, "temp5": temp5, "samples5": samples5_cli, "sampledata5": sampledata5, "best8": best8,
     "sample5": sample5, "samplejoin": samplejoin, "samplesplit": samplesplit, "cands6": cands6, "build6": build6,
     "check6": check6, "smoke6": smoke6, "pairtexts6": pairtexts6, "pairclass6": pairclass6, "ends6c": ends6c,
     "report6c": report6c, "records7": records7, "report7": report7}[args.command](args)


if __name__ == "__main__":
    main()
