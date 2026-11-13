#!/usr/bin/env python3
"""Prefill check of 13 Nov (after the length trial, bt_length.py): does translating a long text fail through a fault in
our MaxText decode path, or through the model? The same code and prompts as parts A and B of the length trial, on
documents whose outcome is known, with only the engine's shapes (prefill and decode length), the MaxText build and the
reuse of decode slots changed. On mbert-mix (v4-32), each worker one process of its 4 chips.

    python prefill_check.py plan --test T1 --docs docs-en.parquet --segments seg/heel/segments-w*.parquet \\
        [--tokenizer TOK] --workers 4 --out DIR
    python prefill_check.py processes --test T3 --segments DIR/T3/segments-w0.parquet
    python prefill_check.py translate --test T1 --segments ... --out gs://... --tokenizer TOK --ckpt CKPT \\
        --work-dir DIR [--max-shards 1]
    python prefill_check.py collect --plan DIR --translations DIR --docs-a docs.parquet --docs-b docs-en.parquet \\
        --out DIR
    python prefill_check.py report --plan DIR --measured measured.parquet --units units.parquet --qe qe-*.parquet \\
        --ref-a DIR --ref-b DIR --logs DIR --out prefill-check.md
    JAX_PLATFORMS=cpu DECOUPLE_GCLOUD=TRUE python prefill_check.py reuse-cpu --work-dir DIR

reuse-cpu (added after the run had started, the CPU side of H-fork): translate_checks.py's tiny Gemma 4 with two
slots; R1 in slot 0 decodes LEAD steps, then R2 goes into slot 1 and decodes; per step of R2 the relative difference
of its decode logits from a train-mode forward over the same tokens. A CASE line per lead (0: R2 starts at cache index
0; REUSE_LEAD: R2 starts at that index and wraps after REUSE_DECODE - REUSE_LEAD steps).

Design and expectation (fixed on 13 Nov before the run, from MaxText's code; W = 1024, the sliding window of the local
layers of TranslateGemma 27B and Gemma 4 31B):

H-025, TranslateGemma on mbert-mix runs MaxText 0.2.5 from PyPI (the FLAN-nl venv), which lacks upstream commit
2bbe8006c (16 Oct 2026, "the decode sliding-window mask was measured from cache slots instead of token positions"). In
0.2.5's AttentionOp.generate_attention_mask a decode step puts the query at the last slot of each cache part
(next_pos = kv_seq_len - 1), so the local layers see only the last W slots of the prefill cache (length P, the padded
prefill length) and of the decode cache (length A = max_decode), whatever the true positions: with P > W the first
P - W prompt slots (BOS, the start of the chat template) are hidden from the first decode step on; with A > W the
first A - W generated tokens are never visible. Part B's d1k (P 1152, A 1536) has both.

H-fork, Gemma 4 runs the fork yhavinga/maxtext @ 2249474 (upstream main of 9 Nov, with 2bbe8006c): the window is
measured in true positions, but the decode cache's write index (cache_ar_index) is one scalar for all slots that wraps
modulo A, and the fix takes the newest token of a slot as its highest valid cache index. For a request inserted in a
slot after other requests have run (index s > 0) that generates more than A - s tokens, that is wrong after the wrap,
and with A > W its most recent tokens fall outside the window. Only with slot reuse (more documents than slots in one
process, or a second shard in the same process). Part A's d2k ran 10 documents a worker in 4 slots.

TESTS below: each test's documents, shapes and MaxText build, and the expected outcome under these two hypotheses
("goed": at least GOOD of the documents done; "fout": less than BAD; otherwise "gemengd"; counted over the documents
whose translation in pieces needed at most the test's decode limit, so that a limit too small is not a failure). The
documents of a test are those of one bin of a part; a test without reuse puts at most `slots` documents in a process.
"""

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

import bt_length as bl  # noqa: E402

W = 1024
GOOD = 0.9
BAD = 0.5
TEMPLATE_ROOM = 128  # prompt tokens beside the source that the document classes allow (translate_models.yml)
PARTS = {
    # part B: TranslateGemma 27B en->nl on the English of part A (gemma3-27b-doc), part A: Gemma 4 31B-it nl->en
    "B": {"source": "en", "target": "nl", "prompt": None, "doc_model": "gemma3-27b-doc"},
    "A": {"source": "nl", "target": "en", "prompt": "natural-v1", "doc_model": "gemma4-31b-doc"},
}
# The slots each class of the trial ran with (the cost tables of bt-length-a.md and bt-length-b.md: part A's d2k ran
# out of memory with 8 and was translated with 4); one shard a worker in every class (10 documents, shard >= 16)
TRIAL_SLOTS = {"A": {"d512": 32, "d1k": 16, "d2k": 4}, "B": {"d512": 56, "d1k": 32, "d2k": 16, "d4k": 8}}


def _test(part, bin_, base, maxtext, prefill, decode, max_source, slots, shard, expect, why, known,
          per_process=False, reuse=False):
    return {"part": part, "bin": bin_, "base": base, "maxtext": maxtext, "prefill": prefill, "decode": decode,
            "max_source": max_source, "slots": slots, "shard": shard, "expect": expect, "why": why, "known": known,
            "per_process": per_process, "reuse": reuse}


TESTS = {
    "T1": _test("B", 512, "gemma3-27b-doc", "0.2.5", 1152, 768, 512, 16, 64, "fout",
                "H-025: P > W, prompt slots 0-127 (BOS, template) hidden", "d512"),
    "T4a": _test("B", 512, "gemma3-27b-doc", "0.2.5", 1024, 768, 512, 16, 64, "goed",
                 "H-025: P = W and A < W, nothing hidden", "d512"),
    "T4b": _test("B", 512, "gemma3-27b-doc", "0.2.5", 1088, 768, 512, 16, 64, "fout",
                 "H-025: P > W, prompt slots 0-63 (BOS, template) hidden", "d512"),
    "T2": _test("B", 1024, "gemma3-27b-doc", "0.2.5", 1024, 1536, 960, 16, 64, "fout",
                "H-025: A > W, generated tokens 0-511 never visible to the local layers", "d1k"),
    "T2b": _test("B", 1024, "gemma3-27b-doc", "0.2.5", 1024, 1024, 960, 16, 64, "goed",
                 "H-025: P = W and A = W, nothing hidden", "d1k"),
    "T5": _test("B", 1024, "gemma3-27b-doc", "fork", 1152, 1536, 1024, 32, 64, "goed",
                "part B's d1k exactly (shapes, slots), on the fork: H-025 does not apply, no reuse", "d1k"),
    "T3": _test("A", 1024, "gemma4-31b-doc", "fork", 2176, 1280, 1024, 6, 5, "goed",
                "fork, P > W padded, no reuse (one shard of <= 5 a process)", "d1k", per_process=True),
    "T6": _test("A", 1024, "gemma4-31b-doc", "fork", 1152, 1280, 1024, 4, 32, "eerste goed, later fout",
                "part A's d1k shapes with 4 slots for 10 documents: H-fork for the documents inserted later", "d1k",
                reuse=True),
}
TG_025 = ("T4a", "T1", "T4b", "T2b", "T2")
FORK = ("T5", "T3", "T6")


def class_name(test):
    return test.lower()


def diag_model(test, models):
    """The translate_models.yml entry of a test: its base model with one class of the test's shapes and slots"""
    t = TESTS[test]
    cls = {"max_source": t["max_source"], "prefill_lengths": [t["prefill"]], "max_decode": t["decode"]}
    return dict(models[t["base"]], length_classes={class_name(test): cls}, slots={class_name(test): t["slots"]})


def model_name(test):
    return f"prefill-check-{test}"


def add_models(models):
    """translate.MODELS with every test's model (in this process only; translate_models.yml stays as it is)"""
    for test in TESTS:
        models[model_name(test)] = diag_model(test, models)
    return models


def verdict(done, n, good=GOOD, bad=BAD):
    if not n:
        return "niet gemeten"
    f = done / n
    return "goed" if f >= good else ("fout" if f < bad else "gemengd")


# ---------------------------------------------------------------------------------------------------- plan

def plan(args):
    import translate as tr

    t = TESTS[args.test]
    part = PARTS[t["part"]]
    models = add_models(dict(tr.MODELS))
    model = tr.with_prompt(models[model_name(args.test)], part["prompt"])
    cls = model["length_classes"][class_name(args.test)]
    segs = {}
    for path in args.segments:
        for r in bl.read_rows(path, ["id", "text", "tokens"]):
            segs[r["id"]] = r
    docs = sorted((d for d in bl.read_rows(args.docs, ["id", "bin"]) if d["bin"] == t["bin"]), key=lambda d: d["id"])
    codec = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        codec = tr.ModelPrompts(AutoTokenizer.from_pretrained(args.tokenizer), model, part["source"], part["target"])
    kept, dropped = [], []
    for d in docs:
        s = segs.get(d["id"])
        if s is None:
            raise SystemExit(f"{args.test}: {d['id']} of bin {t['bin']} has no segment in --segments")
        n = len(codec.encode(s["text"])) if codec else None
        if not 0 < s["tokens"] <= cls["max_source"]:
            dropped.append({"id": d["id"], "tokens": s["tokens"], "prompt_tokens": n, "why": "source over max_source"})
        elif n is not None and n > cls["prefill_lengths"][-1]:
            dropped.append({"id": d["id"], "tokens": s["tokens"], "prompt_tokens": n, "why": "prompt over prefill"})
        elif n is None and s["tokens"] + TEMPLATE_ROOM > cls["prefill_lengths"][-1]:
            raise SystemExit(f"{args.test}: {d['id']} may not fit prefill {cls['prefill_lengths'][-1]}: give "
                             f"--tokenizer")
        else:
            kept.append(dict(s, prompt_tokens=n))
    parts = bl.balanced(kept, args.workers, lambda r: r["tokens"])
    for k, rows in enumerate(parts):
        if not t["reuse"]:
            per_process = t["shard"] if t["per_process"] else len(rows)
            if per_process > t["slots"] or (not t["per_process"] and len(rows) > t["shard"]):
                raise SystemExit(f"{args.test}: worker {k} would reuse slots ({len(rows)} documents, slots "
                                 f"{t['slots']}, shard {t['shard']})")
        rows.sort(key=lambda r: r["id"])
        bl.write_parquet([{"id": r["id"], "text": r["text"], "tokens": r["tokens"]} for r in rows],
                         Path(args.out) / args.test / f"segments-w{k}.parquet")
    out = {"test": args.test, **t, "model": model_name(args.test), "class": class_name(args.test),
           "bin_docs": len(docs), "kept": [r["id"] for r in kept], "dropped": dropped,
           "prompt_tokens_max": max((r["prompt_tokens"] or 0 for r in kept), default=None) if codec else None,
           "workers": [len(p) for p in parts]}
    (Path(args.out) / args.test / "plan.json").write_text(json.dumps(out, indent=1))
    print(f"PLAN {json.dumps({k: out[k] for k in ('test', 'bin_docs', 'workers', 'prompt_tokens_max')})} "
          f"dropped {len(dropped)}", flush=True)


def processes(args):
    """The translate.py processes of one worker's share: one, or one per shard (per_process)"""
    t = TESTS[args.test]
    n = len(bl.read_rows(args.segments, ["id"]))
    print(math.ceil(n / t["shard"]) if t["per_process"] else (1 if n else 0))


# ---------------------------------------------------------------------------------------------------- translate

def translate_argv(test, segments, out, tokenizer, ckpt, work_dir, max_shards=None):
    """translate.py's arguments of a test"""
    t = TESTS[test]
    part = PARTS[t["part"]]
    argv = ["--segments", segments, "--out", out, "--tokenizer", tokenizer, "--model", model_name(test),
            "--length-class", class_name(test), "--source-lang", part["source"], "--target-lang", part["target"],
            "--ckpt", ckpt, "--work-dir", work_dir, "--shard-size", str(t["shard"])]
    if part["prompt"]:
        argv += ["--prompt", part["prompt"]]
    if max_shards is not None:
        argv += ["--max-shards", str(max_shards)]
    return argv


def translate(args):
    """bt_length_translate.py (translate.py with the MEM line) with the tests' models added; ends the process"""
    import bt_length_translate as blt
    import translate as tr

    add_models(tr.MODELS)
    sys.argv = [str(HERE / "bt_length_translate.py")] + translate_argv(
        args.test, args.segments, args.out, args.tokenizer, args.ckpt, args.work_dir, args.max_shards)
    blt.main()


# ---------------------------------------------------------------------------------------------------- collect

def insertion_ranks(rows):
    """Per id its turn in decode_all of one shard (rows in the shard's order): longest prompt first, ties in order"""
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["tokens_in"])
    return {rows[i]["id"]: k for k, i in enumerate(order)}


class _Replay:
    """decode_all's device for wrap_schedule: request i (prompt [i, 0, ...]) ends with EOS at its known length; counts
    the generate calls before each insert"""

    def __init__(self, lengths, slots, eos):
        self.lengths, self.slots, self.eos = lengths, slots, eos
        self.calls, self.owner, self.count, self.start = 0, {}, {}, {}

    def prefill(self, tokens):
        i = tokens[0]
        return i, self.eos if self.lengths[i] == 1 else 2

    def insert(self, prefix, slot):
        self.owner[slot], self.count[slot], self.start[prefix] = prefix, 1, self.calls

    def generate(self):
        self.calls += 1
        out = []
        for s in range(self.slots):
            i = self.owner.get(s)
            if i is not None:
                self.count[s] += 1
            out.append(self.eos if i is not None and self.count[s] == self.lengths[i] else 2)
        return out

    @staticmethod
    def fetch(arrays):
        return arrays


def wrap_schedule(rows, slots, max_decode, sync_every=8, eos=1):
    """Added after the run (T6): translate.decode_all replayed on one shard's rows (in the shard's order) with their
    known prompt and output lengths (tokens_in; tokens_out, the stop token included). Per id: the generate calls
    before its insert (start), the shared decode-cache index it starts at (start mod A, A = max_decode, the length
    of the decode cache), its writes (tokens_out - 1: the first token comes from the prefill) and whether they run past
    the end of the cache and wrap to index 0 (crosses)."""
    import translate as tr

    lengths = [min(r["tokens_out"], max_decode) for r in rows]
    prompts = [[i] + [0] * (r["tokens_in"] - 1) for i, r in enumerate(rows)]
    device = _Replay(lengths, slots, eos)
    tr.decode_all(prompts, slots, device, max_decode, sync_every, [eos])
    out = {}
    for i, r in enumerate(rows):
        start = device.start[i]
        out[r["id"]] = {"start": start, "index": start % max_decode, "writes": lengths[i] - 1,
                        "crosses": start % max_decode + lengths[i] - 1 > max_decode}
    return out


def crossing(rows, slots, max_decode):
    """wrap_schedule over the rows of every worker's shard (one decode_all each; rows of a shard sorted by id, the
    order of translate.py's shards)"""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r.get("worker"), r.get("shard"))].append(r)
    out = {}
    for _, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out.update(wrap_schedule(sorted(rs, key=lambda r: r["id"]), slots, max_decode))
    return out


def read_shards(root, column):
    """{id: row} of translate.py's shards under root, with the worker, the shard and the turn in its decode_all"""
    out = {}
    for path in sorted(Path(root).rglob("shard-*.parquet")) if Path(root).exists() else []:
        rows = bl.read_rows(path)
        ranks = insertion_ranks(rows)
        worker = next((p.name for p in path.parents if re.fullmatch(r"w\d+", p.name)), None)
        for r in rows:
            if r["id"] in out:
                raise SystemExit(f"{r['id']} is in two shards under {root}")
            out[r["id"]] = {"text": r[column], "ended": r["ended"], "tokens_in": r["tokens_in"],
                            "tokens_out": r["tokens_out"], "worker": worker, "shard": path.name,
                            "rank": ranks[r["id"]]}
    return out


def collect(args):
    docs = {"A": {d["id"]: d for d in bl.read_rows(args.docs_a)}, "B": {d["id"]: d for d in bl.read_rows(args.docs_b)}}
    measured, units, missing = [], [], collections.Counter()
    for test in args.tests:
        t = TESTS[test]
        part = PARTS[t["part"]]
        p = json.loads((Path(args.plan) / test / "plan.json").read_text())
        got = read_shards(Path(args.translations) / test, f"text_{part['target']}")
        for doc_id in p["kept"]:
            x = got.get(doc_id)
            if x is None:
                missing[test] += 1
                continue
            d = docs[t["part"]][doc_id]
            row, unit_rows, alignment = bl.measure_doc(d, x["text"], x["ended"], part["source"], part["target"])
            measured.append({"id": doc_id, "source": d["source"], "bin": d["bin"], "condition": test, "test": test,
                             "tokens_source": d["tokens"], "tokens_in": x["tokens_in"], "tokens_out": x["tokens_out"],
                             "translation": x["text"], "worker": x["worker"], "shard": x["shard"], "rank": x["rank"],
                             "first_inserted": x["rank"] < t["slots"], "alignment": json.dumps(alignment), **row})
            for u in unit_rows:
                units.append({"id": f"{doc_id}|{test}|u{u['unit']:04d}", "doc": doc_id, "source": d["source"],
                              "bin": d["bin"], "condition": test, **u})
    bl.write_parquet(measured, Path(args.out) / "measured.parquet")
    bl.write_parquet(units, Path(args.out) / "units.parquet")
    print(f"COLLECT {json.dumps({'measured': len(measured), 'units': len(units), 'missing': dict(missing)})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- report

def read_scores(paths, column="qe_src_cand"):
    scores = {}
    for path in paths or []:
        for r in bl.read_rows(path):
            if r.get(column) is not None:
                scores[r["id"]] = r[column]
    return scores


def similarity(pairs):
    """Of (translation, known translation) pairs: how many are equal, and difflib's ratio on their words (no
    autojunk: on long texts it would drop the common words), median and how many below 0.5"""
    import difflib

    ratios = sorted(difflib.SequenceMatcher(None, (a or "").split(), (b or "").split(), autojunk=False).ratio()
                    for a, b in pairs)
    return {"n": len(pairs), "identical": sum((a or "").strip() == (b or "").strip() for a, b in pairs),
            "median": ratios[len(ratios) // 2] if ratios else None, "below_half": sum(r < 0.5 for r in ratios)}


def stats(rows, qe, cond):
    """bt_length.bin_stats of rows of one condition (a test, or heel/stukken of the trial), whatever their bins; qe
    keyed by (doc, condition), only the rows' documents count"""
    rows = [dict(r, bin=0) for r in rows]
    if any(r["condition"] != cond for r in rows):
        raise ValueError(f"rows of another condition than {cond}")
    return bl.bin_stats(rows, qe, [0], conditions=(cond,))[0][cond]


def load_ref(root):
    """A part's reference: measured, units and scores of the length trial, its heel segments (the workers' order)"""
    root = Path(root)
    units = bl.read_rows(root / "units.parquet", ["id", "doc", "condition", "n_source"])
    return {"measured": bl.read_rows(root / "measured.parquet"),
            "qe": bl.qe_per_paragraph(units, read_scores(sorted(root.glob("qe-*.parquet"))), "qe_src_cand"),
            "segments": {k: bl.read_rows(p, ["id", "tokens"]) for k, p in
                         ((int(p.stem.split("-w")[-1]), p) for p in sorted(root.glob("seg/segments-w*.parquet")))}}


def retro(ref, part):
    """The trial's heel classes split by the turn of their documents in decode_all (one shard a worker): the first
    `slots` documents of a worker start in empty slots, the others in slots used before"""
    import translate as tr

    classes = tr.MODELS[PARTS[part]["doc_model"]]["length_classes"]
    order = list(classes)
    heel = {r["id"]: r for r in ref["measured"] if r["condition"] == "heel"}
    out = {}
    for cls, slots in TRIAL_SLOTS[part].items():
        k = order.index(cls)
        low, high = (classes[order[k - 1]]["max_source"] if k else 0), classes[cls]["max_source"]
        groups = {"first": [], "later": [], "later_cross": [], "later_inside": []}
        for _, segs in sorted(ref["segments"].items()):
            rows = [heel[s["id"]] for s in segs if low < s["tokens"] <= high and s["id"] in heel]
            ranks = insertion_ranks(rows)
            cross = wrap_schedule(rows, slots, classes[cls]["max_decode"])
            for r in rows:
                first = ranks[r["id"]] < slots
                groups["first" if first else "later"].append(r)
                if not first:
                    groups["later_cross" if cross[r["id"]]["crosses"] else "later_inside"].append(r)
        out[cls] = {"slots": slots, **{g: {"n": len(rs), "done": sum(r["done"] for r in rs),
                                           "loop": sum(r["decode_loop"] for r in rs),
                                           "not_ended": sum(not r["ended"] for r in rs)}
                                       for g, rs in groups.items()}}
    return out


def maxtext_of_logs(logs_dir):
    """{log name: the MAXTEXT line's version, url, commit} of translate.py's logs"""
    out = {}
    for path in sorted(Path(logs_dir).glob("*.log")) if logs_dir else []:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("MAXTEXT "):
                m = json.loads(line[len("MAXTEXT "):])
                out[path.name] = {k: m.get(k) for k in ("version", "url", "commit")}
                break
    return out


def report(args):
    measured = bl.read_rows(args.measured)
    units = bl.read_rows(args.units, ["id", "doc", "condition", "n_source"])
    qe = bl.qe_per_paragraph(units, read_scores(args.qe), "qe_src_cand")
    refs = {"A": load_ref(args.ref_a), "B": load_ref(args.ref_b)}
    plans = {test: json.loads((Path(args.plan) / test / "plan.json").read_text())
             for test in TESTS if (Path(args.plan) / test / "plan.json").exists()}
    r = {"tests": {}, "retro": {part: retro(refs[part], part) for part in refs},
         "throughput": bl.throughput(args.logs), "maxtext": maxtext_of_logs(args.logs)}
    for test, p in plans.items():
        t = TESTS[test]
        ref = refs[t["part"]]
        rows = [m for m in measured if m["test"] == test]
        ids = set(p["kept"])
        heel = {m["id"]: m for m in ref["measured"] if m["condition"] == "heel" and m["id"] in ids}
        stukken = {m["id"]: m for m in ref["measured"] if m["condition"] == "stukken" and m["id"] in ids}
        fits = {i for i, m in stukken.items() if m["tokens_out"] <= t["decode"]}
        fit_rows = [m for m in rows if m["id"] in fits]
        e = {"plan": {k: p[k] for k in ("bin_docs", "workers", "prompt_tokens_max")}, "dropped": p["dropped"],
             "kept": len(ids), "measured": len(rows), "fits": len(fits),
             "test": stats(rows, qe, test),
             "ref_heel": stats(list(heel.values()), ref["qe"], "heel"),
             "ref_stukken": stats(list(stukken.values()), ref["qe"], "stukken"),
             "done_fit": sum(m["done"] for m in fit_rows),
             "verdict": verdict(sum(m["done"] for m in fit_rows), len(fit_rows)),
             "same_prompts": sum(m["tokens_in"] == heel[m["id"]]["tokens_in"] for m in rows if m["id"] in heel),
             "both_done": sum(m["done"] and heel[m["id"]]["done"] for m in rows if m["id"] in heel),
             "only_ref_done": sum(heel[m["id"]]["done"] and not m["done"] for m in rows if m["id"] in heel),
             "only_test_done": sum(m["done"] and not heel[m["id"]]["done"] for m in rows if m["id"] in heel),
             "similarity": similarity([(m["translation"], heel[m["id"]]["translation"]) for m in rows
                                       if m["id"] in heel])}
        if t["reuse"]:
            for g, flag in (("first", True), ("later", False)):
                rs = [m for m in fit_rows if m["first_inserted"] == flag]
                e[f"group_{g}"] = {"n": len(rs), "done": sum(m["done"] for m in rs),
                                   "loop": sum(m["decode_loop"] for m in rs),
                                   "not_ended": sum(not m["ended"] for m in rs),
                                   "verdict": verdict(sum(m["done"] for m in rs), len(rs)),
                                   "stats": stats(rs, qe, test)}
            cross = crossing(fit_rows, t["slots"], t["decode"])
            for g, test_of in (("later_cross", lambda m: not m["first_inserted"] and cross[m["id"]]["crosses"]),
                               ("later_inside", lambda m: not m["first_inserted"] and not cross[m["id"]]["crosses"]),
                               ("first_cross", lambda m: m["first_inserted"] and cross[m["id"]]["crosses"])):
                rs = [m for m in fit_rows if test_of(m)]
                e[f"group_{g}"] = {"n": len(rs), "done": sum(m["done"] for m in rs),
                                   "loop": sum(m["decode_loop"] for m in rs),
                                   "not_ended": sum(not m["ended"] for m in rs),
                                   "verdict": verdict(sum(m["done"] for m in rs), len(rs)),
                                   "stats": stats(rs, qe, test),
                                   "similarity": similarity([(m["translation"], heel[m["id"]]["translation"])
                                                             for m in rs if m["id"] in heel])}
        r["tests"][test] = e
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report_md(r))
    Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    print(f"REPORT {args.out}", flush=True)


def _shapes(t):
    return f"{t['prefill']} / {t['decode']}"


def report_md(r):
    num, kn = bl.num, bl.kn
    lines = ["## Prefill-controle: TranslateGemma 27B en→nl en Gemma 4 31B-it nl→en, 13 nov", "",
             "### Ontwerp en verwachting (vastgelegd vóór de run; `prefill_check.py`)", "",
             f"W = {W} (sliding window van de lokale lagen van beide modellen). P = opgevulde prefill-lengte, A = "
             "decodeerlimiet (lengte van de decode-cache). Uitslag per test: **goed** als ten minste "
             f"{int(GOOD * 100)}% van de documenten af is, **fout** onder {int(BAD * 100)}%, anders gemengd; geteld over "
             "de documenten waarvan de vertaling in stukken (lengteproef) hoogstens A tokens had.", "",
             "| test | model, MaxText | documenten | P / A | slots | slot-hergebruik | verwachting | waarom |",
             "|---|---|---|---|---:|---|---|---|"]
    names = {"B": "TranslateGemma 27B en→nl", "A": "Gemma 4 31B-it nl→en"}
    for test, t in TESTS.items():
        lines.append(f"| {test} | {names[t['part']]}, {t['maxtext']} | deel {t['part']}, bak {t['bin']} | "
                     f"{_shapes(t)} | {t['slots']} | {'ja' if t['reuse'] else 'nee'} | {t['expect']} | {t['why']} |")
    lines += ["", "### Uitslag per test [gemeten]", "",
              "| test | P / A | docs (weggelaten) | af | lus | niet af | afgekapt | alinea's behouden | QE gem. | "
              "uitvoertokens med. | bekend heel (zelfde docs): af / QE | in stukken: af / QE | uitslag | verwachting |",
              "|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|"]
    for test, e in r["tests"].items():
        t, s, h, p = TESTS[test], e["test"], e["ref_heel"], e["ref_stukken"]
        lines.append(
            f"| {test} | {_shapes(t)} | {e['measured']} ({len(e['dropped'])}) | {kn(s['done'], s['docs'])} | "
            f"{s['decode_loop']} | {s['not_ended']} | {s['cut']} | {kn(s['paragraphs_kept'], s['docs'])} | "
            f"{num(s['qe_mean'])} | {num(s['tokens_out_median'], 0)} | {kn(h['done'], h['docs'])} / "
            f"{num(h['qe_mean'])} | {kn(p['done'], p['docs'])} / {num(p['qe_mean'])} | "
            f"**{e['verdict']}** ({e['done_fit']}/{e['fits']}) | {t['expect']} |")
    lines += ["", "Per document tegen de bekende uitslag in de klasse van de lengteproef (heel). Gelijkenis (achteraf "
              "toegevoegd, na het zien van T1; niet vooraf vastgelegd): difflib-ratio op woorden van de vertaling nu "
              "en de bekende vertaling van hetzelfde document; 1 is gelijk.", "",
              "| test | prompts gelijk (tokens_in) | beide af | alleen bekend af | alleen nu af | gelijke tekst | "
              "gelijkenis med. | gelijkenis < 0,5 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for test, e in r["tests"].items():
        sim = e["similarity"]
        lines.append(f"| {test} | {e['same_prompts']}/{e['measured']} | {e['both_done']} | {e['only_ref_done']} | "
                     f"{e['only_test_done']} | {sim['identical']}/{sim['n']} | {num(sim['median'], 3)} | "
                     f"{sim['below_half']} |")
    reuse = [(test, e) for test, e in r["tests"].items() if TESTS[test]["reuse"]]
    if reuse:
        lines += ["", "Slot-hergebruik: documenten die in een leeg slot begonnen (eerste `slots` van hun worker) tegen "
                  "documenten die later in een gebruikt slot kwamen [gemeten]:", "",
                  "| test | groep | n | af | lus | niet af | QE gem. | uitslag |", "|---|---|---:|---:|---:|---:|---:|---|"]
        for test, e in reuse:
            for g, label in (("first", "eerst ingevoegd"), ("later", "later ingevoegd")):
                x = e[f"group_{g}"]
                lines.append(f"| {test} | {label} | {x['n']} | {x['done']} | {x['loop']} | {x['not_ended']} | "
                             f"{num(x['stats']['qe_mean'])} | {x['verdict']} |")
        lines += ["", "Achteraf toegevoegd (na het zien van T6; niet vooraf vastgelegd): `decode_all` nagespeeld met de "
                  "gemeten prompt- en uitvoerlengtes (`wrap_schedule`): begint een document op gedeelde index s van de "
                  "decode-cache (lengte A) en schrijft het meer dan A − s tokens, dan loopt de index tijdens zijn "
                  "decode rond (H-fork). Gelijkenis: met de bekende vertaling van hetzelfde document.", "",
                  "| test | groep | n | af | lus | niet af | QE gem. | gelijkenis med. | gelijkenis < 0,5 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for test, e in reuse:
            for g, label in (("first_cross", "eerst ingevoegd, loopt rond"),
                             ("later_cross", "later ingevoegd, loopt rond"),
                             ("later_inside", "later ingevoegd, loopt niet rond")):
                x = e[f"group_{g}"]
                lines.append(f"| {test} | {label} | {x['n']} | {x['done']} | {x['loop']} | {x['not_ended']} | "
                             f"{num(x['stats']['qe_mean'])} | {num(x['similarity']['median'], 3)} | "
                             f"{x['similarity']['below_half']} |")
    lines += ["", "### De lengteproef zelf, per beurt in decode_all [gemeten, retrospectief]", "",
              "Uit `measured.parquet` en de segmenten van de lengteproef: per worker één shard; de eerste `slots` "
              "documenten (langste prompt eerst) beginnen in lege slots, de rest in slots die al gebruikt zijn.", "",
              "| deel | klasse | slots | eerst ingevoegd: n / af / lus / niet af | later ingevoegd: n / af / lus / niet af |",
              "|---|---|---:|---|---|"]
    for part, classes in r["retro"].items():
        for cls, x in classes.items():
            f, la = x["first"], x["later"]
            lines.append(f"| {part} | {cls} | {x['slots']} | {f['n']} / {f['done']} / {f['loop']} / {f['not_ended']} | "
                         f"{la['n']} / {la['done']} / {la['loop']} / {la['not_ended']} |")
    lines += ["", "Achteraf toegevoegd (na het zien van T6): de later ingevoegde documenten gesplitst naar `wrap_schedule` "
              "(loopt de gedeelde index van de decode-cache rond tijdens hun decode). Alleen voor deel A (fork) is dat "
              "het mechanisme van H-fork; deel B draaide op 0.2.5, waar het masker niet op posities rekent.", "",
              "| deel | klasse | later, loopt rond: n / af / lus / niet af | later, loopt niet rond: n / af / lus / niet af |",
              "|---|---|---|---|"]
    for part, classes in r["retro"].items():
        for cls, x in classes.items():
            c, i = x["later_cross"], x["later_inside"]
            lines.append(f"| {part} | {cls} | {c['n']} / {c['done']} / {c['loop']} / {c['not_ended']} | "
                         f"{i['n']} / {i['done']} / {i['loop']} / {i['not_ended']} |")
    lines += ["", "### Kosten en MaxText per log [gemeten]", "",
              "| log | klasse | slots | prefill / decodeerlimiet | segmenten | laden (s) | decodeertijd (s) | "
              "piek HBM per chip (GiB) | geheugen op | MaxText |", "|---|---|---:|---|---:|---:|---:|---:|---|---|"]
    for x in r["throughput"]:
        sh = x["shapes"]
        mt = r["maxtext"].get(x["log"], {})
        build = f"{mt.get('version')} {(mt.get('commit') or 'PyPI')[:9]}" if mt else "–"
        lines.append(f"| {x['log']} | {x['class']} | {sh.get('slots')} | {sh['prefill_lengths'][-1]} / "
                     f"{sh['max_decode']} | {x['segments']} | {x['load_s'] if x['load_s'] is not None else '–'} | "
                     f"{x['seconds']:.0f} | {num(x['peak_gib'])} | {'ja' if x['oom'] else ''} | {build} |")
    dropped = [(test, d) for test, e in r["tests"].items() for d in e["dropped"]]
    if dropped:
        lines += ["", "Weggelaten documenten (prompt past niet in P, of bron boven max_source):", ""]
        lines += [f"- {test}: `{d['id']}`: {d['tokens']} brontokens, prompt {d['prompt_tokens']}, {d['why']}"
                  for test, d in dropped]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- reuse-cpu

REUSE_WINDOW, REUSE_PREFILL, REUSE_DECODE, REUSE_PROMPT, REUSE_LEAD, REUSE_STEPS = 8, 16, 24, 5, 12, 20


def reuse_case(lead, steps, work_dir):
    """Per decode step of R2 (inserted after R1 decoded `lead` steps): max |decode - forward| / max |forward|"""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import translate_checks as tc
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine
    from maxtext.utils import maxtext_utils, model_creation_utils

    seq = REUSE_PREFILL + REUSE_DECODE
    cfg = pyconfig.initialize([""], **dict(tc.TINY, per_device_batch_size=2), sliding_window_size=REUSE_WINDOW,
                              max_prefill_predict_length=REUSE_PREFILL, max_target_length=seq,
                              base_output_directory=f"{work_dir}/maxtext_output")
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(params=0))
    _, params_state, _ = nnx.split(model, nnx.Param, ...)
    leaves, treedef = jax.tree.flatten(params_state)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    params_state = jax.tree.unflatten(treedef, [jax.random.normal(k, x.shape, x.dtype) * 0.2
                                                for k, x in zip(keys, leaves)])
    engine = maxengine.MaxEngine(cfg, jax.devices())
    params = engine.load_params(params=params_state)
    rng = np.random.default_rng(1)

    def prefill(prompt):
        padded = jnp.asarray(prompt + [0] * (REUSE_PREFILL - len(prompt)), dtype=jnp.int32)
        return engine.prefill(params=params, padded_tokens=padded, true_length=len(prompt))

    prefix1, _ = prefill([int(x) for x in rng.integers(3, cfg.vocab_size, REUSE_PROMPT)])
    state = engine.insert(prefix1, engine.init_decode_state(), slot=0)
    for _ in range(lead):
        state, _ = engine.generate(params, state)
    r2 = [int(x) for x in rng.integers(3, cfg.vocab_size, REUSE_PROMPT)]
    prefix2, first = prefill(r2)
    tokens = r2 + [int(first.data[0, 0])]
    logits = [np.asarray(prefix2["logits"], np.float64).reshape(-1)[-cfg.vocab_size:]]
    state = engine.insert(prefix2, state, slot=1)
    for _ in range(steps):
        state, result = engine.generate(params, state)
        logits.append(np.asarray(state["logits"][1], np.float64).reshape(-1)[-cfg.vocab_size:])
        tokens.append(int(result.data[1, 0]))
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        train_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN,
                                                        rngs=nnx.Rngs(params=0))
    nnx.update(train_model, params_state)
    batch = cfg.micro_batch_size_to_train_on
    ids = jnp.tile(jnp.asarray([tokens + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    positions = jnp.tile(jnp.arange(seq, dtype=jnp.int32)[None], (batch, 1))
    segments = jnp.tile(jnp.asarray([[1] * len(tokens) + [0] * (seq - len(tokens))], dtype=jnp.int32), (batch, 1))
    with nn_partitioning.axis_rules(cfg.logical_axis_rules), mesh:
        ref = train_model(ids, positions, decoder_segment_ids=segments, enable_dropout=False,
                          model_mode=MODEL_MODE_TRAIN)
    ref = np.asarray(ref[0] if isinstance(ref, tuple) else ref, np.float64)[0]
    return [float(np.max(np.abs(d - ref[REUSE_PROMPT - 1 + k])) / np.max(np.abs(ref[REUSE_PROMPT - 1 + k])))
            for k, d in enumerate(logits)]


def reuse_cpu(args):
    import translate
    import translate_checks as tc

    print("MAXTEXT " + json.dumps({**translate.maxtext_source(), "patches": translate.patch_maxtext()}), flush=True)
    cases = []
    for lead in (0, REUSE_LEAD):
        rels = reuse_case(lead, REUSE_STEPS, args.work_dir)
        bad = [k for k, x in enumerate(rels) if x > tc.REL_TOL]
        cases.append({"window": REUSE_WINDOW, "max_prefill": REUSE_PREFILL, "decode_cache": REUSE_DECODE,
                      "prompt": REUSE_PROMPT, "lead": lead, "steps": REUSE_STEPS, "wrap_after_step": REUSE_DECODE - lead,
                      "first_step_over_tol": bad[0] if bad else None, "steps_over_tol": len(bad),
                      "rel": [float(f"{x:.2g}") for x in rels]})
        print("CASE " + json.dumps(cases[-1]), flush=True)
    return cases


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--test", required=True, choices=sorted(TESTS))
    a.add_argument("--docs", required=True)
    a.add_argument("--segments", nargs="+", required=True)
    a.add_argument("--tokenizer", default=None, help="The model's tokenizer: drop prompts longer than the prefill")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", required=True)
    a = sub.add_parser("processes")
    a.add_argument("--test", required=True, choices=sorted(TESTS))
    a.add_argument("--segments", required=True)
    a = sub.add_parser("translate")
    a.add_argument("--test", required=True, choices=sorted(TESTS))
    a.add_argument("--segments", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--ckpt", required=True)
    a.add_argument("--work-dir", required=True)
    a.add_argument("--max-shards", type=int, default=None)
    a = sub.add_parser("collect")
    a.add_argument("--plan", required=True)
    a.add_argument("--translations", required=True)
    a.add_argument("--docs-a", required=True)
    a.add_argument("--docs-b", required=True)
    a.add_argument("--tests", nargs="+", default=list(TESTS))
    a.add_argument("--out", required=True)
    a = sub.add_parser("report")
    a.add_argument("--plan", required=True)
    a.add_argument("--measured", required=True)
    a.add_argument("--units", required=True)
    a.add_argument("--qe", nargs="*", default=[])
    a.add_argument("--ref-a", required=True, help="measured, units, qe-* and seg/segments-w* of part A")
    a.add_argument("--ref-b", required=True, help="the same of part B")
    a.add_argument("--logs", default=None)
    a.add_argument("--out", required=True)
    a = sub.add_parser("reuse-cpu")
    a.add_argument("--work-dir", required=True)
    args = p.parse_args(argv)
    {"plan": plan, "processes": processes, "translate": translate, "collect": collect, "report": report,
     "reuse-cpu": reuse_cpu}[args.cmd](args)


if __name__ == "__main__":
    main()
