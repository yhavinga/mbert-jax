#!/usr/bin/env python3
"""Fork fix check of 14 Nov: does the fix of the shared decode-cache index in the fork (yhavinga/maxtext, branch
fix-decode-cache-shared-index @ f4d8615) end the loops of documents that start in a reused decode slot and run past
the end of the decode cache (H-fork of prefill_check.py)? The stages, documents, prompts, checkpoints and measures of
prefill_check.py; only the MaxText build (venv-fork-f4d8615f4 of setup_generic.sh) and the tests differ. On mbert-mix
(v4-32), each worker one process of its 4 chips (run_prefill_check.sh with CHECK=fork_fix_check).

    python fork_fix_check.py plan|processes|translate|collect ...   as prefill_check.py, with the tests below
    python fork_fix_check.py report ... (as prefill_check.py) --before DIR --out fork-fix-check.md
    JAX_PLATFORMS=cpu DECOUPLE_GCLOUD=TRUE python fork_fix_check.py reuse-cpu --work-dir DIR

--before: measured.parquet, units.parquet and qe-*.parquet of the prefill check of 13 Nov (its collect/ and scores/):
the same documents without the fix (T6 for V1, T5 for V3); for V2 the trial's d2k (heel) is the run without the fix.

Design and expectation (fixed on 14 Nov before the run; W = 1024, the sliding window of the local layers):

The fix: AttentionOp._decode_cache_positions takes a slot's newest decode-cache token as the end of the run of its
valid decode-cache slots (a valid slot followed by an invalid one) instead of its highest valid slot; without a wrap
the two are the same. The fork's unit tests (CPU, decode cache 24, window 8): a request inserted into a second slot
halfway through the decode cache follows a train-mode forward pass at every step with the fix (Gemma 2, 3 and 4) and
not without it.

H-fix: with the fix, a document that starts in a reused slot and runs past the end of the decode cache sees the same
window as one in a fresh slot, so the documents that looped through H-fork end as they did in fresh slots:
- V1, T6 with the fix (Gemma 4, part A bin 1024, P 1152 / A 1280, 4 slots for 10 documents a worker): goed, at least
  38 of 40 af, QE within 0.15 of the trial's d1k heel (3.40; 16 slots, every document in a fresh slot). Without the
  fix (T6): 26 of 40, QE 5.13; inserted later and wrapping 2 of 16.
- V2, part A's d2k with the fix (bin 2048, P 2176 / A 2560, 4 slots as in the trial): goed, at least 90% af of the
  documents whose translation in pieces fits A, QE at most 0.3 above in pieces (3.27). Without the fix (the trial's
  d2k): 16 of 40, QE 5.59; inserted later and wrapping 0 of 23.
- V3, TranslateGemma 27B on the fork with slot reuse (part B bin 1024, P 1152 / A 1536 as T5, but 4 slots for 10
  documents a worker instead of 32): goed, as T5 (38 of 40, QE 1.57, no reuse): at least 36 of 40 af, QE within 0.3
  of T5. TranslateGemma has not run on the fork with reuse before; on 0.2.5 the same documents gave 1 of 40 (d1k).
- V4, Gemma 4 bin 4096 whole with 1 slot: not run. Part A's d4k (P 4224 / A 5120) ran out of memory with 1 slot on a
  v4 worker in the trial (len_heel_d4k_g4c1113b-s1, peak 23.88 GiB per chip; bt-length-a.md).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

import bt_length as bl  # noqa: E402
import prefill_check as pc  # noqa: E402

FIX_COMMIT = "f4d8615f478b6c18de9df8f2dff1f57f775a2521"
PC_TESTS = dict(pc.TESTS)
TESTS = {
    "V1": pc._test("A", 1024, "gemma4-31b-doc", "fork-fix", 1152, 1280, 1024, 4, 32, "goed",
                   "T6 with the fix: 4 slots for 10 documents a worker, A > W", "d1k", reuse=True),
    "V2": pc._test("A", 2048, "gemma4-31b-doc", "fork-fix", 2176, 2560, 2048, 4, 32, "goed",
                   "part A's d2k with the fix: 4 slots for 10 documents a worker, A > W", "d2k", reuse=True),
    "V3": pc._test("B", 1024, "gemma3-27b-doc", "fork-fix", 1152, 1536, 1024, 4, 32, "goed",
                   "T5's shapes with the fix and 4 slots for 10 documents a worker, A > W", "d1k", reuse=True),
}
# The same documents without the fix: a test of the prefill check (--before), or the trial's heel and its slots
BEFORE = {"V1": "T6", "V2": "heel", "V3": "T5"}
EXPECT = {"V1": "goed: ≥ 38/40 af, QE ≤ 3,40 + 0,15", "V2": "goed: ≥ 90% af, QE ≤ in stukken + 0,3",
          "V3": "goed: ≥ 36/40 af, QE binnen 0,3 van T5"}
NAMES = {"B": "TranslateGemma 27B en→nl", "A": "Gemma 4 31B-it nl→en"}


def translate(args):
    """prefill_check.py's translate, only on the fork with the fix"""
    import translate as tr

    source = tr.maxtext_source()
    if source["commit"] != FIX_COMMIT:
        raise SystemExit(f"{args.test}: MaxText here is {source}, not the fork at {FIX_COMMIT}")
    pc.translate(args)


# ---------------------------------------------------------------------------------------------------- report

def groups(rows, qe, cond, slots, decode):
    """The rows (one condition) split as prefill_check.py's report splits a test with reuse: first inserted, inserted
    later and wrapping during their decode, inserted later and not wrapping"""
    cross = pc.crossing(rows, slots, decode)
    tests = {"first": lambda m: m["first_inserted"],
             "later_cross": lambda m: not m["first_inserted"] and cross[m["id"]]["crosses"],
             "later_inside": lambda m: not m["first_inserted"] and not cross[m["id"]]["crosses"]}
    out = {}
    for g, test_of in tests.items():
        rs = [m for m in rows if test_of(m)]
        out[g] = {"n": len(rs), "done": sum(m["done"] for m in rs), "loop": sum(m["decode_loop"] for m in rs),
                  "not_ended": sum(not m["ended"] for m in rs), "qe_mean": pc.stats(rs, qe, cond)["qe_mean"]}
    return out


def add_before(r, args):
    """Per test of r: the same documents without the fix (stats, the decode_all groups, and per document) and the
    similarity of each translation to the one without the fix and to the one in pieces"""
    measured = bl.read_rows(args.measured)
    trial = {"A": pc.load_ref(args.ref_a), "B": pc.load_ref(args.ref_b)}
    before = pc.load_ref(args.before)
    for test, e in r["tests"].items():
        t = TESTS[test]
        ids = set(json.loads((Path(args.plan) / test / "plan.json").read_text())["kept"])
        ref = trial[t["part"]]
        stukken = {m["id"]: m for m in ref["measured"] if m["condition"] == "stukken" and m["id"] in ids}
        fits = {i for i, m in stukken.items() if m["tokens_out"] <= t["decode"]}
        rows = {m["id"]: m for m in measured if m["test"] == test}
        b = BEFORE[test]
        if b == "heel":
            prior = {m["id"]: m for m in ref["measured"] if m["condition"] == "heel" and m["id"] in ids}
            qe, slots = ref["qe"], pc.TRIAL_SLOTS[t["part"]][t["known"]]
            retro = r["retro"][t["part"]][t["known"]]
            groups_before = {g: dict(retro[g], qe_mean=None) for g in ("first", "later_cross", "later_inside")}
        else:
            prior = {m["id"]: m for m in before["measured"] if m["condition"] == b and m["id"] in ids}
            qe, slots = before["qe"], PC_TESTS[b]["slots"]
            groups_before = groups([m for m in prior.values() if m["id"] in fits], qe, b, slots, PC_TESTS[b]["decode"])
        fit_prior = [m for m in prior.values() if m["id"] in fits]
        both = [i for i in rows if i in prior]
        e["before"] = {
            "source": b, "slots": slots, "stats": pc.stats(list(prior.values()), qe, b),
            "done_fit": sum(m["done"] for m in fit_prior), "fits": len(fit_prior),
            "verdict": pc.verdict(sum(m["done"] for m in fit_prior), len(fit_prior)), "groups": groups_before,
            "both_done": sum(rows[i]["done"] and prior[i]["done"] for i in both),
            "only_before_done": sum(prior[i]["done"] and not rows[i]["done"] for i in both),
            "only_now_done": sum(rows[i]["done"] and not prior[i]["done"] for i in both),
            "similarity": pc.similarity([(rows[i]["translation"], prior[i]["translation"]) for i in both])}
        e["similarity_stukken"] = pc.similarity([(m["translation"], stukken[i]["translation"])
                                                 for i, m in rows.items() if i in stukken])


def report(args):
    """prefill_check.py's report with the runs without the fix added and this check's text (inside main)"""
    def md(r):
        add_before(r, args)
        return report_md(r)

    pc.report_md = md
    pc.report(args)


def report_md(r):
    num, kn = bl.num, bl.kn
    lines = ["## Fork-fix-controle: de gedeelde index van de decode-cache, Gemma 4 31B-it en TranslateGemma 27B, 14 nov",
             "", "### Ontwerp en verwachting (vastgelegd vóór de run; `fork_fix_check.py`)", "",
             f"W = {pc.W} (sliding window van de lokale lagen). P = opgevulde prefill-lengte, A = decodeerlimiet (lengte "
             f"van de decode-cache). MaxText: de fork yhavinga/maxtext, branch fix-decode-cache-shared-index @ "
             f"`{FIX_COMMIT[:9]}`. Uitslag per test: **goed** als ten minste {int(pc.GOOD * 100)}% van de documenten af "
             f"is, **fout** onder {int(pc.BAD * 100)}%, anders gemengd; geteld over de documenten waarvan de vertaling "
             "in stukken (lengteproef) hoogstens A tokens had. Zonder fix: dezelfde documenten in de prefill-controle "
             "van 13 nov (T6, T5) of in de lengteproef (d2k van deel A, heel).", "",
             "| test | model | documenten | P / A | slots | zonder fix | verwachting | waarom |",
             "|---|---|---|---|---:|---|---|---|"]
    for test, t in TESTS.items():
        lines.append(f"| {test} | {NAMES[t['part']]} | deel {t['part']}, bak {t['bin']} | {t['prefill']} / "
                     f"{t['decode']} | {t['slots']} | {BEFORE[test]} | {EXPECT[test]} | {t['why']} |")
    lines += ["", "V4 (Gemma 4, bak 4096 heel, 1 slot) is niet gedraaid: d4k van deel A (P 4224 / A 5120) gaf in de "
              "lengteproef met 1 slot geheugen-op op een v4-worker (`len_heel_d4k_g4c1113b-s1`, piek 23,88 GiB per chip).",
              "", "### Uitslag per test [gemeten]", "",
              "| test | P / A | docs (weggelaten) | af | lus | niet af | afgekapt | alinea's behouden | QE gem. | "
              "uitvoertokens med. | zonder fix: af / QE | lengteproef heel: af / QE | in stukken: af / QE | uitslag | "
              "zonder fix: uitslag |", "|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|---|"]
    for test, e in r["tests"].items():
        t, s, b = TESTS[test], e["test"], e["before"]
        h, p, bs = e["ref_heel"], e["ref_stukken"], e["before"]["stats"]
        lines.append(
            f"| {test} | {t['prefill']} / {t['decode']} | {e['measured']} ({len(e['dropped'])}) | "
            f"{kn(s['done'], s['docs'])} | {s['decode_loop']} | {s['not_ended']} | {s['cut']} | "
            f"{kn(s['paragraphs_kept'], s['docs'])} | {num(s['qe_mean'])} | {num(s['tokens_out_median'], 0)} | "
            f"{b['source']}: {kn(bs['done'], bs['docs'])} / {num(bs['qe_mean'])} | {kn(h['done'], h['docs'])} / "
            f"{num(h['qe_mean'])} | {kn(p['done'], p['docs'])} / {num(p['qe_mean'])} | **{e['verdict']}** "
            f"({e['done_fit']}/{e['fits']}) | {b['verdict']} ({b['done_fit']}/{b['fits']}) |")
    lines += ["", "Per document. Gelijkenis: difflib-ratio op woorden (1 is gelijk) van de vertaling nu met die zonder fix "
              "en met die in stukken van hetzelfde document.", "",
              "| test | zonder fix | beide af | alleen zonder fix af | alleen nu af | gelijkenis met zonder fix: gelijk / "
              "med. / < 0,5 | gelijkenis met in stukken: med. / < 0,5 |", "|---|---|---:|---:|---:|---|---|"]
    for test, e in r["tests"].items():
        b, sim, st = e["before"], e["before"]["similarity"], e["similarity_stukken"]
        lines.append(f"| {test} | {b['source']} | {b['both_done']} | {b['only_before_done']} | {b['only_now_done']} | "
                     f"{sim['identical']}/{sim['n']} / {num(sim['median'], 3)} / {sim['below_half']} | "
                     f"{num(st['median'], 3)} / {st['below_half']} |")
    lines += ["", "Slot-hergebruik, zoals in de prefill-controle (`wrap_schedule`): eerst ingevoegd (een leeg slot), later "
              "ingevoegd en tijdens de decode over het einde van de decode-cache (loopt rond), later ingevoegd zonder "
              "wrap. Zonder fix: dezelfde indeling van de run zonder fix (bij de lengteproef over alle documenten van "
              "de bak, zonder QE).", "",
              "| test | groep | nu: n / af / lus / niet af | nu: QE gem. | zonder fix: n / af / lus / niet af | "
              "zonder fix: QE gem. |", "|---|---|---|---:|---|---:|"]
    labels = (("first", "eerst ingevoegd"), ("later_cross", "later ingevoegd, loopt rond"),
              ("later_inside", "later ingevoegd, loopt niet rond"))
    for test, e in r["tests"].items():
        for g, label in labels:
            x, y = e[f"group_{g}"], e["before"]["groups"][g]
            lines.append(f"| {test} | {label} | {x['n']} / {x['done']} / {x['loop']} / {x['not_ended']} | "
                         f"{num(x['stats']['qe_mean'])} | {y['n']} / {y['done']} / {y['loop']} / {y['not_ended']} | "
                         f"{num(y['qe_mean'])} |")
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


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    """prefill_check.py's stages read its module-level TESTS (and report its report_md): pointed at this check's for
    the one command, and back after it"""
    argv = list(sys.argv[1:] if argv is None else argv)
    saved = pc.TESTS, pc.report_md
    pc.TESTS = TESTS
    try:
        if argv[:1] == ["translate"]:
            p = argparse.ArgumentParser(prog="fork_fix_check.py translate")
            p.add_argument("--test", required=True, choices=sorted(TESTS))
            for name in ("--segments", "--out", "--tokenizer", "--ckpt", "--work-dir"):
                p.add_argument(name, required=True)
            p.add_argument("--max-shards", type=int, default=None)
            translate(p.parse_args(argv[1:]))
        elif argv[:1] == ["report"]:
            p = argparse.ArgumentParser(prog="fork_fix_check.py report")
            for name in ("--plan", "--measured", "--units", "--ref-a", "--ref-b", "--before", "--out"):
                p.add_argument(name, required=True)
            p.add_argument("--qe", nargs="*", default=[])
            p.add_argument("--logs", default=None)
            report(p.parse_args(argv[1:]))
        else:
            pc.main(argv)
    finally:
        pc.TESTS, pc.report_md = saved


if __name__ == "__main__":
    main()
