"""Meting G4-maat (protocol-g4.md §23, 24 Nov 2026): which automatic signal picks the same side of a pair of
translations as Yeb. The pairs are those of Yeb's two samples (V1: 40 pairs of the packages G4-judge-i15 and
G4-judge-anchor-i15; the mixed sample of G4-8.15: 40 pairs G4-8-s140 against G4-0) and of 14 blind packages with a
Claude judgment (rubric protocol.md §4.2.4). Per side the signals QE (MetricX-24 XXL), COMET-22 (news sets with a
reference), C1, C2, the Gemma 4 judge with the rubric, N1 and N2 (both orders) and the length; per pair the side each
signal picks, four fixed combinations K1-K4, and the agreement with Yeb and with Claude.

    python g4measure.py data --blind DIR --eval MODEL=GLOB ... --mix MODEL=PATH ... --evalsets P --val-prompts GLOB \\
        --tokenizer DIR --out DIR                                                         (w0, CPU)
    python g4measure.py report --refs P --judge GLOB --metricx P --comet P --c1 P --c2 P --yeb1-key J --yeb1-answers T \\
        --yeb2-key J --yeb2-answers T [--val-scores GLOB] [--existing-qe MODEL=GLOB ...] \\
        [--existing-comet MODEL=GLOB ...] [--existing-c1 P ...] [--existing-c2 P ...] --out report.md   (co1, CPU)

data reads the texts (on the VM) and writes the scorers' inputs (with text) and refs.parquet (no text); report reads
only files without text. No command prints a text."""

import argparse
import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pairjudge  # noqa: E402

PROTOCOL = "protocol-g4.md §23"
BASE = "base"
# package: (the Claude file (first round), pairs, the base system, what chose the pairs)
PACKAGES = {
    "G4-7": ("review-74l/judgments.jsonl", 100, "G4-0", ""),
    "G4-7-tg": ("review-74l/judgments.jsonl", 100, "TG", ""),
    "G4-bon-i15c2q": ("review/judgments.jsonl", 100, "G4-0", "C2 + QE"),
    "G4-judge-i15": ("review/judgments.jsonl", 100, "G4-0", "rubriek + QE"),
    "G4-judge-G4-0": ("review/judgments.jsonl", 100, "G4-0", "rubriek + QE"),
    "G4-judge-i15-vs-qe": ("review/judgments.jsonl", 100, "i15-qe", "rubriek + QE tegen laagste QE"),
    "G4-judge-anchor-i15": ("review/judgments.jsonl", 100, "G4-0", "rubriek + QE"),
    "G4-8": ("review/judge-judgments.jsonl", 100, "G4-0", ""),
    "G4-8-tg": ("review/judgments.jsonl", 100, "TG", ""),
    "G4-8-replica": ("review/judgments.jsonl", 100, "G4-0", ""),
    "G4-8-romans-bt": ("review/judgments.jsonl", 98, "G4-0", ""),
    "G4-8-V3": ("review/judgments.jsonl", 100, "G4-0", "rubriek + QE"),
    "G4-9-V3N1": ("review/judgments.jsonl", 100, "G4-0", "N1 + QE"),
    "RB3": ("review-exploratory/judgments.jsonl", 100, "TG", ""),
}
SECOND = {"G4-8": "review/second-reviewer-all.jsonl"}       # a second Claude reviewer, only for Claude against Claude
FIXED = ("G4-7", "G4-7-tg", "G4-8", "G4-8-tg", "RB3")       # the fixed 100 blind passages: blind_id = the eval row
VAL = ("G4-7", "G4-7-tg", "G4-bon-i15c2q", "RB3")            # §17's validation packages of the 14
NOT_EVAL = ("G4-8-V3", "G4-9-V3N1")                          # their anchor is G4-8-data's plan, not $G/eval
GROUPS = {"alle": tuple(PACKAGES), "gemma": tuple(p for p in PACKAGES if PACKAGES[p][2] != "TG"),
          "zonder selectie": ("G4-7", "G4-8", "G4-8-replica", "G4-8-romans-bt")}
EXPECTED = {"pairs": 1398, "noneq": 824, "gemma_noneq": 600, "unselected_noneq": 191}   # [read 24 Nov, labels only]
YEB1 = "steekproef-yeb"
YEB2 = "steekproef-yeb-g4-8-mix"
YEB2_FILE = "steekproef-yeb-g4-8-mix.md"
YEB1_PACKAGES = ("G4-judge-i15", "G4-judge-anchor-i15")
MIX_EVAL_SETS = ("flan", "wmt24pp", "flores", "ntrex", "idiom")
MIX_NEW_SETS = ("novels", "bt")
MIX_MODELS = {"G4-0": "base", "G4-8-s140": "G4-8-s140"}     # the sample's key name -> the --eval/--mix name
NEWS = ("wmt24pp", "flores", "ntrex")                        # COMET-22 with the test set's reference
PROMPTS = {"rubric": pairjudge.PROMPT, "n1": pairjudge.N1_PROMPT, "n2": pairjudge.N2_PROMPT}
VERDICT = {"A beter": "A", "B beter": "B", "gelijk": "gelijk"}
SIDES = ("A", "B")
MARGIN = 0.2                                                 # K1, K2 and the margin table (G4-maat.5, .6.5)


def norm(text):
    """Whitespace normalized: two texts are the same if this is"""
    return " ".join((text or "").split())


def sha(text):
    return hashlib.sha256((text or "").encode()).hexdigest()


def read_jsonl(path):
    return pairjudge.read_jsonl(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------- data

def package_pairs(blind):
    """[(pair row with text, ref row without text)] of the 14 packages, the Claude labels checked"""
    out = []
    for pkg, (review, n, base_system, selected) in PACKAGES.items():
        d = Path(blind) / pkg
        inputs = pairjudge.by_id(read_jsonl(d / "judge-input.jsonl"), f"{pkg}/judge-input")
        key = pairjudge.by_id(read_jsonl(d / "key.jsonl"), f"{pkg}/key")
        claude = pairjudge.by_id(read_jsonl(d / review), f"{pkg}/{review}")
        second = pairjudge.by_id(read_jsonl(d / SECOND[pkg]), f"{pkg}/{SECOND[pkg]}") if pkg in SECOND else None
        if not (set(inputs) == set(key) == set(claude)) or len(inputs) != n or (second and set(second) != set(key)):
            raise SystemExit(f"{pkg}: the files do not hold the same {n} blind ids")
        for b in sorted(inputs):
            x, k, c = inputs[b], key[b], claude[b]
            if c["oordeel"] not in VERDICT or c.get("fout_A") not in ("ja", "nee") or c.get("fout_B") not in ("ja", "nee"):
                raise SystemExit(f"{pkg}/{b}: oordeel {c['oordeel']!r}, fout {c.get('fout_A')!r}/{c.get('fout_B')!r}")
            if (k["A"] == BASE) == (k["B"] == BASE):
                raise SystemExit(f"{pkg}/{b}: not one side {BASE!r} in the key ({k['A']}, {k['B']})")
            if second and second[b]["oordeel"] not in VERDICT:
                raise SystemExit(f"{pkg}/{b}: second reviewer {second[b]['oordeel']!r}")
            pair = f"{pkg}/{b}"
            text = {"pair": pair, "source": x["source"], "A": x["A"], "B": x["B"]}
            ref = {"pair": pair, "origin": "package", "package": pkg, "blind_id": b, "n": None, "set": k["set"],
                   "item": k["item"], "key_a": k["A"], "key_b": k["B"], "base_side": "A" if k["A"] == BASE else "B",
                   "base_system": base_system, "selected_on": selected, "claude": VERDICT[c["oordeel"]],
                   "claude_sure": c.get("zekerheid") == "zeker", "claude_fout_a": c["fout_A"] == "ja",
                   "claude_fout_b": c["fout_B"] == "ja",
                   "claude2": VERDICT[second[b]["oordeel"]] if second else None,
                   "steekproef_n": k.get("steekproef_n")}
            out.append((text, ref))
    noneq = [r for _, r in out if r["claude"] != "gelijk"]
    counts = {"pairs": len(out), "noneq": len(noneq),
              "gemma_noneq": sum(r["package"] in GROUPS["gemma"] for r in noneq),
              "unselected_noneq": sum(r["package"] in GROUPS["zonder selectie"] for r in noneq)}
    if counts != EXPECTED:
        raise SystemExit(f"the packages give {counts}, protocol §23 has {EXPECTED}")
    return out


def read_pairs(paths):
    """{id: row} of collected pairs (evalsets.py collect) of one model"""
    import g4mixsample

    return g4mixsample.read_pairs(paths)


def named(specs, what):
    out = {}
    for s in specs:
        name, _, value = s.partition("=")
        if not name or not value or name in out:
            raise SystemExit(f"{what} {s!r}: NAME=VALUE, each name once")
        out[name] = value
    return out


def mix_pairs(key, eval_pairs, mix, sample_md):
    """[(pair row with text, ref row without text)] of the mixed sample, rebuilt as g4mixsample.pick and checked byte for
    byte against the sample file Yeb read"""
    import g4mixsample

    rows, blocks = [], []
    for k in sorted(key["pairs"], key=lambda p: p["n"]):
        s, i = k["set"], k["item"]
        if s in MIX_NEW_SETS:
            src = mix
        elif s in MIX_EVAL_SETS:
            src = eval_pairs
        else:
            raise SystemExit(f"mix {k['n']}: set {s!r}")
        sides = {}
        for side in SIDES:
            model = MIX_MODELS.get(k[side])
            if model is None or model not in src or i not in src[model]:
                raise SystemExit(f"mix {k['n']}: no {k[side]} pair for {i}")
            sides[side] = src[model][i]
        if {k["A"], k["B"]} != set(MIX_MODELS) or (k["g48"] == "A") != (k["A"] == "G4-8-s140"):
            raise SystemExit(f"mix {k['n']}: its key names {k['A']}, {k['B']}, g48 {k['g48']}")
        source = sides["A"]["source"]
        if sides["B"]["source"] != source:
            raise SystemExit(f"mix {k['n']}: the two models' sources differ")
        a, b = (sides[x]["hypothesis"] or "" for x in SIDES)
        blocks.append({"n": k["n"], "source": source, "A_text": a, "B_text": b})
        pair = f"mix/{k['n']}"
        rows.append(({"pair": pair, "source": source, "A": a, "B": b},
                     {"pair": pair, "origin": "mix", "package": None, "blind_id": None, "n": k["n"], "set": s,
                      "item": i, "key_a": k["A"], "key_b": k["B"], "base_side": "A" if k["A"] == "G4-0" else "B",
                      "base_system": "G4-0", "selected_on": "", "claude": None, "claude_sure": None,
                      "claude_fout_a": None, "claude_fout_b": None, "claude2": None, "steekproef_n": None}))
    if [b["n"] for b in blocks] != list(range(1, 41)):
        raise SystemExit(f"the mixed sample's key has numbers {[b['n'] for b in blocks]}")
    md = Path(sample_md).read_text(encoding="utf-8")
    m = re.match(r"# .*\(40 blinde paren, (.+)\)\n", md)
    if not m:
        raise SystemExit(f"{sample_md}: no title with its date")
    if g4mixsample.sample_text(blocks, m.group(1)) != md:
        raise SystemExit(f"{sample_md}: the rebuilt pairs are not the file Yeb read")
    return rows


def link_shared(pkg_rows, mix_rows):
    """The mixed sample's pairs that are a package pair with the same texts (G4-maat.2): ref['shared'], ref['flip']"""
    by_pair = {r["pair"]: (t, r) for t, r in pkg_rows}
    romans = {r["steekproef_n"]: r["pair"] for _, r in pkg_rows if r["package"] == "G4-8-romans-bt" and r["steekproef_n"]}
    counts = collections.Counter()
    for t, r in mix_rows:
        r["shared"], r["flip"] = None, None
        if r["set"] in MIX_NEW_SETS:
            target = romans.get(r["n"])
        elif r["set"] == "idiom":
            target = f"G4-8/{r['item']}"
            target = target if target in by_pair else None
        else:
            continue
        if target is None:
            counts["not_in_a_package"] += 1
            continue
        pt, pr = by_pair[target]
        # the package's key names G4-0 "base"; the sample's "G4-0"
        system = {side: ("G4-0" if pr[f"key_{side.lower()}"] == BASE else pr[f"key_{side.lower()}"]) for side in SIDES}
        if set(system.values()) != {r["key_a"], r["key_b"]}:
            raise SystemExit(f"{r['pair']} and {target}: other systems ({system}, {r['key_a']}, {r['key_b']})")
        flip = system["A"] != r["key_a"]
        theirs = (pt["B"], pt["A"]) if flip else (pt["A"], pt["B"])
        same = norm(pt["source"]) == norm(t["source"]) and (norm(theirs[0]), norm(theirs[1])) == (norm(t["A"]), norm(t["B"]))
        if (pr["item"] if r["set"] in MIX_NEW_SETS else pr["blind_id"]) != r["item"]:
            same = False
        if not same:
            counts["text_differs"] += 1
            continue
        r["shared"], r["flip"] = target, flip
        counts["shared_" + r["set"]] += 1
    for _, r in pkg_rows:
        r["shared"], r["flip"] = None, None
    return dict(counts)


def reference_of(t, r, evalsets):
    """The test set's reference of a news pair (COMET-22, G4-maat.3.2), its source checked; None elsewhere"""
    if r["set"] not in NEWS:
        return None
    row = evalsets.get(r["blind_id"]) if r["package"] in FIXED else evalsets.get(r["item"])
    if row is None or not row["reference"]:
        raise SystemExit(f"{r['pair']}: no reference row in the evalsets ({r['item']})")
    if row["set"] == "blind":
        if json.loads(row["meta"]).get("blind_set") != r["set"]:
            raise SystemExit(f"{r['pair']}: the blind row {r['blind_id']} is not {r['set']}")
    elif row["set"] != r["set"]:
        raise SystemExit(f"{r['pair']}: the row {r['item']} is of set {row['set']}")
    if norm(row["source"]) != norm(t["source"]):
        raise SystemExit(f"{r['pair']}: the source differs from the evalsets row's")
    return row["reference"]


def eval_match(t, r, eval_pairs):
    """{side: "model|id"} where the side's text is (normalized) an existing evaluation output (G4-maat.6.7c)"""
    out = {}
    for side in SIDES:
        k = r[f"key_{side.lower()}"]
        if r["origin"] == "mix":
            if r["set"] in MIX_NEW_SETS:
                continue
            model, i = MIX_MODELS[k], r["item"]
        else:
            if r["set"] in MIX_NEW_SETS or r["package"] in NOT_EVAL:
                continue
            if k == BASE:
                if r["base_system"] != "G4-0":
                    continue
                model = "base"
            elif k in eval_pairs:
                model = k
            else:
                continue
            i = r["blind_id"] if r["package"] in FIXED else r["item"]
        row = eval_pairs.get(model, {}).get(i)
        out[side] = f"{model}|{i}" if row is not None and norm(row["hypothesis"]) == norm(t[side]) else ""
    return out


def data(args, counter=None):
    import g4arm
    import pyarrow.parquet as pq

    eval_pairs = {m: read_pairs(sorted(glob.glob(p))) for m, p in named(args.eval, "--eval").items()}
    mix = {m: read_pairs([p]) for m, p in named(args.mix, "--mix").items()}
    for need, have in ((("base", "G4-8-s140"), eval_pairs), (("base", "G4-8-s140"), mix)):
        if not set(need) <= set(have):
            raise SystemExit(f"--eval and --mix need {need}, have {sorted(have)}")
    blind = Path(args.blind)
    pkg_rows = package_pairs(blind)
    key2 = json.loads((blind / YEB2 / "key.json").read_text())
    mix_rows = mix_pairs(key2, eval_pairs, mix, blind / YEB2 / YEB2_FILE)
    key1 = json.loads((blind / YEB1 / "key-map.json").read_text())
    pairs_of = {r["pair"] for _, r in pkg_rows}
    if len(key1) != 40 or any(f"{k['package']}/{k['blind_id']}" not in pairs_of or k["package"] not in YEB1_PACKAGES
                              for k in key1):
        raise SystemExit("V1's key-map: not 40 pairs of G4-judge-i15 and G4-judge-anchor-i15")
    shared = link_shared(pkg_rows, mix_rows)
    rows = pkg_rows + mix_rows
    ev = {r["id"]: r for r in pq.read_table(args.evalsets, columns=["id", "set", "source", "reference", "meta"]).to_pylist()}
    counter = counter or g4arm.TokenCounter(args.tokenizer)
    texts = [t[f] for t, _ in rows for f in ("source", "A", "B")]
    lengths = counter.lengths([(x or "").strip() for x in texts])
    val = {}
    for p in sorted(glob.glob(args.val_prompts)):
        for x in pairjudge.read_rows(p):
            val[x["uid"]] = sha(x["text"])
    prompts, qe, comet, nat, refs = [], [], [], [], []
    checked = collections.Counter()
    for j, (t, r) in enumerate(rows):
        src, a, b = ((t[f] or "").strip() for f in ("source", "A", "B"))
        for name, template in PROMPTS.items():
            for uid, text, _ in pairjudge.pair_prompts(f"{r['pair']}/{name}", src, a, b, template):
                prompts.append({"uid": uid, "text": text})
                if name == "rubric" and r["package"] in VAL:
                    old = f"{r['pair']}/{uid.rsplit('/', 1)[1]}"
                    if val.get(old) != sha(text):
                        raise SystemExit(f"{uid}: not §17's prompt {old}")
                    checked["val_prompts"] += 1
        reference = reference_of(t, r, ev)
        for side, text in (("A", a), ("B", b)):
            sid = f"{r['pair']}/{side}"
            qe.append({"id": sid, "source": src, "hypothesis": text})
            nat.append({"model": "g4measure", "set": r["set"], "id": sid, "text": text})
            if reference is not None:
                comet.append({"id": sid, "source": src, "hypothesis": text, "reference": reference.strip()})
        match = eval_match(t, r, eval_pairs)
        ls, la, lb = lengths[3 * j:3 * j + 3]
        refs.append({**r, "tokens_source": ls, "tokens_a": la, "tokens_b": lb, "chars_a": len(a), "chars_b": len(b),
                     "empty_a": not a, "empty_b": not b, "same_text": norm(a) == norm(b),
                     "has_reference": reference is not None, "eval_a": match.get("A"), "eval_b": match.get("B")})
    if checked["val_prompts"] != 2 * 100 * len(VAL):
        raise SystemExit(f"{checked['val_prompts']} of §17's prompts checked, not {2 * 100 * len(VAL)}")
    out = Path(args.out)
    workers = pairjudge.write_prompts(prompts, out)
    pairjudge.write_rows([{"pair": t["pair"], "source": t["source"], "A": t["A"], "B": t["B"]} for t, _ in rows],
                         out / "pairs.parquet")
    pairjudge.write_rows(refs, out / "refs.parquet")
    pairjudge.write_rows(qe, out / "qe-in.parquet")
    pairjudge.write_rows(comet, out / "comet-in.parquet")
    pairjudge.write_rows(nat, out / "texts.parquet")
    eval_found = collections.Counter()
    for r in refs:
        for side in ("a", "b"):
            v = r[f"eval_{side}"]
            eval_found["matched" if v else ("differs" if v == "" else "none")] += 1
    info = {"protocol": PROTOCOL, "pairs": len(rows), "packages": dict(collections.Counter(r["package"] or "mix" for r in refs)),
            "sets": dict(collections.Counter(f"{r['origin']}/{r['set']}" for r in refs)), "prompts": len(prompts),
            "workers": workers, "qe_rows": len(qe), "comet_rows": len(comet), "texts": len(nat), "shared": shared,
            "val_prompts_checked": checked["val_prompts"], "eval_sides": dict(eval_found),
            "empty_sides": sum(r["empty_a"] + r["empty_b"] for r in refs), "same_text": sum(r["same_text"] for r in refs),
            "inputs": {"evalsets": file_sha(args.evalsets), "sample_file": file_sha(blind / YEB2 / YEB2_FILE),
                       "yeb2_key": file_sha(blind / YEB2 / "key.json"), "yeb1_key": file_sha(blind / YEB1 / "key-map.json")}}
    (out / "data.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"MEASUREDATA {json.dumps({k: v for k, v in info.items() if k != 'inputs'})}", flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- choices

def lower(x, y):
    """The side with the lower value; None at a tie or a missing value"""
    if x is None or y is None or x == y:
        return None
    return "A" if x < y else "B"


def higher(x, y):
    if x is None or y is None or x == y:
        return None
    return "A" if x > y else "B"


def other(side):
    return {"A": "B", "B": "A"}.get(side)


SIGNALS = ("QE", "COMET-22", "C1", "C2", "rubriek", "N1", "N2", "lengte (korter)")
JUDGES = {"rubriek": "rubric", "N1": "n1", "N2": "n2"}
COMBOS = ("K1", "K2", "K3", "K4")
REFERENCE_ROWS = ("nieuw systeem", "positie A")
COMBO_LABELS = {"K1": "K1: QE, bij |ΔQE| < 0,2 N1", "K2": "K2: QE, bij |ΔQE| < 0,2 rubriek",
                "K3": "K3: meerderheid QE / rubriek / N1", "K4": "K4: rubriek = N1, anders QE"}


def pair_signals(r, s):
    """{signal: side or None} and {signal: margin} of one pair from its scores s (the frame of the pair)"""
    qa, qb = s["qe"]
    choice = {"QE": lower(qa, qb), "COMET-22": higher(*s["comet"]), "C1": higher(*s["c1"]), "C2": higher(*s["c2"]),
              "lengte (korter)": lower(r["tokens_a"], r["tokens_b"])}
    margin = {"QE": abs(qa - qb) if qa is not None and qb is not None else None}
    for label, name in JUDGES.items():
        p = s["judge"].get(name)
        choice[label] = None if p is None else higher(p["A"], p["B"])
        margin[label] = None if p is None else abs(p["A"] - p["B"])
    small = margin["QE"] is not None and margin["QE"] < MARGIN
    choice["K1"] = choice["N1"] if small else choice["QE"]
    choice["K2"] = choice["rubriek"] if small else choice["QE"]
    votes = collections.Counter(choice[x] for x in ("QE", "rubriek", "N1") if choice[x])
    top = [side for side, n in votes.items() if n >= 2]
    choice["K3"] = top[0] if top else None
    agree = choice["rubriek"] is not None and choice["rubriek"] == choice["N1"]
    choice["K4"] = choice["rubriek"] if agree else choice["QE"]
    choice["nieuw systeem"] = other(r["base_side"])
    choice["positie A"] = "A"
    return choice, margin


def scores_of(refs, judge, metricx, comet, c1, c2):
    """{pair: {"qe": (A, B), "comet": (A, B), "c1": .., "c2": .., "judge": {prompt: P̄ or None}}}"""
    out = {}
    for r in refs:
        p = r["pair"]
        side = lambda d, x: d.get(f"{p}/{x}")  # noqa: E731
        js = {}
        for name in JUDGES.values():
            ab, ba = judge.get(f"{p}/{name}/ab"), judge.get(f"{p}/{name}/ba")
            if ab is None or ba is None:
                raise SystemExit(f"{p}/{name}: no judge score for both orders")
            js[name] = pairjudge.combine(ab, ba)
        out[p] = {"qe": (side(metricx, "A"), side(metricx, "B")), "comet": (side(comet, "A"), side(comet, "B")),
                  "c1": (side(c1, "A"), side(c1, "B")), "c2": (side(c2, "A"), side(c2, "B")), "judge": js}
    return out


# ---------------------------------------------------------------------------------------------------- references

def parse_answers(text, allow_equal):
    """{n: (verdict A/B/=, the side Yeb named 'fout' or None)} of an answers.txt ("1 B, 2 A, 20 B fout A, ...")"""
    out = {}
    for part in [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]:
        m = re.fullmatch(r"(\d+)\s+(A|B|=)(?:\s+fout\s+(A|B))?", part)
        if not m or (m.group(2) == "=" and not allow_equal):
            raise SystemExit(f"answers: {part!r}")
        n = int(m.group(1))
        if n in out:
            raise SystemExit(f"answers: {n} twice")
        out[n] = (m.group(2), m.group(3))
    if sorted(out) != list(range(1, 41)):
        raise SystemExit(f"answers: numbers {sorted(out)}, not 1..40")
    return out


def yeb_refs(key1, answers1, key2, answers2):
    """[{pair, sample, verdict, fout, n}] of Yeb's 80 judgments in the frame of the pair"""
    out = []
    for k in key1:
        v, f = answers1[k["n"]]
        out.append({"pair": f"{k['package']}/{k['blind_id']}", "sample": "V1", "n": k["n"], "verdict": v, "fout": f})
    for k in key2["pairs"]:
        v, f = answers2[k["n"]]
        out.append({"pair": f"mix/{k['n']}", "sample": "gemengd", "n": k["n"], "verdict": v, "fout": f})
    return out


def agreement(items):
    """items: [(reference side, signal side or None)] on the pairs where the reference picks a side"""
    n_ref = len(items)
    chosen = [(x, y) for x, y in items if y is not None]
    k = sum(x == y for x, y in chosen)
    lo, hi = pairjudge.wilson(k, len(chosen))
    return {"n_ref": n_ref, "n": len(chosen), "k": k, "share": k / len(chosen) if chosen else None, "lo": lo, "hi": hi,
            "no_choice": n_ref - len(chosen)}


def claude_in_frame(r, refs_by_pair):
    """Claude's verdict on a pair in its frame: its own for a package pair, the package's (flipped) for a shared one"""
    if r["origin"] == "package":
        return r["claude"]
    if not r.get("shared"):
        return None
    v = refs_by_pair[r["shared"]]["claude"]
    return other(v) if r["flip"] and v in SIDES else v


YEB_GROUPS = {"novels": "romans", "bt": "BT", "flan": "FLAN", "idiom": "idioom", "wmt24pp": "nieuws",
              "flores": "nieuws", "ntrex": "nieuws"}
V1_SETS = {"flan": "FLAN", "wmt24pp": "WMT24++"}
ALL_ROWS = SIGNALS + COMBOS + ("Claude",) + REFERENCE_ROWS


def label(row):
    return COMBO_LABELS.get(row, row)


def report_data(refs, scores, yeb, extra):
    by_pair = {r["pair"]: r for r in refs}
    choices, margins = {}, {}
    for r in refs:
        choices[r["pair"]], margins[r["pair"]] = pair_signals(r, scores[r["pair"]])
        choices[r["pair"]]["Claude"] = (lambda v: v if v in SIDES else None)(claude_in_frame(r, by_pair))

    def table(pairs_refs, rows=ALL_ROWS):
        return {row: agreement([(ref, choices[p][row]) for p, ref in pairs_refs]) for row in rows}

    yeb_noneq = [y for y in yeb if y["verdict"] in SIDES]
    cols = {"Yeb V1": [(y["pair"], y["verdict"]) for y in yeb_noneq if y["sample"] == "V1"],
            "Yeb gemengd": [(y["pair"], y["verdict"]) for y in yeb_noneq if y["sample"] == "gemengd"],
            "Yeb samen": [(y["pair"], y["verdict"]) for y in yeb_noneq]}
    claude_rows = [r for r in refs if r["origin"] == "package" and r["claude"] in SIDES]
    for g, pkgs in GROUPS.items():
        cols[f"Claude {g}"] = [(r["pair"], r["claude"]) for r in claude_rows if r["package"] in pkgs]
    main = {c: table(v, ALL_ROWS if c.startswith("Yeb") else SIGNALS + COMBOS + REFERENCE_ROWS) for c, v in cols.items()}

    per_set = {}
    for y in yeb_noneq:
        r = by_pair[y["pair"]]
        g = YEB_GROUPS[r["set"]]
        per_set.setdefault(f"V1 {V1_SETS[r['set']]}" if y["sample"] == "V1" else f"gemengd {g}", []).append(
            (y["pair"], y["verdict"]))
        if y["sample"] == "V1":
            per_set.setdefault("V1 " + ("judgeq" if r["package"] == "G4-judge-i15" else "janchor"), []).append(
                (y["pair"], y["verdict"]))
        per_set.setdefault(f"samen {g}", []).append((y["pair"], y["verdict"]))
    yeb_sets = {g: table(v) for g, v in sorted(per_set.items())}
    claude_sets = {}
    for r in claude_rows:
        claude_sets.setdefault(f"pakket {r['package']}", []).append((r["pair"], r["claude"]))
        claude_sets.setdefault(f"set {r['set']}", []).append((r["pair"], r["claude"]))
    claude_sets = {g: table(v, SIGNALS + COMBOS + REFERENCE_ROWS) for g, v in claude_sets.items()}

    # Claude against Yeb on the shared pairs; Claude against Claude on G4-8
    cross = collections.Counter()
    for y in yeb:
        c = claude_in_frame(by_pair[y["pair"]], by_pair)
        if c is not None:
            cross[(c, y["verdict"])] += 1
    second = [(r["claude"], r["claude2"]) for r in refs if r["claude2"] is not None]
    cc = agreement([(x, y if y in SIDES else None) for x, y in second if x in SIDES])

    # per margin
    def margin_table(pairs_refs):
        out = {}
        for row in ("QE", "rubriek", "N1", "N2"):
            for lo_hi, test in (("< 0,2", lambda m: m < MARGIN), ("≥ 0,2", lambda m: m >= MARGIN)):
                items = [(ref, choices[p][row]) for p, ref in pairs_refs
                         if margins[p][row] is not None and test(margins[p][row])]
                out[f"{row} {lo_hi}"] = agreement(items)
        return out
    margin = {"Yeb samen": margin_table(cols["Yeb samen"]), "Claude alle": margin_table(cols["Claude alle"])}

    # meaning errors: the side without the error
    def fault_table(items, rows=ALL_ROWS):
        return {row: agreement([(side, choices[p][row]) for p, side in items]) for row in rows}
    yeb_fault = [(y["pair"], other(y["fout"])) for y in yeb if y["fout"]]
    claude_fault = [(r["pair"], "B" if r["claude_fout_a"] else "A") for r in refs
                    if r["origin"] == "package" and r["claude_fout_a"] != r["claude_fout_b"]]
    # Claude against its own error labels says nothing (G4-maat.10): no Claude row there
    faults = {"Yeb (fout aan één kant)": fault_table(yeb_fault),
              "Claude (fout aan één kant)": fault_table(claude_fault, SIGNALS + COMBOS + REFERENCE_ROWS)}

    # G4-maat.10, added after the result: how often a signal picks the side that is not the base (the new system)
    def new_table(pairs_refs):
        return {row: agreement([(choices[p]["nieuw systeem"], choices[p][row]) for p, _ in pairs_refs])
                for row in SIGNALS + COMBOS + ("Claude",)}
    new_system = {c: new_table(v) for c, v in cols.items()}

    coverage = {
        "judge_not_scored": {n: sum(scores[r["pair"]]["judge"][n] is None for r in refs) for n in JUDGES.values()},
        "qe_missing": sum(None in scores[r["pair"]]["qe"] for r in refs),
        "comet_pairs": sum(r["has_reference"] for r in refs),
        "empty_sides": sum(r["empty_a"] + r["empty_b"] for r in refs), "same_text": sum(r["same_text"] for r in refs),
        "shared_yeb_claude": sum(1 for y in yeb if claude_in_frame(by_pair[y["pair"]], by_pair) is not None),
    }
    return {"main": main, "yeb_sets": yeb_sets, "claude_sets": claude_sets,
            "cross": {f"{c}|{v}": n for (c, v), n in sorted(cross.items())}, "claude_vs_claude": cc,
            "margin": margin, "faults": faults, "new_system": new_system, "coverage": coverage, **extra}


# ---------------------------------------------------------------------------------------------------- checks

def max_abs(pairs):
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    return {"n": len(pairs), "max_abs": max((abs(x - y) for x, y in pairs), default=None)}


def checks(refs, scores, judge, metricx, comet, c1, c2, args):
    """G4-maat.6.7 b-e, reported"""
    out = {}
    if args.val_scores:
        val = pairjudge.scores_by_uid(pairjudge.read_glob(args.val_scores))
        got, same = [], 0
        for r in refs:
            if r["package"] in VAL:
                old = pairjudge.combine(val.get(f"{r['pair']}/ab"), val.get(f"{r['pair']}/ba"))
                new = scores[r["pair"]]["judge"]["rubric"]
                if old is None or new is None:
                    continue
                got.append((old["A"], new["A"]))
                same += higher(old["A"], old["B"]) == higher(new["A"], new["B"])
        out["rubric_vs_17"] = {**max_abs(got), "same_direction": same}
    existing = {}
    for what, specs, column in (("qe", args.existing_qe, "qe_source_hypothesis"), ("comet", args.existing_comet, "comet22")):
        for model, pattern in named(specs or [], f"--existing-{what}").items():
            for row in pairjudge.read_glob(pattern, ["id", column]):
                existing.setdefault(what, {})[f"{model}|{row['id']}"] = row[column]
    for what, specs in (("c1", args.existing_c1), ("c2", args.existing_c2)):
        # MODELS=PATH: only those models of that classify run, so that no two runs give the same model
        for models, p in named(specs or [], f"--existing-{what}").items():
            wanted = models.split(",")
            for row in pairjudge.read_rows(p, ["model", "id", "p_original"]):
                if row["model"] in wanted:
                    model = "base" if row["model"] == "G4-0" else row["model"]
                    existing.setdefault(what, {})[f"{model}|{row['id']}"] = row["p_original"]
    fresh = {"qe": metricx, "comet": comet, "c1": c1, "c2": c2}
    for what, old in existing.items():
        got, labels = [], 0
        for r in refs:
            for side in SIDES:
                ref = r[f"eval_{side.lower()}"]
                new = fresh[what].get(f"{r['pair']}/{side}")
                if ref and ref in old and new is not None and old[ref] is not None:
                    got.append((old[ref], new))
                    labels += what in ("c1", "c2") and (old[ref] >= 0.5) == (new >= 0.5)
        out[f"{what}_vs_existing"] = {**max_abs(got), **({"same_label": labels} if what in ("c1", "c2") else {})}
    # (d) the shared mixed pairs: the same scores as the package pair
    dup = collections.defaultdict(list)
    for r in refs:
        if r.get("shared"):
            mine, theirs = scores[r["pair"]], scores[r["shared"]]
            for what in ("qe", "c1", "c2"):
                a, b = mine[what]
                ta, tb = theirs[what][::-1] if r["flip"] else theirs[what]
                dup[what] += [(a, ta), (b, tb)]
            for name in JUDGES.values():
                p, q = mine["judge"][name], theirs["judge"][name]
                if p is not None and q is not None:
                    dup[name].append((p["A"], q["B"] if r["flip"] else q["A"]))
    out["shared_same_scores"] = {k: max_abs(v) for k, v in dup.items()}
    return out


# ---------------------------------------------------------------------------------------------------- the report

def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}".replace(".", ",") + "%"


def cell(a):
    if not a["n"]:
        return "–"
    return f"{pct(a['share'])} ({a['k']}/{a['n']}) [{pct(a['lo'], 0)}; {pct(a['hi'], 0)}]"


def report_text(r):
    m = r["main"]
    cols = list(m)
    out = [f"# G4-maat: welk signaal kiest dezelfde kant als Yeb ({PROTOCOL})", "",
           "Per signaal: het aandeel paren waar het signaal dezelfde kant kiest als de referentie (Yeb of de blinde "
           "Claude-beoordelaar), op de paren waar de referentie A of B kiest en het signaal een keuze maakt; k/n en een "
           "95%-Wilson-interval. Rapportage; beslist niets. n is klein (Yeb: 74 niet-gelijke oordelen).", "",
           "## 1. Hoofdtabel", "",
           "| signaal | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for row in ALL_ROWS:
        out.append(f"| {label(row)} | " + " | ".join(cell(m[c][row]) if row in m[c] else "" for c in cols) + " |")
    out += ["", "n_ref (niet-gelijke oordelen van de referentie): " +
            ", ".join(f"{c} {m[c]['QE']['n_ref']}" for c in cols) + ".", "",
            "Geen keuze (gelijke waarden, geen referentie voor COMET-22, niet gescoord; per kolom n_ref − n):", "",
            "| signaal | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for row in ALL_ROWS:
        out.append(f"| {label(row)} | " + " | ".join(str(m[c][row]["no_choice"]) if row in m[c] else "" for c in cols) + " |")
    for title, sets in (("2. Yeb per set en pakket", r["yeb_sets"]), ("3. Claude per pakket en set", r["claude_sets"])):
        groups = list(sets)
        out += ["", f"## {title}", "", "Alle groepen met n; bij n < 10 geen lezing.", "",
                "| signaal | " + " | ".join(groups) + " |", "|---|" + "---:|" * len(groups)]
        for row in next(iter(sets.values())):
            out.append(f"| {label(row)} | " + " | ".join(cell(sets[g][row]) for g in groups) + " |")
        out.append("| n_ref | " + " | ".join(str(sets[g]["QE"]["n_ref"]) for g in groups) + " |")
    out += ["", "## 4. Claude tegen Yeb (gedeelde paren) en Claude tegen Claude", "",
            "| Claude \\ Yeb | A | B | = |", "|---|---:|---:|---:|"]
    for c in ("A", "B", "gelijk"):
        out.append(f"| {c} | " + " | ".join(str(r["cross"].get(f"{c}|{v}", 0)) for v in ("A", "B", "=")) + " |")
    cc = r["claude_vs_claude"]
    out += ["", f"G4-8, eerste Claude-ronde tegen de tweede beoordelaar, op de paren waar de eerste een kant kiest: "
            f"{cell(cc)}; de tweede zegt daar {cc['no_choice']} keer gelijk.", "",
            "## 5. Per marge", "", "| signaal en marge | Yeb samen | Claude alle |", "|---|---:|---:|"]
    for k in r["margin"]["Yeb samen"]:
        out.append(f"| {k} | {cell(r['margin']['Yeb samen'][k])} | {cell(r['margin']['Claude alle'][k])} |")
    out += ["", "## 6. Betekenisfouten: kiest het signaal de kant zonder fout?", "",
            "| signaal | " + " | ".join(r["faults"]) + " |", "|---|" + "---:|" * len(r["faults"])]
    for row in ALL_ROWS:
        out.append(f"| {label(row)} | " + " | ".join(cell(r["faults"][g][row]) if row in r["faults"][g] else ""
                                                    for g in r["faults"]) + " |")
    out += ["", "## 7. Controles en dekking", "", "| controle | waarde |", "|---|---|"]
    for k, v in {**r["coverage"], **r["checks"]}.items():
        out.append(f"| {k} | {json.dumps(v, ensure_ascii=False)} |")
    ns = r["new_system"]
    out += ["", "## 8. Na de uitslag toegevoegd (niet vooraf vastgelegd, §23 G4-maat.10): hoe vaak kiest een signaal "
            "het nieuwe systeem?", "",
            "Het aandeel paren (waar de referentie een kant kiest en het signaal kiest) waar het signaal de kant kiest "
            "die niet `base` is (V1: de judge-keuze; gemengd: G4-8-s140). Een hoog aandeel op V1 betekent dat de "
            "overeenstemming met Yeb daar vooral die van de selectie is.", "",
            "| signaal | " + " | ".join(ns) + " |", "|---|" + "---:|" * len(ns)]
    for row in SIGNALS + COMBOS + ("Claude",):
        out.append(f"| {label(row)} | " + " | ".join(cell(ns[c][row]) for c in ns) + " |")
    out += ["", "## 9. Wat het niet zegt", "",
            "- n is klein; veel vergelijkingen zonder correctie; geen drempel, geen winnaar (§23 G4-maat.7).",
            "- V1 is gekozen met de rubriek-judge en QE ≤ QE₀: rubriek en QE vallen daar grotendeels samen met "
            "'nieuw systeem'. Zo ook de pakketten met een selectie (§23 G4-maat.2).",
            "- Claude oordeelde met dezelfde rubriek als de rubriek-judge.",
            "- Yeb is één beoordelaar; zijn oordeel is hier de referentie, geen grondwaarheid.", ""]
    return "\n".join(out)


def read_scores(path, column, key="id"):
    return {r[key]: r[column] for r in pairjudge.read_rows(path, [key, column])}


def report(args):
    refs = pairjudge.read_rows(args.refs)
    text_columns = {"source", "A", "B", "text", "hypothesis", "reference"} & set(refs[0])
    if text_columns:
        raise SystemExit(f"{args.refs} holds text columns {sorted(text_columns)}")
    judge = pairjudge.scores_by_uid(pairjudge.read_glob(args.judge))
    metricx = read_scores(args.metricx, "qe_source_hypothesis")
    comet = read_scores(args.comet, "comet22")
    c1 = read_scores(args.c1, "p_original")
    c2 = read_scores(args.c2, "p_original")
    expected = {r["pair"] for r in refs}
    want = {f"{p}/{n}/{o}" for p in expected for n in JUDGES.values() for o in pairjudge.ORDERS}
    if set(judge) != want:
        raise SystemExit(f"the judge scores: {len(want - set(judge))} missing, {len(set(judge) - want)} extra")
    sides = {f"{p}/{s}" for p in expected for s in SIDES}
    for name, d in (("metricx", metricx), ("c1", c1), ("c2", c2)):
        if set(d) != sides:
            raise SystemExit(f"{name}: {len(sides - set(d))} sides missing, {len(set(d) - sides)} extra")
    news = {f"{r['pair']}/{s}" for r in refs if r["has_reference"] for s in SIDES}
    if set(comet) != news:
        raise SystemExit(f"comet: {len(news - set(comet))} missing, {len(set(comet) - news)} extra")
    scores = scores_of(refs, judge, metricx, comet, c1, c2)
    key1 = json.loads(Path(args.yeb1_key).read_text())
    key2 = json.loads(Path(args.yeb2_key).read_text())
    a1 = parse_answers(Path(args.yeb1_answers).read_text(), allow_equal=False)
    a2 = parse_answers(Path(args.yeb2_answers).read_text(), allow_equal=True)
    yeb = yeb_refs(key1, a1, key2, a2)
    if not {y["pair"] for y in yeb} <= expected:
        raise SystemExit("Yeb's pairs are not all in refs.parquet")
    r = report_data(refs, scores, yeb, {"protocol": PROTOCOL})
    r["checks"] = checks(refs, scores, judge, metricx, comet, c1, c2, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text(r), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    m = r["main"]
    print(f"MEASUREREPORT {json.dumps({c: {row: m[c][row]['share'] for row in ('QE', 'rubriek', 'N1', 'N2', 'C1', 'C2')} for c in m})}",
          flush=True)
    return r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("data")
    a.add_argument("--blind", required=True)
    a.add_argument("--eval", action="append", required=True, help="MODEL=GLOB of evalsets.py collect's pairs")
    a.add_argument("--mix", action="append", required=True, help="MODEL=PATH of the mixed sample's new pairs")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--val-prompts", required=True, help="GLOB of §17's prompts-w<k>.parquet")
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("report")
    for f in ("refs", "judge", "metricx", "comet", "c1", "c2", "yeb1-key", "yeb1-answers", "yeb2-key", "yeb2-answers", "out"):
        a.add_argument(f"--{f}", required=True)
    a.add_argument("--val-scores", default=None)
    for f in ("existing-qe", "existing-comet", "existing-c1", "existing-c2"):
        a.add_argument(f"--{f}", action="append", default=[])
    args = parser.parse_args(argv)
    {"data": data, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
