"""Meting G4-maat2 (protocol-g4.md §24, part 1, 24 Nov 2026): two new prompts for the Gemma 4 judge, A1 (fidelity to the
source only) and J-nat (naturalness first, a clear meaning error as a veto), and two fixed combinations, A1-veto -> N1 and
A1-veto -> N2 (A1 decides when |P̄_A1(A) - P̄_A1(B)| >= 0.2, else N1 or N2), on §20's 1,000 fiction/BT pairs (the human
or original side against the machine side) and on §23's 1,438 pairs (Yeb's two samples and the 14 blind Claude
packages), beside §20's, FB.8's and §23's existing scores; a test (a) > 70% human on fiction/BT, (b) >= 65% the
error-free side on Claude's one-sided error pairs, (c) flips <= 20%, reported; a reproduction of 200 §23 pairs.

    python g4measure2.py data1 --fb-refs P --fb-prompts GLOB --fb-n1-prompts GLOB --fb-n2-prompts GLOB \\
        --pairs P --m23-prompts GLOB --out DIR                                            (w0, CPU)
    python g4measure2.py report1 --refs P --judge GLOB --metricx P --comet P --c1 P --c2 P --new GLOB --fb-refs P \\
        --fb-scores rubric=GLOB --fb-scores n1=GLOB --fb-scores n2=GLOB --data-json J --yeb1-key J \\
        --yeb1-answers T --yeb2-key J --yeb2-answers T --out report.md                    (co1, CPU)

data1 reads texts (on the VM: §20's and §23's prompts, §23's pairs) and writes the judge's prompts; report1 reads only
files without text. No command prints a text."""

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import g4measure as gm  # noqa: E402
import pairjudge  # noqa: E402

PROTOCOL = "protocol-g4.md §24"
# G4-maat2.2, letterlijk: A1, fidelity to the source only
A1_PROMPT = (
    "Je beoordeelt vertalingen uit het Engels in het Nederlands. Je krijgt de Engelse bron en twee Nederlandse "
    "vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.\n"
    "\n"
    "Oordeel alleen op trouw aan de bron: welke vertaling geeft de betekenis van de Engelse bron het volledigst en het "
    "juistst weer? Tel alleen betekenisfouten:\n"
    "- een weglating: iets uit de bron ontbreekt in de vertaling;\n"
    "- een toevoeging: de vertaling zegt iets wat niet in de bron staat;\n"
    "- een verkeerde betekenis: de vertaling zegt iets anders dan de bron, ook als een idioom of vaste uitdrukking zo is "
    "vertaald dat de betekenis verandert.\n"
    "Laat stijl, woordkeus, zinsbouw en natuurlijkheid uitdrukkelijk buiten beschouwing: een stijve of letterlijke "
    "vertaling is even goed als een vloeiende, zolang de betekenis klopt. \"gelijk\" als er geen duidelijk verschil in "
    "betekenisfouten is.\n"
    "\n"
    "Engelse bron:\n"
    "{bron}\n"
    "\n"
    "Vertaling A:\n"
    "{A}\n"
    "\n"
    "Vertaling B:\n"
    "{B}\n"
    "\n"
    "Welke vertaling is getrouwer aan de bron? Antwoord met één woord: A, B of gelijk.")
# G4-maat2.2, letterlijk: J-nat, naturalness first, an error as a veto
JNAT_PROMPT = (
    "Je beoordeelt vertalingen uit het Engels in het Nederlands. Je krijgt de Engelse bron en twee Nederlandse "
    "vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.\n"
    "\n"
    "Kies de vertaling met het natuurlijkste Nederlands: de tekst die het meest leest als tekst die oorspronkelijk in het "
    "Nederlands is geschreven, en het minst als een vertaling. Alleen als die vertaling een duidelijke betekenisfout of "
    "weglating heeft (iets uit de bron ontbreekt, of de vertaling zegt iets anders dan de bron) en de andere niet, kies je "
    "de andere. Woordkeus, stijl en zinsbouw zijn geen betekenisfout zolang de betekenis klopt. \"gelijk\" als er geen "
    "duidelijk verschil is.\n"
    "\n"
    "Engelse bron:\n"
    "{bron}\n"
    "\n"
    "Vertaling A:\n"
    "{A}\n"
    "\n"
    "Vertaling B:\n"
    "{B}\n"
    "\n"
    "Welke vertaling kies je? Antwoord met één woord: A, B of gelijk.")
NEW_PROMPTS = {"a1": A1_PROMPT, "jnat": JNAT_PROMPT}
OLD_PROMPTS = {"rubric": pairjudge.PROMPT, "n1": pairjudge.N1_PROMPT, "n2": pairjudge.N2_PROMPT}
ALL_PROMPTS = {**OLD_PROMPTS, **NEW_PROMPTS}
ORDERS = pairjudge.ORDERS
VETO = 0.2                                   # G4-maat2.3: A1 decides at |P̄_A1(A) - P̄_A1(B)| >= 0.2
FB_PAIRS = 1000
M23_PAIRS = 1438
REPRO_N = 200
REPRO_SEED = "en-nl-g4arm/0/G4-maat2/repro"
REPRO_TOL = 1e-5                             # float32 precision of the probabilities (G4-maat2.7)
TEST_A, TEST_B, TEST_C = 0.70, 0.65, 0.20    # G4-maat2.5
CLAUDE_FAULT_PAIRS = 263
NEW_JUDGES = {"A1": "a1", "J-nat": "jnat"}
COMBOS2 = {"A1-veto → N1": "n1", "A1-veto → N2": "n2"}
NEW_ROWS = tuple(NEW_JUDGES) + tuple(COMBOS2)
OLD_JUDGES = {"rubriek": "rubric", "N1": "n1", "N2": "n2"}
TEST_ROWS = NEW_ROWS + tuple(OLD_JUDGES)
FB_NAMES = {"fiction": "fictie", "bt": "BT"}


def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------- data1

def template_parts(template):
    """(head, between bron and A, between A and B, tail) of a prompt template"""
    head, rest = template.split("{bron}")
    mid1, rest = rest.split("{A}")
    mid2, tail = rest.split("{B}")
    return head, mid1, mid2, tail


def parse_prompt(text, template=pairjudge.PROMPT, what="prompt"):
    """(bron, A, B) of a prompt made with pairjudge.user_text from template; it stops unless the split is unique and
    the three texts give exactly this prompt again (G4-maat2.4)"""
    head, mid1, mid2, tail = template_parts(template)
    if not text.startswith(head) or not text.endswith(tail) or len(text) < len(head) + len(tail):
        raise SystemExit(f"{what}: not a prompt of this template")
    body = text[len(head):len(text) - len(tail)]
    if body.count(mid1) != 1 or body.count(mid2) != 1:
        raise SystemExit(f"{what}: the separators of A and B are not there exactly once")
    source, rest = body.split(mid1)
    if rest.count(mid2) != 1:
        raise SystemExit(f"{what}: the separator of B is not after that of A")
    a, b = rest.split(mid2)
    if pairjudge.user_text(source, a, b, template) != text:
        raise SystemExit(f"{what}: its texts do not give the same prompt")
    return source, a, b


def texts_of(pattern):
    out = {}
    for r in pairjudge.read_glob(pattern, ["uid", "text"]):
        if r["uid"] in out:
            raise SystemExit(f"{pattern}: {r['uid']} twice")
        out[r["uid"]] = r["text"]
    return out


def fb_prompts(refs, rubric, n1, n2):
    """The new prompts of §20's pairs (uid fb/<prompt>/<test>/<id>/<order>), their texts parsed from §20's rubric prompts
    and checked against §20's and FB.8's prompts per uid"""
    keys = [f"{r['package']}/{r['blind_id']}" for r in refs]
    if len(keys) != FB_PAIRS or len(set(keys)) != len(keys):
        raise SystemExit(f"§20's refs: {len(keys)} pairs ({len(set(keys))} distinct), not {FB_PAIRS}")
    want = {f"{k}/{o}" for k in keys for o in ORDERS}
    for name, have in (("§20's rubric", rubric), ("FB.8's N1", n1), ("FB.8's N2", n2)):
        if set(have) != want:
            raise SystemExit(f"{name} prompts: {len(want - set(have))} missing, {len(set(have) - want)} extra")
    prompts = []
    for k in keys:
        source, a, b = parse_prompt(rubric[f"{k}/ab"], what=f"{k}/ab")
        for name, template, have in (("rubric", pairjudge.PROMPT, rubric), ("n1", pairjudge.N1_PROMPT, n1),
                                     ("n2", pairjudge.N2_PROMPT, n2)):
            for uid, text, _ in pairjudge.pair_prompts(k, source, a, b, template):
                if have[uid] != text:
                    raise SystemExit(f"{uid}: the parsed texts do not give the {name} prompt of §20/FB.8")
        for name, template in NEW_PROMPTS.items():
            prompts += [{"uid": uid, "text": text}
                        for uid, text, _ in pairjudge.pair_prompts(f"fb/{name}/{k}", source, a, b, template)]
    return prompts


def m23_prompts(pairs, m23):
    """The new prompts of §23's pairs (uid <pair>/<prompt>/<order>), the rubric, N1 and N2 prompts rebuilt from the same
    texts checked against §23's per uid; and the reproduction's prompts (§23's texts, uid repro/<pair>/<prompt>/<order>)"""
    ids = [p["pair"] for p in pairs]
    if len(ids) != M23_PAIRS or len(set(ids)) != len(ids):
        raise SystemExit(f"§23's pairs: {len(ids)} ({len(set(ids))} distinct), not {M23_PAIRS}")
    want = {f"{i}/{n}/{o}" for i in ids for n in OLD_PROMPTS for o in ORDERS}
    if set(m23) != want:
        raise SystemExit(f"§23's prompts: {len(want - set(m23))} missing, {len(set(m23) - want)} extra")
    prompts = []
    for p in pairs:
        source, a, b = ((p[f] or "").strip() for f in ("source", "A", "B"))
        for name, template in OLD_PROMPTS.items():
            for uid, text, _ in pairjudge.pair_prompts(f"{p['pair']}/{name}", source, a, b, template):
                if sha(m23[uid]) != sha(text):
                    raise SystemExit(f"{uid}: not §23's prompt")
        for name, template in NEW_PROMPTS.items():
            prompts += [{"uid": uid, "text": text}
                        for uid, text, _ in pairjudge.pair_prompts(f"{p['pair']}/{name}", source, a, b, template)]
    repro = random.Random(REPRO_SEED).sample(sorted(ids), REPRO_N)
    prompts += [{"uid": f"repro/{i}/{n}/{o}", "text": m23[f"{i}/{n}/{o}"]} for i in repro for n in OLD_PROMPTS
                for o in ORDERS]
    return prompts, repro


def data1(args):
    refs = pairjudge.read_rows(args.fb_refs)
    fb = fb_prompts(refs, texts_of(args.fb_prompts), texts_of(args.fb_n1_prompts), texts_of(args.fb_n2_prompts))
    pairs = pairjudge.read_rows(args.pairs, ["pair", "source", "A", "B"])
    m23, repro = m23_prompts(pairs, texts_of(args.m23_prompts))
    prompts = fb + m23
    if len({p["uid"] for p in prompts}) != len(prompts):
        raise SystemExit("a uid twice")
    out = Path(args.out)
    workers = pairjudge.write_prompts(prompts, out)
    info = {"protocol": PROTOCOL, "fb_pairs": len(refs), "m23_pairs": len(pairs), "prompts": len(prompts),
            "fb_prompts": len(fb), "m23_prompts": len(m23) - 6 * REPRO_N, "repro_prompts": 6 * REPRO_N,
            "repro_pairs": repro, "workers": workers,
            "inputs": {"fb_refs": file_sha(args.fb_refs), "pairs": file_sha(args.pairs)}}
    (out / "data.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"MEASURE2DATA {json.dumps({k: v for k, v in info.items() if k not in ('inputs', 'repro_pairs')})}", flush=True)
    return info


# ---------------------------------------------------------------------------------------------------- choices

def combo_side(p_a1, p_other, veto=VETO):
    """A1-veto -> N (G4-maat2.3) on P̄ (or one order's probabilities): A1's side if |A - B| >= veto, else the other's"""
    if p_a1 is None or p_other is None:
        return None
    if abs(p_a1["A"] - p_a1["B"]) >= veto:
        return gm.higher(p_a1["A"], p_a1["B"])
    return gm.higher(p_other["A"], p_other["B"])


def order_view(row, order):
    """One order's normalized probabilities in the frame of order ab; None if not scored"""
    if row is None or row.get("skipped"):
        return None
    if order == "ab":
        return {"A": row["p_a"], "B": row["p_b"], "gelijk": row["p_eq"]}
    return {"A": row["p_b"], "B": row["p_a"], "gelijk": row["p_eq"]}


def flip_of_prompt(ab, ba):
    """G4-judge.5.3: the greedy answers of both orders, put back, name opposite winners; None if not scored"""
    if ab is None or ba is None or ab.get("skipped") or ba.get("skipped"):
        return None
    return {pairjudge.greedy_of(ab, "ab"), pairjudge.greedy_of(ba, "ba")} == {"A", "B"}


def flip_of_combo(a1_ab, a1_ba, other_ab, other_ba):
    """G4-maat2.5 (c): the combination on each order's probabilities picks opposite sides"""
    sides = [combo_side(order_view(a, o), order_view(x, o)) for a, x, o in ((a1_ab, other_ab, "ab"), (a1_ba, other_ba, "ba"))]
    if None in [order_view(r, o) for r, o in ((a1_ab, "ab"), (a1_ba, "ba"), (other_ab, "ab"), (other_ba, "ba"))]:
        return None
    return set(sides) == {"A", "B"}


def share(values):
    values = [v for v in values if v is not None]
    return pairjudge.share(values)


# ---------------------------------------------------------------------------------------------------- fiction/BT

def fb_views(refs, scores):
    """Per pair of §20 the per-prompt rows (both orders) and P̄: {key: {prompt: (ab, ba, P̄)}}"""
    out = {}
    for r in refs:
        k = f"{r['package']}/{r['blind_id']}"
        out[k] = {}
        for name, s in scores.items():
            ab, ba = s.get(f"{k}/ab"), s.get(f"{k}/ba")
            if ab is None or ba is None:
                raise SystemExit(f"{name} {k}: no score of both orders")
            out[k][name] = (ab, ba, pairjudge.combine(ab, ba))
    return out


def fb_signal(views, test, label):
    """Per signal on the pairs of one test (or None: both): direction (judge: P̄ credit, combination: its side), flips"""
    keys = [k for k in views if test is None or k.split("/", 1)[0] == test]
    direction, flips, verdicts, skipped = [], [], collections.Counter(), 0
    for k in keys:
        v = views[k]
        if label in COMBOS2:
            a1, other = v["a1"], v[COMBOS2[label]]
            if a1[2] is None or other[2] is None:
                skipped += 1
                continue
            side = combo_side(a1[2], other[2])
            direction.append(1.0 if side == "A" else 0.0 if side == "B" else 0.5)
            flips.append(flip_of_combo(a1[0], a1[1], other[0], other[1]))
            verdicts[{"A": "mens", "B": "machine"}.get(side, "geen keuze")] += 1
        else:
            name = {**NEW_JUDGES, **OLD_JUDGES}[label]
            ab, ba, p = v[name]
            if p is None:
                skipped += 1
                continue
            direction.append(pairjudge.direction_credit(p, "A"))
            flips.append(flip_of_prompt(ab, ba))
            verdicts[{"A": "mens", "B": "machine", "gelijk": "gelijk"}[pairjudge.verdict_of(p)]] += 1
    return {"n": len(direction), "skipped": skipped, "direction": share(direction), "flip": share(flips),
            "verdicts": dict(verdicts)}


# ---------------------------------------------------------------------------------------------------- §23's pairs

def m23_flips(refs, judge23, new):
    """Flips on §23's pairs per prompt (G4-judge.5.3) and per combination (G4-maat2.5 c)"""
    out = {}
    rows = {**judge23, **new}
    for label, name in {**NEW_JUDGES, **OLD_JUDGES}.items():
        out[label] = share(flip_of_prompt(rows.get(f"{r['pair']}/{name}/ab"), rows.get(f"{r['pair']}/{name}/ba"))
                           for r in refs)
    for label, name in COMBOS2.items():
        out[label] = share(flip_of_combo(rows.get(f"{r['pair']}/a1/ab"), rows.get(f"{r['pair']}/a1/ba"),
                                         rows.get(f"{r['pair']}/{name}/ab"), rows.get(f"{r['pair']}/{name}/ba"))
                           for r in refs)
    return out


def extend_choices(r, s, choice, margin):
    """g4measure.report_data's hook: the new judges' sides and margins, and the two combinations"""
    for label, name in NEW_JUDGES.items():
        p = s["judge"].get(name)
        choice[label] = None if p is None else gm.higher(p["A"], p["B"])
        margin[label] = None if p is None else abs(p["A"] - p["B"])
    for label, name in COMBOS2.items():
        choice[label] = combo_side(s["judge"].get("a1"), s["judge"].get(name))


def repro_check(repro, judge23, pairs):
    """G4-maat2.7 / Pool.7: the reproduction's scores are §23's (|Δp| <= REPRO_TOL, the same greedy verdict, per pair
    the same side); a deviation stops"""
    want = {f"{i}/{n}/{o}" for i in pairs for n in OLD_PROMPTS for o in ORDERS}
    if set(repro) != want:
        raise SystemExit(f"the reproduction: {len(want - set(repro))} scores missing, {len(set(repro) - want)} extra")
    worst, bad = 0.0, []
    for uid, row in sorted(repro.items()):
        old = judge23.get(uid)
        if old is None:
            raise SystemExit(f"the reproduction: no §23 score of {uid}")
        if bool(row.get("skipped")) != bool(old.get("skipped")):
            bad.append(uid)
            continue
        if row.get("skipped"):
            continue
        d = max(abs(row[c] - old[c]) for c in ("p_a", "p_b", "p_eq"))
        worst = max(worst, d)
        order = uid.rsplit("/", 1)[1]
        if d > REPRO_TOL or pairjudge.greedy_of(row, order) != pairjudge.greedy_of(old, order):
            bad.append(uid)
    sides = 0
    for i in pairs:
        for n in OLD_PROMPTS:
            new = pairjudge.combine(repro[f"{i}/{n}/ab"], repro[f"{i}/{n}/ba"])
            old = pairjudge.combine(judge23[f"{i}/{n}/ab"], judge23[f"{i}/{n}/ba"])
            if (new is None) != (old is None) or (new and gm.higher(new["A"], new["B"]) != gm.higher(old["A"], old["B"])):
                bad.append(f"{i}/{n} (side)")
            sides += 1
    result = {"pairs": len(pairs), "prompts": len(repro), "max_abs": worst, "tolerance": REPRO_TOL, "sides": sides,
              "deviations": len(bad)}
    if bad:
        raise SystemExit(f"the reproduction deviates from §23 on {len(bad)} prompts or pairs (max |Δp| {worst:.3g}): "
                         f"{bad[:5]}")
    return result


def split_new(new):
    """The new scores by kind: {prompt: {<test>/<id>/<order>: row}} of fiction/BT, §23's {uid: row}, the reproduction"""
    fb, m23, repro = {n: {} for n in NEW_PROMPTS}, {}, {}
    for uid, row in new.items():
        if uid.startswith("fb/"):
            _, name, rest = uid.split("/", 2)
            if name not in fb:
                raise SystemExit(f"{uid}: prompt {name}")
            fb[name][rest] = row
        elif uid.startswith("repro/"):
            repro[uid[len("repro/"):]] = row
        else:
            m23[uid] = row
    return fb, m23, repro


def report1(args):
    refs = pairjudge.read_rows(args.refs)
    text_columns = {"source", "A", "B", "text", "hypothesis", "reference"} & set(refs[0])
    if text_columns:
        raise SystemExit(f"{args.refs} holds text columns {sorted(text_columns)}")
    info = json.loads(Path(args.data_json).read_text())
    judge23 = pairjudge.scores_by_uid(pairjudge.read_glob(args.judge))
    new = pairjudge.scores_by_uid(pairjudge.read_glob(args.new))
    fb_new, m23_new, repro = split_new(new)
    pairs = [r["pair"] for r in refs]
    want = {f"{p}/{n}/{o}" for p in pairs for n in NEW_PROMPTS for o in ORDERS}
    if set(m23_new) != want:
        raise SystemExit(f"the new §23 scores: {len(want - set(m23_new))} missing, {len(set(m23_new) - want)} extra")
    checks = {"repro": repro_check(repro, judge23, info["repro_pairs"])}
    # fiction/BT: §20's rubric, FB.8's N1 and N2, the new prompts
    fb_refs = pairjudge.read_rows(args.fb_refs)
    fb_scores = {name: pairjudge.scores_by_uid(pairjudge.read_glob(p))
                 for name, p in pairjudge.named(args.fb_scores, "--fb-scores", tuple(OLD_PROMPTS)).items()}
    fb_scores.update(fb_new)
    views = fb_views(fb_refs, fb_scores)
    fb = {label: {t: fb_signal(views, t, label) for t in (None, "fiction", "bt")} for label in TEST_ROWS}
    fb_detail = {}
    for label, name in {**NEW_JUDGES, **OLD_JUDGES}.items():
        res, _ = pairjudge.fb_result(fb_refs, fb_scores[name], {"fiction": {}, "bt": {}}, {})
        fb_detail[label] = {t: {k: v for k, v in m.items() if k not in ("strata", "bands")} for t, m in res["tests"].items()}
    # §23's tables with the new rows, §23's scores as they are
    metricx = gm.read_scores(args.metricx, "qe_source_hypothesis")
    comet = gm.read_scores(args.comet, "comet22")
    c1 = gm.read_scores(args.c1, "p_original")
    c2 = gm.read_scores(args.c2, "p_original")
    scores = gm.scores_of(refs, judge23, metricx, comet, c1, c2)
    for p in pairs:
        for name in NEW_PROMPTS:
            scores[p]["judge"][name] = pairjudge.combine(m23_new[f"{p}/{name}/ab"], m23_new[f"{p}/{name}/ba"])
    yeb = gm.yeb_refs(json.loads(Path(args.yeb1_key).read_text()),
                      gm.parse_answers(Path(args.yeb1_answers).read_text(), allow_equal=False),
                      json.loads(Path(args.yeb2_key).read_text()),
                      gm.parse_answers(Path(args.yeb2_answers).read_text(), allow_equal=True))
    tables = gm.report_data(refs, scores, yeb, {}, added=NEW_ROWS, added_margins=tuple(NEW_JUDGES),
                            extend=extend_choices)
    flips23 = m23_flips(refs, judge23, m23_new)
    fault = tables["faults"]["Claude (fout aan één kant)"]
    if fault["QE"]["n_ref"] != CLAUDE_FAULT_PAIRS:
        raise SystemExit(f"Claude's one-sided error pairs: {fault['QE']['n_ref']}, §24 has {CLAUDE_FAULT_PAIRS}")
    test = {}
    for label in TEST_ROWS:
        a = fb[label][None]["direction"]
        b = fault[label]
        c_fb, c_23 = fb[label][None]["flip"], flips23[label]
        test[label] = {"a": a, "a_pass": a["share"] is not None and a["share"] > TEST_A,
                       "a_fiction": fb[label]["fiction"]["direction"], "a_bt": fb[label]["bt"]["direction"],
                       "b": b, "b_pass": b["share"] is not None and b["share"] >= TEST_B,
                       "c_fb": c_fb, "c_23": c_23,
                       "c_pass": c_fb["share"] is not None and c_23["share"] is not None
                       and c_fb["share"] <= TEST_C and c_23["share"] <= TEST_C}
    checks["coverage"] = {
        "new_not_scored_23": {n: sum(scores[p]["judge"][n] is None for p in pairs) for n in NEW_PROMPTS},
        "new_not_scored_fb": {n: sum(v[n][2] is None for v in views.values()) for n in NEW_PROMPTS},
        "fb_pairs": len(fb_refs), "m23_pairs": len(pairs), "data": {k: info[k] for k in ("prompts", "workers")}}
    r = {"protocol": PROTOCOL, "test": test, "fb": fb, "fb_detail": fb_detail, "flips23": flips23, "tables": tables,
         "checks": checks}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report1_text(r), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    print("MEASURE2REPORT " + json.dumps({label: {"a": t["a"]["share"], "b": t["b"]["share"], "c_fb": t["c_fb"]["share"],
                                                  "c_23": t["c_23"]["share"]} for label, t in test.items()}), flush=True)
    return r


# ---------------------------------------------------------------------------------------------------- the report

def share_cell(s):
    if not s or not s.get("n"):
        return "–"
    k = round(s["share"] * s["n"], 1)
    k = int(k) if k == int(k) else k
    return f"{gm.pct(s['share'])} ({str(k).replace('.', ',')}/{s['n']}) [{gm.pct(s['lo'], 0)}; {gm.pct(s['hi'], 0)}]"


def yes(x):
    return "ja" if x else "nee"


def table_lines(cols, rows, cell_of):
    out = ["| signaal | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for row in rows:
        out.append(f"| {gm.label(row)} | " + " | ".join(cell_of(c, row) for c in cols) + " |")
    return out


def report1_text(r):
    t, fb, tb = r["test"], r["fb"], r["tables"]
    L = [f"# G4-maat2: twee nieuwe judge-prompts, A1 en J-nat ({PROTOCOL}, deel 1)", "",
         "Gemma 4 31B-it als judge (één prefill, P(A)/P(B)/P(gelijk), beide volgordes) met twee nieuwe prompts: **A1** "
         "(alleen trouw aan de bron, stijl genegeerd) en **J-nat** (natuurlijkheid eerst, een duidelijke betekenisfout of "
         "weglating als veto); de combinaties **A1-veto → N1** en **A1-veto → N2** (A1 beslist bij |P̄_A1(A) − "
         "P̄_A1(B)| ≥ 0,2, anders N1 of N2). Rubriek, N1 en N2 ter vergelijking, uit de scores van §20, FB.8 en §23 "
         "(niet opnieuw berekend). Rapportage; beslist niets.", "",
         "## 1. Vooraf vastgelegde toets (G4-maat2.5)", "",
         "(a) richting mens/origineel op de 1.000 fictie/BT-paren, > 70%; (b) de foutloze kant op Claude's 263 paren "
         "met precies één kant `fout`, ≥ 65% (k/n, n = waar het signaal kiest); (c) omklappen bij verwisselen ≤ 20% "
         "op de fictie/BT-paren én op de §23-paren.", "",
         "| signaal | (a) fictie + BT | fictie | BT | (a) | (b) Claude-fouten | (b) | (c) omklappen fictie/BT | "
         "(c) omklappen §23 | (c) |",
         "|---|---:|---:|---:|---|---:|---|---:|---:|---|"]
    for label in TEST_ROWS:
        x = t[label]
        L.append(f"| {label} | {share_cell(x['a'])} | {share_cell(x['a_fiction'])} | {share_cell(x['a_bt'])} | "
                 f"{yes(x['a_pass'])} | {gm.cell(x['b'])} | {yes(x['b_pass'])} | {share_cell(x['c_fb'])} | "
                 f"{share_cell(x['c_23'])} | {yes(x['c_pass'])} |")
    L += ["", "Rubriek, N1 en N2 staan er ter vergelijking; de toets is vastgelegd voor de vier nieuwe rijen.", "",
          "## 2. Fictie en BT per proef", "",
          "Judge: oordeel = de grootste P̄; richting = aandeel P̄(mens) > P̄(machine) (gelijk ½). Combinatie: richting = "
          "aandeel waar ze de menselijke kant kiest (geen keuze ½); oordeel 'geen keuze' bij gelijke kansen.", "",
          "| proef | signaal | n | mens | machine | gelijk / geen keuze | richting (95%-BI) | omklappen |",
          "|---|---|---:|---:|---:|---:|---|---:|"]
    for test in ("fiction", "bt"):
        for label in TEST_ROWS:
            m = fb[label][test]
            v = m["verdicts"]
            L.append(f"| {FB_NAMES[test]} | {label} | {m['n']} | {v.get('mens', 0)} | {v.get('machine', 0)} | "
                     f"{v.get('gelijk', 0) + v.get('geen keuze', 0)} | {share_cell(m['direction'])} | "
                     f"{share_cell(m['flip'])} |")
    L += ["", "Gemiddelde P̄ en Δ = P̄(mens) − P̄(machine) per prompt (bootstrap-interval als §20):", "",
          "| proef | prompt | gem. P̄(mens) | gem. P̄(machine) | gem. P̄(gelijk) | gem. Δ [95%-BI] | inconsistent |",
          "|---|---|---:|---:|---:|---|---:|"]
    for test in ("fiction", "bt"):
        for label, d in r["fb_detail"].items():
            m = d[test]
            ci = m.get("delta_ci95") or [None, None]
            L.append(f"| {FB_NAMES[test]} | {label} | {pairjudge.num(m['mean_p_human'])} | "
                     f"{pairjudge.num(m['mean_p_machine'])} | {pairjudge.num(m['mean_p_gelijk'])} | "
                     f"{pairjudge.num(m['mean_delta'])} [{pairjudge.num(ci[0])}; {pairjudge.num(ci[1])}] | "
                     f"{share_cell(m['inconsistent'])} |")
    m = tb["main"]
    cols = list(m)
    rows = gm.SIGNALS + gm.COMBOS + NEW_ROWS + ("Claude",) + gm.REFERENCE_ROWS
    L += ["", "## 3. Hoofdtabel: dezelfde kant als Yeb en Claude (§23's paren)", "",
          "Als §23 tabel 1, met de nieuwe rijen; k/n en 95%-Wilson-interval; n = waar het signaal kiest.", ""]
    L += table_lines(cols, rows, lambda c, row: gm.cell(m[c][row]) if row in m[c] else "")
    L += ["", "n_ref: " + ", ".join(f"{c} {m[c]['QE']['n_ref']}" for c in cols) + ".", "",
          "Geen keuze (n_ref − n):", ""]
    L += table_lines(cols, rows, lambda c, row: str(m[c][row]["no_choice"]) if row in m[c] else "")
    for title, sets in (("4. Yeb per set en pakket", tb["yeb_sets"]), ("5. Claude per pakket en set", tb["claude_sets"])):
        groups = list(sets)
        L += ["", f"## {title}", "", "Alle groepen met n; bij n < 10 geen lezing.", ""]
        L += table_lines(groups, list(next(iter(sets.values()))), lambda g, row: gm.cell(sets[g][row]))
        L.append("| n_ref | " + " | ".join(str(sets[g]["QE"]["n_ref"]) for g in groups) + " |")
    L += ["", "## 6. Per marge", "", "| signaal en marge | Yeb samen | Claude alle |", "|---|---:|---:|"]
    for k in tb["margin"]["Yeb samen"]:
        L.append(f"| {k} | {gm.cell(tb['margin']['Yeb samen'][k])} | {gm.cell(tb['margin']['Claude alle'][k])} |")
    faults = tb["faults"]
    L += ["", "## 7. Betekenisfouten: kiest het signaal de kant zonder fout?", ""]
    L += table_lines(list(faults), rows, lambda g, row: gm.cell(faults[g][row]) if row in faults[g] else "")
    ns = tb["new_system"]
    L += ["", "## 8. Hoe vaak kiest een signaal het nieuwe systeem (als §23 tabel 8)", ""]
    L += table_lines(list(ns), gm.SIGNALS + gm.COMBOS + NEW_ROWS + ("Claude",), lambda c, row: gm.cell(ns[c][row]))
    L += ["", "## 9. Omklappen op de §23-paren", "", "| signaal | omklappen |", "|---|---:|"]
    for label, s in r["flips23"].items():
        L.append(f"| {label} | {share_cell(s)} |")
    rp = r["checks"]["repro"]
    L += ["", "## 10. Controles", "",
          f"- Reproductie (G4-maat2.7): {rp['pairs']} §23-paren, {rp['prompts']} prompts (rubriek, N1, N2) opnieuw "
          f"gescoord: grootste |Δp| {rp['max_abs']:.3g} (grens {rp['tolerance']:g}), {rp['deviations']} afwijkingen in "
          f"greedy oordeel of kant ({rp['sides']} paar-prompts).",
          "- De teksten van de fictie/BT-paren zijn uit §20's prompts ontleed en geven per uid precies §20's rubriek- en "
          "FB.8's N1- en N2-prompts; de §23-prompts (rubriek, N1, N2) uit `pairs.parquet` zijn per uid die van §23 "
          "(data1 stopt anders).",
          "- Dekking: " + json.dumps(r["checks"]["coverage"], ensure_ascii=False) + ".",
          "", "## 11. Wat het niet zegt", "",
          "- n van Yeb is klein (74 niet-gelijke oordelen); veel vergelijkingen zonder correctie; geen winnaar.",
          "- V1 en de pakketten met een selectie zijn met de rubriek gekozen; Claude oordeelde met de rubriek.",
          "- Fictie/BT: de herkomst is de 'waarheid', niet de kwaliteit; het BT-Engels is Gemma 4's vertaling van het "
          "origineel.",
          "- De judge is Gemma 4 31B-it, het model achter de meeste vertalingen; één formulering per prompt, zero-shot.",
          ""]
    return "\n".join(L)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("data1")
    for f in ("fb-refs", "fb-prompts", "fb-n1-prompts", "fb-n2-prompts", "pairs", "m23-prompts", "out"):
        a.add_argument(f"--{f}", required=True)
    a = sub.add_parser("report1")
    for f in ("refs", "judge", "metricx", "comet", "c1", "c2", "new", "fb-refs", "data-json", "yeb1-key", "yeb1-answers",
              "yeb2-key", "yeb2-answers", "out"):
        a.add_argument(f"--{f}", required=True)
    a.add_argument("--fb-scores", action="append", required=True, help="rubric=GLOB, n1=GLOB, n2=GLOB")
    args = parser.parse_args(argv)
    {"data1": data1, "report1": report1}[args.command](args)


if __name__ == "__main__":
    main()
