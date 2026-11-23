#!/usr/bin/env python3
"""Meting G4-judge (protocol-g4.md §17, 23 Nov 2026): Gemma 4 31B-it as a zero-shot judge of two translations of one
English source (the rubric of protocol.md §4.2.4 with a one-word answer A, B or gelijk), its probabilities read from
the logits of one prefill, every pair in both orders. Step 1 validates the judge against the Claude judgments of five
blind packages and tests the thresholds D1-D3; step 2 (only if they hold) chooses among the 8 samples of G4-0 and
G4-7-i15 on G4-bon's search part with the judge against the anchor G4-0 greedy (judge, judgeq), measures the choice
with C1 and the samples' existing scores, and builds a blind package.

    python pairjudge.py valdata --blind DIR --out DIR                         (CPU)
    python pairjudge.py score --prompts P --out gs://DIR --ckpt C --tokenizer T --work-dir W   (TPU, one worker)
    python pairjudge.py valreport --refs R --scores GLOB --out report.md      (CPU)
    python pairjudge.py bonpairs --model NAME=SAMPLES_DIR,CHOICE ... --anchor DIR --evalsets E --split S --out DIR
    python pairjudge.py select --model NAME --samples DIR --choice C --scores GLOB --evalsets E --split S --out DIR
    python pairjudge.py bonreport --evalsets E --split S --refqe R --anchor DIR --predictions P \\
        --probe-predictions PP --bon-predictions BP --model NAME=GREEDY,T10S1,BONSELECT,JSELECT ... --out report.md
    python pairjudge.py blind --model NAME --select DIR --anchor DIR --evalsets E --split S --out DIR

A prompt row is (uid, text): text is the user turn; score puts the model's chat turn around it. The texts stay in the
bucket and on the VM: no command prints a text, the reports hold counts and scores only."""

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
for _path in (HERE.parent / "flan-nl", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PROTOCOL = "protocol-g4.md §17"
MODEL = "gemma4-31b"                              # translate_models.yml: Gemma 4 31B-it, its turn and checkpoint
CHIPS = 4
# G4-judge.3, letterlijk; {bron}, {A}, {B} gestript ingevuld
PROMPT = (
    "Je beoordeelt vertalingen uit het Engels in het Nederlands. Je krijgt de Engelse bron en twee Nederlandse "
    "vertalingen, A en B. Je weet niet hoe of door wie ze zijn gemaakt; probeer dat niet te raden.\n"
    "\n"
    "Oordeel eerst op juistheid: de vertaling zonder betekenisfout, of met de minst ernstige, is beter. Een "
    "betekenisfout is:\n"
    "- een weglating: iets uit de bron ontbreekt in de vertaling;\n"
    "- een toevoeging: de vertaling zegt iets wat niet in de bron staat;\n"
    "- een verkeerde betekenis: de vertaling zegt iets anders dan de bron;\n"
    "- een misplaatste uitdrukking: een idioom of vaste uitdrukking uit de bron is letterlijk vertaald waar dat in "
    "het Nederlands niet werkt, of vertaald met een Nederlandse uitdrukking die iets anders betekent.\n"
    "Woordkeus, stijl en zinsbouw zijn geen betekenisfout zolang de betekenis klopt.\n"
    "\n"
    "Zijn ze even juist, oordeel dan op natuurlijk, oorspronkelijk klinkend Nederlands: de vertaling die het meest "
    "leest als tekst die in het Nederlands is geschreven en niet als vertaling. \"gelijk\" als er op beide punten "
    "geen duidelijk verschil is.\n"
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
    "Welke vertaling is beter? Antwoord met één woord: A, B of gelijk.")
ANSWERS = ("A", "B", "gelijk")
# G4-judge.2 [measured in tokenizer.json @ 842da37]; score checks them against the tokenizer on the VM
ANSWER_TOKENS = {"A": ("A", "▁A"), "B": ("B", "▁B"), "gelijk": ("gelijk", "▁gelijk")}
ANSWER_IDS = {"A": (236776, 562), "B": (236799, 603), "gelijk": (161224, 190506)}
PREFILL_LENGTHS = (512, 1024, 1536, 2048, 3072, 4096)
MAX_DECODE = 16                                   # the engine's decode limit; nothing is decoded
ORDERS = ("ab", "ba")
WORKERS = 4
TOP_K = 5

# step 1: the packages, their Claude judgments (fresh subagent, rubric §4.2.4) and the extra references
PACKAGES = {"G4-7": "review-74l/judgments.jsonl", "G4-7-tg": "review-74l/judgments.jsonl",
            "G4-bon-i15c2q": "review/judgments.jsonl", "RB3": "review-exploratory/judgments.jsonl",
            "RB2": "review-exploratory/judgments.jsonl"}
FINAL = {"RB3": "review-exploratory/final-judgments.jsonl", "RB2": "review-exploratory/final-judgments.jsonl"}
CLOSE = ("G4-7", "G4-bon-i15c2q")                 # two Gemma 4 outputs close to each other (D2)
EXPECTED = {"pairs": 500, "noneq": 340, "noneq_sure": 125, "close_noneq": 94}   # [read 23 Nov, labels only]
D1, D2, D3 = 0.70, 0.60, 0.20
VERDICT = {"A beter": "A", "B beter": "B", "gelijk": "gelijk"}

# step 2
BON_MODELS = ("G4-0", "G4-7-i15")
SAMPLE_NAMES = tuple(f"t10s{k}" for k in range(1, 9))
JUDGE_SELECTIONS = ("judge", "judgeq")
OUTPUTS = ("greedy", "t10s1", "qe", "c2q", "judge", "judgeq")
CONTRASTS = (("judge", "qe"), ("judgeq", "qe"), ("judge", "c2q"), ("judgeq", "c2q"))
QE_TOL = 1e-9
BOOT_NAME = "G4-judge"                            # g4bon.paired_ci's seed: en-nl-g4arm/0/G4-bon/boot/G4-judge/<cmp>
BLIND_PER_SET = 50
BLIND_TARGET = "G4-7-i15-judgeq"


# ---------------------------------------------------------------------------------------------------- the prompt

def user_text(source, a, b):
    """The prompt with the three texts in one pass over the template: a text holding "{A}" stays as it is"""
    values = {"bron": source.strip(), "A": a.strip(), "B": b.strip()}
    return re.sub(r"\{(bron|A|B)\}", lambda m: values[m.group(1)], PROMPT)


def worker_of(uid, workers=WORKERS):
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "little") % workers


def pair_prompts(key, source, first, second):
    """The two prompts of a pair (first is A in order ab): [(uid, text, order)]"""
    return [(f"{key}/ab", user_text(source, first, second), "ab"), (f"{key}/ba", user_text(source, second, first), "ba")]


def write_rows(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_rows(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def read_glob(pattern, columns=None):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no file matches {pattern}")
    return [r for p in paths for r in read_rows(p, columns)]


def write_prompts(rows, out, workers=WORKERS):
    """prompts-w<k>.parquet (uid, text): every row on the worker of its uid; {worker: rows}"""
    per = collections.defaultdict(list)
    for r in rows:
        per[worker_of(r["uid"], workers)].append({"uid": r["uid"], "text": r["text"]})
    for k in range(workers):
        write_rows(per.get(k, []), Path(out) / f"prompts-w{k}.parquet")
    return {f"w{k}": len(per.get(k, [])) for k in range(workers)}


# ---------------------------------------------------------------------------------------------------- step 1: data

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(rows, what):
    out = {}
    for r in rows:
        if r["blind_id"] in out:
            raise SystemExit(f"{what}: {r['blind_id']} twice")
        out[r["blind_id"]] = r
    return out


def valdata(args):
    """The prompts of the five packages in both orders (G4-judge.5) and the references without text"""
    blind = Path(args.blind)
    prompts, refs = [], []
    for package, review in PACKAGES.items():
        d = blind / package
        inputs = by_id(read_jsonl(d / "judge-input.jsonl"), f"{package}/judge-input")
        key = by_id(read_jsonl(d / "key.jsonl"), f"{package}/key")
        claude = by_id(read_jsonl(d / review), f"{package}/{review}")
        final = by_id(read_jsonl(d / FINAL[package]), f"{package}/final") if package in FINAL else None
        if not (set(inputs) == set(key) == set(claude)) or len(inputs) != 100 or (final and set(final) != set(key)):
            raise SystemExit(f"{package}: the files do not hold the same 100 blind ids")
        for b in sorted(inputs):
            x, k, c = inputs[b], key[b], claude[b]
            if c["oordeel"] not in VERDICT or c["zekerheid"] not in ("zeker", "twijfel"):
                raise SystemExit(f"{package}/{b}: oordeel {c['oordeel']!r}, zekerheid {c['zekerheid']!r}")
            if not all((x[f] or "").strip() for f in ("source", "A", "B")):
                raise SystemExit(f"{package}/{b}: an empty source or translation")
            for uid, text, order in pair_prompts(f"{package}/{b}", x["source"], x["A"], x["B"]):
                prompts.append({"uid": uid, "text": text})
            ref = {"package": package, "blind_id": b, "set": k["set"], "key_a": k["A"], "key_b": k["B"],
                   "claude": VERDICT[c["oordeel"]], "sure": c["zekerheid"] == "zeker",
                   "fout_a": c["fout_A"] == "ja", "fout_b": c["fout_B"] == "ja",
                   "final": VERDICT[final[b]["oordeel"]] if final else None}
            refs.append(ref)
    counts = {"pairs": len(refs), "noneq": sum(r["claude"] != "gelijk" for r in refs),
              "noneq_sure": sum(r["claude"] != "gelijk" and r["sure"] for r in refs),
              "close_noneq": sum(r["claude"] != "gelijk" and r["package"] in CLOSE for r in refs)}
    if counts != EXPECTED:
        raise SystemExit(f"the packages give {counts}, protocol §17 has {EXPECTED}")
    out = Path(args.out)
    workers = write_prompts(prompts, out)
    write_rows(refs, out / "refs.parquet")
    info = {"protocol": PROTOCOL, "counts": counts, "prompts": len(prompts), "workers": workers,
            "per_package": dict(collections.Counter(r["package"] for r in refs))}
    (out / "valdata.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"VALDATA {json.dumps(info)}", flush=True)


# ---------------------------------------------------------------------------------------------------- the scorer

def logsumexp(values):
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


def answer_view(logp_a, logp_b, logp_eq):
    """The mass and the normalized probabilities of the three answers from their log probabilities"""
    lse = logsumexp([logp_a, logp_b, logp_eq])
    return {"mass": math.exp(lse), "p_a": math.exp(logp_a - lse), "p_b": math.exp(logp_b - lse),
            "p_eq": math.exp(logp_eq - lse)}


def bucket_of(n, lengths=PREFILL_LENGTHS):
    return next((length for length in lengths if n <= length), None)


def check_answer_tokens(tokenizer):
    for answer, tokens in ANSWER_TOKENS.items():
        got = tuple(tokenizer.convert_ids_to_tokens(list(ANSWER_IDS[answer])))
        if got != tokens:
            raise SystemExit(f"the ids of {answer} are {got} in this tokenizer, not {tokens}")


def check_template(tokenizer, turn, texts):
    """Our prompt ids are those of the model's chat template (one user turn, the generation prompt)"""
    for text in texts:
        expected = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                                 add_generation_prompt=True)
        if isinstance(expected, dict) or hasattr(expected, "keys"):
            expected = expected["input_ids"]
        ours = tokenizer(turn["prefix"] + text + turn["suffix"], add_special_tokens=False)["input_ids"]
        if list(expected) != list(ours):
            raise SystemExit("the prompt ids differ from the chat template's")


def score_logits(logits, top_k=TOP_K):
    """{logp_a, logp_b, logp_eq, top_ids, top_logp} of one position's logits (numpy, float32)"""
    import numpy as np

    x = logits.astype(np.float64)
    m = float(x.max())
    lse = m + math.log(float(np.exp(x - m).sum()))
    logp = x - lse
    out = {f"logp_{k}": logsumexp([float(logp[i]) for i in ANSWER_IDS[a]])
           for k, a in (("a", "A"), ("b", "B"), ("eq", "gelijk"))}
    top = np.argpartition(-logp, top_k)[:top_k]
    top = top[np.argsort(-logp[top])]
    out["top_ids"] = [int(i) for i in top]
    out["top_logp"] = [round(float(logp[i]), 4) for i in top]
    return out


def shards_of(order, size):
    return [(f"shard-{n:05d}.parquet", order[i:i + size]) for n, i in enumerate(range(0, len(order), size))]


def score(args):
    """Every prompt of --prompts through one prefill of the judge, in shards (sorted by length) written to --out; a
    restart skips the shards there"""
    import numpy as np
    import translate

    rows = read_rows(args.prompts)
    if len({r["uid"] for r in rows}) != len(rows):
        raise SystemExit(f"{args.prompts}: a uid twice")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    check_answer_tokens(tokenizer)
    turn = translate.MODELS[MODEL]["turn"]
    if not turn.get("check"):
        raise SystemExit(f"{MODEL}'s turn is not checked against the template")
    check_template(tokenizer, turn, [r["text"] for r in rows[:2] + rows[-2:]])
    ids = tokenizer([turn["prefix"] + r["text"] + turn["suffix"] for r in rows], add_special_tokens=False)["input_ids"]
    order = sorted(range(len(rows)), key=lambda i: (len(ids[i]), rows[i]["uid"]))
    shards = shards_of(order, args.shard_size)
    client = translate.gcs_client()
    done = translate.done_shards(args.out, client)
    lengths = [len(x) for x in ids]
    print(f"PROMPTS {json.dumps({'rows': len(rows), 'shards': len(shards), 'done': len(done & {n for n, _ in shards}), 'tokens_median': statistics.median(lengths) if lengths else 0, 'tokens_max': max(lengths, default=0), 'too_long': sum(bucket_of(n) is None for n in lengths), 'buckets': dict(sorted(collections.Counter(bucket_of(n) or 0 for n in lengths).items()))})}",
          flush=True)
    if all(n in done for n, _ in shards):
        return
    import jax
    import jax.numpy as jnp

    if (jax.device_count(), jax.process_count()) != (CHIPS, 1):
        raise SystemExit(f"needs one process with {CHIPS} chips, has {jax.process_count()} with {jax.device_count()}")
    print(f"MAXTEXT {json.dumps({'model': MODEL, 'ckpt': args.ckpt, **translate.maxtext_source(), 'patches': translate.patch_maxtext()})}",
          flush=True)
    started = time.time()
    ns = argparse.Namespace(model=MODEL, ckpt=args.ckpt, work_dir=args.work_dir, tokenizer=args.tokenizer, copies=None)
    engine, params = translate.make_engine(ns, {"prefill_lengths": list(PREFILL_LENGTHS), "slots": CHIPS,
                                                "max_decode": MAX_DECODE}, CHIPS)
    print(f"engine ready in {time.time() - started:.0f} s", flush=True)
    local = Path(args.work_dir) / "scores"
    local.mkdir(parents=True, exist_ok=True)
    compiled = set()
    for name, members in shards:
        if name in done:
            continue
        t0 = time.time()
        out = []
        for i in members:
            n = len(ids[i])
            rec = {"uid": rows[i]["uid"], "prompt_tokens": n, "bucket": bucket_of(n), "skipped": None}
            if rec["bucket"] is None:
                out.append({**rec, "skipped": "too_long"})
                continue
            padded = np.zeros(rec["bucket"], np.int32)
            padded[:n] = ids[i]
            t1 = time.time()
            prefix, _ = engine.prefill(params=params, padded_tokens=jnp.asarray(padded), true_length=n)
            logits = np.asarray(jax.device_get(prefix["logits"]), dtype=np.float32).reshape(-1)
            first = int(np.asarray(jax.device_get(prefix["tokens"])).reshape(-1)[0])
            if rec["bucket"] not in compiled:
                compiled.add(rec["bucket"])
                print(f"COMPILE {json.dumps({'bucket': rec['bucket'], 'seconds': round(time.time() - t1, 1)})}", flush=True)
            s = score_logits(logits)
            if first != s["top_ids"][0]:
                raise SystemExit(f"{rec['uid']}: the engine's first token {first} is not the top token {s['top_ids'][0]}")
            out.append({**rec, **s, **answer_view(s["logp_a"], s["logp_b"], s["logp_eq"]), "first_token": first})
        path = local / name
        write_rows(out, path)
        translate.upload(path, f"{args.out}/{name}", client)
        path.unlink()
        secs = time.time() - t0
        print(f"SHARD {json.dumps({'shard': name, 'rows': len(out), 'seconds': round(secs, 1), 'per_s': round(len(out) / max(secs, 1e-9), 2), 'tokens_max': max(r['prompt_tokens'] for r in out)})}",
              flush=True)
    print(f"SCORED {json.dumps({'rows': len(rows), 'shards': len(shards), 'seconds': round(time.time() - started)})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- the combination

def scores_by_uid(rows):
    out = {}
    for r in rows:
        if r["uid"] in out:
            raise SystemExit(f"the scores hold {r['uid']} twice")
        out[r["uid"]] = r
    return out


def combine(ab, ba):
    """P̄ in the frame of order ab (G4-judge.4): {"A", "B", "gelijk"}; None if a prompt was not scored"""
    if ab is None or ba is None or ab.get("skipped") or ba.get("skipped"):
        return None
    return {"A": 0.5 * (ab["p_a"] + ba["p_b"]), "B": 0.5 * (ab["p_b"] + ba["p_a"]),
            "gelijk": 0.5 * (ab["p_eq"] + ba["p_eq"])}


def verdict_of(p):
    """The largest of A, B, gelijk; gelijk at an exact tie with it, gelijk too when A and B tie at the top"""
    top = max(p.values())
    if p["gelijk"] == top or p["A"] == p["B"] == top:
        return "gelijk"
    return "A" if p["A"] == top else "B"


def greedy_of(row, order):
    """The greedy answer of one prompt in the frame of order ab"""
    v = verdict_of({"A": row["p_a"], "B": row["p_b"], "gelijk": row["p_eq"]})
    if order == "ba" and v != "gelijk":
        v = "B" if v == "A" else "A"
    return v


def direction_credit(p, winner):
    other = "B" if winner == "A" else "A"
    return 1.0 if p[winner] > p[other] else 0.5 if p[winner] == p[other] else 0.0


def wilson(k, n, z=1.959964):
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def share(values):
    values = list(values)
    if not values:
        return {"n": 0, "share": None, "lo": None, "hi": None}
    k = sum(values)
    lo, hi = wilson(k, len(values))
    return {"n": len(values), "share": k / len(values), "lo": lo, "hi": hi}


def kappa(a, b, classes=ANSWERS):
    """Cohen's kappa of two label lists and their cross table {a: {b: n}}"""
    n = len(a)
    table = {x: {y: 0 for y in classes} for x in classes}
    for x, y in zip(a, b):
        table[x][y] += 1
    if n == 0:
        return None, table
    po = sum(table[x][x] for x in classes) / n
    pe = sum(sum(table[x].values()) * sum(table[y][x] for y in classes) for x in classes) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None, table


# ---------------------------------------------------------------------------------------------------- step 1: report

def val_pairs(refs, scores):
    """Per reference row the judge's view: P̄, verdict, greedy per order, mass per order; skipped pairs apart"""
    out, skipped = [], []
    for r in refs:
        key = f"{r['package']}/{r['blind_id']}"
        ab, ba = scores.get(f"{key}/ab"), scores.get(f"{key}/ba")
        if ab is None or ba is None:
            raise SystemExit(f"{key}: no score of {'ab' if ab is None else 'ba'}")
        p = combine(ab, ba)
        if p is None:
            skipped.append(key)
            continue
        out.append({**r, "p": p, "judge": verdict_of(p), "greedy_ab": greedy_of(ab, "ab"),
                    "greedy_ba": greedy_of(ba, "ba"), "mass_ab": ab["mass"], "mass_ba": ba["mass"],
                    "pa_ab": ab["p_a"], "pb_ab": ab["p_b"], "pa_ba": ba["p_a"], "pb_ba": ba["p_b"],
                    "top_ab": ab["top_ids"][0], "top_ba": ba["top_ids"][0],
                    "raw_greedy_ab": verdict_of({"A": ab["p_a"], "B": ab["p_b"], "gelijk": ab["p_eq"]}),
                    "raw_greedy_ba": verdict_of({"A": ba["p_a"], "B": ba["p_b"], "gelijk": ba["p_eq"]})})
    return out, skipped


def val_measures(rows, reference="claude"):
    """The measures of G4-judge.5 over rows (one package or several), against the reference column"""
    rows = [r for r in rows if r[reference] is not None]
    noneq = [r for r in rows if r[reference] != "gelijk"]
    sure = [r for r in noneq if r["sure"]]
    k3, table = kappa([r["judge"] for r in rows], [r[reference] for r in rows])
    k3s, _ = kappa([r["judge"] for r in rows if r["sure"]], [r[reference] for r in rows if r["sure"]])
    out = {"n": len(rows),
           "direction_noneq": share(direction_credit(r["p"], r[reference]) for r in noneq),
           "direction_noneq_sure": share(direction_credit(r["p"], r[reference]) for r in sure),
           "agree3": share(r["judge"] == r[reference] for r in rows),
           "agree3_sure": share(r["judge"] == r[reference] for r in rows if r["sure"]),
           "kappa": k3, "kappa_sure": k3s, "table": table,
           "flip": share({r["greedy_ab"], r["greedy_ba"]} == {"A", "B"} for r in rows),
           "inconsistent": share(r["greedy_ab"] != r["greedy_ba"] for r in rows),
           "judge_verdicts": dict(collections.Counter(r["judge"] for r in rows)),
           "reference_verdicts": dict(collections.Counter(r[reference] for r in rows))}
    return out


def position_measures(rows):
    out = {}
    for o in ORDERS:
        raw = [r[f"raw_greedy_{o}"] for r in rows]
        mass = [r[f"mass_{o}"] for r in rows]
        out[o] = {"first_position": sum(v == "A" for v in raw) / len(rows), "gelijk": sum(v == "gelijk" for v in raw) / len(rows),
                  "second_position": sum(v == "B" for v in raw) / len(rows),
                  "mean_pa_minus_pb": statistics.fmean(r[f"pa_{o}"] - r[f"pb_{o}"] for r in rows),
                  "mass_median": statistics.median(mass), "mass_below_half": sum(m < 0.5 for m in mass)}
    return out


def fault_measure(rows):
    """5a: pairs where Claude gave exactly one side a meaning error; the judge's direction for the side without"""
    picked = [r for r in rows if r["fout_a"] != r["fout_b"]]
    return share(direction_credit(r["p"], "B" if r["fout_a"] else "A") for r in picked)


def system_preference(rows):
    """5b: per package the share of non-gelijk verdicts for the system that is not 'base', judge and Claude"""
    out = {}
    for package in PACKAGES:
        rs = [r for r in rows if r["package"] == package]
        if not rs:
            continue
        other = sorted(({r["key_a"] for r in rs} | {r["key_b"] for r in rs}) - {"base"})

        def for_other(v, r):
            return (v == "A" and r["key_a"] != "base") or (v == "B" and r["key_b"] != "base")
        j = [r for r in rs if r["judge"] != "gelijk"]
        c = [r for r in rs if r["claude"] != "gelijk"]
        out[package] = {"system": other, "judge": share(for_other(r["judge"], r) for r in j),
                        "claude": share(for_other(r["claude"], r) for r in c)}
    return out


def gate(rows):
    d1 = share(direction_credit(r["p"], r["claude"]) for r in rows if r["claude"] != "gelijk" and r["sure"])
    d2 = share(direction_credit(r["p"], r["claude"]) for r in rows if r["claude"] != "gelijk" and r["package"] in CLOSE)
    d3 = share({r["greedy_ab"], r["greedy_ba"]} == {"A", "B"} for r in rows)
    ok = {"D1": d1["share"] is not None and d1["share"] >= D1, "D2": d2["share"] is not None and d2["share"] >= D2,
          "D3": d3["share"] is not None and d3["share"] <= D3}
    return {"D1": {**d1, "threshold": D1, "holds": ok["D1"]}, "D2": {**d2, "threshold": D2, "holds": ok["D2"]},
            "D3": {**d3, "threshold": D3, "holds": ok["D3"]}, "pass": all(ok.values())}


def valreport(args):
    refs = read_rows(args.refs)
    scores = scores_by_uid(read_glob(args.scores))
    if len(scores) != 2 * len(refs):
        raise SystemExit(f"{len(scores)} scores for {len(refs)} pairs (expected {2 * len(refs)})")
    rows, skipped = val_pairs(refs, scores)
    result = {"protocol": PROTOCOL, "pairs": len(refs), "scored_pairs": len(rows), "skipped_pairs": skipped,
              "gate": gate(rows), "all": val_measures(rows), "packages": {}, "sets": {},
              "position": position_measures(rows), "fault": fault_measure(rows),
              "system_preference": system_preference(rows),
              "top_tokens": {o: dict(collections.Counter(r[f"top_{o}"] for r in rows).most_common(8)) for o in ORDERS},
              "final": {}}
    for package in PACKAGES:
        result["packages"][package] = val_measures([r for r in rows if r["package"] == package])
    for s in sorted({r["set"] for r in rows}):
        result["sets"][s] = val_measures([r for r in rows if r["set"] == s])
    for package in FINAL:
        result["final"][package] = val_measures([r for r in rows if r["package"] == package], "final")
    result["close"] = val_measures([r for r in rows if r["package"] in CLOSE])
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    Path(args.out).write_text(valreport_text(result), encoding="utf-8")
    g = result["gate"]
    print(f"VALREPORT {json.dumps({k: round(g[k]['share'], 4) if g[k]['share'] is not None else None for k in ('D1', 'D2', 'D3')})}",
          flush=True)
    print(f"GATE {'pass' if g['pass'] else 'fail'} {json.dumps({k: g[k]['holds'] for k in ('D1', 'D2', 'D3')})}", flush=True)


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}%".replace(".", ",")


def num(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def share_text(s):
    if s["share"] is None:
        return "– (n = 0)"
    return f"{pct(s['share'])} [{pct(s['lo'])}; {pct(s['hi'])}] (n = {s['n']})"


def valreport_text(r):
    g = r["gate"]
    L = ["# Meting G4-judge, stap 1: Gemma 4 31B-it als judge tegen de Claude-oordelen (rapporteren; drempel voor stap 2)",
         "", f"Gemaakt door `pairjudge.py valreport` ({PROTOCOL}). {r['pairs']} paren uit vijf blinde pakketten, elk in "
         "beide volgordes; P̄ = gemiddelde van de genormaliseerde kansen van A, B en gelijk over de twee volgordes (in het "
         "kader van het pakket). Richting = teken van P̄(A beter) − P̄(B beter). Intervallen: 95% Wilson. Referentie: "
         "het oordeel van de verse Claude-subagent (rubriek §4.2.4), vóór elke vervanging van twijfelgevallen.", "",
         f"Gescoorde paren: {r['scored_pairs']} van {r['pairs']}" + (f"; niet gescoord (prompt > 4.096 tokens): "
                                                                     f"{len(r['skipped_pairs'])}" if r['skipped_pairs'] else "") + ".", "",
         "## Drempel voor stap 2 (G4-judge.6)", "", "| toets | maat | waarde (95%-BI) | grens | gehaald |",
         "|---|---|---|---:|---|",
         f"| D1 | richting, zekere niet-gelijke paren, vijf pakketten | {share_text(g['D1'])} | ≥ {pct(D1, 0)} | {'ja' if g['D1']['holds'] else 'nee'} |",
         f"| D2 | richting, alle niet-gelijke paren van G4-7 en G4-bon-i15c2q | {share_text(g['D2'])} | ≥ {pct(D2, 0)} | {'ja' if g['D2']['holds'] else 'nee'} |",
         f"| D3 | omklappen (greedy, tegengestelde winnaars), alle paren | {share_text(g['D3'])} | ≤ {pct(D3, 0)} | {'ja' if g['D3']['holds'] else 'nee'} |",
         "", f"**Uitslag: {'gehaald — stap 2 draait' if g['pass'] else 'niet gehaald — stap 2 draait niet'}.**", "",
         "## Tabel 1: per pakket", "",
         "| pakket | n | richting niet-gelijk | richting zeker niet-gelijk | 3 klassen gelijk | κ | κ zeker | omklappen | inconsistent |",
         "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for name, m in list(r["packages"].items()) + [("G4-7 + G4-bon-i15c2q", r["close"]), ("alle", r["all"])]:
        L.append(f"| {name} | {m['n']} | {share_text(m['direction_noneq'])} | {share_text(m['direction_noneq_sure'])} | "
                 f"{pct(m['agree3']['share'])} | {num(m['kappa'])} | {num(m['kappa_sure'])} | {pct(m['flip']['share'])} | "
                 f"{pct(m['inconsistent']['share'])} |")
    L += ["", "## Tabel 2: kruistabel, alle paren (rij: judge, kolom: Claude)", "", "| judge \\ Claude | A | B | gelijk |",
          "|---|---:|---:|---:|"]
    for x in ANSWERS:
        L.append(f"| {x} | " + " | ".join(str(r["all"]["table"][x][y]) for y in ANSWERS) + " |")
    L += ["", "## Tabel 3: positie en antwoordmassa (per volgorde, greedy antwoord van de prompt zelf)", "",
          "| volgorde | eerste positie (A) | tweede positie (B) | gelijk | gem. p_A − p_B | massa mediaan | massa < 0,5 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for o, x in r["position"].items():
        L.append(f"| {o} | {pct(x['first_position'])} | {pct(x['second_position'])} | {pct(x['gelijk'])} | "
                 f"{num(x['mean_pa_minus_pb'])} | {num(x['mass_median'])} | {x['mass_below_half']} |")
    L += ["", "Meest voorkomende top-tokens (id: aantal): " + "; ".join(
        f"{o}: " + ", ".join(f"{k}: {v}" for k, v in t.items()) for o, t in r["top_tokens"].items()) + ".", "",
          "## Tabel 4: per set (alle pakketten)", "", "| set | n | richting niet-gelijk | richting zeker niet-gelijk | κ |",
          "|---|---:|---|---|---:|"]
    for s, m in r["sets"].items():
        L.append(f"| {s} | {m['n']} | {share_text(m['direction_noneq'])} | {share_text(m['direction_noneq_sure'])} | {num(m['kappa'])} |")
    L += ["", "## Tabel 5: gerapporteerd, beslist niet mee", "",
          f"- 5a. Paren waar Claude precies één kant een betekenisfout gaf: de richting van de judge kiest de kant "
          f"zonder fout in {share_text(r['fault'])}.", "",
          "| pakket | niet-basissysteem | judge: niet-gelijke oordelen voor dat systeem | Claude: idem |", "|---|---|---|---|"]
    for p, x in r["system_preference"].items():
        L.append(f"| {p} | {', '.join(x['system'])} | {share_text(x['judge'])} | {share_text(x['claude'])} |")
    L += ["", "Tegen de `final-judgments` (twijfelgevallen daar opnieuw beoordeeld; de zekere oordelen zijn gelijk):", "",
          "| pakket | richting niet-gelijk | 3 klassen gelijk | κ |", "|---|---|---:|---:|"]
    for p, m in r["final"].items():
        L.append(f"| {p} | {share_text(m['direction_noneq'])} | {pct(m['agree3']['share'])} | {num(m['kappa'])} |")
    L += ["", "## Wat dit niet zegt", "",
          "- Claude's oordelen zijn geen grondwaarheid; bij veel paren twijfelde de beoordelaar zelf. Overeenstemming kan "
          "ook gedeelde voorkeuren van taalmodellen weerspiegelen.",
          "- Gemma 4 31B-it beoordeelt hier ook uitvoer van zijn eigen model (G4-0) en van LoRA's erop.",
          "- Eén vaste prompt, zero-shot; geen zoektocht naar een betere prompt op deze paren.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- step 2: pairs

def search_part(evalsets, split):
    import g4bon

    ids = g4bon.search_ids(g4bon.load_evalsets(evalsets), g4bon.load_split(split))
    return ids, [i for s in g4bon.SETS for i in ids[s]], {i: s for s in g4bon.SETS for i in ids[s]}


def hypotheses(pattern, ids):
    import armeval

    rows = armeval.read_rows(pattern, ["id", "source", "hypothesis"])
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{pattern}: no row for {len(missing)} search segments, e.g. {missing[:3]}")
    return rows


def read_choice(path, ids):
    rows = {r["id"]: r for r in read_rows(path)}
    if set(rows) != set(ids):
        raise SystemExit(f"{path}: {len(rows)} segments, not the {len(ids)} of the search part")
    for i, r in rows.items():
        if len(r["qe"]) != len(SAMPLE_NAMES) or len(r["fail"]) != len(SAMPLE_NAMES):
            raise SystemExit(f"{path}: {i} has no 8 samples")
    return rows


def bonpairs(args):
    """Every sample that is no fail against the anchor (G4-0 greedy), both orders (G4-judge.7)"""
    _, all_ids, set_of = search_part(args.evalsets, args.split)
    anchor = hypotheses(f"{args.anchor}/pairs/pairs-w*.parquet", all_ids)
    prompts, info = [], {}
    for spec in args.model:
        model, paths = spec.split("=", 1)
        samples_dir, choice_path = paths.split(",")
        if model not in BON_MODELS:
            raise SystemExit(f"--model {model}: one of {BON_MODELS}")
        choice = read_choice(choice_path, all_ids)
        c = collections.Counter()
        for k, name in enumerate(SAMPLE_NAMES):
            rows = hypotheses(f"{samples_dir}/{name}/pairs/pairs-w*.parquet", all_ids)
            for i in all_ids:
                if (rows[i]["source"] or "").strip() != (anchor[i]["source"] or "").strip():
                    raise SystemExit(f"{model}/{name} {i}: the source is not the anchor's")
                if choice[i]["fail"][k]:
                    c[f"fail_{set_of[i]}"] += 1
                    continue
                hyp = rows[i]["hypothesis"] or ""
                if not hyp.strip():
                    raise SystemExit(f"{model}/{name} {i}: an empty hypothesis that is no fail")
                c[f"pairs_{set_of[i]}"] += 1
                c["same_as_anchor"] += hyp.strip() == (anchor[i]["hypothesis"] or "").strip()
                for uid, text, _ in pair_prompts(f"{model}/{i}/{name}", anchor[i]["source"], hyp, anchor[i]["hypothesis"]):
                    prompts.append({"uid": uid, "text": text})
        info[model] = dict(sorted(c.items()))
    out = Path(args.out)
    workers = write_prompts(prompts, out)
    summary = {"protocol": PROTOCOL, "models": info, "prompts": len(prompts), "workers": workers}
    (out / "bonpairs.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"BONPAIRS {json.dumps(summary)}", flush=True)


# ---------------------------------------------------------------------------------------------------- step 2: choice

def sample_judgements(model, i, scores):
    """Per sample of one source: P̄(sample better), P̄(anchor better), or None (a fail: not scored; or not scored)"""
    out = []
    for name in SAMPLE_NAMES:
        key = f"{model}/{i}/{name}"
        ab, ba = scores.get(f"{key}/ab"), scores.get(f"{key}/ba")
        p = combine(ab, ba)
        out.append(None if p is None else {"sample": p["A"], "anchor": p["B"], "gelijk": p["gelijk"]})
    return out


def choose_judge(judged, fails, qes, qe0, qe_choice):
    """{judge: (k, fallback), judgeq: (k, fallback)} and the number of judgeq's candidates (G4-judge.7)"""
    def qe_key(k):
        return math.inf if qes[k] is None else qes[k]

    ok = [k for k in range(len(judged)) if not fails[k] and judged[k] is not None]
    bound = [k for k in ok if qes[k] is not None and qes[k] <= qe0 + QE_TOL]

    def best(cands):
        return min(cands, key=lambda k: (-judged[k]["sample"], qe_key(k), k))
    return {"judge": (best(ok), False) if ok else (qe_choice, True),
            "judgeq": (best(bound), False) if bound else (qe_choice, True)}, len(bound)


def select(args):
    import armeval
    import g4arm

    _, all_ids, set_of = search_part(args.evalsets, args.split)
    choice = read_choice(args.choice, all_ids)
    scores = scores_by_uid(read_glob(args.scores))
    raw = {n: {"pairs": armeval.read_rows(f"{args.samples}/{n}/pairs/pairs-w*.parquet"),
               "metricx": armeval.read_rows(f"{args.samples}/{n}/scores/metricx-w*.parquet"),
               "comet": armeval.read_rows(f"{args.samples}/{n}/scores/comet-w*.parquet")
               if glob.glob(f"{args.samples}/{n}/scores/comet-w*.parquet") else {}} for n in SAMPLE_NAMES}
    used, rows_out = set(), []
    chosen = {s: {"pairs": [], "metricx": [], "comet": []} for s in JUDGE_SELECTIONS}
    for i in all_ids:
        ch = choice[i]
        judged = sample_judgements(args.model, i, scores)
        for k, name in enumerate(SAMPLE_NAMES):
            key = f"{args.model}/{i}/{name}"
            present = [f"{key}/{o}" in scores for o in ORDERS]
            if (bool(ch["fail"][k]) and any(present)) or (not ch["fail"][k] and not all(present)):
                raise SystemExit(f"{key}: {'a fail with' if ch['fail'][k] else 'no fail without'} a judge score "
                                 f"(orders present: {present})")
            used.update({f"{key}/{o}" for o in ORDERS} if all(present) else set())
        picks, n_bound = choose_judge(judged, [bool(f) for f in ch["fail"]], ch["qe"], ch["qe0"],
                                      SAMPLE_NAMES.index(ch["qe_choice"]))
        rec = dict(ch)
        rec.update({"p_sample": [None if j is None else j["sample"] for j in judged],
                    "p_anchor": [None if j is None else j["anchor"] for j in judged],
                    "p_gelijk": [None if j is None else j["gelijk"] for j in judged],
                    "judged": sum(j is not None for j in judged), "judgeq_eligible": n_bound})
        for sel, (k, fallback) in picks.items():
            name = SAMPLE_NAMES[k]
            rec[f"{sel}_choice"] = name
            rec[f"{sel}_fallback"] = fallback
            chosen[sel]["pairs"].append(raw[name]["pairs"][i])
            chosen[sel]["metricx"].append(raw[name]["metricx"][i])
            if i in raw[name]["comet"]:
                chosen[sel]["comet"].append(raw[name]["comet"][i])
        rows_out.append(rec)
    extra = set(u for u in scores if u.startswith(f"{args.model}/")) - used
    if extra:
        raise SystemExit(f"{len(extra)} judge scores of {args.model} belong to no sample of the search part, e.g. {sorted(extra)[:2]}")
    out = Path(args.out)
    for sel, x in chosen.items():
        g4arm.write_parquet(x["pairs"], out / sel / "pairs" / "pairs-wall.parquet")
        g4arm.write_parquet(x["metricx"], out / sel / "scores" / "metricx-wall.parquet")
        if x["comet"]:
            g4arm.write_parquet(x["comet"], out / sel / "scores" / "comet-wall.parquet")
    g4arm.write_parquet(rows_out, out / "choice.parquet")
    stats = select_stats(rows_out, set_of)
    (out / "select.json").write_text(json.dumps({"protocol": PROTOCOL, "model": args.model, "stats": stats}, indent=1) + "\n")
    print(f"JSELECT {json.dumps({'model': args.model, 'fallbacks': {s: {sel: x[sel]['fallback'] for sel in JUDGE_SELECTIONS} for s, x in stats.items()}})}",
          flush=True)


def select_stats(rows, set_of):
    import g4bon

    out = {}
    for st in g4bon.SETS:
        rs = [r for r in rows if set_of[r["id"]] == st]
        x = {}
        for sel in JUDGE_SELECTIONS:
            ks = [SAMPLE_NAMES.index(r[f"{sel}_choice"]) for r in rs]
            ps = [(r["p_sample"][k], r["p_anchor"][k]) for r, k in zip(rs, ks) if r["p_sample"][k] is not None]
            x[sel] = {"fallback": sum(r[f"{sel}_fallback"] for r in rs),
                      "same_as_qe": sum(r[f"{sel}_choice"] == r["qe_choice"] for r in rs) / len(rs),
                      "same_as_c2q": sum(r[f"{sel}_choice"] == r["c2q_choice"] for r in rs) / len(rs),
                      "chosen": dict(sorted(collections.Counter(r[f"{sel}_choice"] for r in rs).items())),
                      "p_sample_mean": statistics.fmean(a for a, _ in ps) if ps else None,
                      "beats_anchor": sum(a > b for a, b in ps) / len(rs)}
        x["judged_mean"] = statistics.fmean(r["judged"] for r in rs)
        x["judgeq_eligible_mean"] = statistics.fmean(r["judgeq_eligible"] for r in rs)
        x["all_fail"] = sum(all(f for f in r["fail"]) for r in rs)
        out[st] = x
    return out


# ---------------------------------------------------------------------------------------------------- step 2: report

LABELS = {"greedy": "greedy", "t10s1": "één sample (T = 1, seed 1)", "qe": "best-of-8: laagste QE (G4-bon)",
          "c2q": "best-of-8: hoogste C2 met QE ≤ QE₀ (G4-bon)", "judge": "best-of-8: hoogste P̄(sample beter) (judge)",
          "judgeq": "best-of-8: judge met QE ≤ QE₀"}
SELECTION_MEASURES = {"greedy": [], "t10s1": [], "qe": ["QE", "MetricX-ref (zelfde model als QE)"],
                      "c2q": ["C2", "QE", "MetricX-ref (zelfde model als QE)", "fail-vlaggen"],
                      "judge": ["judge", "fail-vlaggen (QE > 7,5 en lengteband)"],
                      "judgeq": ["judge", "QE", "MetricX-ref (zelfde model als QE)", "fail-vlaggen"]}


def bonreport(args):
    import armeval
    import g4arm
    import g4bon

    ids, all_ids, _ = search_part(args.evalsets, args.split)
    refqe = {r["id"]: r["qe_source_reference"] for r in g4arm.read_rows(args.refqe, ["id", "qe_source_reference"])}
    c1 = {(r["model"], r["id"]): r["p_original"] for r in g4arm.read_rows(args.predictions, ["model", "id", "p_original"])}
    anchor_rows = armeval.scored(args.anchor)
    anchors = {i: g4bon.anchor_of(anchor_rows[i]) for i in all_ids}
    result = {"protocol": PROTOCOL, "part": g4bon.PART, "n": g4bon.N_SEARCH, "human_qe_wmt24pp":
              g4arm.mean(refqe[i] for i in ids["wmt24pp"]), "boot": g4bon.BOOT, "models": {}, "checks": {}}
    views, selects = {}, {}
    for spec in args.model:
        model, dirs = spec.split("=", 1)
        greedy_dir, t1_dir, bon_dir, j_dir = dirs.split(",")
        bon_choice = {r["id"]: r for r in g4arm.read_rows(f"{bon_dir}/choice.parquet")}
        choice = {r["id"]: r for r in g4arm.read_rows(f"{j_dir}/choice.parquet")}
        if set(choice) != set(all_ids) or set(bon_choice) != set(all_ids):
            raise SystemExit(f"{model}: a choice without the {len(all_ids)} segments of the search part")
        differ = [i for i in all_ids if any(choice[i][f"{s}_choice"] != bon_choice[i][f"{s}_choice"] for s in ("qe", "c2q"))]
        if differ:
            raise SystemExit(f"{model}: qe or c2q differs from G4-bon's choice at {len(differ)} segments")
        scored = {"greedy": armeval.scored(greedy_dir), "t10s1": armeval.scored(t1_dir),
                  "qe": armeval.scored(f"{bon_dir}/qe"), "c2q": armeval.scored(f"{bon_dir}/c2q"),
                  **{s: armeval.scored(f"{j_dir}/{s}") for s in JUDGE_SELECTIONS}}
        views[model] = {o: g4bon.output_rows(model, o, scored[o], choice, c1, anchors) for o in OUTPUTS}
        selects[model] = json.loads(Path(f"{j_dir}/select.json").read_text())
    if g4bon.BASE not in views:
        raise SystemExit(f"no {g4bon.BASE} among the models")
    g40 = views[g4bon.BASE]["greedy"]
    for model, v in views.items():
        outs = {}
        for o in OUTPUTS:
            x = {"sets": {s: g4bon.measures(v[o], ids[s], s == "wmt24pp") for s in g4bon.SETS},
                 "c1_pooled": sum(v[o][i]["c1"] >= 0.5 for i in all_ids) / len(all_ids),
                 "c2_pooled": sum(v[o][i]["logit"] is not None and v[o][i]["logit"] >= 0 for i in all_ids) / len(all_ids),
                 "selection_measures": SELECTION_MEASURES[o]}
            if o != "greedy":
                x["c1_vs_greedy"] = g4bon.c1_delta(v[o], v["greedy"], ids, f"{BOOT_NAME}/{model}/{o}-greedy")
            if not (model == g4bon.BASE and o == "greedy"):
                x["c1_vs_g40"] = g4bon.c1_delta(v[o], g40, ids, f"{BOOT_NAME}/{model}/{o}-G4-0")
            outs[o] = x
        contrasts = {}
        for a, b in CONTRASTS:
            n = f"{BOOT_NAME}/{model}/{a}-{b}"
            contrasts[f"{a}-{b}"] = {"c1": g4bon.c1_delta(v[a], v[b], ids, n),
                                     "comet22": g4bon.value_delta(v[a], v[b], ids, "comet22", f"{n}/comet"),
                                     "metricx_ref": g4bon.value_delta(v[a], v[b], ids, "metricx_ref", f"{n}/mxref")}
        for o in ("qe", "c2q") + JUDGE_SELECTIONS:
            n = f"{BOOT_NAME}/{model}/{o}-greedy"
            contrasts[f"{o}-greedy"] = {"comet22": g4bon.value_delta(v[o], v["greedy"], ids, "comet22", f"{n}/comet"),
                                        "metricx_ref": g4bon.value_delta(v[o], v["greedy"], ids, "metricx_ref", f"{n}/mxref")}
        result["models"][model] = {"outputs": outs, "contrasts": contrasts, "select": selects[model]}
    result["checks"] = c1_checks(views, all_ids, c1, args)
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    Path(args.out).write_text(bonreport_text(result), encoding="utf-8")
    print(f"JBONREPORT {json.dumps({m: {o: round(x['outputs'][o]['c1_pooled'], 4) for o in OUTPUTS} for m, x in result['models'].items()})}",
          flush=True)


def c1_checks(views, all_ids, c1, args):
    """C1 of this run against the sampling probe's (G4-0) and G4-bon's run (G4-7-i15): shares and flipped labels"""
    import g4arm

    def compare(here_name, there, there_name):
        got = [(c1[(here_name, i)] >= 0.5, there[(there_name, i)] >= 0.5) for i in all_ids if (there_name, i) in there]
        if len(got) != len(all_ids):
            raise SystemExit(f"the other run has {len(got)} of the {len(all_ids)} search segments of {there_name}")
        return {"share_here": sum(a for a, _ in got) / len(got), "share_there": sum(b for _, b in got) / len(got),
                "flipped": sum(a != b for a, b in got)}

    probe = {(r["model"], r["id"]): r["p_original"] for r in g4arm.read_rows(args.probe_predictions, ["model", "id", "p_original"])}
    bon = {(r["model"], r["id"]): r["p_original"] for r in g4arm.read_rows(args.bon_predictions, ["model", "id", "p_original"])}
    out = {}
    if "G4-0" in views:
        for here, there in (("G4-0", "G4-0"), ("G4-0/t10s1", "G4-0/t10s1"), ("G4-0/qe", "G4-0/best8")):
            out[f"{here} ~ sampling-proef {there}"] = compare(here, probe, there)
    for m in [m for m in views if m != "G4-0"]:
        for o in ("", "/t10s1", "/qe", "/c2q"):
            out[f"{m}{o} ~ G4-bon {m}{o}"] = compare(f"{m}{o}", bon, f"{m}{o}")
    return out


def bonreport_text(r):
    import g4bon

    ci = g4bon.ci_text

    def p(x):
        return pct(x)

    L = ["# Meting G4-judge, stap 2: best-of-8 met de keuze op Gemma 4 31B-it als judge (zoekdeel; rapporteren, beslist niets)",
         "", f"Gemaakt door `pairjudge.py bonreport` ({PROTOCOL}). Zoekdeel van G4-5.4: WMT24++ ({r['n']['wmt24pp']}) en "
         f"FLAN ({r['n']['flan']}). Per bron 8 samples bij T = 1 (die van G4-bon); judge: het hoogste P̄(sample beter "
         "dan G4-0 greedy) onder de samples zonder fail, judgeq: idem met QE ≤ QE₀; qe en c2q uit G4-bon. C1 "
         "(\"origineel\" = P ≥ 0,5) uit één classify-run over alle uitvoer. Intervallen: 95%, gepaarde bootstrap "
         f"({r['boot']} trekkingen, per set binnen de set).", "",
         "## Tabel 1: aandeel \"origineel\" (C1)", "",
         "| model | uitvoer | WMT24++ | FLAN | samen | Δ eigen greedy (95%-BI) | Δ G4-0 greedy (95%-BI) | C2 samen |",
         "|---|---|---:|---:|---:|---|---|---:|"]
    for m, x in r["models"].items():
        for o in OUTPUTS:
            y = x["outputs"][o]
            L.append(f"| {m} | {LABELS[o]} | {p(y['sets']['wmt24pp']['c1'])} | {p(y['sets']['flan']['c1'])} | "
                     f"{p(y['c1_pooled'])} | {ci(y['c1_vs_greedy']) if 'c1_vs_greedy' in y else '–'} | "
                     f"{ci(y['c1_vs_g40']) if 'c1_vs_g40' in y else '–'} | {p(y['c2_pooled'])} |")
    L += ["", f"## Tabel 2: kwaliteit en vorm (QE van de menselijke referentie op WMT24++ van het zoekdeel: "
          f"{num(r['human_qe_wmt24pp'])})", "",
          "| model | uitvoer | set | QE | MetricX-ref | COMET-22 | fail (G4-7) | niet af | lus | Engels over | leeg | "
          "tekens / bron | ℓ/ℓ₀ mediaan | C2-logit gem. | niet onafhankelijk (selectiemaat) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m, x in r["models"].items():
        for o in OUTPUTS:
            y = x["outputs"][o]
            for s in g4bon.SETS:
                z = y["sets"][s]
                L.append(f"| {m} | {o} | {s} | {num(z['qe'])} | {num(z.get('metricx_ref'))} | {num(z.get('comet22'), 4)} | "
                         f"{p(z['fail'])} | {p(z['not_ended'])} | {p(z['decode_loop'])} | {p(z['english_left'])} | "
                         f"{p(z['empty'])} | {num(z['chars_source'])} | {num(z['ratio_median'])} | {num(z['c2_logit'])} | "
                         f"{', '.join(y['selection_measures']) or '–'} |")
    L += ["", "## Tabel 3: contrasten binnen een model (C1 samen; COMET-22 en MetricX-ref op WMT24++)", "",
          "| model | contrast | ΔC1 samen (95%-BI) | ΔCOMET-22 (95%-BI) | ΔMetricX-ref (95%-BI) |", "|---|---|---|---|---|"]
    for m, x in r["models"].items():
        for name, c in x["contrasts"].items():
            L.append(f"| {m} | {name} | {ci(c['c1']) if 'c1' in c else '–'} | {ci(c['comet22'], 1.0, '', 4)} | "
                     f"{ci(c['metricx_ref'], 1.0, '', 3)} |")
    L += ["", "## Tabel 4: de selectie", "",
          "| model | set | selectie | terugval op qe | gelijk aan qe | gelijk aan c2q | gem. P̄(sample beter) van de keuze | "
          "keuze P̄(sample) > P̄(anker) | gekozen samples |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for m, x in r["models"].items():
        for s, st in x["select"]["stats"].items():
            for sel in JUDGE_SELECTIONS:
                y = st[sel]
                L.append(f"| {m} | {s} | {sel} | {y['fallback']} | {p(y['same_as_qe'])} | {p(y['same_as_c2q'])} | "
                         f"{num(y['p_sample_mean'])} | {p(y['beats_anchor'])} | "
                         f"{', '.join(f'{k} {v}' for k, v in y['chosen'].items())} |")
            L.append(f"| {m} | {s} | (gem. beoordeelde samples {num(st['judged_mean'], 2)}; gem. kandidaten judgeq "
                     f"{num(st['judgeq_eligible_mean'], 2)}; bronnen waar alle 8 falen {st['all_fail']}) | | | | | | |")
    L += ["", "## Controles (gerapporteerd)", ""]
    for k, v in r["checks"].items():
        L.append(f"- C1 {k}: hier {p(v['share_here'])}, daar {p(v['share_there'])}, label anders bij {v['flipped']} segmenten.")
    L += ["", "## Wat dit niet zegt", "",
          "- Een selectie uit 8 samples is geen getraind model.",
          "- De judge is Gemma 4 31B-it zelf: hij beoordeelt samples van zichzelf (G4-0) en van een LoRA erop tegen zijn "
          "eigen greedy vertaling; een voorkeur voor de eigen stijl is niet uitgesloten.",
          "- Een C1-winst na selectie op de judge is geen bewijs van natuurlijker Nederlands; de blinde controle van het "
          "pakket G4-judge-i15 doet een Claude-subagent met de rubriek waartegen de judge gevalideerd is.",
          "- Alleen het zoekdeel (n = 480 + 500); G4-7-i15 is op dit deel gekozen; intervallen per vergelijking.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- step 2: blind

def blind(args):
    """The blind package of (model, judgeq) against G4-0 greedy (G4-judge.7): 50 WMT24++ + 50 FLAN where they differ"""
    import armeval
    import g4bon

    ids, all_ids, set_of = search_part(args.evalsets, args.split)
    choice = {r["id"]: r for r in read_rows(f"{args.select}/choice.parquet")}
    target = armeval.read_rows(f"{args.select}/judgeq/pairs/pairs-w*.parquet", ["id", "source", "hypothesis"])
    anchor = hypotheses(f"{args.anchor}/pairs/pairs-w*.parquet", all_ids)
    if set(choice) != set(all_ids) or set(target) != set(all_ids):
        raise SystemExit(f"{args.select}: not the {len(all_ids)} segments of the search part")
    inputs, keys, report = [], [], {"protocol": PROTOCOL, "model": args.model, "target": BLIND_TARGET, "sets": {}}
    n = 0
    for s in g4bon.SETS:
        differ = sorted(i for i in ids[s] if (target[i]["hypothesis"] or "").strip() != (anchor[i]["hypothesis"] or "").strip())
        seed = f"g4judge/blind/{s}"
        picked = random.Random(seed).sample(differ, min(BLIND_PER_SET, len(differ)))
        report["sets"][s] = {"segments": len(ids[s]), "differ": len(differ), "picked": len(picked), "seed": seed}
        for i in picked:
            n += 1
            b = f"judge-{n:03d}"
            ab_seed = f"g4judge/blind/ab/{b}"
            base_first = random.Random(ab_seed).random() < 0.5
            base, own = anchor[i]["hypothesis"], target[i]["hypothesis"]
            a, bb = (base, own) if base_first else (own, base)
            inputs.append({"blind_id": b, "source": anchor[i]["source"], "A": a, "B": bb})
            keys.append({"blind_id": b, "A": "base" if base_first else BLIND_TARGET,
                         "B": BLIND_TARGET if base_first else "base", "seed": ab_seed, "set": s, "item": i})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "judge-input.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in inputs)
    with open(out / "key.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in keys)
    report["a_is_base"] = sum(k["A"] == "base" for k in keys)
    report["n"] = len(keys)
    (out / "select-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"JBLIND {json.dumps(report)}", flush=True)


# ---------------------------------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("valdata")
    a.add_argument("--blind", required=True, help="a directory with <package>/judge-input.jsonl, key.jsonl, reviews")
    a.add_argument("--out", required=True)
    a = sub.add_parser("score")
    a.add_argument("--prompts", required=True)
    a.add_argument("--out", required=True, help="gs:// directory of the score shards")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--work-dir", required=True)
    a.add_argument("--shard-size", type=int, default=256)
    a = sub.add_parser("valreport")
    a.add_argument("--refs", required=True)
    a.add_argument("--scores", required=True, help="GLOB of the score shards")
    a.add_argument("--out", required=True)
    a = sub.add_parser("bonpairs")
    a.add_argument("--model", action="append", required=True, help="NAME=SAMPLES_DIR,CHOICE (G4-bon's choice.parquet)")
    a.add_argument("--anchor", required=True, help="G4-0's evaluation (pairs/)")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("select")
    a.add_argument("--model", required=True)
    a.add_argument("--samples", required=True, help="the directory of t10s1..t10s8 (pairs/, scores/)")
    a.add_argument("--choice", required=True, help="G4-bon's choice.parquet of the model")
    a.add_argument("--scores", required=True, help="GLOB of the judge's score shards")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("bonreport")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--refqe", required=True)
    a.add_argument("--anchor", required=True)
    a.add_argument("--predictions", required=True, help="C1 of this measurement's classify run")
    a.add_argument("--probe-predictions", required=True, help="C1 of the sampling probe's classify run")
    a.add_argument("--bon-predictions", required=True, help="C1 of G4-bon's classify run G4-0_G4-7-i15")
    a.add_argument("--model", action="append", required=True, help="NAME=GREEDY_DIR,T10S1_DIR,BON_SELECT_DIR,JUDGE_SELECT_DIR")
    a.add_argument("--out", required=True)
    a = sub.add_parser("blind")
    a.add_argument("--model", required=True)
    a.add_argument("--select", required=True, help="the judge's select directory of the model")
    a.add_argument("--anchor", required=True, help="G4-0's evaluation (pairs/)")
    a.add_argument("--evalsets", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"valdata": valdata, "score": score, "valreport": valreport, "bonpairs": bonpairs, "select": select,
     "bonreport": bonreport, "blind": blind}[args.cmd](args)


if __name__ == "__main__":
    main()
