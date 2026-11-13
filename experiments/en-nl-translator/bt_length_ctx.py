#!/usr/bin/env python3
"""Length trial of 13 Nov, part A, third condition ctx (Yeb, 13 Nov ~21:30: "bij vertalen in stukken het vorige stuk
als context meegeven"): the stretches of bins 2048-16384 of part A in the same pieces as the condition stukken, each
piece translated by Gemma 4 31B-it (natural-v1-ctx, greedy) with the end of the English already made of the piece
before as context; the first piece without context (natural-v1, the prompt of stukken). Compared with part A's
stukken (same documents, pieces, hardware and code) by the decision rule of mbert-ops/translator-sft/bt-length-trial.md
(section "Deel A, conditie ctx").

    python bt_length_ctx.py plan --docs docs.parquet --segments seg/stukken/segments-w*.parquet --tokenizer TOK \\
        --workers 4 --out DIR
    python bt_length_ctx.py translate --plan plan-w0.parquet --out gs://.../ctx/tr/w0 --tokenizer TOK --ckpt CKPT \\
        [--slots 20] [--max-docs 3 --max-rounds 2]
    python bt_length_ctx.py collect --docs docs.parquet --plan plan-w*.parquet --stukken DIR --ctx DIR \\
        --measured-a measured.parquet --workers 4 --out DIR
    python bt_length_ctx.py extract --queries queries-w0.parquet --out answers-w0.parquet --tokenizer TOK --ckpt CKPT
    python bt_length_ctx.py report --collect DIR --units-a units.parquet --qe qe-*.parquet --answers answers-w*.parquet \\
        --logs DIR --setup setup.json --out bt-length-ctx.md
    python bt_length_ctx.py blind --collect DIR --docs docs.parquet --out DIR

plan: the pieces of each stretch of the bins (seg/stukken of part A: the same pieces, joined they are the stretch),
every stretch whole on one worker (a piece needs the English of the one before), split by number of pieces; checks that
every prompt fits the class c512 of translate_models.yml gemma4-31b-ctx (a piece with the smallest context: at most the
largest prefill minus MIN_CONTEXT tokens).

translate: rounds on one engine: round k translates piece k of every stretch of this worker that has one (continuous
batching over the slots), with as context the end of the English of piece k-1 (whole paragraphs from its end, as many
as keep the prompt within the largest prefill length, 1,152); round 0 without context. A round is one shard in the
bucket (shard-r<k>.parquet, translate.py's columns plus the context's size), so a start again resumes after the last
round there. A ROUND line per round, MEM (peak HBM per chip) at the end.

collect: per stretch and condition (stukken: part A's measured rows and pieces; ctx: the rounds) the measures of
bt_length.measure_doc, the units for MetricX (ctx; stukken was scored in part A), per piece the copy of context (COPY:
the share of the piece's English word 6-grams that occur in the English of the piece before, the same detector on
stukken as its baseline) and the prompt's markup in the output (tag_leak), the terms of the consistency measure (TERMS)
and their questions for Gemma, split over the workers.

extract: Gemma 4 31B-it (greedy) answers each question: the exact English words that translate the Dutch term in that
piece's English, or NONE.

report: the tables per bin (stukken against ctx), the consistency measure, the copy of context, the rule per bin, the
costs; blind: the material for a blind comparison of bin 4096 (A/B in a random order, the key apart, a rubric).
"""

import argparse
import collections
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

import bt_length as bl  # noqa: E402

BINS = (2048, 4096, 8192, 16384)
MODEL = "gemma4-31b-ctx"
CLASS = "c512"
QCLASS = "xq"
CONDITIONS = ("stukken", "ctx")
MIN_CONTEXT = 64            # tokens of room for context every piece must leave in the largest prefill (plan's check)
NGRAM = 6                   # COPY: word n-grams
COPY_SHARE = 0.2            # a piece copies its context when >= this share of its n-grams are in the piece before
TAG_MARKS = ("previous_translation>", "<previous_translation", "text to translate:")
NAMES_PER_DOC = 4           # TERMS: names (capitalized words and runs, abbreviations) per stretch
TERMS_PER_DOC = 4           # TERMS: lowercase terms per stretch
TERM_MIN_LETTERS = 8
TERM_MIN_COUNT = 3          # a lowercase term occurs at least this often in the stretch
TERM_MAX_DF = 0.4           # a lowercase term is in at most this share of all documents of docs.parquet
MAX_QUERY_PIECES = 8        # pieces asked per term (evenly spread over the pieces it is in, first and last included)
RULE = {"inconsistency_ratio": 0.75, "sign_p": 0.05, "done_docs": 1, "omission_pp": 1.0, "qe": 0.2, "copy_pp": 1.0}
BLIND_BIN = 4096
BLIND_DOCS = 20
SEED = "bt-length-ctx/0"
CONNECTORS = {"van", "de", "der", "den", "het", "voor", "op", "ter", "ten", "tot", "aan", "in", "'t", "te", "du", "la",
              "le", "von"}
# Dutch words that start a sentence with a capital (a run of capitals that starts a sentence loses its first word when
# it is one of these), and capitalized words that are no name
STARTERS = set("""de het een in op voor van dit deze die dat er ook bij na met als maar want om zo wij ik u hij zij we
ze uit aan tot over door naar onder tegen sinds volgens daarom daarnaast echter toen nu dan hier daar wat wie waar hoe
waarom welke mijn onze ons uw hun zijn haar geen niet wel nog al alle elk iedere ieder men je jij jullie dus toch
verder tevens bovendien inmiddels ten ter mevrouw meneer heer dames heren beste geachte zoals omdat indien wanneer
terwijl nadat voordat hoewel tenzij alleen zowel ofwel noch of en ja nee oké goed dank bedankt artikel""".split())
NOT_NAMES = {"U", "Uw", "Gij", "Ik"}
TERM_SUFFIXES = ("lijk", "lijke", "lijker", "ig", "ige", "iger", "isch", "ische", "end", "ende", "baar", "bare")
TERM_PREFIXES = ("daar", "hier", "waar")
TOKEN = re.compile(r"\w[\w&'’-]*\w|\w")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
QUESTION = ("Below is a sentence from a Dutch text and the English translation of the passage it comes from. How is "
            "the Dutch term \"{term}\" translated in that English translation?\n\nDutch:\n{dutch}\n\nEnglish "
            "translation:\n{english}\n\nAnswer with the exact words of the English translation that translate "
            "\"{term}\", copied as they appear there, and nothing else. If the term is not translated in the English "
            "text, answer NONE.")


# ---------------------------------------------------------------------------------------------------- plan

def piece_index(piece_id):
    """(stretch id, k) of a piece id <stretch>#c<k>"""
    doc, sep, k = piece_id.rpartition("#c")
    if not sep or not k.isdigit():
        raise ValueError(f"not a piece id: {piece_id!r}")
    return doc, int(k)


def stretches_of(segments, documents):
    """{stretch id: [piece rows in order]} for the stretches of documents; every stretch complete (pieces 0..n-1) and
    its pieces joined are its text"""
    pieces = collections.defaultdict(dict)
    for r in segments:
        doc, k = piece_index(r["id"])
        if doc in documents:
            if k in pieces[doc]:
                raise SystemExit(f"piece {r['id']} twice in the segments")
            pieces[doc][k] = r
    out = {}
    for doc_id, d in documents.items():
        ks = sorted(pieces.get(doc_id, {}))
        if not ks or ks != list(range(len(ks))):
            raise SystemExit(f"{doc_id}: pieces {ks[:5]}... are not 0..n-1")
        rows = [pieces[doc_id][k] for k in ks]
        if bl.join([r["text"] for r in rows]) != bl.join(d["paragraphs"]):
            raise SystemExit(f"{doc_id}: its pieces joined are not the stretch")
        out[doc_id] = rows
    return out


def prompt_codec(tokenizer):
    import translate as tr

    return tr.ModelPrompts(tokenizer, tr.MODELS[MODEL], "nl", "en")


def shapes_of(cls, slots=None):
    import translate as tr

    model = tr.MODELS[MODEL]
    return dict(model["length_classes"][cls], slots=slots or model["slots"][cls])


def plan(args):
    from transformers import AutoTokenizer

    bins = [int(b) for b in args.bins.split(",")]
    documents = {d["id"]: d for d in bl.read_rows(args.docs) if d["bin"] in bins}
    segments = [r for path in args.segments for r in bl.read_rows(path)]
    stretches = stretches_of(segments, documents)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = prompt_codec(tokenizer)
    shapes = shapes_of(CLASS)
    limit = shapes["prefill_lengths"][-1]
    texts = [rows[k]["text"] for rows in list(stretches.values())[:2] for k in range(2)]
    codec.check(texts[:2], None)
    codec.check(texts[1:3], [[{"source": "", "translation": texts[0]}]] * 2)
    worst = {"plain": 0, "context_room": limit}
    for doc_id, rows in stretches.items():
        for r in rows:
            if r["tokens"] > shapes["max_source"]:
                raise SystemExit(f"{r['id']}: {r['tokens']} tokens, more than class {CLASS}'s {shapes['max_source']}")
            plain = len(codec.encode(r["text"]))
            room = limit - len(codec.encode(r["text"], [{"source": "", "translation": "."}]))
            if plain > limit or room < MIN_CONTEXT:
                raise SystemExit(f"{r['id']}: a prompt of {plain} tokens, {room} tokens of room for context in a prefill "
                                 f"of {limit} (at least {MIN_CONTEXT})")
            worst = {"plain": max(worst["plain"], plain), "context_room": min(worst["context_room"], room)}
    out = Path(args.out)
    split = bl.balanced(sorted(stretches), args.workers, lambda d: len(stretches[d]))
    summary = {"model": MODEL, "class": CLASS, "shapes": shapes, "bins": bins, "workers": []}
    for w, docs in enumerate(split):
        rows = [{"id": r["id"], "doc": d, "k": k, "pieces": len(stretches[d]), "bin": documents[d]["bin"],
                 "source": documents[d]["source"], "text": r["text"], "tokens": r["tokens"]}
                for d in sorted(docs) for k, r in enumerate(stretches[d])]
        bl.write_parquet(rows, out / f"plan-w{w}.parquet")
        summary["workers"].append({"worker": w, "stretches": len(docs), "pieces": len(rows),
                                   "rounds": max((len(stretches[d]) for d in docs), default=0),
                                   "bins": dict(collections.Counter(documents[d]["bin"] for d in docs))})
    summary.update(stretches=len(stretches), pieces=sum(len(v) for v in stretches.values()), **worst)
    (out / "plan.json").write_text(json.dumps(summary, indent=1))
    print(f"PLAN {json.dumps(summary)}", flush=True)


# ---------------------------------------------------------------------------------------------------- translate

def context_for(encode, text, previous, limit):
    """The context of a piece: the end of previous (the English of the piece before) that keeps the prompt within limit
    tokens, as {"context": [{"source": "", "translation": ...}] or None, "paragraphs": whole paragraphs taken,
    "full": all of previous, "cut": the last paragraph's end only}. Whole paragraphs from the end, as many as fit; when
    not even the last one fits, its last words. encode(text, context) gives the prompt's ids."""
    paras = bl.paragraphs_of_text(previous or "")
    none = {"context": None, "paragraphs": 0, "full": False, "cut": False, "tokens": 0}
    if not paras:
        return none

    def element(ctx):
        return [{"source": "", "translation": ctx}]

    def fits(ctx):
        return len(encode(text, element(ctx))) <= limit

    taken = 0
    while taken < len(paras) and fits(bl.join(paras[-(taken + 1):])):
        taken += 1
    if taken:
        ctx = bl.join(paras[-taken:])
        return {"context": element(ctx), "paragraphs": taken, "full": taken == len(paras), "cut": False,
                "tokens": len(encode(text, element(ctx))) - len(encode(text, element("")))}
    words = paras[-1].split(" ")
    lo, hi, best = 1, len(words), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(" ".join(words[-mid:])):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    if not best:
        raise RuntimeError(f"no room for context in a prompt of at most {limit} tokens")
    ctx = " ".join(words[-best:])
    return {"context": element(ctx), "paragraphs": 0, "full": False, "cut": True,
            "tokens": len(encode(text, element(ctx))) - len(encode(text, element("")))}


def stretches_of_plan(rows, max_docs=None):
    """{stretch: [plan rows in order of k]}; with max_docs, the first max_docs stretches (by id) of at least 2 pieces"""
    out = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["doc"], r["k"])):
        out[r["doc"]].append(r)
    for doc, rs in out.items():
        if [r["k"] for r in rs] != list(range(len(rs))):
            raise SystemExit(f"{doc}: the plan's pieces are not 0..n-1")
    if max_docs is not None:
        out = {d: out[d] for d in sorted(d for d in out if len(out[d]) >= 2)[:max_docs]}
    return dict(sorted(out.items()))


def round_name(k):
    return f"shard-r{k:03d}.parquet"


def run_rounds(stretches, rounds, done, encode, decode, detokenize, store, limit, log=print):
    """Rounds 0..rounds-1 over the stretches ({stretch: plan rows}). done: {k: rows of a round stored before} (a
    prefix of the rounds: a round needs the English of the one before). decode(prompts) -> [(tokens, ended)] in order;
    store(k, rows, seconds) keeps a round. Returns {stretch: {k: English}}."""
    english = collections.defaultdict(dict)
    pending = [k for k in range(rounds) if k not in done]
    if pending and any(k in done for k in range(pending[0], rounds)):
        raise SystemExit(f"rounds done before {sorted(done)} are not a prefix of 0..{rounds - 1}")
    for k in sorted(done):
        for r in done[k]:
            english[r["doc"]][k] = r["text_en"]
    for k in pending:
        items = [(doc, rows[k]) for doc, rows in stretches.items() if k < len(rows)]
        prompts, contexts = [], []
        for doc, r in items:
            c = context_for(encode, r["text"], english[doc][k - 1], limit) if k else \
                {"context": None, "paragraphs": 0, "full": False, "cut": False, "tokens": 0}
            ids = encode(r["text"], c["context"])
            if len(ids) > limit:
                raise RuntimeError(f"{r['id']}: a prompt of {len(ids)} tokens, more than {limit}")
            prompts.append(ids)
            contexts.append(c)
        started = time.time()
        outputs = decode(prompts)
        seconds = time.time() - started
        rows = {"id": [], "doc": [], "k": [], "text_en": [], "ended": [], "tokens_in": [], "tokens_out": [],
                "context_tokens": [], "context_paragraphs": [], "context_full": [], "context_cut": []}
        for (doc, r), ids, c, (completion, ended) in zip(items, prompts, contexts, outputs):
            text = detokenize(completion).strip()
            english[doc][k] = text
            for key, value in (("id", r["id"]), ("doc", doc), ("k", k), ("text_en", text), ("ended", bool(ended)),
                               ("tokens_in", len(ids)), ("tokens_out", len(completion) + int(bool(ended))),
                               ("context_tokens", c["tokens"]), ("context_paragraphs", c["paragraphs"]),
                               ("context_full", c["full"]), ("context_cut", c["cut"])):
                rows[key].append(value)
        store(k, rows, seconds)
        n_out = sum(rows["tokens_out"])
        log(f"ROUND {json.dumps({'round': k, 'prompts': len(items), 'tokens_in': sum(rows['tokens_in']), 'tokens_out': n_out, 'seconds': round(seconds, 1), 'out_per_s': round(n_out / max(seconds, 1e-9)), 'not_ended': rows['ended'].count(False), 'context_tokens_mean': round(statistics.fmean(rows['context_tokens']), 1) if items else 0})}")
    return english


def download(url, path, client):
    import translate as tr

    bucket, name = tr.split_gs(url)
    client.bucket(bucket).blob(name).download_to_filename(str(path))


def load_engine(args, shapes):
    import translate as tr

    print(f"MAXTEXT {json.dumps({'model': MODEL, 'ckpt': args.ckpt, **tr.maxtext_source(), 'patches': tr.patch_maxtext()})}",
          flush=True)
    started = time.time()
    engine, params = tr.make_engine(argparse.Namespace(model=MODEL, ckpt=args.ckpt, work_dir=args.work_dir,
                                                       tokenizer=args.tokenizer), shapes, tr.MODELS[MODEL]["chips"])
    device = tr.EngineSlots(engine, params, shapes["prefill_lengths"])
    print(f"engine ready in {time.time() - started:.0f} s", flush=True)
    return device


def check_chips():
    import jax
    import translate as tr

    chips = tr.MODELS[MODEL]["chips"]
    if (jax.device_count(), jax.process_count()) != (chips, 1):
        raise SystemExit(f"needs one process with {chips} chips, has {jax.process_count()} process(es) with "
                         f"{jax.device_count()} devices")


def translate(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import translate as tr
    from transformers import AutoTokenizer

    stretches = stretches_of_plan(bl.read_rows(args.plan), args.max_docs)
    rounds = max((len(v) for v in stretches.values()), default=0)
    if args.max_rounds is not None:
        rounds = min(rounds, args.max_rounds)
    shapes = shapes_of(CLASS, args.slots)
    client = tr.gcs_client()
    names = tr.done_shards(args.out, client)
    local = Path(args.work_dir) / "ctx-rounds"
    local.mkdir(parents=True, exist_ok=True)
    done = {}
    for k in range(rounds):
        if round_name(k) in names:
            download(f"{args.out}/{round_name(k)}", local / round_name(k), client)
            done[k] = bl.read_rows(local / round_name(k))
    print(f"{CLASS}: {len(stretches)} stretches, {sum(map(len, stretches.values()))} pieces, {rounds} rounds, "
          f"{len(done)} done before; shapes {json.dumps(shapes)}", flush=True)
    if len(done) == rounds:
        return
    check_chips()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = prompt_codec(tokenizer)
    first = [rows[0]["text"] for rows in stretches.values()][:2]
    codec.check(first, None)
    codec.check(first, [[{"source": "", "translation": first[0]}]] * len(first))
    device = load_engine(args, shapes)
    eos = tr.MODELS[MODEL]["eos_ids"]
    totals = {"rounds": 0, "segments": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}

    def decode(prompts):
        return tr.decode_all(prompts, shapes["slots"], device, shapes["max_decode"], args.sync_every, eos)

    def store(k, rows, seconds):
        path = local / round_name(k)
        pq.write_table(pa.table(dict(rows, slots=[shapes["slots"]] * len(rows["id"]))), path, compression="zstd")
        tr.upload(path, f"{args.out}/{round_name(k)}", client)
        for key, value in (("rounds", 1), ("segments", len(rows["id"])), ("tokens_in", sum(rows["tokens_in"])),
                           ("tokens_out", sum(rows["tokens_out"])), ("seconds", seconds)):
            totals[key] += value

    run_rounds(stretches, rounds, done, codec.encode,
               decode, lambda ids: tokenizer.decode(ids, skip_special_tokens=True), store,
               shapes["prefill_lengths"][-1], log=lambda line: print(line, flush=True))
    print(f"DONE {json.dumps({'class': CLASS, **totals, 'out_per_s': round(totals['tokens_out'] / max(totals['seconds'], 1e-9))})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- copy

def words(text):
    return re.findall(r"\w+", (text or "").lower())


def ngrams(text, n=NGRAM):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def copy_share(output, previous, n=NGRAM):
    """The share of the output's word n-grams that occur in previous (0 for an output of fewer than n words)"""
    out = ngrams(output, n)
    return len(out & ngrams(previous, n)) / len(out) if out else 0.0


def tag_leak(output):
    low = (output or "").lower()
    return any(m in low for m in TAG_MARKS)


def piece_rows(stretch, condition, english, source_pieces):
    """Per piece of a stretch: its English, the copy of the piece before (k >= 1) and markup of the prompt"""
    out = []
    for k, piece in enumerate(source_pieces):
        e = english[k]
        share = copy_share(e["text"], english[k - 1]["text"]) if k else None
        out.append({"doc": stretch["id"], "bin": stretch["bin"], "source": stretch["source"], "condition": condition,
                    "k": k, "text_nl": piece, "text_en": e["text"], "ended": bool(e["ended"]),
                    "tokens_out": e["tokens_out"], "copy_share": share,
                    "copy": share is not None and share >= COPY_SHARE, "tag_leak": tag_leak(e["text"]),
                    "context_tokens": e.get("context_tokens"), "context_full": e.get("context_full"),
                    "context_cut": e.get("context_cut")})
    return out


# ---------------------------------------------------------------------------------------------------- terms

def sentence_start(text, start):
    prev = text[:start].rstrip()
    return not prev or prev[-1] in ".!?:;•*-–—)(\"'“”‘’[]"


def candidates(paragraph):
    """[(term, kind, start)] of a Dutch paragraph: names (runs of capitalized words joined by connectors such as van
    and der, a run that starts a sentence without its first word when that is a common starter; a capitalized word
    that does not start a sentence; an abbreviation of two or more capitals) and lowercase words (kind term)"""
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(paragraph)]
    out, i = [], 0
    while i < len(toks):
        t, s, e = toks[i]
        if t[0].isupper():
            run, j = [i], i + 1
            while j < len(toks):
                if paragraph[toks[j - 1][2]:toks[j][1]] != " ":
                    break
                if toks[j][0][0].isupper():
                    run.append(j)
                    j += 1
                elif (toks[j][0] in CONNECTORS and j + 1 < len(toks) and toks[j + 1][0][0].isupper()
                      and paragraph[toks[j][2]:toks[j + 1][1]] == " "):
                    run += [j, j + 1]
                    j += 2
                else:
                    break
            at_start = sentence_start(paragraph, s)
            first = run[0]
            if at_start and toks[first][0].lower() in STARTERS:
                run = run[1:]
                while run and toks[run[0]][0] in CONNECTORS:
                    run = run[1:]
            caps = [k for k in run if toks[k][0][0].isupper()]
            if len(caps) >= 2:
                out.append((paragraph[toks[run[0]][1]:toks[run[-1]][2]], "name", toks[run[0]][1]))
            elif len(caps) == 1:
                word, ws, _ = toks[caps[0]]
                abbr = sum(c.isupper() for c in word) >= 2
                if word not in NOT_NAMES and any(c.isalpha() for c in word) and len(word) >= 2 and (
                        abbr or not sentence_start(paragraph, ws)):
                    out.append((word, "name", ws))
            i = j
            continue
        if t.islower() and t.replace("-", "").isalpha():
            out.append((t, "term", s))
        i += 1
    return out


def lowercase_term_ok(word):
    letters = sum(c.isalpha() for c in word)
    return (letters >= TERM_MIN_LETTERS and not word.endswith(TERM_SUFFIXES) and not word.startswith(TERM_PREFIXES)
            and word not in STARTERS)


def document_frequency(documents):
    """{lowercase word: number of documents it is in}"""
    df = collections.Counter()
    for d in documents:
        df.update({w for p in d["paragraphs"] for w, kind, _ in candidates(p) if kind == "term"})
    return df


def spread(items, n):
    """At most n of items, evenly spread, the first and the last included"""
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def terms_of(stretch_id, pieces, df, n_docs):
    """The consistency measure's terms of a stretch (pieces: its Dutch pieces): [{term, kind, pieces, count,
    asked}]: up to NAMES_PER_DOC names and TERMS_PER_DOC lowercase terms that occur in two or more pieces (lowercase
    ones also TERM_MIN_COUNT times, in at most TERM_MAX_DF of all documents, and lowercase_term_ok); asked: the pieces
    whose rendering is asked, at most MAX_QUERY_PIECES"""
    where = collections.defaultdict(list)
    count = collections.Counter()
    for k, text in enumerate(pieces):
        for p in bl.paragraphs_of_text(text):
            for term, kind, _ in candidates(p):
                count[(term, kind)] += 1
                if not where[(term, kind)] or where[(term, kind)][-1] != k:
                    where[(term, kind)].append(k)
    names = [(t, kd) for (t, kd) in where if kd == "name" and len(where[(t, kd)]) >= 2]
    names.sort(key=lambda x: (-len(where[x]), -count[x], x[0]))
    terms = [(t, kd) for (t, kd) in where if kd == "term" and len(where[(t, kd)]) >= 2
             and count[(t, kd)] >= TERM_MIN_COUNT and lowercase_term_ok(t) and df[t] <= TERM_MAX_DF * n_docs]
    terms.sort(key=lambda x: (-count[x], -len(where[x]), x[0]))
    return [{"doc": stretch_id, "term": t, "kind": kd, "pieces": where[(t, kd)], "count": count[(t, kd)],
             "asked": spread(where[(t, kd)], MAX_QUERY_PIECES)}
            for t, kd in names[:NAMES_PER_DOC] + terms[:TERMS_PER_DOC]]


def sentence_with(text, term, kind):
    """(Dutch, scope) of the question: the first sentence of a Dutch piece in which the term is a candidate of its kind
    (scope sentence), or, where the split into sentences changes what starts a sentence, the first paragraph in which
    it is one (scope paragraph). A term found in no paragraph of the piece is an error."""
    for p in bl.paragraphs_of_text(text):
        if not any(t == term and kd == kind for t, kd, _ in candidates(p)):
            continue
        for s in SENTENCE_END.split(p):
            if any(t == term and kd == kind for t, kd, _ in candidates(s)):
                return s.strip(), "sentence"
        return p, "paragraph"
    raise ValueError(f"{term!r} ({kind}) is in no paragraph of the piece")


def queries_of(terms, pieces_nl, english):
    """The questions of a stretch's terms for each condition: english {condition: [English of piece k]}"""
    out = []
    for i, t in enumerate(terms):
        for k in t["asked"]:
            dutch, scope = sentence_with(pieces_nl[k], t["term"], t["kind"])
            for cond in CONDITIONS:
                out.append({"id": f"{t['doc']}|{cond}|t{i:02d}|c{k:03d}", "doc": t["doc"], "condition": cond,
                            "term_index": i, "term": t["term"], "kind": t["kind"], "k": k, "dutch": dutch,
                            "dutch_scope": scope, "english": english[cond][k]})
    return out


def normalize(answer):
    """A rendering for comparison: lowercase, without quotes or end punctuation, without a leading article or a
    possessive 's"""
    a = " ".join((answer or "").replace("’", "'").split()).lower()
    a = a.strip(" \"'`*.,;:!?()[]{}“”‘")
    a = re.sub(r"^(the|a|an) ", "", a)
    a = re.sub(r"'s$", "", a)
    return a.strip()


def parse_answer(raw, english):
    """(status, rendering): status none (NONE), invalid (not in the English), empty, or valid"""
    line = (raw or "").strip().splitlines()[0].strip() if (raw or "").strip() else ""
    line = line.strip("`*\"'“” ").strip()
    if not line:
        return "empty", None
    if line.rstrip(".").upper() == "NONE":
        return "none", None
    rendering = normalize(line)
    if not rendering:
        return "empty", None
    hay = " ".join((english or "").replace("’", "'").split()).lower()
    if rendering not in hay:
        return "invalid", rendering
    return "valid", rendering


# ---------------------------------------------------------------------------------------------------- collect

def collect(args):
    bins = [int(b) for b in args.bins.split(",")]
    documents = {d["id"]: d for d in bl.read_rows(args.docs)}
    df = document_frequency(documents.values())
    n_docs = len(documents)
    planned = stretches_of_plan([r for path in args.plan for r in bl.read_rows(path)])
    stukken = bl.read_translations(args.stukken, "text_en")
    ctx = {}
    for path in sorted(Path(args.ctx).rglob("shard-r*.parquet")):
        for r in bl.read_rows(path):
            if r["id"] in ctx:
                raise SystemExit(f"{r['id']} is in two rounds under {args.ctx}")
            ctx[r["id"]] = {"text": r["text_en"], "ended": r["ended"], "tokens_in": r["tokens_in"],
                            "tokens_out": r["tokens_out"], "context_tokens": r["context_tokens"],
                            "context_full": r["context_full"], "context_cut": r["context_cut"]}
    measured_a = {(r["id"], r["condition"]): r for r in bl.read_rows(args.measured_a)}
    measured, units, pieces_out, terms_out, queries = [], [], [], [], []
    for doc_id, rows in planned.items():
        d = documents[doc_id]
        if d["bin"] not in bins:
            continue
        ids = [r["id"] for r in rows]
        missing = {c: [i for i in ids if i not in src] for c, src in (("stukken", stukken), ("ctx", ctx))}
        for cond, m in missing.items():
            if m:
                raise SystemExit(f"{doc_id}: {len(m)} of {len(ids)} pieces without a translation in {cond} ({m[0]})")
        english = {"stukken": [stukken[i] for i in ids], "ctx": [ctx[i] for i in ids]}
        a = measured_a.get((doc_id, "stukken"))
        if a is None:
            raise SystemExit(f"{doc_id}: not in part A's measured rows (stukken)")
        joined = bl.join([(e["text"] or "").strip() for e in english["stukken"]])
        if joined != a["translation"]:
            raise SystemExit(f"{doc_id}: the pieces of stukken joined differ from part A's translation")
        for cond in CONDITIONS:
            ps = piece_rows(d, cond, english[cond], [r["text"] for r in rows])
            pieces_out += ps
            later = [p for p in ps if p["k"]]
            extra = {"copy_pieces": sum(p["copy"] for p in later), "pieces_after_first": len(later),
                     "copy_share_mean": statistics.fmean(p["copy_share"] for p in later) if later else None,
                     "tag_leak_pieces": sum(p["tag_leak"] for p in ps)}
            if cond == "stukken":
                measured.append(dict(a, **extra))
                continue
            text = bl.join([(e["text"] or "").strip() for e in english["ctx"]])
            row, unit_rows, units_al = bl.measure_doc(d, text, all(e["ended"] for e in english["ctx"]), "nl", "en")
            measured.append({"id": doc_id, "source": d["source"], "bin": d["bin"], "condition": "ctx",
                             "tokens_source": d["tokens"], "tokens_in": sum(e["tokens_in"] for e in english["ctx"]),
                             "tokens_out": sum(e["tokens_out"] for e in english["ctx"]), "translation": text,
                             "pieces": len(ids), "alignment": json.dumps(units_al), **row, **extra})
            for u in unit_rows:
                units.append({"id": f"{doc_id}|ctx|u{u['unit']:04d}", "doc": doc_id, "source": d["source"],
                              "bin": d["bin"], "condition": "ctx", **u})
        terms = terms_of(doc_id, [r["text"] for r in rows], df, n_docs)
        for i, t in enumerate(terms):
            terms_out.append(dict(t, bin=d["bin"], source=d["source"], term_index=i))
        queries += queries_of(terms, [r["text"] for r in rows],
                              {c: [(e["text"] or "").strip() for e in english[c]] for c in CONDITIONS})
    out = Path(args.out)
    bl.write_parquet(measured, out / "measured.parquet")
    bl.write_parquet(units, out / "units.parquet")
    bl.write_parquet(pieces_out, out / "pieces.parquet")
    bl.write_parquet(terms_out, out / "terms.parquet")
    for w, part in enumerate(bl.balanced(queries, args.workers, lambda q: len(q["english"]) + len(q["dutch"]))):
        part.sort(key=lambda q: q["id"])
        bl.write_parquet(part, out / f"queries-w{w}.parquet")
    summary = {"measured": len(measured), "units_ctx": len(units), "pieces": len(pieces_out), "terms": len(terms_out),
               "queries": len(queries)}
    (out / "collect.json").write_text(json.dumps(summary))
    print(f"COLLECT {json.dumps(summary)}", flush=True)


# ---------------------------------------------------------------------------------------------------- extract

def question_codec(tokenizer):
    import translate as tr

    prompt = {"kind": "instruct", "plain": "{text}"}
    return tr.TurnCodec(tokenizer, prompt, tr.MODELS[MODEL]["turn"], "nl", "en")


def question_ids(codec, q, limit):
    """The prompt ids of a question within limit tokens: the Dutch sentence shortened to the words around the term
    when needed; None when even the term alone does not fit"""
    dutch = q["dutch"]
    ids = codec.encode(QUESTION.format(term=q["term"], dutch=dutch, english=q["english"]))
    if len(ids) <= limit:
        return ids
    w = dutch.split(" ")
    at = next((i for i, x in enumerate(w) if q["term"].split(" ")[0] in x), 0)
    for half in (40, 20, 10, 5, 0):
        part = " ".join(w[max(0, at - half):at + len(q["term"].split(" ")) + half])
        ids = codec.encode(QUESTION.format(term=q["term"], dutch=part, english=q["english"]))
        if len(ids) <= limit:
            return ids
    return None


def extract(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import translate as tr
    from transformers import AutoTokenizer

    queries = bl.read_rows(args.queries)
    if not queries:
        print("EXTRACT: no question", flush=True)
        return
    shapes = shapes_of(QCLASS, args.slots)
    print(f"{QCLASS}: {len(queries)} questions; shapes {json.dumps(shapes)}", flush=True)
    check_chips()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = question_codec(tokenizer)
    codec.check([QUESTION.format(term=q["term"], dutch=q["dutch"], english=q["english"]) for q in queries[:2]])
    limit = shapes["prefill_lengths"][-1]
    prompts, kept, skipped = [], [], 0
    for q in queries:
        ids = question_ids(codec, q, limit)
        if ids is None:
            skipped += 1
            continue
        prompts.append(ids)
        kept.append(q)
    device = load_engine(args, shapes)
    started = time.time()
    outputs = tr.decode_all(prompts, shapes["slots"], device, shapes["max_decode"], args.sync_every,
                            tr.MODELS[MODEL]["eos_ids"])
    seconds = time.time() - started
    rows = {"id": [q["id"] for q in kept], "answer": [], "ended": [], "tokens_in": [len(p) for p in prompts]}
    for completion, ended in outputs:
        rows["answer"].append(tokenizer.decode(completion, skip_special_tokens=True).strip())
        rows["ended"].append(bool(ended))
    pq.write_table(pa.table(rows), args.out, compression="zstd")
    print(f"EXTRACT {json.dumps({'questions': len(queries), 'asked': len(kept), 'too_long': skipped, 'seconds': round(seconds, 1), 'tokens_in': sum(rows['tokens_in']), 'not_ended': rows['ended'].count(False)})}",
          flush=True)


# ---------------------------------------------------------------------------------------------------- report

def sign_test(better, worse):
    """One-sided exact sign test: P(X >= better) for X ~ Binomial(better + worse, 1/2); 1.0 without discordant pairs"""
    n = better + worse
    if not n:
        return 1.0
    return sum(math.comb(n, i) for i in range(better, n + 1)) / 2 ** n


def consistency(terms, queries, answers):
    """Per (doc, term_index, condition): the renderings of the asked pieces and whether they agree. answers {id: raw}.
    Returns {(doc, term_index): {condition: {...}}} and the answer statuses per condition"""
    by_id = {q["id"]: q for q in queries}
    renderings = collections.defaultdict(lambda: collections.defaultdict(list))
    status = {c: collections.Counter() for c in CONDITIONS}
    for qid, q in by_id.items():
        if qid not in answers:
            status[q["condition"]]["not_asked"] += 1
            continue
        st, rendering = parse_answer(answers[qid], q["english"])
        status[q["condition"]][st] += 1
        if st == "valid":
            renderings[(q["doc"], q["term_index"])][q["condition"]].append(rendering)
    out = {}
    for t in terms:
        key = (t["doc"], t["term_index"])
        out[key] = {}
        for cond in CONDITIONS:
            r = renderings[key][cond]
            out[key][cond] = {"renderings": r, "measurable": len(r) >= 2, "consistent": len(r) >= 2 and len(set(r)) == 1,
                              "distinct": len(set(r))}
    return out, status


def consistency_stats(terms, per_term, bins, kinds=("name", "term")):
    stats = {}
    for b in bins:
        ts = [t for t in terms if t["bin"] == b and t["kind"] in kinds]
        s = {}
        for cond in CONDITIONS:
            m = [per_term[(t["doc"], t["term_index"])][cond] for t in ts]
            meas = [x for x in m if x["measurable"]]
            s[cond] = {"terms": len(ts), "measurable": len(meas), "consistent": sum(x["consistent"] for x in meas),
                       "share": sum(x["consistent"] for x in meas) / len(meas) if meas else None,
                       "distinct_mean": statistics.fmean(x["distinct"] for x in meas) if meas else None}
        both = [per_term[(t["doc"], t["term_index"])] for t in ts]
        both = [x for x in both if x["stukken"]["measurable"] and x["ctx"]["measurable"]]
        better = sum(x["ctx"]["consistent"] and not x["stukken"]["consistent"] for x in both)
        worse = sum(x["stukken"]["consistent"] and not x["ctx"]["consistent"] for x in both)
        s["paired"] = {"terms": len(both), "ctx_better": better, "ctx_worse": worse, "p": sign_test(better, worse),
                       "share_stukken": sum(x["stukken"]["consistent"] for x in both) / len(both) if both else None,
                       "share_ctx": sum(x["ctx"]["consistent"] for x in both) / len(both) if both else None}
        stats[b] = s
    return stats


def copy_stats(pieces, bins):
    out = {}
    for b in bins:
        out[b] = {}
        for cond in CONDITIONS:
            ps = [p for p in pieces if p["bin"] == b and p["condition"] == cond]
            later = [p for p in ps if p["k"]]
            ctx = [p for p in later if p["context_tokens"] is not None]
            out[b][cond] = {"pieces": len(ps), "pieces_after_first": len(later), "copy": sum(p["copy"] for p in later),
                            "copy_share_mean": statistics.fmean(p["copy_share"] for p in later) if later else None,
                            "tag_leak": sum(p["tag_leak"] for p in ps),
                            "not_ended": sum(not p["ended"] for p in ps),
                            "context_tokens_mean": statistics.fmean(p["context_tokens"] for p in ctx) if ctx else None,
                            "context_full": sum(bool(p["context_full"]) for p in ctx) if ctx else None,
                            "context_cut": sum(bool(p["context_cut"]) for p in ctx) if ctx else None}
        first = {(p["doc"], p["condition"]): p["text_en"] for p in pieces if p["bin"] == b and p["k"] == 0}
        docs = sorted({d for d, _ in first})
        out[b]["first_piece_identical"] = sum(first.get((d, "stukken")) == first.get((d, "ctx")) for d in docs)
        out[b]["first_piece_docs"] = len(docs)
    return out


def rule(stats, cons, copy, spec=RULE):
    """Per bin the parts of the rule of ctx against stukken (exact on k/n)"""
    verdict = {}
    for b in stats:
        p, c = stats[b].get("stukken"), stats[b].get("ctx")
        if not p or not p["docs"] or not c or not c["docs"]:
            verdict[b] = {"measured": False, "pass": False}
            continue
        k = cons[b]
        inc_p = 1 - k["stukken"]["share"] if k["stukken"]["share"] is not None else None
        inc_c = 1 - k["ctx"]["share"] if k["ctx"]["share"] is not None else None
        i_ratio = inc_p is not None and inc_c is not None and inc_c <= spec["inconsistency_ratio"] * inc_p + 1e-12 \
            and inc_p > 0
        i_sign = k["paired"]["ctx_better"] > k["paired"]["ctx_worse"] and k["paired"]["p"] <= spec["sign_p"] + 1e-12
        ii = c["done"] >= p["done"] - spec["done_docs"]
        om_p, om_c = p["omitted"] / p["paragraphs"], c["omitted"] / c["paragraphs"]
        iii = (om_c - om_p) * 100 <= spec["omission_pp"] + 1e-12
        qe_known = p["qe_mean"] is not None and c["qe_mean"] is not None
        iv = qe_known and c["qe_mean"] - p["qe_mean"] <= spec["qe"] + 1e-12
        cp, cc = copy[b]["stukken"], copy[b]["ctx"]
        cop_p = cp["copy"] / cp["pieces_after_first"] if cp["pieces_after_first"] else 0.0
        cop_c = cc["copy"] / cc["pieces_after_first"] if cc["pieces_after_first"] else 0.0
        v = (cop_c - cop_p) * 100 <= spec["copy_pp"] + 1e-12
        verdict[b] = {"measured": True, "i_ratio": bool(i_ratio), "i_sign": bool(i_sign), "i": bool(i_ratio and i_sign),
                      "ii": ii, "iii": iii, "iv": iv, "v": v, "qe_known": qe_known,
                      "inconsistency": {"stukken": inc_p, "ctx": inc_c},
                      "done": {"stukken": p["done"], "ctx": c["done"]}, "omission_diff_pp": (om_c - om_p) * 100,
                      "qe_diff": c["qe_mean"] - p["qe_mean"] if qe_known else None,
                      "copy_diff_pp": (cop_c - cop_p) * 100,
                      "pass": bool(i_ratio and i_sign and ii and iii and iv and v)}
    return verdict


def costs(logs_dir):
    """Per log of translate and extract: the engine's load time, rounds, prompts, tokens, seconds, peak HBM"""
    out = []
    for path in sorted(Path(logs_dir).glob("*.log")) if logs_dir else []:
        t = {"log": path.name, "load_s": None, "rounds": 0, "prompts": 0, "tokens_in": 0, "tokens_out": 0,
             "seconds": 0.0, "round_seconds": [], "peak_gib": None, "oom": False, "slots": None, "extract": None}
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^engine ready in (\d+) s$", line)
            if m:
                t["load_s"] = int(m.group(1))
            elif line.startswith("ROUND "):
                s = json.loads(line[len("ROUND "):])
                t["rounds"] += 1
                t["prompts"] += s["prompts"]
                t["tokens_in"] += s["tokens_in"]
                t["tokens_out"] += s["tokens_out"]
                t["seconds"] += s["seconds"]
                t["round_seconds"].append(s["seconds"])
            elif line.startswith("EXTRACT "):
                t["extract"] = json.loads(line[len("EXTRACT "):])
            elif line.startswith("MEM "):
                t["peak_gib"] = json.loads(line[len("MEM "):]).get("peak_gib_max")
            elif "shapes {" in line:
                t["slots"] = json.loads(line.split("shapes ", 1)[1]).get("slots")
            elif "RESOURCE_EXHAUSTED" in line or "Ran out of memory" in line:
                t["oom"] = True
        out.append(t)
    return out


def report(args):
    c = Path(args.collect)
    measured = bl.read_rows(c / "measured.parquet")
    bins = sorted({r["bin"] for r in measured})
    units = [u for u in bl.read_rows(args.units_a, ["id", "doc", "condition", "n_source"]) if u["condition"] == "stukken"]
    units += bl.read_rows(c / "units.parquet", ["id", "doc", "condition", "n_source"])
    scores = {}
    for path in args.qe or []:
        for r in bl.read_rows(path):
            if r.get("qe_src_cand") is not None:
                scores[r["id"]] = r["qe_src_cand"]
    qe = bl.qe_per_paragraph(units, scores, "qe_src_cand")
    stats = bl.bin_stats(measured, qe, bins, conditions=CONDITIONS)
    pieces = bl.read_rows(c / "pieces.parquet")
    copy = copy_stats(pieces, bins)
    terms = bl.read_rows(c / "terms.parquet")
    queries = [q for path in sorted(c.glob("queries-w*.parquet")) for q in bl.read_rows(path)]
    answers = {r["id"]: r["answer"] for path in args.answers or [] for r in bl.read_rows(path)}
    per_term, status = consistency(terms, queries, answers)
    cons = consistency_stats(terms, per_term, bins)
    cons_kind = {kind: consistency_stats(terms, per_term, bins, (kind,)) for kind in ("name", "term")}
    verdict = rule(stats, cons, copy)
    r = {"title": args.title, "stats": stats, "copy": copy, "consistency": cons, "consistency_by_kind": cons_kind,
         "answers": {k: dict(v) for k, v in status.items()}, "verdict": verdict, "rule": RULE,
         "passing_bins": [b for b in bins if verdict[b]["pass"]], "costs": costs(args.logs),
         "setup": json.loads(Path(args.setup).read_text()) if args.setup else {},
         "examples": inconsistent_examples(terms, per_term)}
    md = report_md(r, bins)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    print(md)
    print(f"VERDICT {json.dumps({str(b): verdict[b]['pass'] for b in bins})}")


def inconsistent_examples(terms, per_term, n=12):
    """Terms consistent in one condition and not in the other, a few of each (for reading, not for the rule)"""
    out = {"ctx_better": [], "ctx_worse": []}
    for t in sorted(terms, key=lambda t: (t["bin"], t["doc"], t["term_index"])):
        x = per_term[(t["doc"], t["term_index"])]
        if not (x["stukken"]["measurable"] and x["ctx"]["measurable"]):
            continue
        key = ("ctx_better" if x["ctx"]["consistent"] and not x["stukken"]["consistent"] else
               "ctx_worse" if x["stukken"]["consistent"] and not x["ctx"]["consistent"] else None)
        if key and len(out[key]) < n:
            out[key].append({"bin": t["bin"], "doc": t["doc"], "term": t["term"], "kind": t["kind"],
                             "stukken": sorted(collections.Counter(x["stukken"]["renderings"]).items()),
                             "ctx": sorted(collections.Counter(x["ctx"]["renderings"]).items())})
    return out


def report_md(r, bins):
    stats, copy, cons, verdict = r["stats"], r["copy"], r["consistency"], r["verdict"]
    lines = [f"## {r['title']}", ""]
    lines += [f"- {k}: {v}" for k, v in r["setup"].items()] + ([""] if r["setup"] else [])
    lines += ["| maat | " + " | ".join(f"{b} stukken | {b} ctx" for b in bins) + " |", "|---|" + "---:|" * (2 * len(bins))]

    def row(label, f, table=stats):
        cells = []
        for b in bins:
            for cond in CONDITIONS:
                s = table[b].get(cond)
                cells.append(f(s) if s else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    kn, num, pct = bl.kn, bl.num, bl.pct
    row("documenten", lambda s: str(s["docs"]))
    row("brontokens, mediaan", lambda s: num(s["tokens_source_median"], 0))
    row("**af (zonder lus of afkapping)**", lambda s: kn(s["done"], s["docs"]))
    row("niet af (decodeerlimiet)", lambda s: str(s["not_ended"]))
    row("lus", lambda s: str(s["decode_loop"]))
    row("afgekapt (laatste alinea weg)", lambda s: str(s["cut"]))
    row("brontaal over", lambda s: str(s["source_left"]))
    row("inleiding of noot (`chatter`)", lambda s: str(s["chatter"]))
    row("toegevoegde opmaak (documenten)", lambda s: str(s["added_formatting"]))
    row("alinea-aantal behouden", lambda s: kn(s["paragraphs_kept"], s["docs"]))
    row("bronalinea's", lambda s: str(s["paragraphs"]))
    row("**weglatingen**", lambda s: kn(s["omitted"], s["paragraphs"]))
    row("waarvan op niets uitgelijnd", lambda s: str(s["omitted_none"]))
    row("waarvan vertaling te kort (R2)", lambda s: str(s["omitted_short"]))
    row("waarvan vertaling te lang (R2)", lambda s: str(s["omitted_long"]))
    row("toegevoegde alinea's", lambda s: str(s["added_paragraphs"]))
    row("**MetricX-24-QE nl->en per alinea, gem.**", lambda s: num(s["qe_mean"]))
    row("idem, mediaan", lambda s: num(s["qe_median"]))
    row("alinea's met score / documenten met score", lambda s: f"{s['qe_paragraphs']} / {s['qe_docs']}")
    row("NL/EN-woorden per document, mediaan", lambda s: num(s["word_ratio_median"], 3))
    row("stukken (k ≥ 1)", lambda s: str(s["pieces_after_first"]), copy)
    row("**kopie van context (stukken k ≥ 1)**", lambda s: kn(s["copy"], s["pieces_after_first"]), copy)
    row("gedeelde 6-grammen met het vorige stuk, gem.", lambda s: pct(s["copy_share_mean"]), copy)
    row("opmaak van de prompt in de uitvoer (stukken)", lambda s: str(s["tag_leak"]), copy)
    row("contexttokens per stuk, gem.", lambda s: num(s["context_tokens_mean"], 0), copy)
    row("vorige vertaling helemaal als context", lambda s: kn(s["context_full"], s["pieces_after_first"])
        if s["context_full"] is not None else "–", copy)
    row("**consistente termen**", lambda s: kn(s["consistent"], s["measurable"]), cons)
    for kind, label in (("name", "waarvan namen"), ("term", "waarvan kleine-letter-termen")):
        row(label, lambda s: kn(s["consistent"], s["measurable"]), r["consistency_by_kind"][kind])
    row("verschillende weergaven per term, gem.", lambda s: num(s["distinct_mean"]), cons)
    lines.append("")
    lines += ["Antwoorden van Gemma op de termvragen: " + "; ".join(
        f"{cond}: " + ", ".join(f"{k} {v}" for k, v in sorted(r["answers"][cond].items())) for cond in CONDITIONS), ""]
    lines += ["Eerste stuk (zonder context in beide condities) letterlijk gelijk: " + ", ".join(
        f"{b}: {copy[b]['first_piece_identical']}/{copy[b]['first_piece_docs']}" for b in bins), ""]
    lines += ["### Beslisregel per bak (ctx tegen in stukken)", "",
              "| bak | (i) inconsistentie ≤ 0,75 × stukken | (i) tekentoets p ≤ 0,05 | (ii) af ≥ stukken − 1 | (iii) weglatingen ≤ +1 pp | (iv) QE ≤ +0,2 | (v) kopie ≤ +1 pp | haalt de regel |",
              "|---:|---|---|---|---|---|---|---|"]
    for b in bins:
        v = verdict[b]
        if not v["measured"]:
            lines.append(f"| {b} | niet gemeten | | | | | | nee |")
            continue
        k = cons[b]
        inc = v["inconsistency"]
        lines.append(
            f"| {b} | {'ja' if v['i_ratio'] else 'nee'} ({pct(inc['ctx'])} tegen {pct(inc['stukken'])}) "
            f"| {'ja' if v['i_sign'] else 'nee'} ({k['paired']['ctx_better']} beter, {k['paired']['ctx_worse']} slechter van "
            f"{k['paired']['terms']}; p = {num(k['paired']['p'], 4)}) "
            f"| {'ja' if v['ii'] else 'nee'} ({v['done']['ctx']} tegen {v['done']['stukken']}) "
            f"| {'ja' if v['iii'] else 'nee'} ({num(v['omission_diff_pp'], 2, True)} pp) "
            f"| {'ja' if v['iv'] else 'nee'} ({num(v['qe_diff'], 3, True) if v['qe_known'] else 'geen QE'}) "
            f"| {'ja' if v['v'] else 'nee'} ({num(v['copy_diff_pp'], 2, True)} pp) "
            f"| **{'ja' if v['pass'] else 'nee'}** |")
    passing = r["passing_bins"]
    lines += ["", f"**ctx haalt de regel in: {', '.join(map(str, passing)) if passing else 'geen bak'}**", ""]
    lines += ["### Termen die in één conditie consistent zijn en in de andere niet (voorbeelden)", "",
              "| bak | document | term | soort | stukken | ctx | |", "|---:|---|---|---|---|---|---|"]
    for key, label in (("ctx_better", "ctx consistent"), ("ctx_worse", "stukken consistent")):
        for e in r["examples"][key]:
            fmt = "; ".join(f"{w} ×{n}" for w, n in e["stukken"])
            fmt2 = "; ".join(f"{w} ×{n}" for w, n in e["ctx"])
            lines.append(f"| {e['bin']} | `{e['doc']}` | {e['term']} | {e['kind']} | {fmt} | {fmt2} | {label} |")
    lines += ["", "### Kosten (bt_length_ctx.py per proces van 4 chips)", "",
              "| log | slots | laden (s) | rondes | prompts | s per ronde, gem. | totale tijd (s) | invoertokens | uitvoertokens/s | piek HBM per chip (GiB) | geheugen op |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for t in r["costs"]:
        if t["extract"] is not None:
            e = t["extract"]
            lines.append(f"| {t['log']} | {t['slots'] or '–'} | {t['load_s'] if t['load_s'] is not None else '–'} | – "
                         f"| {e['asked']} | – | {num(e['seconds'], 0)} | {e['tokens_in']} | – "
                         f"| {num(t['peak_gib'], 2) if t['peak_gib'] is not None else '–'} | {'ja' if t['oom'] else ''} |")
            continue
        lines.append(
            f"| {t['log']} | {t['slots'] or '–'} | {t['load_s'] if t['load_s'] is not None else '–'} | {t['rounds']} "
            f"| {t['prompts']} | {num(statistics.fmean(t['round_seconds']), 1) if t['round_seconds'] else '–'} "
            f"| {num(t['seconds'], 0)} | {t['tokens_in']} "
            f"| {num(t['tokens_out'] / t['seconds'], 0) if t['seconds'] else '–'} "
            f"| {num(t['peak_gib'], 2) if t['peak_gib'] is not None else '–'} | {'ja' if t['oom'] else ''} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- blind

RUBRIC = """# Blinde vergelijking: twee Engelse vertalingen van hetzelfde Nederlandse document

Elk bestand `doc-NN.md` bevat een Nederlands document (een aaneengesloten stuk van een echt document, ~3.000-4.000
tokens) en twee Engelse vertalingen, **A** en **B**. Beide zijn in dezelfde stukken vertaald (opeenvolgende hele
alinea's tot ~512 tokens); de grenzen tussen de stukken zijn in alle drie de teksten gemarkeerd met een regel
`— stukgrens k —`. Welke vertaling op welke manier gemaakt is, staat niet in het bestand.

Beoordeel per document, en noteer bij elk oordeel een of twee concrete plaatsen (citaat) als onderbouwing:

1. **Consistentie** van termen, namen, titels en aanspreekvormen over het hele document, vooral over de stukgrenzen
   heen (wordt hetzelfde Nederlandse begrip steeds hetzelfde vertaald?): A beter / B beter / gelijk.
2. **Samenhang** over de stukgrenzen: verwijzingen (voornaamwoorden, "deze", "voornoemde"), verbindingswoorden,
   tijden en register lopen door; geen herhaalde of weggevallen zin bij een stukgrens: A beter / B beter / gelijk.
3. **Betekenisfouten**: tel per vertaling de fouten die de betekenis veranderen (verkeerd vertaald, weggelaten,
   toegevoegd); geef het aantal voor A en voor B.
4. **Herhaling of kopie**: staat in A of B aan het begin van een stuk tekst die al aan het eind van het vorige stuk
   stond? ja/nee per vertaling, met de plaats.
5. **Voorkeur** over alles: A / B / gelijk.

Antwoord per document als één regel JSON:
`{"doc": "doc-NN", "consistentie": "A|B|gelijk", "samenhang": "A|B|gelijk", "fouten_A": n, "fouten_B": n,
"kopie_A": true|false, "kopie_B": true|false, "voorkeur": "A|B|gelijk", "onderbouwing": "..."}`
"""


def blind(args):
    c = Path(args.collect)
    documents = {d["id"]: d for d in bl.read_rows(args.docs)}
    pieces = [p for p in bl.read_rows(c / "pieces.parquet") if p["bin"] == args.bin]
    by_doc = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in pieces:
        by_doc[p["doc"]][p["condition"]][p["k"]] = p
    rng = random.Random(f"{SEED}/blind/{args.bin}")
    per_source = collections.defaultdict(list)
    for doc in sorted(by_doc):
        per_source[documents[doc]["source"]].append(doc)
    chosen = []
    sources = [s for s in bl.SOURCE_ORDER if per_source[s]]
    for s in sources:
        rng.shuffle(per_source[s])
    while len(chosen) < args.n and any(per_source[s] for s in sources):
        for s in sources:
            if per_source[s] and len(chosen) < args.n:
                chosen.append(per_source[s].pop(0))
    rng.shuffle(chosen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {}

    def marked(parts):
        text = []
        for k, part in enumerate(parts):
            if k:
                text.append(f"— stukgrens {k} —")
            text.append(part.strip())
        return "\n\n".join(text)

    index = ["# Blinde vergelijking, bak " + str(args.bin), "", "Rubriek: `rubric.md`. Documenten:", ""]
    for n, doc in enumerate(chosen, 1):
        name = f"doc-{n:02d}"
        conds = list(CONDITIONS)
        rng.shuffle(conds)
        key[name] = {"doc": doc, "A": conds[0], "B": conds[1], "source": documents[doc]["source"]}
        ks = sorted(by_doc[doc]["stukken"])
        nl = marked([by_doc[doc]["stukken"][k]["text_nl"] for k in ks])
        a = marked([by_doc[doc][conds[0]][k]["text_en"] for k in ks])
        b = marked([by_doc[doc][conds[1]][k]["text_en"] for k in ks])
        (out / f"{name}.md").write_text(f"# {name}\n\n## Nederlands (bron)\n\n{nl}\n\n## Vertaling A\n\n{a}\n\n"
                                        f"## Vertaling B\n\n{b}\n")
        index.append(f"- `{name}.md` ({len(ks)} stukken, {documents[doc]['tokens']} brontokens)")
    (out / "index.md").write_text("\n".join(index) + "\n")
    (out / "rubric.md").write_text(RUBRIC)
    (out.parent / f"blind-key-{args.bin}.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"BLIND {json.dumps({'bin': args.bin, 'docs': len(chosen), 'out': str(out)})}", flush=True)


# ---------------------------------------------------------------------------------------------------- main

def with_memory_line(fn, args):
    """fn(args), then the MEM line of bt_length_translate.py (also after a failure); ends the process with os._exit,
    whatever threads MaxText left (translate.py's __main__)"""
    import bt_length_translate as blt

    code, error = 0, None
    try:
        fn(args)
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException as e:  # noqa: BLE001 - the process ends here with the error's traceback and code
        traceback.print_exc()
        code, error = 1, f"{type(e).__name__}: {str(e)[:300]}"
    try:
        import jax

        print(blt.memory_line(jax.local_devices(), error), flush=True)
    except Exception as e:  # noqa: BLE001 - the line is a measurement; its failure must not hide the code
        print(f"MEM {json.dumps({'peak_gib_max': None, 'unavailable': str(e)[:200]})}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--docs", required=True)
    a.add_argument("--segments", nargs="+", required=True)
    a.add_argument("--tokenizer", required=True)
    a.add_argument("--bins", default=",".join(map(str, BINS)))
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", required=True)
    for name in ("translate", "extract"):
        a = sub.add_parser(name)
        if name == "translate":
            a.add_argument("--plan", required=True)
            a.add_argument("--max-docs", type=int, default=None)
            a.add_argument("--max-rounds", type=int, default=None)
        else:
            a.add_argument("--queries", required=True)
        a.add_argument("--out", required=True, help="translate: bucket directory of the rounds; extract: local parquet")
        a.add_argument("--tokenizer", required=True)
        a.add_argument("--ckpt", required=True)
        a.add_argument("--slots", type=int, default=None)
        a.add_argument("--sync-every", type=int, default=8)
        a.add_argument("--work-dir", default="/dev/shm/translate-generic/work-ctx")
    a = sub.add_parser("collect")
    a.add_argument("--docs", required=True)
    a.add_argument("--plan", nargs="+", required=True)
    a.add_argument("--stukken", required=True)
    a.add_argument("--ctx", required=True)
    a.add_argument("--measured-a", required=True)
    a.add_argument("--bins", default=",".join(map(str, BINS)))
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", required=True)
    a = sub.add_parser("report")
    a.add_argument("--collect", required=True)
    a.add_argument("--units-a", required=True)
    a.add_argument("--qe", nargs="*", default=[])
    a.add_argument("--answers", nargs="*", default=[])
    a.add_argument("--logs", default=None)
    a.add_argument("--setup", default=None)
    a.add_argument("--title", default="Deel A, conditie ctx: Gemma 4 31B-it nl→en in stukken met het vorige stuk als context, tegen in stukken")
    a.add_argument("--out", required=True)
    a = sub.add_parser("blind")
    a.add_argument("--collect", required=True)
    a.add_argument("--docs", required=True)
    a.add_argument("--bin", type=int, default=BLIND_BIN)
    a.add_argument("--n", type=int, default=BLIND_DOCS)
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    fn = {"plan": plan, "translate": translate, "collect": collect, "extract": extract, "report": report,
          "blind": blind}[args.cmd]
    if args.cmd in ("translate", "extract"):
        with_memory_line(fn, args)
    fn(args)


if __name__ == "__main__":
    main()
