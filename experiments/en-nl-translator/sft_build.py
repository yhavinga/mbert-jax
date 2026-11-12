#!/usr/bin/env python3
"""The training data of the fine-tune's arms (experiments/en-nl-translator/protocol.md; hammond
mbert/designs/en-nl-translator.md §3 step 3, §7 point 6, §9): a query on the pairs tables of the main run and the long
run, the FLAN-nl replay, and the user turn of translate.user_text.

    python sft_build.py build --run RUN --long LONG --replay REPLAY.parquet --flan-eval EVAL-ROWS.parquet \\
        --evalsets EVALSETS.parquet --tokenizer TOK --bt auto|all|N [--qe-max 7.5] [--replay-tag none|conventional] \\
        [--len-factor F] [--block-jsonl blind100.jsonl] [--real REAL.parquet [--real-share 0.2 | --real-all] \\
        [--real-without-qe] [--real-qe-max Q | --real-qe-unbounded] [--real-align-max A] [--bt-per-real K \\
        [--bt-match-set SET [--bt-match-bins B]]] \\
        [--real2 REAL2.parquet --real2-from SET/train-meta.parquet --real2-per-real K2]] \\
        --sets B=none N-token=token+replay R2=none+real ... --out OUT
    python sft_build.py check --data OUT/<set> --tokenizer TOK   (SFT venv: MaxText's own SFT steps on every example)

RUN holds the main run's pairs.parquet (mainset/run/pairs/), passages.parquet and context-turns.parquet
(mainset/run/select/); LONG the long run's pairs.parquet and passages.parquet.

The query (QUERY; Yeb, 9 Nov: strict on the source, mild on the pair). A back-translation pair (kind passage) is
eligible when
- hard_fails is empty (not ended, decode_loop, Dutch left in the English, length ratio, structure; pairs.py);
- its source passed the selection: Gemma's source score >= 0.5, for FineWeb-2 also the origin score >= 0.5 (the
  selection chose no other; checked here);
- MetricX-24-QE en->nl <= --qe-max (QE_MAX by default), a wide bound against outliers near p99 of the training
  passages; chrF and the other scores have no bound;
- it shares no 13-gram of words with an evaluation set (WMT24++, FLORES-200, NTREX-128, Europarl-direct: source and
  reference). FLAN's evaluation rows are kept out of the replay by their segments (as the design has it), not by
  13-grams: their templates' fixed sentences recur in a quarter of the replay (131,273 of 521,329 pairs shared a
  13-gram with them, 10 Nov), so a 13-gram block would take whole tasks out; the report counts that overlap;
- with --len-factor F (arm R2, amendment of 11 Nov): its length ratio, the Dutch characters over the English ones, lies
  within a factor F of the median ratio of its source (median / F to median x F; the medians over the source's
  training passages that pass hard_fails), a bound against outliers on both sides that leaves each source's own level
  as it is;
- it shares no 13-gram with the passages of --block-jsonl (the blind assessment's sources);
- the whole example fits MAX_TOKENS in every prompt of the build.
The split is the selection's (per document): train pairs train, eval pairs eval.

Real pairs (--real, arm R2): English-original text with a human Dutch translation (real_pairs.py: Global Voices,
Europarl-direct after 2003), in TranslateGemma's own prompt without context or tag, all in train. A real pair is eligible
when it passed real_pairs.py's hard checks (its fails empty), its MetricX-24-QE en->nl is <= --qe-max (unless
--real-without-qe, a dry run whose report says so), its length ratio lies within --len-factor of its source's median
(over the source's pairs that pass the hard checks), it shares no 13-gram with an evaluation set or --block-jsonl, and it
fits MAX_TOKENS. A set with +real takes at most --real-share of its examples from them (real = BT x share / (1 -
share)): every eligible Global Voices pair first, then Europarl drawn with a seed to fill the share; fewer, then all.
With --real-all (arm RB, amendment of 11 Nov ~12:00: the Dutch books) it takes every eligible real pair, whatever share
they make. --real-qe-max gives the real pairs a MetricX-QE bound of their own (arm RB2, amendment of 11 Nov ~21:30: the
books at 7.5, the back-translations at --qe-max 5). --bt-per-real K (arm RB2: 4, back-translations : books 80 : 20), with
--bt all and --real-all, then draws K back-translation examples per real pair from every eligible one, uniformly with the
seed: a passage drawn with its context stays a context example, so the context share is that of --bt all in expectation.
--real2 (arm RBR, amendment of 12 Nov ~13:10: the books, and R2's Global Voices and Europarl pairs) adds a second table
of real pairs to a --real-all build: the query of --real at the back-translations' --qe-max (R2's bound) and the length
bound around each of its own sources' medians, as R2's build had them; of those, only the real examples of
--real2-from (R2's train-meta.parquet), each of which must pass that query here; round(--real2-per-real x the --real
pairs) of them drawn uniformly with the seed, so their sources keep R2's shares in expectation.
Arm RL (amendment of 12 Nov ~15:30): --real-qe-unbounded takes the real pairs (the books) without a bound on MetricX-QE
(their QE is still read and reported); --real-align-max A fails a real pair whose sentence alignment (align_edges: Gale
and Church's alignment by length) costs more than A at its start or its end, the shifted passages a QE bound caught
before. --bt-match-set SET draws the --bt-per-real back-translation examples so that their ratio of Dutch to English
words follows that of the real examples of SET's train split (R2's: Europarl-direct and Global Voices), by histogram
matching on log(ratio) (bt_match_draw): --bt-match-bins bins at the target's quantiles, each filled to its share of the
target, without replacement and with the seed; fewer than K per real pair when the sparsest bin for its share does not
hold that many.

Context examples (design §9): a training passage the selection marked as a context example keeps its turns before it,
nearest first, as long as each turn is eligible by the same query; the first that fails ends the context there. Without
a turn left it is a plain example. The share of context examples is CONTEXT_SHARE of the back-translation examples, in
every arm alike: --bt auto takes every eligible context example and as many plain ones as make them that share; --bt N
draws that share of N; --bt all takes every eligible training passage (Yeb, 10 Nov), its context examples whatever
share they make (~7%). The plain examples are drawn per source in proportion to the eligible passages of each source,
so the sources keep their shares.

Prompts (translate_models.yml): a plain example gets chat-v1 without context, which is TranslateGemma's own prompt word
for word; a context example gets chat-v1 with its earlier turns. The tag of a set picks the variant of chat-v1 for its
back-translation examples (TAG_PROMPTS): none, the unused token <unused0> before the text, or one sentence in the
instruction. The FLAN-nl replay (a set with +replay: arm N) is REPLAY_SHARE of the set's examples, without context, in
the prompt of --replay-tag: untagged chat-v1 (none), or with the tag turned round (Yeb, 10 Nov: the back-translations
untagged, the replay with the sentence of chat-v1-tag-conventional, the mode of a close, conventional translation;
at use nothing is added); it is drawn from the replay pairs that are none of FLAN's evaluation rows' segments and share no
13-gram with an evaluation set.

build writes per set OUT/<set>/: train.parquet and eval.parquet (one column `messages`, user then assistant, for
MaxText's SFT pipeline; train shuffled), train-meta.parquet and eval-meta.parquet (row by row: id, kind bt/bt_context/
replay, source, tag, prompt, the English, the context, the label, token counts, QE, chrF), template.jinja (the user
turn as built, in Gemma 3's turn) and manifest.json; and OUT/report.json, OUT/report.md: what each bound costs, the
counts per set, source and kind, and the tokens (per source too).
"""

import argparse
import bisect
import collections
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# translate.py (and translate_models.yml) of FLAN-nl in the repo, or beside this file on a VM
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

import select_passages as sp  # noqa: E402
import translate  # noqa: E402

QE_MAX = 7.5  # p99 of the eligible training passages before this bound: 7.416 (10 Nov)
SOURCE_MIN = 0.5
ORIGIN_MIN = 0.5
QUERY = {"hard_fails": "empty", "source_score_min": SOURCE_MIN, "origin_score_min (fineweb)": ORIGIN_MIN,
         "qe_en_nl_max": QE_MAX, "chrf_min": None, "eval_13gram": "none shared"}
CONTEXT_SHARE = 0.2
REPLAY_SHARE = 0.2
REAL_SHARE = 0.2  # arm R2: real pairs at most this share of the examples (Yeb's main session, 11 Nov)
REAL_FIRST = "globalvoices"  # every eligible pair of this source first, the others fill the share
REAL_PROMPT = "chat-v1"  # TranslateGemma's own prompt, no tag (amendment of 11 Nov)
MAX_TOKENS = 2048  # the fine-tune's sequence length (design §7 point 7)
TAG_PROMPTS = {"none": "chat-v1", "token": "chat-v1-tag-token", "instruction": "chat-v1-tag-instruction",
               "conventional": "chat-v1-tag-conventional"}
TAG_TOKEN = "<unused0>"
TAG_SENTENCE = "Translate into natural, original-sounding Dutch."
CONVENTIONAL_SENTENCE = "Translate closely, as a conventional translation."  # Yeb's wording, 10 Nov
TAG_SENTENCES = {"instruction": TAG_SENTENCE, "conventional": CONVENTIONAL_SENTENCE}
SEED = "en-nl-sft/0"
MODEL = "gemma3-27b"  # its user turn; TranslateGemma 12B's is the same
# The turn strings of Gemma 3, in MaxText's chat template: what translate.py's TurnCodec puts around the user turn
TEMPLATE = (
    "{{ bos_token }}\n"
    "{%- if (messages[0]['role'] != 'user') -%}{{ raise_exception('Conversations must start with a user prompt.') }}"
    "{%- endif -%}\n"
    "{%- for message in messages -%}\n"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif -%}\n"
    "{%- if message['content'] is none or message['content'] is not string -%}"
    "{{ raise_exception('content must be a string') }}{%- endif -%}\n"
    "{%- if message['role'] == 'assistant' -%}{{ '<start_of_turn>model\\n' }}{{ message['content'] | trim }}\n"
    "{%- elif message['role'] == 'user' -%}{{ '<start_of_turn>user\\n' }}{{ message['content'] | trim }}\n"
    "{%- else -%}{{ raise_exception('Conversations must only contain user or assistant roles.') }}{%- endif -%}\n"
    "{{ '<end_of_turn>\\n' }}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\\n' }}{%- endif -%}\n"
)
# arm RL's sentence alignment (Gale & Church 1993, "A Program for Aligning Sentences in Bilingual Corpora",
# Computational Linguistics 19(1)): the priors of the beads (theirs; 1-0/0-1 and 2-1/1-2 split equally over the two
# directions, so that they sum to 1) and the variance of the length difference per character (6.8); the expected ratio
# is the source's median ratio of characters (length_bounds), not their 1
ALIGN_PRIORS = {(1, 1): 0.89, (1, 0): 0.0099 / 2, (0, 1): 0.0099 / 2, (2, 1): 0.089 / 2, (1, 2): 0.089 / 2,
                (2, 2): 0.011}
ALIGN_VARIANCE = 6.8
# a sentence ends at . ! ? or … (closing quotes, brackets, Markdown's _ and * after it), white space, and a capital, a
# digit or an opening quote, bracket or dash
SENTENCE_END = re.compile(r"[.!?…]+[\"'’”»)\]_*]*\s+(?=[\"'‘“„«(\[—–\-_*]*\s*[A-Z0-9À-Ý])")
# a full stop after these (lower case), after an initial or after a bare number ("5." of a note) ends no sentence
NOT_A_SENTENCE_END = {"mr", "mrs", "ms", "dr", "st", "prof", "jr", "sr", "lt", "col", "gen", "capt", "rev", "vs", "mevr",
                      "dhr", "drs", "ir", "ing", "bijv", "nr", "ca", "jhr", "mgr"}
QE_BANDS = (5, 7.5, 10, 15, 20, 23)  # the report of the alignment check per band of MetricX-QE
META_COLUMNS = ["id", "kind", "source", "length_class", "tag", "prompt", "en", "context", "context_ids", "label",
                "tokens_prompt", "tokens_target", "qe_en_nl", "chrf"]


def speaker_label(speaker, party):
    """'De heer Thijssen (PvdA)', 'De voorzitter' or None (gemma4_pilot.speaker_label)."""
    speaker, party = (speaker or "").strip(), (party or "").strip()
    return f"{speaker} ({party})" if speaker and party else speaker or None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocked_ngrams(evalsets):
    """The 13-grams no training example may share: source and reference of every evaluation set."""
    blocked = set()
    for r in evalsets:
        blocked |= sp.ngrams(r["source"] or "") | sp.ngrams(r["reference"] or "")
    return blocked


def flan_eval_ngrams(flan_eval):
    """The 13-grams of FLAN's evaluation rows (inputs and targets): counted in the replay, not blocked."""
    grams = set()
    for r in flan_eval:
        grams |= sp.ngrams(r["inputs"] or "") | sp.ngrams(r["targets"] or "")
    return grams


def char_ratio(nl, en):
    """The Dutch characters over the English ones (stripped)"""
    return len(nl.strip()) / max(1, len(en.strip()))


def word_ratio(nl, en):
    """The Dutch words over the English ones (split on white space), the measure of the main session (11 Nov)"""
    return len(nl.split()) / max(1, len(en.split()))


def median(values):
    return quantile(sorted(values), 0.5)


def length_bounds(rows, factor):
    """{source: {median, lo, hi}}: the bounds of --len-factor around the median char_ratio of each source's rows"""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(char_ratio(r["nl"], r["en"]))
    out = {}
    for source, values in sorted(by.items()):
        m = median(values)
        out[source] = {"median": round(m, 4), "lo": round(m / factor, 4), "hi": round(m * factor, 4), "n": len(values)}
    return out


def length_fails(pair, bounds):
    """True when the pair's char_ratio lies outside its source's bounds (no bounds: no length bound)"""
    if bounds is None:
        return False
    b = bounds.get(pair["source"])
    if b is None:
        raise SystemExit(f"{pair.get('id')}: no length bound for source {pair['source']}")
    return not b["lo"] <= char_ratio(pair["nl"], pair["en"]) <= b["hi"]


def sentences(text):
    """The sentences of a text: every line, cut at SENTENCE_END unless the piece ends in an abbreviation, an initial or a
    bare number"""
    out = []
    for line in text.split("\n"):
        line, start, pieces = line.strip(), 0, []
        for m in SENTENCE_END.finditer(line):
            pieces.append(line[start:m.end()].strip())
            start = m.end()
        pieces.append(line[start:].strip())
        merged = []
        for piece in (p for p in pieces if p):
            if merged and _no_sentence_end(merged[-1]):
                merged[-1] += " " + piece
            else:
                merged.append(piece)
        out += merged
    return out


def _no_sentence_end(piece):
    words = piece.rstrip(".").split()
    last = words[-1] if words else ""
    return (piece.endswith(".") and (last.lower() in NOT_A_SENTENCE_END or (len(last) == 1 and last.isupper())
                                     or piece.rstrip(".").isdigit()))


def bead_cost(n_en, n_nl, ratio):
    """-log P(the length difference) of a bead of n_en English and n_nl Dutch characters (Gale & Church's delta, with
    the mean of n_en and n_nl / ratio under the root)"""
    mean = (n_en + n_nl / ratio) / 2
    z = abs(n_nl - n_en * ratio) / math.sqrt(ALIGN_VARIANCE * mean)
    return -math.log(max(math.erfc(z / math.sqrt(2)), 1e-300))  # 2 (1 - Phi(z))


def sentence_alignment(en, nl, ratio):
    """[(sentences of en, sentences of nl, cost)] in order: the alignment of least cost (-log prior - log P(length))
    with the beads of ALIGN_PRIORS, by dynamic programming"""
    if not en or not nl:
        raise ValueError("an alignment needs sentences on both sides")
    len_en, len_nl = [len(s) for s in en], [len(s) for s in nl]
    best = {(0, 0): (0.0, None)}
    for i in range(len(en) + 1):
        for j in range(len(nl) + 1):
            for (di, dj), prior in ALIGN_PRIORS.items():
                if (i, j) == (0, 0) or i < di or j < dj or (i - di, j - dj) not in best:
                    continue
                cost = -math.log(prior) + bead_cost(sum(len_en[i - di:i]), sum(len_nl[j - dj:j]), ratio)
                total = best[(i - di, j - dj)][0] + cost
                if (i, j) not in best or total < best[(i, j)][0]:
                    best[(i, j)] = (total, (di, dj, cost))
    beads, i, j = [], len(en), len(nl)
    while (i, j) != (0, 0):
        di, dj, cost = best[(i, j)][1]
        beads.append((di, dj, cost))
        i, j = i - di, j - dj
    return beads[::-1]


def align_edges(en, nl, ratio):
    """(start, end): the mean cost of the first two beads and of the last two of the pair's sentence alignment. A shifted
    passage (the Dutch begins with a sentence the English lacks, or the English ends with one the Dutch lacks) pairs
    sentences of unlike length there"""
    costs = [c for _, _, c in sentence_alignment(sentences(en), sentences(nl), ratio)]
    return sum(costs[:2]) / len(costs[:2]), sum(costs[-2:]) / len(costs[-2:])


def qe_band(qe):
    """'<=5', '5-7.5', ..., '>23': the band of QE_BANDS a MetricX-QE score falls in"""
    for k, hi in enumerate(QE_BANDS):
        if qe <= hi:
            return f"<={hi}" if k == 0 else f"{QE_BANDS[k - 1]}-{hi}"
    return f">{QE_BANDS[-1]}"


def query_fails(pair, passage, blocked, qe_max=QE_MAX, bounds=None):
    """The parts of the query a back-translation pair fails (empty: eligible). passage is its row of the selection
    (None for a context turn, whose score is in context-turns.parquet). bounds: length_bounds, or None for none."""
    fails = []
    if pair["hard_fails"]:
        fails.append("hard_fails")
    if passage is not None:
        if passage["source_score"] is None or passage["source_score"] < SOURCE_MIN:
            fails.append("source_score")
        if passage["source"] == "fineweb" and (passage["origin_score"] is None or passage["origin_score"] < ORIGIN_MIN):
            fails.append("origin_score")
    if pair["qe_en_nl"] is None or pair["qe_en_nl"] > qe_max:
        fails.append("qe")
    if length_fails(pair, bounds):
        fails.append("length")
    if (sp.ngrams(pair["nl"]) | sp.ngrams(pair["en"])) & blocked:
        fails.append("eval_13gram")
    return fails


def context_of(passage, context_pairs, turns, blocked, qe_max=QE_MAX, bounds=None):
    """(context for user_text, turn ids): the passage's earlier turns, oldest first, cut at the nearest turn that fails
    the query (a turn with a source score below SOURCE_MIN fails too)."""
    kept = []
    for turn_id in reversed(passage["context_ids"] or []):
        pair, turn = context_pairs.get(turn_id), turns.get(turn_id)
        if pair is None or turn is None:
            raise SystemExit(f"{passage['pid']}: context turn {turn_id} has no pair or no row in context-turns.parquet")
        if query_fails(pair, None, blocked, qe_max, bounds) or turn["source_score"] is None or \
                turn["source_score"] < SOURCE_MIN:
            break
        kept.append((turn_id, {"label": speaker_label(turn["speaker"], turn["party"]), "source": pair["en"].strip(),
                               "translation": pair["nl"].strip()}))
    kept.reverse()
    return [c for _, c in kept], [i for i, _ in kept]


def allocate(total, counts):
    """total split over the keys of counts in proportion to them, by largest remainder (sums to total)."""
    whole = sum(counts.values())
    if not whole:
        raise ValueError("nothing to allocate over")
    exact = {k: total * v / whole for k, v in counts.items()}
    out = {k: math.floor(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (out[k] - exact[k], k))[:total - sum(out.values())]:
        out[k] += 1
    return out


def bt_draw(eligible, context_ids, bt, seed):
    """The back-translation examples of a build: (chosen context examples, {source: chosen plain examples}, info).

    eligible: {source: [pid]} of the eligible training passages; context_ids: the pids among them that have a context
    left. bt: 'auto' (every context example, CONTEXT_SHARE of the total), 'all' (every eligible passage) or a total."""
    n_ctx = len(context_ids)
    if bt == "all":
        total = sum(len(v) for v in eligible.values())
    else:
        total = round(n_ctx / CONTEXT_SHARE) if bt == "auto" else int(bt)
    ctx_used = min(n_ctx, round(CONTEXT_SHARE * total))
    targets = allocate(total, {s: len(v) for s, v in eligible.items()})
    rng = random.Random(f"{seed}/context")
    chosen_ctx = sorted(rng.sample(sorted(context_ids), ctx_used))
    ctx_set = set(chosen_ctx)
    sources = {p: s for s, pids in eligible.items() for p in pids}
    plain = {}
    for s in sorted(eligible):
        want = targets[s] - sum(sources[p] == s for p in chosen_ctx)
        pool = sorted(p for p in eligible[s] if p not in ctx_set)
        if want < 0 or want > len(pool):
            raise SystemExit(f"{s}: {want} plain examples wanted, {len(pool)} in the pool (total {total})")
        plain[s] = sorted(random.Random(f"{seed}/plain/{s}").sample(pool, want))
    info = {"bt_total": total, "context_available": n_ctx, "context_used": ctx_used, "targets": targets,
            "context_share": round(ctx_used / total, 4) if total else None}
    return chosen_ctx, plain, info


def user_turn(prompt_name, en, context=None, label=None):
    return translate.user_text(translate.PROMPTS[prompt_name], en, "en", "nl", context or None,
                               label if context else None)


class TokenCounter:
    """Token counts of a prompt and a target as MaxText's template makes them (Gemma 3's turn around the user turn)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.turn = translate.MODELS[MODEL]["turn"]

    def prompt_ids(self, user):
        return self.tokenizer(self.turn["prefix"] + user + self.turn["suffix"], add_special_tokens=False)["input_ids"]

    def target_ids(self, nl):
        return self.tokenizer(nl + "<end_of_turn>\n", add_special_tokens=False)["input_ids"]

    # A fast tokenizer encodes a batch in parallel; one string at a time costs ~1 ms (10 Nov), too slow for ~1M texts
    def prompt_lengths(self, users, batch=4096):
        return self._lengths([self.turn["prefix"] + u + self.turn["suffix"] for u in users], batch)

    def target_lengths(self, nls, batch=4096):
        return self._lengths([nl + "<end_of_turn>\n" for nl in nls], batch)

    def _lengths(self, texts, batch):
        out = []
        for k in range(0, len(texts), batch):
            out += [len(ids) for ids in self.tokenizer(texts[k:k + batch], add_special_tokens=False)["input_ids"]]
        return out


def candidates_bt(args, blocked, report):
    """The back-translation pairs of both runs with what the query says of them; the context examples' contexts."""
    import pyarrow.parquet as pq

    cols = ["id", "pid", "kind", "source", "split", "nl", "en", "tokens_nl", "tokens_en", "hard_fails", "qe_en_nl", "chrf"]
    pairs = pq.read_table(Path(args.run) / "pairs.parquet", columns=cols).to_pylist()
    long = pq.read_table(Path(args.long) / "pairs.parquet", columns=cols + ["length_class"]).to_pylist()
    pcols = ["pid", "source", "split", "source_score", "origin_score", "context_ids", "speaker", "party"]
    passages = {p["pid"]: p for p in pq.read_table(Path(args.run) / "passages.parquet", columns=pcols).to_pylist()}
    for p in pq.read_table(Path(args.long) / "passages.parquet", columns=pcols).to_pylist():
        if p["pid"] in passages:
            raise SystemExit(f"{p['pid']} is in both selections")
        p["context_ids"] = []  # the long run has no context turns (--no-context)
        passages[p["pid"]] = p
    turns = {t["id"]: t for t in pq.read_table(Path(args.run) / "context-turns.parquet",
                                               columns=["id", "speaker", "party", "source_score"]).to_pylist()}
    for r in pairs:
        r.setdefault("length_class", "short")
    for r in long:
        r["length_class"] = r.get("length_class") or "long"
    context_pairs = {r["pid"]: r for r in pairs if r["kind"] == "context"}
    rows = [r for r in pairs + long if r["kind"] == "passage"]
    for r in rows + list(context_pairs.values()):
        for side in ("nl", "en"):
            if TAG_TOKEN in r[side] or "<unused" in r[side]:
                raise SystemExit(f"{r['pid']}: its {side} has the tag token's text")
    fails = collections.Counter()
    alone = collections.Counter()
    by_split = collections.Counter()
    bounds = None
    if getattr(args, "len_factor", None):
        bounds = length_bounds([r for r in rows if r["split"] == "train" and not r["hard_fails"]], args.len_factor)
        report["length_bounds_bt"] = bounds
    for r in rows:
        p = passages.get(r["pid"])
        if p is None:
            raise SystemExit(f"{r['pid']}: a pair without a passage in the selections")
        if p["split"] != r["split"]:
            raise SystemExit(f"{r['pid']}: split {r['split']} in the pairs, {p['split']} in the selection")
        r["fails"] = query_fails(r, p, blocked, args.qe_max, bounds)
        r["passage"] = p
        by_split[r["split"]] += 1
        for f in r["fails"]:
            alone[(r["split"], f)] += 1
        if r["fails"]:
            fails[(r["split"], r["fails"][0])] += 1
    qe_hard = sorted(r["qe_en_nl"] for r in rows if r["split"] == "train" and not r["hard_fails"])
    report["pairs"] = {"passages": dict(by_split), "context_turns": len(context_pairs)}
    report["query_cost"] = {
        "first_failed": {f"{s}/{f}": n for (s, f), n in sorted(fails.items())},
        "alone": {f"{s}/{f}": n for (s, f), n in sorted(alone.items())},
        "qe_train_after_hard": {"n": len(qe_hard), **{f"p{q}": round(quantile(qe_hard, q / 100), 3)
                                                      for q in (50, 90, 95, 99, 99.5, 99.9)},
                                "max": qe_hard[-1] if qe_hard else None,
                                "over": {str(c): sum(v > c for v in qe_hard)
                                         for c in (3, 4, 5, 6, 7, 7.5, 8, 9, 10, 12, 15)}},
    }
    ctx_stats = collections.Counter()
    for r in rows:
        r["context"], r["context_ids"] = [], []
        r["label"] = speaker_label(r["passage"]["speaker"], r["passage"]["party"]) if r["source"] == "handelingen" \
            else None
        if r["fails"] or not r["passage"]["context_ids"]:
            continue
        ctx_stats[f"{r['split']}/marked"] += 1
        r["context"], r["context_ids"] = context_of(r["passage"], context_pairs, turns, blocked, args.qe_max, bounds)
        dropped = len(r["passage"]["context_ids"]) - len(r["context_ids"])
        ctx_stats[f"{r['split']}/turns_dropped"] += dropped
        ctx_stats[f"{r['split']}/{'with_context' if r['context'] else 'no_turn_left'}"] += 1
    report["context"] = dict(sorted(ctx_stats.items()))
    return rows


def candidates_replay(args, blocked, report, flan_grams=frozenset()):
    import pyarrow.parquet as pq

    replay = pq.read_table(args.replay, columns=["id", "task", "en", "nl", "tokens_en", "tokens_nl"]).to_pylist()
    eval_rows = pq.read_table(args.flan_eval, columns=["segments"]).to_pylist()
    eval_segments = {s for r in eval_rows for s in r["segments"]}
    stats = collections.Counter()
    out = []
    for r in replay:
        for side in ("nl", "en"):
            if "<unused" in r[side]:
                raise SystemExit(f"replay {r['id']}: its {side} has the tag token's text")
        if r["id"] in eval_segments:
            stats["eval_segment"] += 1
            continue
        if (sp.ngrams(r["nl"]) | sp.ngrams(r["en"])) & blocked:
            stats["eval_13gram"] += 1
            continue
        out.append(r)
    report["replay"] = {"pairs": len(replay), "dropped": dict(stats), "candidates": len(out),
                        "kept_sharing_a_13gram_with_flan_eval_rows": sum(
                            bool((sp.ngrams(r["nl"]) | sp.ngrams(r["en"])) & flan_grams) for r in out)}
    return out


def candidates_real(args, blocked, report, path=None, qe_max=None, key="real"):
    """The real pairs of --real (or of path, with its own QE bound, reported under key: --real2) with what the query
    says of them (first and alone per source, as for the back-translations); returns the eligible ones (tokens are
    counted in build)."""
    import pyarrow.parquet as pq

    path = path or args.real
    rows = pq.read_table(path).to_pylist()
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit(f"{path}: real pairs with the same id")
    without_qe = getattr(args, "real_without_qe", False) and key == "real"
    if not without_qe and "qe_en_nl" not in rows[0]:
        raise SystemExit(f"{path} has no qe_en_nl (real_pairs.py join-qe), and --real-without-qe is not set")
    hard_ok = [r for r in rows if not r["fails"]]
    bounds = length_bounds(hard_ok, args.len_factor) if getattr(args, "len_factor", None) else None
    qe_max = real_qe_max(args) if key == "real" else qe_max
    align_max = getattr(args, "real_align_max", None) if key == "real" else None
    if align_max is not None and bounds is None:
        raise SystemExit("--real-align-max aligns at the source's median ratio: it needs --len-factor")
    first, alone, out = collections.Counter(), collections.Counter(), []
    align = {"max": align_max, "edges": [], "by_qe": collections.defaultdict(lambda: [0, 0])}
    for r in rows:
        for side in ("nl", "en"):
            if "<unused" in r[side]:
                raise SystemExit(f"real {r['id']}: its {side} has the tag token's text")
        fails = ["hard:" + f for f in r["fails"]]
        if not r["fails"]:
            if not without_qe:
                if r["qe_en_nl"] is None:
                    raise SystemExit(f"real {r['id']}: passed the hard checks without a MetricX-QE")
                if qe_max is not None and r["qe_en_nl"] > qe_max:
                    fails.append("qe")
            if length_fails(r, bounds):
                fails.append("length")
            if align_max is not None:
                edge = max(align_edges(r["en"], r["nl"], bounds[r["source"]]["median"]))
                align["edges"].append(edge)
                band = align["by_qe"][qe_band(r["qe_en_nl"])] if r["qe_en_nl"] is not None else [0, 0]
                band[0] += 1
                if edge > align_max:
                    fails.append("align")
                    band[1] += 1
            if (sp.ngrams(r["nl"]) | sp.ngrams(r["en"])) & blocked:
                fails.append("eval_13gram")
        for f in fails:
            alone[f"{r['source']}/{f}"] += 1
        if fails:
            first[f"{r['source']}/{fails[0]}"] += 1
        else:
            out.append(r)
    report[key] = {"pairs": dict(collections.Counter(r["source"] for r in rows)),
                      "query_cost": {"first_failed": dict(sorted(first.items())), "alone": dict(sorted(alone.items()))},
                      "eligible": dict(collections.Counter(r["source"] for r in out)), "without_qe": without_qe,
                      "qe_max": None if without_qe else qe_max,
                      "qe_unbounded": not without_qe and qe_max is None,
                      "length_bounds": bounds,
                      "qe_hard_ok": None if without_qe else {
                          s: {"n": len(v), **{f"p{q}": round(quantile(v, q / 100), 3) for q in (50, 90, 95, 99)},
                              "over": {str(c): sum(x > c for x in v) for c in (3, 4, 5, 7.5, 10)}}
                          for s, v in sorted(_by_source(hard_ok, "qe_en_nl").items())}}
    if align_max is not None:
        edges = sorted(align["edges"])
        report[key]["align"] = {
            "max": align_max, "ratio": {s: b["median"] for s, b in bounds.items()}, "checked": len(edges),
            "failed": sum(e > align_max for e in edges),
            "quantiles": {f"p{q}": round(quantile(edges, q / 100), 3) for q in (50, 90, 95, 97, 98, 99)},
            "by_qe": {b: {"pairs": n, "failed": k} for b, (n, k) in sorted(
                align["by_qe"].items(), key=lambda kv: float(kv[0].lstrip("<=>").split("-")[0]))}}
    return out


def real_qe_max(args):
    """The real pairs' bound on MetricX-QE: --real-qe-max, else the back-translations' --qe-max; None (no bound) with
    --real-qe-unbounded"""
    if getattr(args, "real_qe_unbounded", False):
        return None
    own = getattr(args, "real_qe_max", None)
    return args.qe_max if own is None else own


def bt_per_real_draw(bt_rows, n_real, per_real, seed):
    """(the back-translation rows kept, in their order, info): round(per_real x n_real) of bt_rows, drawn uniformly by
    pid with the seed; more wanted than there are: SystemExit"""
    want = round(per_real * n_real)
    if want > len(bt_rows):
        raise SystemExit(f"--bt-per-real {per_real}: {want} back-translation examples wanted for {n_real} real pairs, "
                         f"{len(bt_rows)} eligible")
    keep = set(random.Random(f"{seed}/bt-per-real").sample(sorted(r["pid"] for r in bt_rows), want))
    kept = [r for r in bt_rows if r["pid"] in keep]
    return kept, {"per_real": per_real, "real": n_real, "drawn": want, "of": len(bt_rows),
                  "drawn_by_source": dict(sorted(collections.Counter(r["source"] for r in kept).items()))}


def match_target(set_dir):
    """[(source, ratio)]: the ratio of Dutch to English words (word_ratio) of every real example of a set's train split
    (--bt-match-set; arm RL: R2's, Europarl-direct and Global Voices)"""
    import pyarrow.parquet as pq

    d = Path(set_dir)
    metas = pq.read_table(d / "train-meta.parquet", columns=["kind", "source", "en"]).to_pylist()
    msgs = pq.read_table(d / "train.parquet").column("messages").to_pylist()
    if len(metas) != len(msgs):
        raise SystemExit(f"{d}: {len(msgs)} examples, {len(metas)} meta rows")
    out = [(m["source"], word_ratio(x[1]["content"], m["en"])) for m, x in zip(metas, msgs) if m["kind"] == "real"]
    if not out:
        raise SystemExit(f"{d}: no real examples to match")
    return out


def bt_match_draw(bt_rows, target, want, bins, seed):
    """(the back-translation rows kept, in their order, info): a sample whose log(word_ratio) follows the target's
    (arm RL), by histogram matching. The bins are cut at the target's quantiles k / bins (the outer two open; equal cuts
    merge); bin b gets n x its share of the target (largest remainder), drawn uniformly without replacement with the
    seed. n is want when every bin holds its part, else the largest n for which they all do (and the info says so)."""
    if any(r <= 0 for _, r in target):
        raise SystemExit("a target ratio of 0")
    logs = sorted(math.log(r) for _, r in target)
    cuts = sorted({quantile(logs, k / bins) for k in range(1, bins)})
    pool = collections.defaultdict(list)
    for r in bt_rows:
        ratio = word_ratio(r["nl"], r["en"])
        if ratio <= 0:
            raise SystemExit(f"{r['pid']}: a back-translation without Dutch words")
        pool[bisect.bisect_right(cuts, math.log(ratio))].append(r)
    share = collections.Counter(bisect.bisect_right(cuts, x) for x in logs)
    empty = sorted(b for b in share if not pool[b])
    if empty:
        raise SystemExit(f"--bt-match-set: no back-translation in bin(s) {empty} of the target")
    most = min(want, min(math.floor(len(pool[b]) * len(logs) / share[b]) for b in share))
    n = most
    while any(k > len(pool[b]) for b, k in allocate(n, dict(share)).items()):
        n -= 1
    parts = allocate(n, dict(share))
    keep = set()
    for b in sorted(parts):
        keep |= set(random.Random(f"{seed}/bt-match/{b}").sample(sorted(r["pid"] for r in pool[b]), parts[b]))
    kept = [r for r in bt_rows if r["pid"] in keep]
    edges = [None] + [round(math.exp(c), 4) for c in cuts] + [None]

    def ratios(values):
        values = sorted(values)
        return {"n": len(values), **{f"p{q}": round(quantile(values, q / 100), 4) for q in (5, 10, 25, 50, 75, 90, 95)}}

    info = {"method": "histogram matching on log(words nl / words en)", "bins": len(cuts) + 1, "bins_asked": bins,
            "cuts": edges[1:-1], "want": want, "most": most, "drawn": n, "limited": n < want,
            "per_bin": [{"lo": edges[b], "hi": edges[b + 1], "target": share[b],
                         "target_share": round(share[b] / len(logs), 4), "pool": len(pool[b]), "drawn": parts.get(b, 0)}
                        for b in range(len(cuts) + 1)],
            "ratio": {"target": ratios(r for _, r in target),
                      "pool": ratios(word_ratio(r["nl"], r["en"]) for r in bt_rows),
                      "drawn": ratios(word_ratio(r["nl"], r["en"]) for r in kept)},
            "target_by_source": dict(sorted(collections.Counter(s for s, _ in target).items())),
            "pool_by_source": dict(sorted(collections.Counter(r["source"] for r in bt_rows).items())),
            "drawn_by_source": dict(sorted(collections.Counter(r["source"] for r in kept).items()))}
    return kept, info


def _by_source(rows, key):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            by[r["source"]].append(r[key])
    return {s: sorted(v) for s, v in by.items()}


def real_draw(eligible, cap, seed):
    """(chosen real pairs, info): every eligible pair of REAL_FIRST (a draw of cap when there are more), then the other
    sources' pairs drawn to fill cap; fewer than cap: all of them."""
    first = sorted((r for r in eligible if r["source"] == REAL_FIRST), key=lambda r: r["id"])
    rest = sorted((r for r in eligible if r["source"] != REAL_FIRST), key=lambda r: r["id"])
    if len(first) > cap:
        first = random.Random(f"{seed}/real/{REAL_FIRST}").sample(first, cap)
    need = cap - len(first)
    if len(rest) > need:
        rest = random.Random(f"{seed}/real/rest").sample(rest, need)
    chosen = first + rest
    info = {"cap": cap, "eligible": dict(collections.Counter(r["source"] for r in eligible)),
            "chosen": dict(collections.Counter(r["source"] for r in chosen))}
    return chosen, info


def real2_from_ids(meta_path):
    """The ids of the real examples (kind real) of a set's train-meta.parquet: --real2-from"""
    import pyarrow.parquet as pq

    ids = [m["id"] for m in pq.read_table(meta_path, columns=["id", "kind"]).to_pylist() if m["kind"] == "real"]
    if not ids or len(set(ids)) != len(ids):
        raise SystemExit(f"{meta_path}: {len(ids)} real examples, {len(set(ids))} ids (none, or one twice)")
    return ids


def real2_draw(fitting, from_ids, n_real, per_real, seed):
    """(chosen pairs of --real2 in id order, info): round(per_real x n_real) of the pairs whose id is in from_ids, drawn
    uniformly with the seed. Every id of from_ids must be among fitting (the pairs that pass the query and fit here):
    they passed the same query in that set's build, so a missing one means the bounds differ (SystemExit)."""
    by_id = {r["id"]: r for r in fitting}
    missing = sorted(set(from_ids) - set(by_id))
    if missing:
        raise SystemExit(f"--real2-from: {len(missing)} of its {len(from_ids)} real pairs do not pass the query here "
                         f"(first: {missing[:3]})")
    want = round(per_real * n_real)
    if want > len(from_ids):
        raise SystemExit(f"--real2-per-real {per_real}: {want} pairs wanted for {n_real} real pairs, "
                         f"{len(from_ids)} in --real2-from")
    keep = set(random.Random(f"{seed}/real2").sample(sorted(from_ids), want))
    chosen = sorted((by_id[i] for i in keep), key=lambda r: r["id"])
    info = {"per_real": per_real, "real": n_real, "drawn": want, "of": len(from_ids),
            "eligible": dict(sorted(collections.Counter(r["source"] for r in fitting).items())),
            "from": dict(sorted(collections.Counter(by_id[i]["source"] for i in from_ids).items())),
            "chosen": dict(sorted(collections.Counter(r["source"] for r in chosen).items()))}
    return chosen, info


def quantile(values, q):
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def parse_sets(specs):
    """'N-token=token+replay' -> {name: (tag, replay)}; 'R2=none+real' -> {name: (tag, 'real')} (the real pairs)"""
    sets = {}
    for spec in specs:
        name, _, form = spec.partition("=")
        tag, _, extra = form.partition("+")
        if not name or tag not in TAG_PROMPTS or extra not in ("", "replay", "real"):
            raise SystemExit(f"--sets: NAME=TAG[+replay|+real] with TAG one of {sorted(TAG_PROMPTS)}, not {spec!r}")
        sets[name] = (tag, "real") if extra == "real" else (tag, extra == "replay")
    return sets


def build(args, tokenizer=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = parse_sets(args.sets)
    tags = sorted({tag for tag, _ in sets.values()})
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    counter = TokenCounter(tokenizer)
    tag_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN)
    if counter.prompt_ids(TAG_TOKEN + "The House.").count(tag_id) != 1 or \
            tokenizer(TAG_TOKEN, add_special_tokens=False)["input_ids"] != [tag_id]:
        raise SystemExit(f"{TAG_TOKEN} is not one token of {args.tokenizer}")
    query = QUERY | {"qe_en_nl_max": args.qe_max}
    len_factor = getattr(args, "len_factor", None)
    if len_factor:
        query |= {"len_factor": len_factor}
    if getattr(args, "real_qe_max", None) is not None:
        query |= {"real_qe_en_nl_max": args.real_qe_max}
    if getattr(args, "real_qe_unbounded", False):
        query |= {"real_qe_en_nl_max": None}
    if getattr(args, "real_align_max", None) is not None:
        query |= {"real_align_max": args.real_align_max}
    if getattr(args, "bt_match_set", None):
        query |= {"bt_match": {"bins": args.bt_match_bins, "per_real": args.bt_per_real}}
    if getattr(args, "real2", None):
        query |= {"real2_qe_en_nl_max": args.qe_max}
    real_path = getattr(args, "real", None)
    if any(extra == "real" for _, extra in sets.values()) != bool(real_path):
        raise SystemExit("--real goes with a set NAME=TAG+real, and such a set needs --real")
    replay_prompt = TAG_PROMPTS[args.replay_tag]
    report = {"query": query, "context_share": CONTEXT_SHARE, "replay_share": REPLAY_SHARE, "max_tokens": MAX_TOKENS,
              "seed": args.seed, "bt_arg": args.bt, "sets": {}, "tag_token": {"text": TAG_TOKEN, "id": tag_id},
              "tag_sentences": TAG_SENTENCES, "tag_prompts": TAG_PROMPTS, "replay_tag": args.replay_tag,
              "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in (
                  ("pairs_main", Path(args.run) / "pairs.parquet"), ("pairs_long", Path(args.long) / "pairs.parquet"),
                  ("passages_main", Path(args.run) / "passages.parquet"),
                  ("passages_long", Path(args.long) / "passages.parquet"),
                  ("context_turns", Path(args.run) / "context-turns.parquet"), ("replay", Path(args.replay)),
                  ("flan_eval", Path(args.flan_eval)), ("evalsets", Path(args.evalsets)))}}
    for key in ("real", "real2", "block_jsonl"):
        if getattr(args, key, None):
            report["inputs"][key] = {"path": str(getattr(args, key)), "sha256": sha256(getattr(args, key))}
    evalsets = pq.read_table(args.evalsets, columns=["source", "reference"]).to_pylist()
    flan_eval = pq.read_table(args.flan_eval, columns=["inputs", "targets"]).to_pylist()
    blocked = blocked_ngrams(evalsets)
    report["blocked_13grams"] = len(blocked)
    if getattr(args, "block_jsonl", None):
        extra = set()
        for line in Path(args.block_jsonl).read_text(encoding="utf-8").split("\n"):
            if line.strip():
                extra |= sp.ngrams(json.loads(line)["source"] or "")
        report["blocked_13grams_extra"] = len(extra - blocked)
        blocked |= extra
    print(f"PROGRESS {len(blocked)} blocked 13-grams", flush=True)
    rows = candidates_bt(args, blocked, report)

    # Tokens in every prompt the build uses: an example joins only when it fits MAX_TOKENS in all of them, so that the
    # arms keep the same pairs
    too_long = collections.Counter()
    eligible = [r for r in rows if not r["fails"]]

    def longest_prompt(part):
        lengths = [counter.prompt_lengths([user_turn(TAG_PROMPTS[t], r["en"], r["context"], r["label"]) for r in part])
                   for t in tags]
        return [max(x) for x in zip(*lengths)]

    print(f"PROGRESS {len(eligible)} eligible pairs, counting tokens", flush=True)
    for r, n in zip(eligible, counter.target_lengths([r["nl"].strip() for r in eligible])):
        r["tokens_target"] = n
    for r, n in zip(eligible, longest_prompt(eligible)):
        r["fits"] = n + r["tokens_target"] <= MAX_TOKENS
    # a context that makes an example too long goes; the passage stays, as a plain example
    shortened = [r for r in eligible if not r["fits"] and r["context"]]
    for r in shortened:
        too_long[f"{r['split']}/context_too_long"] += 1
        r["context"], r["context_ids"] = [], []
    for r, n in zip(shortened, longest_prompt(shortened)):
        r["fits"] = n + r["tokens_target"] <= MAX_TOKENS
    for r in eligible:
        if not r["fits"]:
            too_long[f"{r['split']}/too_long"] += 1
    report["too_long"] = dict(too_long)
    train = [r for r in eligible if r["split"] == "train" and r["fits"]]
    evals = [r for r in eligible if r["split"] == "eval" and r["fits"]]
    by_source = collections.defaultdict(list)
    for r in train:
        by_source[r["source"]].append(r["pid"])
    ctx_pids = [r["pid"] for r in train if r["context"]]
    chosen_ctx, plain, info = bt_draw(by_source, ctx_pids, args.bt, args.seed)
    report["bt"] = info | {"eligible_train": {s: len(v) for s, v in sorted(by_source.items())},
                           "eligible_eval": len(evals)}
    by_pid = {r["pid"]: r for r in train}
    ctx_set = set(chosen_ctx)
    # a passage with a context that is drawn as a plain example goes without its context
    bt_rows = [by_pid[p] for p in chosen_ctx] + [by_pid[p] for s in sorted(plain) for p in plain[s]]
    n_replay = round(info["bt_total"] * REPLAY_SHARE / (1 - REPLAY_SHARE))
    replay = []
    if any(extra is True for _, extra in sets.values()):
        pool = candidates_replay(args, blocked, report, flan_eval_ngrams(flan_eval))
        print(f"PROGRESS {len(pool)} replay candidates, counting tokens", flush=True)
        prompts = counter.prompt_lengths([user_turn(replay_prompt, r["en"]) for r in pool])
        targets = counter.target_lengths([r["nl"].strip() for r in pool])
        fitting = [r | {"tokens_prompt": p, "tokens_target": t} for r, p, t in zip(pool, prompts, targets)
                   if p + t <= MAX_TOKENS]
        report["replay"]["fitting"] = len(fitting)
        if n_replay > len(fitting):
            raise SystemExit(f"{n_replay} replay pairs wanted, {len(fitting)} candidates")
        replay = random.Random(f"{args.seed}/replay").sample(fitting, n_replay)
        report["replay"]["drawn"] = n_replay

    def fit_real(pool, key):
        """the pairs of pool that fit MAX_TOKENS in TranslateGemma's own prompt, with their token counts; the others
        counted as too_long in report[key]"""
        print(f"PROGRESS {len(pool)} eligible real pairs ({key}), counting tokens", flush=True)
        prompts = counter.prompt_lengths([user_turn(REAL_PROMPT, r["en"]) for r in pool])
        targets = counter.target_lengths([r["nl"].strip() for r in pool])
        fitting = []
        for r, n_prompt, n_target in zip(pool, prompts, targets):
            if n_prompt + n_target <= MAX_TOKENS:
                fitting.append(r | {"tokens_prompt": n_prompt, "tokens_target": n_target})
            else:
                report[key]["query_cost"]["first_failed"][f"{r['source']}/too_long"] = \
                    report[key]["query_cost"]["first_failed"].get(f"{r['source']}/too_long", 0) + 1
        return fitting

    real = []
    if real_path:
        fitting = fit_real(candidates_real(args, blocked, report), "real")
        if getattr(args, "real_all", False):
            real = sorted(fitting, key=lambda r: r["id"])
            by_source = dict(collections.Counter(r["source"] for r in real))
            share, draw = "all", {"cap": None, "eligible": by_source, "chosen": by_source}
        else:
            share = getattr(args, "real_share", REAL_SHARE)
            cap = round(info["bt_total"] * share / (1 - share))
            real, draw = real_draw(fitting, cap, args.seed)
        report["real"]["share"] = share
        report["real"]["draw"] = draw
    real2 = []
    if getattr(args, "real2", None):
        if not real_path or not getattr(args, "real_all", False):
            raise SystemExit("--real2 adds to --real-all's real pairs")
        pool2 = candidates_real(args, blocked, report, path=args.real2, qe_max=args.qe_max, key="real2")
        real2, report["real2"]["draw"] = real2_draw(fit_real(pool2, "real2"), real2_from_ids(args.real2_from),
                                                    len(real), args.real2_per_real, args.seed)
        report["inputs"]["real2_from"] = {"path": str(args.real2_from), "sha256": sha256(args.real2_from)}
        if {r["id"] for r in real} & {r["id"] for r in real2}:
            raise SystemExit("--real and --real2 share an id")
    if getattr(args, "bt_per_real", None):
        if args.bt != "all" or not getattr(args, "real_all", False) or \
                any(extra != "real" for _, extra in sets.values()):
            raise SystemExit("--bt-per-real draws from --bt all against --real-all's real pairs, in sets with +real only")
        if getattr(args, "bt_match_set", None):
            print(f"PROGRESS matching the back-translations to {args.bt_match_set}", flush=True)
            bt_rows, report["bt"]["match_draw"] = bt_match_draw(bt_rows, match_target(args.bt_match_set),
                                                                round(args.bt_per_real * len(real)), args.bt_match_bins,
                                                                args.seed)
            report["bt"]["match_draw"]["context"] = sum(r["pid"] in ctx_set for r in bt_rows)
            report["inputs"]["bt_match_set"] = {"path": str(args.bt_match_set), **{
                f: sha256(Path(args.bt_match_set) / f) for f in ("train.parquet", "train-meta.parquet")}}
        else:
            bt_rows, report["bt"]["per_real_draw"] = bt_per_real_draw(bt_rows, len(real), args.bt_per_real, args.seed)
            report["bt"]["per_real_draw"]["context"] = sum(r["pid"] in ctx_set for r in bt_rows)

    def real_example(r):
        user = user_turn(REAL_PROMPT, r["en"])
        return ({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["nl"].strip()}]},
                {"id": r["id"], "kind": "real", "source": r["source"], "length_class": None, "tag": "none",
                 "prompt": REAL_PROMPT, "en": r["en"].strip(), "context": [], "context_ids": [], "label": None,
                 "tokens_prompt": r["tokens_prompt"], "tokens_target": r["tokens_target"],
                 "qe_en_nl": r.get("qe_en_nl"), "chrf": None})

    def bt_examples(part, tag, with_context):
        users = [user_turn(TAG_PROMPTS[tag], r["en"], r["context"] if c else [], r["label"])
                 for r, c in zip(part, with_context)]
        return [bt_example(r, tag, c, u, n)
                for r, c, u, n in zip(part, with_context, users, counter.prompt_lengths(users))]

    def bt_example(r, tag, with_context, user, tokens_prompt):
        context = r["context"] if with_context else []
        prompt = TAG_PROMPTS[tag]
        return ({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["nl"].strip()}]},
                {"id": r["id"], "kind": "bt_context" if context else "bt", "source": r["source"],
                 "length_class": r["length_class"], "tag": tag, "prompt": prompt, "en": r["en"].strip(),
                 "context": context, "context_ids": r["context_ids"] if context else [],
                 "label": r["label"] if context else None, "tokens_prompt": tokens_prompt,
                 "tokens_target": r["tokens_target"], "qe_en_nl": r["qe_en_nl"], "chrf": r["chrf"]})

    def replay_example(r):
        user = user_turn(replay_prompt, r["en"])
        return ({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["nl"].strip()}]},
                {"id": f"replay:{r['id']}", "kind": "replay", "source": f"flan:{r['task']}", "length_class": None,
                 "tag": args.replay_tag, "prompt": replay_prompt, "en": r["en"].strip(), "context": [], "context_ids": [],
                 "label": None, "tokens_prompt": r["tokens_prompt"], "tokens_target": r["tokens_target"],
                 "qe_en_nl": None, "chrf": None})

    for name, (tag, extra) in sets.items():
        with_replay = extra is True
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        examples = bt_examples(bt_rows, tag, [r["pid"] in ctx_set for r in bt_rows])
        if with_replay:
            examples += [replay_example(r) for r in replay]
        if extra == "real":
            examples += [real_example(r) for r in real + real2]
        random.Random(f"{args.seed}/shuffle").shuffle(examples)
        eval_examples = bt_examples(evals, tag, [bool(r["context"]) for r in evals])
        for split, part in (("train", examples), ("eval", eval_examples)):
            pq.write_table(pa.table({"messages": [m["messages"] for m, _ in part]}), d / f"{split}.parquet")
            pq.write_table(pa.Table.from_pylist([meta for _, meta in part]), d / f"{split}-meta.parquet")
        (d / "template.jinja").write_text(TEMPLATE, encoding="utf-8")
        stats = set_stats(examples, eval_examples)
        manifest = {"set": name, "tag": tag, "replay": with_replay, "prompts": sorted({m["prompt"] for _, m in examples}),
                    "replay_tag": args.replay_tag if with_replay else None, "query": query, "seed": args.seed,
                    "bt": args.bt, "real": extra == "real",
                    "real_without_qe": bool(getattr(args, "real_without_qe", False)) if extra == "real" else None,
                    **stats,
                    "files": {f.name: sha256(f) for f in sorted(d.iterdir()) if f.suffix in (".parquet", ".jinja")}}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        report["sets"][name] = {"tag": tag, "replay": with_replay, "replay_tag": args.replay_tag if with_replay else None,
                                "real": extra == "real", **stats}
        print(f"SET {name} {json.dumps(stats['train'])}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n",
                                     encoding="utf-8")
    (out / "report.md").write_text(report_md(report), encoding="utf-8")


def set_stats(train, evals):
    def part(rows):
        metas = [m for _, m in rows]
        kinds = collections.Counter(m["kind"] for m in metas)
        sources = collections.Counter(m["source"] if m["kind"] != "replay" else "replay" for m in metas)
        tokens = sum(m["tokens_prompt"] + m["tokens_target"] for m in metas)
        loss = sum(m["tokens_target"] for m in metas)
        bt = kinds["bt"] + kinds["bt_context"]
        by_source = collections.defaultdict(lambda: [0, 0])
        for m in metas:
            s = by_source[m["source"] if m["kind"] != "replay" else "replay"]
            s[0] += m["tokens_prompt"] + m["tokens_target"]
            s[1] += m["tokens_target"]
        ratios = collections.defaultdict(lambda: ([], []))
        for msgs, m in rows:
            nl = msgs["messages"][1]["content"]
            group = "real" if m["kind"] == "real" else "replay" if m["kind"] == "replay" else "bt"
            for key in (group, "all", f"source/{m['source'] if m['kind'] != 'replay' else 'replay'}"):
                ratios[key][0].append(word_ratio(nl, m["en"]))
                ratios[key][1].append(char_ratio(nl, m["en"]))
        length = {k: {"n": len(w), "median_words": round(median(w), 4), "median_chars": round(median(c), 4)}
                  for k, (w, c) in sorted(ratios.items())}
        return {"examples": len(metas), "kinds": dict(sorted(kinds.items())), "sources": dict(sorted(sources.items())),
                "length_ratio": length, "real_share": round(kinds["real"] / len(metas), 4) if metas else None,
                "tokens_by_source": {k: v[0] for k, v in sorted(by_source.items())},
                "loss_tokens_by_source": {k: v[1] for k, v in sorted(by_source.items())},
                "long": sum(m["length_class"] == "long" for m in metas),
                "context_share_of_bt": round(kinds["bt_context"] / bt, 4) if bt else None,
                "replay_share": round(kinds["replay"] / len(metas), 4) if metas else None,
                "tokens": tokens, "loss_tokens": loss,
                "max_tokens": max((m["tokens_prompt"] + m["tokens_target"] for m in metas), default=0)}

    return {"train": part(train), "eval": part(evals)}


def _n(x):
    return f"{x:,}".replace(",", ".") if isinstance(x, int) else str(x)


def report_md(report):
    lines = ["# Trainingsdata van de fine-tune (sft_build.py)", "",
             f"Query: {json.dumps(report['query'], ensure_ascii=False)}; contextaandeel {report['context_share']}, "
             f"replay-aandeel {report['replay_share']} (arm N), hoogstens {report['max_tokens']} tokens per voorbeeld; "
             f"seed `{report['seed']}`, `--bt {report['bt_arg']}`. Geblokkeerde 13-grammen van de evaluatiesets: "
             f"{_n(report['blocked_13grams'])}.", "",
             "## Wat elke grens kost (passages; `first_failed`: de eerste grens die een paar niet haalt, in de volgorde "
             "hard_fails, bronscore, herkomst, QE, 13-gram; `alone`: alle paren die die grens niet halen)", "",
             "| split/grens | eerste | alleen |", "|---|---:|---:|"]
    cost = report["query_cost"]
    for key in sorted(set(cost["first_failed"]) | set(cost["alone"])):
        lines.append(f"| {key} | {_n(cost['first_failed'].get(key, 0))} | {_n(cost['alone'].get(key, 0))} |")
    q = cost["qe_train_after_hard"]
    lines += ["", f"MetricX-24-QE van de trainpassages die door de harde controles komen (n = {_n(q['n'])}): "
              + ", ".join(f"{k} {q[k]}" for k in ("p50", "p90", "p95", "p99", "p99.5", "p99.9", "max")) + ".", "",
              "| QE > | paren |", "|---:|---:|"] + [f"| {c} | {_n(n)} |" for c, n in q["over"].items()]
    if "length_bounds_bt" in report:
        lines += ["", "Lengtegrens (`--len-factor`, tekens Nederlands / tekens Engels binnen een factor van de mediaan van de "
                  "bron; de medianen over de trainpassages die door `hard_fails` komen):", "",
                  "| bron | n | mediaan | ondergrens | bovengrens |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {k} | {_n(v['n'])} | {v['median']} | {v['lo']} | {v['hi']} |"
                  for k, v in report["length_bounds_bt"].items()]
    if "blocked_13grams_extra" in report:
        lines += ["", f"13-grammen van `--block-jsonl` (de blinde passages) erbij: {_n(report['blocked_13grams_extra'])}."]
    lines += ["", f"Context: {json.dumps(report['context'])}; te lang: {json.dumps(report['too_long'])}.",
              f"BT: {json.dumps(report['bt'], ensure_ascii=False)}."]
    if "replay" in report:
        lines.append(f"Replay: {json.dumps(report['replay'])}.")
    if "real" in report:
        r = report["real"]
        lines += ["", "## Echte vertaalparen (`--real`)", ""]
        if r["without_qe"]:
            lines += ["**Proefbouw zonder MetricX-QE op de echte paren (`--real-without-qe`): niet de set voor de "
                      "training.**", ""]
        lines += ["Wat elke grens kost (paren; `hard:` = de harde controles van `real_pairs.py`; eerste grens die faalt / "
                  "alle paren die hem niet halen):", "", "| bron/grens | eerste | alleen |", "|---|---:|---:|"]
        cost = r["query_cost"]
        for key in sorted(set(cost["first_failed"]) | set(cost["alone"])):
            lines.append(f"| {key} | {_n(cost['first_failed'].get(key, 0))} | {_n(cost['alone'].get(key, 0))} |")
        drawn = ("alle paren die in aanmerking komen (`--real-all`)" if r.get("share") == "all" else
                 f"getrokken (hoogstens {r.get('share')} van de voorbeelden, {REAL_FIRST} eerst)")
        lines += ["", f"Paren: {json.dumps(r['pairs'])}; in aanmerking: {json.dumps(r['eligible'])}; {drawn}: "
                  f"{json.dumps(r.get('draw'))}."]
        if r.get("qe_max") is not None:
            lines += ["", f"MetricX-24-QE-grens van de echte paren: {r['qe_max']}."]
        if r.get("qe_unbounded"):
            lines += ["", "Geen MetricX-24-QE-grens op de echte paren (`--real-qe-unbounded`); hun QE staat in de meta."]
        if r.get("align"):
            a = r["align"]
            lines += ["", f"Uitlijncontrole (`--real-align-max {a['max']}`): zinsuitlijning naar lengte (Gale en Church "
                      f"1993) met de mediane tekenverhouding van de bron ({json.dumps(a['ratio'])}); een paar valt af als "
                      f"het gemiddelde van de kosten van de eerste twee of de laatste twee paren zinnen hoger is dan "
                      f"{a['max']}. Gecontroleerd {_n(a['checked'])}, afgevallen {_n(a['failed'])}; kosten aan de rand "
                      f"{json.dumps(a['quantiles'])}.", "", "| MetricX-QE | paren | afgevallen | aandeel |",
                      "|---|---:|---:|---:|"]
            lines += [f"| {b} | {_n(v['pairs'])} | {_n(v['failed'])} | {v['failed'] / v['pairs']:.1%} |"
                      for b, v in a["by_qe"].items()]
        if "match_draw" in report["bt"]:
            d = report["bt"]["match_draw"]
            lines += ["", f"BT hersampled (`--bt-match-set`, `--bt-per-real`): {d['method']}, {d['bins']} bakken op de "
                      f"kwantielen van het doel; gewenst {_n(d['want'])}, hoogstens {_n(d['most'])}, getrokken "
                      f"{_n(d['drawn'])}{' (minder dan gewenst: de dunste bak begrenst)' if d['limited'] else ''} van "
                      f"de BT-voorbeelden die in aanmerking komen ({_n(d['context'])} met context); per bron "
                      f"{json.dumps(d['drawn_by_source'])} (in aanmerking {json.dumps(d['pool_by_source'])}).", "",
                      "| bak (woorden NL / EN) | doel | aandeel doel | in aanmerking | getrokken |",
                      "|---|---:|---:|---:|---:|"]
            lines += [f"| {b['lo'] or '…'} – {b['hi'] or '…'} | {_n(b['target'])} | {b['target_share']:.3f} | "
                      f"{_n(b['pool'])} | {_n(b['drawn'])} |" for b in d["per_bin"]]
            lines += ["", "| verdeling | n | p5 | p10 | p25 | p50 | p75 | p90 | p95 |", "|---|---:|" + "---:|" * 7]
            lines += [f"| {k} | {_n(v['n'])} | " + " | ".join(f"{v[f'p{q}']:.3f}" for q in (5, 10, 25, 50, 75, 90, 95))
                      + " |" for k, v in d["ratio"].items()]
        if "per_real_draw" in report["bt"]:
            d = report["bt"]["per_real_draw"]
            lines += ["", f"BT getrokken (`--bt-per-real {d['per_real']}`): {_n(d['drawn'])} van {_n(d['of'])} "
                      f"BT-voorbeelden die in aanmerking komen ({d['per_real']} × {_n(d['real'])} echte paren; "
                      f"{_n(d['context'])} met context); per bron {json.dumps(d['drawn_by_source'])}."]
        if r["length_bounds"]:
            lines += ["", "| bron | n | mediaan | ondergrens | bovengrens |", "|---|---:|---:|---:|---:|"]
            lines += [f"| {k} | {_n(v['n'])} | {v['median']} | {v['lo']} | {v['hi']} |" for k, v in r["length_bounds"].items()]
        if r["qe_hard_ok"]:
            lines += ["", f"MetricX-24-QE van de paren die door de harde controles komen: {json.dumps(r['qe_hard_ok'])}."]
    if "real2" in report:
        r = report["real2"]
        d = r["draw"]
        lines += ["", "## Tweede bron van echte paren (`--real2`)", "",
                  f"De query van `--real` met de QE-grens van de BT-paren ({r['qe_max']}) en de lengtegrens rond de "
                  "mediaan van elke eigen bron; daarvan alleen de echte voorbeelden van `--real2-from`, die hier allemaal "
                  "door de query moeten komen.", "",
                  "| bron/grens | eerste | alleen |", "|---|---:|---:|"]
        cost = r["query_cost"]
        for key in sorted(set(cost["first_failed"]) | set(cost["alone"])):
            lines.append(f"| {key} | {_n(cost['first_failed'].get(key, 0))} | {_n(cost['alone'].get(key, 0))} |")
        lines += ["", f"Paren: {json.dumps(r['pairs'])}; in aanmerking: {json.dumps(d['eligible'])}; in `--real2-from`: "
                  f"{json.dumps(d['from'])}; getrokken (`--real2-per-real {d['per_real']}`, {d['per_real']} × "
                  f"{_n(d['real'])} paren van `--real`, uniform): {_n(d['drawn'])} van {_n(d['of'])}, per bron "
                  f"{json.dumps(d['chosen'])}."]
        if r["length_bounds"]:
            lines += ["", "| bron | n | mediaan | ondergrens | bovengrens |", "|---|---:|---:|---:|---:|"]
            lines += [f"| {k} | {_n(v['n'])} | {v['median']} | {v['lo']} | {v['hi']} |" for k, v in r["length_bounds"].items()]
    lines += ["", "## Sets", "", f"Tag van de replay (`--replay-tag`): {report['replay_tag']} "
              f"(`{report['tag_prompts'][report['replay_tag']]}`).", "",
              "| set | tag | replay | split | voorbeelden | bt | bt_context | replay | lang | "
              "tokens | verliestokens | max |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in report["sets"].items():
        for split in ("train", "eval"):
            p = s[split]
            k = p["kinds"]
            lines.append(f"| {name} | {s['tag']} | {'ja' if s['replay'] else 'nee'} | {split} | {_n(p['examples'])} | "
                         f"{_n(k.get('bt', 0))} | {_n(k.get('bt_context', 0))} | {_n(k.get('replay', 0))} | "
                         f"{_n(p['long'])} | {_n(p['tokens'])} | {_n(p['loss_tokens'])} | {_n(p['max_tokens'])} |")
    lines += ["", "Bronnen per set (train): " + "; ".join(f"{n}: {json.dumps(s['train']['sources'])}"
                                                        for n, s in report["sets"].items()), ""]
    lines += ["## Tokens per bron (train; tokens: prompt + doel, doeltokens: het Nederlands + `<end_of_turn>\\n`)", "",
              "| set | bron | voorbeelden | tokens | doeltokens | doeltokens per voorbeeld |", "|---|---|---:|---:|---:|---:|"]
    for name, s in report["sets"].items():
        p = s["train"]
        for source, n in p["sources"].items():
            loss = p["loss_tokens_by_source"][source]
            lines.append(f"| {name} | {source} | {_n(n)} | {_n(p['tokens_by_source'][source])} | {_n(loss)} | "
                         f"{loss / n:.0f} |")
    lines += ["", "## Lengteverhouding (train; mediaan per voorbeeld: woorden Nederlands / woorden Engels, en tekens)", "",
              "| set | deel | n | woorden | tekens |", "|---|---|---:|---:|---:|"]
    for name, s in report["sets"].items():
        for key, v in s["train"].get("length_ratio", {}).items():
            lines.append(f"| {name} | {key} | {_n(v['n'])} | {v['median_words']:.3f} | {v['median_chars']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def check(args):
    """MaxText's own SFT steps (apply_chat_template, tokenization, SFTPromptMasking, as hf_data_processing runs them)
    on every example of a set: the user turn is user_text of its prompt (recomputed from the meta), the inputs are
    Gemma 3's turn around it followed by the Dutch and <end_of_turn>\\n, the targets mask the prompt; an untagged plain
    example's prompt ids are TranslateGemma's own template's; the tag token is in a tagged prompt once, elsewhere
    never."""
    import numpy as np
    import pyarrow.parquet as pq
    import transformers
    from maxtext.input_pipeline import input_pipeline_utils as ipu

    data = Path(args.data)
    template = (data / "template.jinja").read_text(encoding="utf-8")
    mt_tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer, add_bos_token=False, add_eos_token=False,
                                                        legacy=False, extra_special_tokens={})
    mt_tok.chat_template = template
    own = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    counter = TokenCounter(own)
    codec = translate.PromptCodec(own, "en", "nl")
    tag_id = own.convert_tokens_to_ids(TAG_TOKEN)
    pad_id = mt_tok.pad_token_id
    stats = {}
    for split in ("train", "eval"):
        rows = pq.read_table(data / f"{split}.parquet").column("messages").to_pylist()
        metas = pq.read_table(data / f"{split}-meta.parquet").to_pylist()
        if len(rows) != len(metas):
            raise SystemExit(f"{split}: {len(rows)} examples, {len(metas)} meta rows")
        masking = ipu.SFTPromptMasking("messages", completion_only=True, max_target_length=10 ** 9, unk_id=pad_id)
        lengths, kinds = [], collections.Counter()
        for k, (msgs, meta) in enumerate(zip(rows, metas)):
            user, assistant = msgs[0]["content"], msgs[1]["content"]
            if user != user_turn(meta["prompt"], meta["en"], meta["context"], meta["label"]):
                raise SystemExit(f"{split} row {k} ({meta['id']}): the user turn is not user_text of its prompt")
            example = ipu.apply_chat_template({"messages": msgs}, mt_tok, "messages")
            example = ipu.tokenization(example, mt_tok, truncation=False, max_length=MAX_TOKENS,
                                       column_names=["messages"])
            got = masking.map(example)
            prompt = counter.prompt_ids(user)
            if meta["tag"] == "none" and not meta["context"] and prompt != codec.encode(meta["en"]):
                raise SystemExit(f"{split} row {k} ({meta['id']}): not TranslateGemma's own prompt ids")
            if prompt.count(tag_id) != (1 if meta["tag"] == "token" else 0):
                raise SystemExit(f"{split} row {k} ({meta['id']}): the tag token {prompt.count(tag_id)} times")
            for tag, sentence in TAG_SENTENCES.items():
                if user.count(sentence) != (1 if meta["tag"] == tag else 0):
                    raise SystemExit(f"{split} row {k} ({meta['id']}): the {tag} sentence {user.count(sentence)} times")
            completion = counter.target_ids(assistant)
            full = prompt + completion
            if got["inputs"].tolist() != full or got["targets"].tolist() != [pad_id] * len(prompt) + completion:
                raise SystemExit(f"{split} row {k} ({meta['id']}): MaxText's SFT ids differ from the prompt + Dutch + "
                                 f"<end_of_turn>\n")
            if len(full) > MAX_TOKENS or len(prompt) != meta["tokens_prompt"] or \
                    len(completion) != meta["tokens_target"]:
                raise SystemExit(f"{split} row {k} ({meta['id']}): {len(full)} tokens, meta says "
                                 f"{meta['tokens_prompt']} + {meta['tokens_target']}")
            lengths.append(len(full))
            kinds[meta["kind"]] += 1
        stats[split] = {"examples": len(rows), "kinds": dict(kinds), "tokens": int(np.sum(lengths)),
                        "max_tokens": int(np.max(lengths)) if lengths else 0}
    print("CHECK " + json.dumps({"data": str(data), **stats}))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Training data of the fine-tune's arms")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", required=True, help="pairs.parquet, passages.parquet, context-turns.parquet of the main run")
    b.add_argument("--long", required=True, help="pairs.parquet, passages.parquet of the long run")
    b.add_argument("--replay", required=True, help="mainset/replay/pairs.parquet")
    b.add_argument("--flan-eval", required=True, help="mainset/flan-eval/eval-rows.parquet")
    b.add_argument("--evalsets", required=True, help="eval/data/evalsets.parquet")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--bt", default="auto", help="'auto' (every context example, CONTEXT_SHARE of the total), 'all' "
                   "(every eligible training passage) or a total")
    b.add_argument("--qe-max", type=float, default=QE_MAX, help="the query's bound on MetricX-24-QE en->nl")
    b.add_argument("--replay-tag", default="none", choices=sorted(TAG_PROMPTS), help="the tag of the replay's prompt")
    b.add_argument("--len-factor", type=float, default=None,
                   help="the bound on the length ratio (Dutch / English characters): within this factor of its "
                   "source's median; none by default")
    b.add_argument("--block-jsonl", default=None, help="JSONL with a source per line (blind100.jsonl): 13-grams blocked")
    b.add_argument("--real", default=None, help="real_pairs.py's real pairs (with qe_en_nl from join-qe)")
    b.add_argument("--real-share", type=float, default=REAL_SHARE)
    b.add_argument("--real-all", action="store_true", help="every eligible real pair, no share (arm RB)")
    b.add_argument("--real-without-qe", action="store_true", help="a dry run: no MetricX-QE bound on the real pairs")
    b.add_argument("--real-qe-max", type=float, default=None,
                   help="the real pairs' own bound on MetricX-24-QE en->nl (arm RB2: 7.5); --qe-max by default")
    b.add_argument("--real-qe-unbounded", action="store_true",
                   help="no bound on the real pairs' MetricX-QE (arm RL: the books); their QE is still required")
    b.add_argument("--real-align-max", type=float, default=None,
                   help="fail a real pair whose sentence alignment costs more than this at its start or end (arm RL)")
    b.add_argument("--bt-match-set", default=None,
                   help="a set's directory: the --bt-per-real draw follows the word ratio of its real examples (arm RL)")
    b.add_argument("--bt-match-bins", type=int, default=10, help="the bins of --bt-match-set (at the target's quantiles)")
    b.add_argument("--bt-per-real", type=float, default=None,
                   help="with --bt all and --real-all: draw this many back-translation examples per real pair (arm RB2: 4)")
    b.add_argument("--real2", default=None,
                   help="with --real-all: a second table of real pairs (with qe_en_nl), at --qe-max (arm RBR: R2's)")
    b.add_argument("--real2-from", default=None,
                   help="a set's train-meta.parquet: --real2 takes only its real examples (arm RBR: R2's)")
    b.add_argument("--real2-per-real", type=float, default=None,
                   help="draw this many --real2 pairs per --real pair, uniformly (arm RBR: 1)")
    b.add_argument("--sets", nargs="+", required=True,
                   help="NAME=TAG[+replay|+real], TAG none, token, instruction or conventional")
    b.add_argument("--seed", default=SEED)
    b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--tokenizer", required=True)
    args = parser.parse_args(argv)
    if args.command == "build" and args.bt not in ("auto", "all") and not re.fullmatch(r"[1-9]\d*", args.bt):
        parser.error("--bt is 'auto', 'all' or a positive number")
    if args.command == "build" and args.real_all and args.real_share != REAL_SHARE:
        parser.error("--real-all takes every eligible real pair: no --real-share with it")
    if args.command == "build" and args.bt_per_real is not None and (args.bt != "all" or not args.real_all
                                                                     or args.bt_per_real <= 0):
        parser.error("--bt-per-real is a positive number, with --bt all and --real-all")
    if args.command == "build" and args.real_without_qe and args.real_qe_max is not None:
        parser.error("--real-without-qe has no QE bound on the real pairs: no --real-qe-max with it")
    if args.command == "build" and args.real_qe_unbounded and (args.real_without_qe or args.real_qe_max is not None
                                                               or not args.real):
        parser.error("--real-qe-unbounded goes with --real, without --real-without-qe or --real-qe-max")
    if args.command == "build" and args.real_align_max is not None and (args.real_align_max <= 0 or not args.real
                                                                        or not args.len_factor):
        parser.error("--real-align-max is a positive number, with --real and --len-factor")
    if args.command == "build" and (args.bt_match_set or args.bt_match_bins != 10) and (
            not args.bt_match_set or args.bt_per_real is None or args.bt_match_bins < 2):
        parser.error("--bt-match-set goes with --bt-per-real (and --bt-match-bins of 2 or more)")
    if args.command == "build" and (args.real2 or args.real2_from or args.real2_per_real is not None) and (
            not (args.real2 and args.real2_from and args.real2_per_real and args.real2_per_real > 0)
            or not args.real or not args.real_all or args.real_without_qe):
        parser.error("--real2, --real2-from and a positive --real2-per-real go together, with --real and --real-all, "
                     "not with --real-without-qe")
    {"build": build, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
