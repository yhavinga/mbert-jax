#!/usr/bin/env python3
"""The training data of the fine-tune's arms (experiments/en-nl-translator/protocol.md; hammond
mbert/designs/en-nl-translator.md §3 step 3, §7 point 6, §9): a query on the pairs tables of the main run and the long
run, the FLAN-nl replay, and the user turn of translate.user_text.

    python sft_build.py build --run RUN --long LONG --replay REPLAY.parquet --flan-eval EVAL-ROWS.parquet \\
        --evalsets EVALSETS.parquet --tokenizer TOK --bt auto|N --sets B=none N-token=token+replay ... --out OUT
    python sft_build.py check --data OUT/<set> --tokenizer TOK   (SFT venv: MaxText's own SFT steps on every example)

RUN holds the main run's pairs.parquet (mainset/run/pairs/), passages.parquet and context-turns.parquet
(mainset/run/select/); LONG the long run's pairs.parquet and passages.parquet.

The query (QUERY; Yeb, 9 Nov: strict on the source, mild on the pair). A back-translation pair (kind passage) is
eligible when
- hard_fails is empty (not ended, decode_loop, Dutch left in the English, length ratio, structure; pairs.py);
- its source passed the selection: Gemma's source score >= 0.5, for FineWeb-2 also the origin score >= 0.5 (the
  selection chose no other; checked here);
- MetricX-24-QE en->nl <= QE_MAX, a wide bound against outliers near p99 of the training passages; chrF and the other
  scores have no bound;
- it shares no 13-gram of words with an evaluation set (WMT24++, FLORES-200, NTREX-128, Europarl-direct: source and
  reference) or with FLAN's evaluation rows;
- the whole example fits MAX_TOKENS in every prompt of the build.
The split is the selection's (per document): train pairs train, eval pairs eval.

Context examples (design §9): a training passage the selection marked as a context example keeps its turns before it,
nearest first, as long as each turn is eligible by the same query; the first that fails ends the context there. Without
a turn left it is a plain example. The share of context examples is CONTEXT_SHARE of the back-translation examples, in
every arm alike: --bt auto takes every eligible context example and as many plain ones as make them that share; --bt N
draws that share of N. The plain examples are drawn per source in proportion to the eligible passages of each source,
so the sources keep their shares.

Prompts (translate_models.yml): a plain example gets chat-v1 without context, which is TranslateGemma's own prompt word
for word; a context example gets chat-v1 with its earlier turns. The tag of a set picks the variant of chat-v1 for its
back-translation examples (TAG_PROMPTS): none, the unused token <unused0> before the text, or one sentence in the
instruction. The FLAN-nl replay (a set with +replay: arm N) is REPLAY_SHARE of the set's examples, untagged, in chat-v1
without context; it is drawn from the replay pairs that are none of FLAN's evaluation rows' segments and share no
13-gram with the evaluation texts.

build writes per set OUT/<set>/: train.parquet and eval.parquet (one column `messages`, user then assistant, for
MaxText's SFT pipeline; train shuffled), train-meta.parquet and eval-meta.parquet (row by row: id, kind bt/bt_context/
replay, source, tag, prompt, the English, the context, the label, token counts, QE, chrF), template.jinja (the user
turn as built, in Gemma 3's turn) and manifest.json; and OUT/report.json, OUT/report.md: what each bound costs, the
counts per set, source and kind, and the tokens.
"""

import argparse
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
MAX_TOKENS = 2048  # the fine-tune's sequence length (design §7 point 7)
TAG_PROMPTS = {"none": "chat-v1", "token": "chat-v1-tag-token", "instruction": "chat-v1-tag-instruction"}
TAG_TOKEN = "<unused0>"
TAG_SENTENCE = "Translate into natural, original-sounding Dutch."
REPLAY_PROMPT = "chat-v1"
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
META_COLUMNS = ["id", "kind", "source", "length_class", "tag", "prompt", "en", "context", "context_ids", "label",
                "tokens_prompt", "tokens_target", "qe_en_nl", "chrf"]


def speaker_label(speaker, party):
    """'De heer Thijssen (PvdA)', 'De voorzitter' or None (gemma4_pilot.speaker_label)."""
    speaker, party = (speaker or "").strip(), (party or "").strip()
    return f"{speaker} ({party})" if speaker and party else speaker or None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocked_ngrams(evalsets, flan_eval):
    """The 13-grams of the evaluation texts no training example may share: source and reference of every evaluation
    set, the inputs and targets of FLAN's evaluation rows."""
    blocked = set()
    for r in evalsets:
        blocked |= sp.ngrams(r["source"] or "") | sp.ngrams(r["reference"] or "")
    for r in flan_eval:
        blocked |= sp.ngrams(r["inputs"] or "") | sp.ngrams(r["targets"] or "")
    return blocked


def query_fails(pair, passage, blocked):
    """The parts of the query a back-translation pair fails (empty: eligible). passage is its row of the selection
    (None for a context turn, whose score is in context-turns.parquet)."""
    fails = []
    if pair["hard_fails"]:
        fails.append("hard_fails")
    if passage is not None:
        if passage["source_score"] is None or passage["source_score"] < SOURCE_MIN:
            fails.append("source_score")
        if passage["source"] == "fineweb" and (passage["origin_score"] is None or passage["origin_score"] < ORIGIN_MIN):
            fails.append("origin_score")
    if pair["qe_en_nl"] is None or pair["qe_en_nl"] > QE_MAX:
        fails.append("qe")
    if (sp.ngrams(pair["nl"]) | sp.ngrams(pair["en"])) & blocked:
        fails.append("eval_13gram")
    return fails


def context_of(passage, context_pairs, turns, blocked):
    """(context for user_text, turn ids): the passage's earlier turns, oldest first, cut at the nearest turn that fails
    the query (a turn with a source score below SOURCE_MIN fails too)."""
    kept = []
    for turn_id in reversed(passage["context_ids"] or []):
        pair, turn = context_pairs.get(turn_id), turns.get(turn_id)
        if pair is None or turn is None:
            raise SystemExit(f"{passage['pid']}: context turn {turn_id} has no pair or no row in context-turns.parquet")
        if query_fails(pair, None, blocked) or turn["source_score"] is None or turn["source_score"] < SOURCE_MIN:
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
    left. bt: 'auto' (every context example, CONTEXT_SHARE of the total) or a total."""
    n_ctx = len(context_ids)
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
    for r in rows:
        p = passages.get(r["pid"])
        if p is None:
            raise SystemExit(f"{r['pid']}: a pair without a passage in the selections")
        if p["split"] != r["split"]:
            raise SystemExit(f"{r['pid']}: split {r['split']} in the pairs, {p['split']} in the selection")
        r["fails"] = query_fails(r, p, blocked)
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
                                "over": {str(c): sum(v > c for v in qe_hard) for c in (6, 7, 7.5, 8, 9, 10, 12, 15)}},
    }
    ctx_stats = collections.Counter()
    for r in rows:
        r["context"], r["context_ids"] = [], []
        r["label"] = speaker_label(r["passage"]["speaker"], r["passage"]["party"]) if r["source"] == "handelingen" \
            else None
        if r["fails"] or not r["passage"]["context_ids"]:
            continue
        ctx_stats[f"{r['split']}/marked"] += 1
        r["context"], r["context_ids"] = context_of(r["passage"], context_pairs, turns, blocked)
        dropped = len(r["passage"]["context_ids"]) - len(r["context_ids"])
        ctx_stats[f"{r['split']}/turns_dropped"] += dropped
        ctx_stats[f"{r['split']}/{'with_context' if r['context'] else 'no_turn_left'}"] += 1
    report["context"] = dict(sorted(ctx_stats.items()))
    return rows


def candidates_replay(args, blocked, report):
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
    report["replay"] = {"pairs": len(replay), "dropped": dict(stats), "candidates": len(out)}
    return out


def quantile(values, q):
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def parse_sets(specs):
    """'N-token=token+replay' -> {name: (tag, replay)}"""
    sets = {}
    for spec in specs:
        name, _, form = spec.partition("=")
        tag, _, extra = form.partition("+")
        if not name or tag not in TAG_PROMPTS or extra not in ("", "replay"):
            raise SystemExit(f"--sets: NAME=TAG[+replay] with TAG one of {sorted(TAG_PROMPTS)}, not {spec!r}")
        sets[name] = (tag, extra == "replay")
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
    report = {"query": QUERY, "context_share": CONTEXT_SHARE, "replay_share": REPLAY_SHARE, "max_tokens": MAX_TOKENS,
              "seed": args.seed, "bt": args.bt, "sets": {}, "tag_token": {"text": TAG_TOKEN, "id": tag_id},
              "tag_sentence": TAG_SENTENCE, "tag_prompts": TAG_PROMPTS,
              "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in (
                  ("pairs_main", Path(args.run) / "pairs.parquet"), ("pairs_long", Path(args.long) / "pairs.parquet"),
                  ("passages_main", Path(args.run) / "passages.parquet"),
                  ("passages_long", Path(args.long) / "passages.parquet"),
                  ("context_turns", Path(args.run) / "context-turns.parquet"), ("replay", Path(args.replay)),
                  ("flan_eval", Path(args.flan_eval)), ("evalsets", Path(args.evalsets)))}}
    evalsets = pq.read_table(args.evalsets, columns=["source", "reference"]).to_pylist()
    flan_eval = pq.read_table(args.flan_eval, columns=["inputs", "targets"]).to_pylist()
    blocked = blocked_ngrams(evalsets, flan_eval)
    report["blocked_13grams"] = len(blocked)
    rows = candidates_bt(args, blocked, report)

    # Tokens in every prompt the build uses: an example joins only when it fits MAX_TOKENS in all of them, so that the
    # arms keep the same pairs
    too_long = collections.Counter()
    eligible = [r for r in rows if not r["fails"]]
    for r in eligible:
        r["tokens_target"] = len(counter.target_ids(r["nl"].strip()))
        r["tokens_prompt"] = {t: len(counter.prompt_ids(user_turn(TAG_PROMPTS[t], r["en"], r["context"], r["label"])))
                              for t in tags}
        r["fits"] = max(r["tokens_prompt"].values()) + r["tokens_target"] <= MAX_TOKENS
        if not r["fits"] and r["context"]:
            # the context is what makes it long: the passage stays, as a plain example
            too_long[f"{r['split']}/context_too_long"] += 1
            r["context"], r["context_ids"] = [], []
            r["tokens_prompt"] = {t: len(counter.prompt_ids(user_turn(TAG_PROMPTS[t], r["en"]))) for t in tags}
            r["fits"] = max(r["tokens_prompt"].values()) + r["tokens_target"] <= MAX_TOKENS
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
    if any(with_replay for _, with_replay in sets.values()):
        pool = candidates_replay(args, blocked, report)
        fitting = []
        for r in pool:
            prompt = len(counter.prompt_ids(user_turn(REPLAY_PROMPT, r["en"])))
            target = len(counter.target_ids(r["nl"].strip()))
            if prompt + target <= MAX_TOKENS:
                fitting.append(r | {"tokens_prompt": prompt, "tokens_target": target})
        report["replay"]["fitting"] = len(fitting)
        if n_replay > len(fitting):
            raise SystemExit(f"{n_replay} replay pairs wanted, {len(fitting)} candidates")
        replay = random.Random(f"{args.seed}/replay").sample(fitting, n_replay)
        report["replay"]["drawn"] = n_replay

    def bt_example(r, tag, with_context):
        context = r["context"] if with_context else []
        prompt = TAG_PROMPTS[tag]
        user = user_turn(prompt, r["en"], context, r["label"])
        return ({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["nl"].strip()}]},
                {"id": r["id"], "kind": "bt_context" if context else "bt", "source": r["source"],
                 "length_class": r["length_class"], "tag": tag, "prompt": prompt, "en": r["en"].strip(),
                 "context": context, "context_ids": r["context_ids"] if context else [],
                 "label": r["label"] if context else None, "tokens_prompt": len(counter.prompt_ids(user)),
                 "tokens_target": r["tokens_target"], "qe_en_nl": r["qe_en_nl"], "chrf": r["chrf"]})

    def replay_example(r):
        user = user_turn(REPLAY_PROMPT, r["en"])
        return ({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["nl"].strip()}]},
                {"id": f"replay:{r['id']}", "kind": "replay", "source": f"flan:{r['task']}", "length_class": None,
                 "tag": "none", "prompt": REPLAY_PROMPT, "en": r["en"].strip(), "context": [], "context_ids": [],
                 "label": None, "tokens_prompt": r["tokens_prompt"], "tokens_target": r["tokens_target"],
                 "qe_en_nl": None, "chrf": None})

    for name, (tag, with_replay) in sets.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        examples = [bt_example(r, tag, r["pid"] in ctx_set) for r in bt_rows]
        if with_replay:
            examples += [replay_example(r) for r in replay]
        random.Random(f"{args.seed}/shuffle").shuffle(examples)
        eval_examples = [bt_example(r, tag, bool(r["context"])) for r in evals]
        for split, part in (("train", examples), ("eval", eval_examples)):
            pq.write_table(pa.table({"messages": [m["messages"] for m, _ in part]}), d / f"{split}.parquet")
            pq.write_table(pa.Table.from_pylist([meta for _, meta in part]), d / f"{split}-meta.parquet")
        (d / "template.jinja").write_text(TEMPLATE, encoding="utf-8")
        stats = set_stats(examples, eval_examples)
        manifest = {"set": name, "tag": tag, "replay": with_replay, "prompts": sorted({m["prompt"] for _, m in examples}),
                    "query": QUERY, "seed": args.seed, "bt": args.bt, **stats,
                    "files": {f.name: sha256(f) for f in sorted(d.iterdir()) if f.suffix in (".parquet", ".jinja")}}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        report["sets"][name] = {"tag": tag, "replay": with_replay, **stats}
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
        return {"examples": len(metas), "kinds": dict(sorted(kinds.items())), "sources": dict(sorted(sources.items())),
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
             f"seed `{report['seed']}`, `--bt {report['bt']}`. Geblokkeerde 13-grammen van de evaluatieteksten: "
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
    lines += ["", f"Context: {json.dumps(report['context'])}; te lang: {json.dumps(report['too_long'])}.",
              f"BT: {json.dumps(report['bt'], ensure_ascii=False)}."]
    if "replay" in report:
        lines.append(f"Replay: {json.dumps(report['replay'])}.")
    lines += ["", "## Sets", "", "| set | tag | replay | split | voorbeelden | bt | bt_context | replay | lang | "
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
            if user.count(TAG_SENTENCE) != (1 if meta["tag"] == "instruction" else 0):
                raise SystemExit(f"{split} row {k} ({meta['id']}): the tag sentence {user.count(TAG_SENTENCE)} times")
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
    b.add_argument("--bt", default="auto", help="'auto' (every context example, CONTEXT_SHARE of the total) or a total")
    b.add_argument("--sets", nargs="+", required=True, help="NAME=TAG[+replay], TAG none, token or instruction")
    b.add_argument("--seed", default=SEED)
    b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--tokenizer", required=True)
    args = parser.parse_args(argv)
    if args.command == "build" and args.bt != "auto" and not re.fullmatch(r"[1-9]\d*", args.bt):
        parser.error("--bt is 'auto' or a positive number")
    {"build": build, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
