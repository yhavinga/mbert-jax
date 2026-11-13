#!/usr/bin/env python3
"""The memory and step test of LoRA SFT on Gemma 4 26B-A4B-it (MoE) and Gemma 4 31B-it on mbert-mix (Yeb, 13 Nov
~12:50: "een test met lora op de MoE"), with the translator's SFT stack (MaxText 0.2.5, Tunix, qwix: run_sft.sh,
sft_lora.py). A measurement: 8 steps per variant; no arm is trained. run_g4lora.sh runs these on a worker:

  data     the test's training set: the first N examples without context of arm RL's set ($S/data/rl-b5/RL), each one's
           English in natural-v1's user turn (translate.py, as the Gemma 4 pilot translated), its Dutch as the
           completion, in Gemma 4's chat turn (template.jinja: the empty thought channel closed, as the model's own
           template does with thinking off); checked with MaxText's own SFT steps against each model's tokenizer and
           its own chat template; and the Gemma 4 : TranslateGemma token ratio of the RL set and of a sample of R2's,
           for the epoch estimate
  result   one variant's outcome from its run directory (the four workers' logs, summary.json): ok, oom or error; the
           peak HBM per chip, step time, tokens/s, the loss of steps 1-8, the LoRA parameters per module type
  best     the variant to repeat at max_target_length 4096 (the advice's rule, fixed before the test)
  report   the table and the advice (markdown and json)
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# translate.py (and translate_models.yml) of FLAN-nl in the repo, or beside this file on a worker
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

import translate  # noqa: E402

PROMPT = "natural-v1"         # the Gemma 4 pilot's en->nl instruction (bt-pilot-natural-english.md §3)
MODELS = ("gemma4-26b", "gemma4-31b")
END = "<turn|>\n"             # the end of a Gemma 4 turn; <turn|> (106) is one of the model's stop tokens
MEM_MAX_GIB = 29.5            # the fine-tune's bound on the peak per chip (run_arms.sh's memory test, 11 Nov)
SEED = "en-nl-g4lora/0"
TEST_STEPS = 8
MIN_BATCHES = 20              # per process at 4096: 8 steps + sft_lora.py's multi-host margin of 8, and some room


def model_turn(model):
    """The model's user turn (translate_models.yml): prefix and suffix around the user text."""
    turn = translate.MODELS[model]["turn"]
    return turn["prefix"], turn["suffix"]


def _turn_strings():
    prefixes = {model_turn(m) for m in MODELS}
    if len(prefixes) != 1:
        raise SystemExit(f"the models' turns differ: {prefixes}")
    prefix, suffix = prefixes.pop()
    bos = "<bos>"
    if not prefix.startswith(bos) or not suffix.startswith(END):
        raise SystemExit(f"unexpected turn {prefix!r} ... {suffix!r}")
    return prefix[len(bos):], suffix[len(END):]


USER_START, MODEL_START = _turn_strings()   # "<|turn>user\n", "<|turn>model\n<|channel>thought\n<channel|>"


def _jinja_literal(text):
    return text.replace("\n", "\\n")


# sft_build.py's template with Gemma 4's turn strings: the user turn as given (trimmed), the model's turn opened with
# the empty thought channel closed, every turn ending in "<turn|>\n"; with add_generation_prompt, the model's turn as
# translate.py's TurnCodec ends the prompt
TEMPLATE = (
    "{{ bos_token }}\n"
    "{%- if (messages[0]['role'] != 'user') -%}{{ raise_exception('Conversations must start with a user prompt.') }}"
    "{%- endif -%}\n"
    "{%- for message in messages -%}\n"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif -%}\n"
    "{%- if message['content'] is none or message['content'] is not string -%}"
    "{{ raise_exception('content must be a string') }}{%- endif -%}\n"
    "{%- if message['role'] == 'assistant' -%}{{ '" + _jinja_literal(MODEL_START) + "' }}"
    "{{ message['content'] | trim }}\n"
    "{%- elif message['role'] == 'user' -%}{{ '" + _jinja_literal(USER_START) + "' }}{{ message['content'] | trim }}\n"
    "{%- else -%}{{ raise_exception('Conversations must only contain user or assistant roles.') }}{%- endif -%}\n"
    "{{ '" + _jinja_literal(END) + "' }}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}{{ '" + _jinja_literal(MODEL_START) + "' }}{%- endif -%}\n"
)


def user_turn(en):
    return translate.user_text(translate.PROMPTS[PROMPT], en, "en", "nl")


def prompt_text(en):
    """The prompt as the model's chat template makes it (one user turn, generation prompt, thinking off)."""
    return "<bos>" + USER_START + user_turn(en) + END + MODEL_START


def completion_text(nl):
    return nl.strip() + END


def is_plain(meta):
    """An example without context or tag: natural-v1 has no context form (translate_models.yml)."""
    return not meta.get("context") and meta.get("tag", "none") == "none"


def select(metas, n):
    """The first n examples without context, in the set's order (its build drew the order)."""
    chosen = [k for k, m in enumerate(metas) if is_plain(m)][:n]
    if len(chosen) < n:
        raise SystemExit(f"only {len(chosen)} examples without context, need {n}")
    return chosen


def batches_estimate(tokens, length, processes=4, chips=4, fill=0.9):
    """Batches per process of a packed set: its tokens over the slots of a step of one process, at the packing fill of
    the arms (RL: 3,095,665 tokens of process 0 in 413 batches of 4 x 2,048 = 0.915)."""
    return tokens / processes / (chips * length * fill)


class G4Counter:
    """Token counts of the test's examples in a Gemma 4 tokenizer (prompt and completion as MaxText makes them)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def lengths(self, texts, batch=4096):
        out = []
        for k in range(0, len(texts), batch):
            out += [len(x) for x in self.tokenizer(texts[k:k + batch], add_special_tokens=False)["input_ids"]]
        return out

    def example_lengths(self, ens, nls):
        return [p + c for p, c in zip(self.lengths([prompt_text(e) for e in ens]),
                                      self.lengths([completion_text(n) for n in nls]))]


def read_set(directory):
    import pyarrow.parquet as pq

    rows = pq.read_table(Path(directory) / "train.parquet").column("messages").to_pylist()
    metas = pq.read_table(Path(directory) / "train-meta.parquet").to_pylist()
    if len(rows) != len(metas):
        raise SystemExit(f"{directory}: {len(rows)} examples, {len(metas)} meta rows")
    return rows, metas


def token_ratio(rows, metas, counter, sample=None, seed=SEED):
    """Gemma 4 tokens (natural-v1, Gemma 4's turn) over TranslateGemma's (the set's own prompt and target, meta) on the
    set's examples without context, all or a seeded sample."""
    plain = [k for k, m in enumerate(metas) if is_plain(m)]
    if sample and sample < len(plain):
        plain = sorted(random.Random(seed).sample(plain, sample))
    g4 = counter.example_lengths([metas[k]["en"] for k in plain], [rows[k][1]["content"] for k in plain])
    tg = [metas[k]["tokens_prompt"] + metas[k]["tokens_target"] for k in plain]
    return {"examples": len(metas), "plain": sum(1 for m in metas if is_plain(m)), "counted": len(plain),
            "tokens_g4": int(sum(g4)), "tokens_tg": int(sum(tg)), "ratio": round(sum(g4) / sum(tg), 4)}


def data(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import transformers
    from maxtext.input_pipeline import hf_data_processing as hfp
    from maxtext.input_pipeline import input_pipeline_utils as ipu

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, metas = read_set(args.rl)
    chosen = select(metas, args.n)
    messages = [[{"role": "user", "content": user_turn(metas[k]["en"])},
                 {"role": "assistant", "content": rows[k][1]["content"]}] for k in chosen]
    if any(rows[k][1]["role"] != "assistant" or rows[k][0]["role"] != "user" for k in chosen):
        raise SystemExit("an example of the set is not one user and one assistant turn")
    pq.write_table(pa.Table.from_pylist([{"messages": m} for m in messages]), out / "train.parquet")
    pq.write_table(pa.Table.from_pylist([{"id": metas[k]["id"], "kind": metas[k]["kind"], "row": k} for k in chosen]),
                   out / "train-meta.parquet")
    (out / "template.jinja").write_text(TEMPLATE, encoding="utf-8")
    kinds = {}
    for k in chosen:
        kinds[metas[k]["kind"]] = kinds.get(metas[k]["kind"], 0) + 1
    report = {"prompt": PROMPT, "source": str(args.rl), "examples": len(chosen), "kinds": kinds,
              "first_row": chosen[0], "last_row": chosen[-1], "template": TEMPLATE, "models": {}}
    r2_rows, r2_metas = read_set(args.r2)
    for spec in args.tokenizer:
        model, path = spec.split("=", 1)
        # as hf_data_processing makes MaxText's tokenizer for SFT, and the model's own (its chat template)
        mt_tok = transformers.AutoTokenizer.from_pretrained(path, add_bos_token=False, add_eos_token=False,
                                                            legacy=False, extra_special_tokens={})
        mt_tok.chat_template = TEMPLATE
        own = transformers.AutoTokenizer.from_pretrained(path)
        pad_id = hfp._get_pad_id(mt_tok)  # noqa: SLF001 - the id MaxText masks the prompt with
        masking = ipu.SFTPromptMasking("messages", completion_only=True, max_target_length=10 ** 9, unk_id=pad_id)
        lengths, loss_tokens = [], 0
        for k, msgs in enumerate(messages):
            example = ipu.apply_chat_template({"messages": msgs}, mt_tok, "messages")
            example = ipu.tokenization(example, mt_tok, truncation=False, max_length=10 ** 9, column_names=["messages"])
            got = masking.map(example)
            expected = own.apply_chat_template([msgs[0]], tokenize=True, add_generation_prompt=True)
            if isinstance(expected, dict) or hasattr(expected, "keys"):
                expected = expected["input_ids"]
            prompt = list(expected)
            if prompt != own(prompt_text(metas[chosen[k]]["en"]), add_special_tokens=False)["input_ids"]:
                raise SystemExit(f"{model} row {chosen[k]}: the model's chat template differs from Gemma 4's turn")
            completion = own(completion_text(msgs[1]["content"]), add_special_tokens=False)["input_ids"]
            if got["inputs"].tolist() != prompt + completion or \
                    got["targets"].tolist() != [pad_id] * len(prompt) + completion:
                raise SystemExit(f"{model} row {chosen[k]}: MaxText's SFT ids differ from the prompt of the model's own "
                                 f"chat template + Dutch + <turn|>\\n")
            lengths.append(len(prompt) + len(completion))
            loss_tokens += len(completion)
        counter = G4Counter(own)
        total = int(np.sum(lengths))
        report["models"][model] = {
            "tokenizer": path, "pad_id": pad_id, "tokens": total, "loss_tokens": loss_tokens,
            "max_tokens": int(np.max(lengths)), "over_2048": int(sum(1 for x in lengths if x > 2048)),
            "batches_per_process_est": {str(L): round(batches_estimate(total, L), 1) for L in (2048, 4096)},
            "ratio": {"RL": token_ratio(rows, metas, counter), "R2": token_ratio(r2_rows, r2_metas, counter,
                                                                                sample=args.r2_sample)},
        }
        est = report["models"][model]["batches_per_process_est"]["4096"]
        if est < MIN_BATCHES:
            raise SystemExit(f"{model}: ~{est} batches per process at 4096, fewer than {MIN_BATCHES}: take more examples")
        print(f"DATA {model} {json.dumps({k: v for k, v in report['models'][model].items() if k != 'tokenizer'})}",
              flush=True)
    (out / "data.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"DATA {json.dumps({k: report[k] for k in ('examples', 'kinds', 'first_row', 'last_row')})}")


# ------------------------------------------------------------------------------------------------------- result

OOM = re.compile(r"RESOURCE_EXHAUSTED|[Oo]ut of memory|Ran out of memory|[Ee]xceeded hbm capacity")
HBM = re.compile(r"Used ([0-9.]+)([GM]) of ([0-9.]+)([GM]) hbm")
ERROR = re.compile(r"^\s*(\w+\.)*\w*(Error|Exception|Failure)\b.*|.*\bFAILED_PRECONDITION\b.*|.*\bINTERNAL:.*")


def parse_log(text):
    """One worker's log of run_sft.sh: its EXIT code, the SUMMARY, LORA_MODULES and EPOCH lines, the first out-of-memory
    line with the HBM numbers, the last error line."""
    lines = text.splitlines()
    exits = [ln for ln in lines if re.fullmatch(r"EXIT -?[0-9]+", ln.strip())]
    out = {"exit": int(exits[-1].split()[1]) if exits else None}
    for tag in ("SUMMARY", "LORA_MODULES", "EPOCH"):
        found = [ln for ln in lines if ln.startswith(tag + " ")]
        out[tag.lower()] = json.loads(found[-1][len(tag) + 1:]) if found else None
    oom = [ln.strip() for ln in lines if OOM.search(ln)]
    out["oom"] = oom[0][:400] if oom else None
    hbm = next((HBM.search(ln) for ln in lines if HBM.search(ln)), None)
    if hbm:
        gib = [float(hbm.group(i)) / (1024 if hbm.group(i + 1) == "M" else 1) for i in (1, 3)]
        out["hbm_used_of"] = [round(gib[0], 2), round(gib[1], 2)]
    errors = [ln.strip() for ln in lines if ERROR.match(ln)]
    out["error"] = errors[-1][:400] if errors else None
    return out


def result(spec, logs, summary):
    """A variant's outcome (spec: variant, run, model, rank, alpha, length, extra) from its four workers' logs and, when
    the run ended, process 0's summary.json. ok: every worker EXIT 0 with a SUMMARY; oom: a worker ran out of HBM; error:
    anything else that ended; incomplete: a worker without EXIT."""
    parsed = [parse_log(text) for text in logs]
    exits = [p["exit"] for p in parsed]
    out = dict(spec, exits=exits)
    lora = next((p["lora_modules"] for p in parsed if p["lora_modules"]), None)
    if lora:
        out["lora_params"] = lora["params"]
        out["lora_by_module"] = lora["by_type"]
        out["matched_without_lora"] = lora["matched_without_lora"]
    epoch = next((p["epoch"] for p in parsed if p["epoch"]), None)
    if epoch:
        out["batches_per_process"] = epoch.get("batches_per_process")
    if any(e is None for e in exits):
        out["status"] = "incomplete"
        return out
    if all(e == 0 for e in exits) and all(p["summary"] for p in parsed):
        s = [p["summary"] for p in parsed]
        out["status"] = "ok"
        out["peak_gib"] = max(x["peak_gib_max"] for x in s)
        out["peak_gib_by_worker"] = [x["peak_gib_max"] for x in s]
        out["fits"] = out["peak_gib"] <= MEM_MAX_GIB
        out["step_s_median"] = max(x["throughput"]["step_s_median"] for x in s)
        if summary:
            t = summary["throughput"]
            out.update(step_s_mean=t["step_s_mean"], tokens_per_s=t["tokens_per_s"],
                       tokens_per_s_per_chip=t.get("tokens_per_s_per_chip"), loss_tokens_per_s=t["loss_tokens_per_s"],
                       loss=summary["loss"], times=summary["times"], tokens_per_step=summary["tokens"]["per_step"],
                       loss_tokens_per_step=summary["tokens"]["loss_per_step"], moe=summary.get("moe"),
                       sharding={k: summary["config"].get(k) for k in (
                           "ici_fsdp_parallelism", "ici_tensor_parallelism", "dcn_data_parallelism",
                           "dcn_fsdp_parallelism", "per_device_batch_size", "remat_policy", "attention",
                           "weight_dtype", "dtype", "scan_layers")},
                       devices=summary["devices"], processes=summary["processes"])
            out["loss_finite"] = len(summary["loss"]) == TEST_STEPS and all(math.isfinite(v) for _, v in summary["loss"])
        else:
            out["loss_finite"] = False
        return out
    oom = next((p["oom"] for p in parsed if p["oom"]), None)
    if oom:
        out["status"] = "oom"
        out["oom_line"] = oom
        hbm = [p["hbm_used_of"] for p in parsed if p.get("hbm_used_of")]
        if hbm:
            out["hbm_used_of"] = max(hbm)
        return out
    out["status"] = "error"
    out["error_line"] = next((p["error"] for p in parsed if p["error"]), None)
    return out


def cmd_result(args):
    run = Path(args.run_dir)
    logs = []
    for w in range(4):
        f = run / f"{args.run}-w{w}.log"
        if not f.exists():
            raise SystemExit(f"no {f}")
        logs.append(f.read_text(errors="replace"))
    summary_file = run / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else None
    spec = json.loads(args.spec)
    print("RESULT " + json.dumps(result(spec, logs, summary)), flush=True)


# --------------------------------------------------------------------------------------------- best and report

def experts_reached(r):
    """LoRA on the routed experts: a module type of the MoE block's own expert weights in the run's LoRA report."""
    return any("MoeBlock_0" in kind for kind in (r.get("lora_by_module") or {}))


def usable(r):
    return bool(r) and r.get("status") == "ok" and r.get("fits") and r.get("loss_finite")


def final(results):
    """The last attempt of each variant (a fallback run replaces the first), at 2,048."""
    out = {}
    for r in results:
        if r.get("length", 2048) == 2048:
            out[r["variant"]] = r
    return out


# The advice's rule, fixed on 13 Nov before the test: the 31B before the 26B (blind review 56/22/22 against 49/25/26,
# MetricX-QE WMT24++ 2.230 against 2.289, FLAN 3.07 against 3.15: gemma4-sizes.md §5, g4-blind); within a model the most
# LoRA that fits: 31B at rank 64 (TranslateGemma's modules), else at rank 32; 26B with the experts (only when LoRA
# reached them), else attention + the dense MLP, else attention only. Usable: all four workers EXIT 0, peak per chip
# <= 29.5 GiB, 8 finite losses.
BEST_ORDER = ("v3", "v3b", "v2", "v2b", "v1")


def best(results):
    last = final(results)
    for v in BEST_ORDER:
        r = last.get(v)
        if v == "v2" and r and not experts_reached(r):
            continue
        if usable(r):
            return v
    return None


def epoch_estimate(r, tg_summary, ratio, length=2048):
    """Steps and hours of one epoch of a set for the variant r, from TranslateGemma's epoch of that set (its steps at
    2,048) times the Gemma 4 : TranslateGemma token ratio, at r's median step time, plus r's own setup and compile."""
    steps = tg_summary["steps"] * ratio * 2048 / length
    seconds = steps * r["step_s_median"] + r.get("times", {}).get("setup_s", 0) + \
        (r.get("times", {}).get("compile_s_est") or 0)
    return {"steps": round(steps), "hours": round(seconds / 3600, 2)}


def _gib(x):
    return "–" if x is None else f"{x:.2f}".replace(".", ",")


def _num(x, digits=0):
    if x is None:
        return "–"
    text = f"{x:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


VARIANT_TEXT = {
    "v1": "26B-A4B, attention (q, k, v, o)",
    "v2": "26B-A4B, attention + routed experts (router bevroren)",
    "v2b": "26B-A4B, attention + dichte MLP (`shared_experts`)",
    "v3": "31B, TranslateGemma-modules (attention + MLP)",
    "v3b": "31B, TranslateGemma-modules, rank 32",
    "v4": "beste variant bij 4.096",
}


def table(results):
    lines = ["| variant | run | model | rank / alpha | lengte | uitkomst | piek HBM per chip (GiB) | ≤ 29,5 | staptijd "
             "mediaan (s) | tokens/s (16 chips) | verlies stap 1 → 8 | LoRA-parameters | fallback |",
             "|---|---|---|---|---:|---|---:|---|---:|---:|---|---:|---|"]
    for r in results:
        loss = r.get("loss") or []
        loss_text = " → ".join(f"{v:.3f}".replace(".", ",") for _, v in (loss[:1] + loss[-1:])) if loss else "–"
        if r["status"] == "oom":
            outcome = "geheugen op" + (f" ({_gib(r['hbm_used_of'][0])} van {_gib(r['hbm_used_of'][1])} GiB)"
                                       if r.get("hbm_used_of") else "")
        elif r["status"] == "error":
            outcome = "fout: " + (r.get("error_line") or "?")[:120].replace("|", "/")
        else:
            outcome = r["status"]
        lines.append(f"| {r['variant']} ({VARIANT_TEXT.get(r['variant'], '')}) | `{r['run']}` | {r['model']} | "
                     f"{r['rank']} / {r['alpha']} | {_num(r['length'])} | {outcome} | {_gib(r.get('peak_gib'))} | "
                     f"{'ja' if r.get('fits') else ('nee' if r.get('peak_gib') is not None else '–')} | "
                     f"{_gib(r.get('step_s_median'))} | {_num(r.get('tokens_per_s'))} | {loss_text} | "
                     f"{_num(r.get('lora_params'))} | {r.get('extra') or '–'} |")
    return lines


def module_lines(results):
    lines = ["| variant | moduletype | modules met LoRA | LoRA-parameters |", "|---|---|---:|---:|"]
    for r in results:
        for kind, e in (r.get("lora_by_module") or {}).items():
            lines.append(f"| {r['variant']} | `{kind}` | {e['modules']} | {_num(e['params'])} |")
        for kind, n in (r.get("matched_without_lora") or {}).items():
            lines.append(f"| {r['variant']} | `{kind}` (regex raakt de module, **geen LoRA**) | 0 van {n} | 0 |")
    return lines


def report(results, data_json, tg):
    """The generated part of the report: table, LoRA per module type, the per-step loss, the advice by the fixed rule."""
    lines = ["## Uitkomst (gegenereerd door `g4lora.py report`)", "", *table(results), "",
             "### LoRA per moduletype (regel `LORA_MODULES`, vóór de eerste stap)", "", *module_lines(results), ""]
    lines += ["### Verlies per stap", "", "| variant | stap 1–8 |", "|---|---|"]
    for r in results:
        if r.get("loss"):
            lines.append(f"| {r['variant']} | " + " / ".join(f"{v:.3f}".replace(".", ",") for _, v in r["loss"]) + " |")
    lines.append("")
    lines += ["### Sharding en MoE-instellingen", "", "| variant | sharding | MoE |", "|---|---|---|"]
    for r in results:
        if r.get("sharding"):
            lines.append(f"| {r['variant']} | {json.dumps(r['sharding'])} | {json.dumps(r.get('moe'))} |")
    lines.append("")
    choice = best(results)
    advice = {"variant": choice}
    lines += ["### Advies volgens de vooraf vastgelegde regel (§2.4)", ""]
    if choice is None:
        lines.append("Geen variant is bruikbaar (alle vier EXIT 0, piek ≤ 29,5 GiB, 8 eindige verliezen): geen advies "
                     "voor een arm op Gemma 4 met deze stack op mbert-mix.")
    else:
        r = final(results)[choice]
        advice.update(model=r["model"], rank=r["rank"], alpha=r["alpha"], modules=VARIANT_TEXT[choice],
                      regex=r.get("regex"), peak_gib=r.get("peak_gib"), step_s_median=r.get("step_s_median"))
        lines.append(f"- **Model en modules:** {VARIANT_TEXT[choice]} (variant {choice}), rank {r['rank']}, alpha "
                     f"{r['alpha']}, LR 1e-5, `max_target_length` 2.048; piek {_gib(r.get('peak_gib'))} GiB per chip, "
                     f"staptijd {_gib(r.get('step_s_median'))} s (mediaan stap 2–8).")
        long = [x for x in results if x["variant"] == "v4"]
        if long:
            x = long[-1]
            fits = usable(x)
            advice["fits_4096"] = fits
            lines.append(f"- **Bij 4.096:** {'past' if fits else 'past niet of traint niet'} ({x['status']}, piek "
                         f"{_gib(x.get('peak_gib'))} GiB, staptijd {_gib(x.get('step_s_median'))} s).")
        model = data_json["models"].get(r["model"], {})
        for name in ("RL", "R2"):
            ratio = model.get("ratio", {}).get(name, {}).get("ratio")
            if ratio and name in tg:
                est = epoch_estimate(r, tg[name], ratio)
                advice[f"epoch_{name}"] = est
                lines.append(f"- **Eén epoch {name}** [schatting]: TranslateGemma {_num(tg[name]['steps'])} stappen × "
                             f"tokenverhouding Gemma 4 : TG {_num(ratio, 3)} ≈ {_num(est['steps'])} "
                             f"stappen × {_gib(r['step_s_median'])} s + laden en compileren ≈ "
                             f"**{_gib(est['hours'])} uur** (zonder evaluatie; TG: "
                             f"{_gib(tg[name]['train_s'] / 3600)} uur training).")
    lines.append("")
    return "\n".join(lines), advice


VARIANT_ORDER = ("v1", "v2", "v2b", "v3", "v3b", "v4")


def read_results(paths):
    """RESULT lines from the chain's markers (one JSON object per marker, maybe after other text), in the variants'
    order, a variant's first attempt before its fallback."""
    out = []
    for p in paths:
        text = Path(p).read_text()
        m = re.search(r"RESULT (\{.*\})", text)
        if not m:
            raise SystemExit(f"{p}: no RESULT")
        out.append(json.loads(m.group(1)))
    order = {v: k for k, v in enumerate(VARIANT_ORDER)}
    return sorted(out, key=lambda r: (order.get(r["variant"], len(order)), r.get("fallback", 0)))


def cmd_report(args):
    results = read_results(args.results)
    data_json = json.loads(Path(args.data).read_text())
    tg = {}
    for spec in args.tg:
        name, path = spec.split("=", 1)
        s = json.loads(Path(path).read_text())
        tg[name] = {"steps": s["steps"], "train_s": s["times"]["train_s"],
                    "step_s_median": s["throughput"]["step_s_median"]}
    head = Path(args.head).read_text() if args.head else ""
    body, advice = report(results, data_json, tg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + body)
    out.with_suffix(".json").write_text(json.dumps({"results": results, "advice": advice, "tg": tg,
                                                    "data": {k: v for k, v in data_json.items() if k != "template"}},
                                                   indent=1) + "\n")
    print(f"REPORT {json.dumps(advice)}")


def cmd_best(args):
    print(best(read_results(args.results)) or "none")


def main(argv=None):
    parser = argparse.ArgumentParser(description="LoRA memory and step test on Gemma 4 26B-A4B and 31B")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data")
    p.add_argument("--rl", required=True, help="arm RL's set (train.parquet, train-meta.parquet)")
    p.add_argument("--r2", required=True, help="arm R2's set, for its token ratio")
    p.add_argument("--tokenizer", action="append", required=True, help="MODEL=DIR (tokenizer.json, chat_template.jinja)")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--r2-sample", type=int, default=20000)
    p.add_argument("--out", required=True)
    p = sub.add_parser("result")
    p.add_argument("--run-dir", required=True, help="the run's files: <run>-w0..3.log and, if any, summary.json")
    p.add_argument("--run", required=True)
    p.add_argument("--spec", required=True, help="JSON: variant, run, model, rank, alpha, length, extra, regex")
    p = sub.add_parser("best")
    p.add_argument("results", nargs="+")
    p = sub.add_parser("report")
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tg", action="append", default=[], help="NAME=summary.json of TranslateGemma's arm on that set")
    p.add_argument("--head", help="the report's fixed part (markdown), put before the generated part")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"data": data, "result": cmd_result, "best": cmd_best, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
