#!/usr/bin/env python3
"""The checks of G4-8-s140's Hugging Face release (protocol-g4.md §26), on a worker's CPU. Texts stay in the files;
everything printed is ids, counts and scores.

  select   the sources of the checks from the MaxText evaluation's pairs of G4-8-s140 (arm) and G4-0 (base): per set of
           FLORES, NTREX and WMT24++ (no novels, no FLAN) a seeded sample of the segments that both decoded cleanly (ended,
           not empty, no decode loop, no English left) with prompt + output at most MAX_TOKENS tokens on both sides; and
           a seeded sample of FLORES segments for the GGUF smoke-test. Writes ids.json (ids and sets only).
  check    teacher-forced argmax agreement in transformers (bf16, CPU): the prompt of the model's own chat template
           (checked against Gemma 4's turn around natural-v1, as training and the MaxText evaluation made it), then a
           greedy output of the MaxText evaluation and <turn|>; per position the argmax of the logits against the next
           token. Runs: adapter:arm (base + PEFT adapter against G4-8-s140's output), base:base (the base model, adapter
           disabled, against G4-0's), base:arm (the base model against G4-8-s140's: how far the adapter moves the model),
           model:arm (a merged model directory against G4-8-s140's). --require RUN>=X stops (exit 3) below X, after the
           report is written.
  example  the transformers + peft usage example of the adapter's model card, run as written (its model ids replaced by
           local directories, or as written to pull from the Hub): its translate() on the smoke-test's FLORES sources,
           greedy; non-empty, language by langid, chrF against G4-8-s140's MaxText output, exact matches.
  smoke    a GGUF in llama-server (llama.cpp) on the CPU: the chat template's prompt (POST /apply-template) against Gemma
           4's turn, then /v1/chat/completions with temperature 0 on the smoke-test's FLORES sources; the same scores.
  cards    the prompts in the model cards (the usage example's PROMPT, the ```text block) against natural-v1.
"""

import argparse
import contextlib
import glob
import io
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

from g4lora import prompt_text, user_turn  # noqa: E402

SEED = "en-nl-g4arm/0/G4-8-hf"
SETS = ("flores", "ntrex", "wmt24pp")
MAX_TOKENS = 1000          # prompt + output, below Gemma 4's sliding window of 1,024
END = "<turn|>"            # the end of the model's turn: the last position of each check
ARM, BASE = "arm", "base"
RUN = re.compile(r"^(adapter|base|model):(arm|base)$")


def read_pairs(directory):
    """{id: row} of a MaxText evaluation's pairs (pairs/pairs-w*.parquet)."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(Path(directory) / "pairs" / "pairs-w*.parquet")))
    if not files:
        raise SystemExit(f"no pairs/pairs-w*.parquet in {directory}")
    rows = {}
    for f in files:
        for row in pq.read_table(f).to_pylist():
            if row["id"] in rows:
                raise SystemExit(f"{row['id']} twice in {directory}")
            rows[row["id"]] = row
    return rows


def clean(row):
    return (row["ended"] and not row["empty"] and not row["decode_loop"] and not row["english_left"]
            and row["tokens_in"] + row["tokens_out"] <= MAX_TOKENS)


def select(arm, base, per_set, smoke):
    """(check ids by set, smoke ids): per set a seeded sample of the ids clean on both sides and with the same
    source; the smoke-test's from FLORES, drawn apart from the check's."""
    out, pool = {}, {}
    for name in SETS:
        ids = sorted(i for i, r in arm.items() if r["set"] == name and i in base and clean(r) and clean(base[i])
                     and base[i]["source"] == r["source"])
        if len(ids) < per_set:
            raise SystemExit(f"{name}: {len(ids)} clean segments, need {per_set}")
        out[name] = sorted(random.Random(f"{SEED}/check/{name}").sample(ids, per_set))
        pool[name] = ids
    rest = [i for i in pool["flores"] if i not in set(out["flores"])]
    smoke_ids = sorted(random.Random(f"{SEED}/smoke/flores").sample(rest, smoke))
    return out, smoke_ids


def cmd_select(args):
    arm, base = read_pairs(args.arm), read_pairs(args.base)
    check, smoke = select(arm, base, args.per_set, args.smoke)
    out = {"seed": SEED, "max_tokens": MAX_TOKENS, "per_set": args.per_set, "check": check, "smoke": smoke,
           "arm": str(args.arm), "base": str(args.base)}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("SELECT " + json.dumps({"check": {k: len(v) for k, v in check.items()}, "smoke": len(smoke)}), flush=True)


# ------------------------------------------------------------------------------------------------------------ check

def chat_prompt_ids(tok, en):
    """The prompt ids of the model's own chat template (one user turn, generation prompt, thinking off); they must be
    Gemma 4's turn around natural-v1's user text, as training and the MaxText evaluation made the prompt."""
    got = tok.apply_chat_template([{"role": "user", "content": user_turn(en)}], add_generation_prompt=True,
                                  tokenize=True)
    if hasattr(got, "keys"):
        got = got["input_ids"]
    want = tok(prompt_text(en), add_special_tokens=False)["input_ids"]
    if list(got) != list(want):
        raise SystemExit("the chat template's prompt differs from Gemma 4's turn around natural-v1")
    return list(want)


def target_ids(tok, nl):
    return list(tok(nl.strip() + END, add_special_tokens=False)["input_ids"])


def agreement(model, prompt, target, torch):
    """(equal positions, positions, first differing position or None): the argmax after each prefix of the target."""
    ids = torch.tensor([prompt + target])
    with torch.inference_mode():
        logits = model(input_ids=ids, logits_to_keep=len(target) + 1).logits[0]
    pred = logits[:-1].argmax(-1).tolist()
    if len(pred) != len(target):
        raise SystemExit(f"{len(pred)} predictions for {len(target)} target tokens")
    equal = [p == t for p, t in zip(pred, target)]
    return sum(equal), len(target), next((k for k, e in enumerate(equal) if not e), None)


def summary(rows):
    eq, n = sum(r["equal"] for r in rows), sum(r["tokens"] for r in rows)
    return {"sources": len(rows), "tokens": n, "equal": eq, "agreement": eq / n if n else None,
            "sources_all_equal": sum(r["equal"] == r["tokens"] for r in rows),
            "min_source_agreement": min(r["equal"] / r["tokens"] for r in rows) if rows else None}


def parse_require(specs):
    out = {}
    for spec in specs:
        m = re.match(r"^([a-z]+:[a-z]+)>=([0-9.]+)$", spec)
        if not m or not RUN.match(m.group(1)):
            raise SystemExit(f"--require {spec}: RUN>=X with RUN adapter|base|model:arm|base")
        out[m.group(1)] = float(m.group(2))
    return out


def peft_load_check(model, adapter_dir, torch):
    """Every LoRA module of the PeftModel holds the adapter file's factors (a key PEFT did not load keeps its random
    init and would not show in the agreement); the LoRA modules are exactly the file's."""
    from safetensors.torch import load_file

    want = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    seen, differ = 0, []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
            seen += 1
            key = re.sub(r"^base_model\.model\.", "", name)
            for f, layer in (("A", module.lora_A["default"]), ("B", module.lora_B["default"])):
                w = want.get(f"base_model.model.{key}.lora_{f}.weight")
                if w is None or not torch.equal(layer.weight.detach().to(torch.float32), w.to(torch.float32)):
                    differ.append(f"{key}.lora_{f}")
    if seen * 2 != len(want) or differ:
        raise SystemExit(f"PEFT load: {seen} LoRA modules for {len(want) // 2} in the file, {len(differ)} differ, "
                         f"e.g. {differ[:3]}")
    return {"lora_modules": seen, "equal": True}


def cmd_check(args):
    import torch
    import transformers

    require = parse_require(args.require)
    runs = args.runs.split(",")
    if any(not RUN.match(r) for r in runs):
        raise SystemExit(f"--runs {args.runs}: each adapter|base|model:arm|base")
    if args.adapter_dir and any(r.startswith("model:") for r in runs) or \
            not args.adapter_dir and any(not r.startswith("model:") for r in runs):
        raise SystemExit("adapter: and base: runs need --adapter-dir, model: runs a model without it")
    torch.set_num_threads(args.threads)
    ids = json.loads(Path(args.ids).read_text())
    outputs = {ARM: read_pairs(args.arm), BASE: read_pairs(args.base)}
    tok = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    model.eval()
    report = {"model_dir": str(args.model_dir), "class": type(model).__name__, "adapter_dir": args.adapter_dir,
              "threads": args.threads, "versions": {m.__name__: m.__version__ for m in (torch, transformers)},
              "runs": {}, "require": require}
    if args.adapter_dir:
        import peft

        model = peft.PeftModel.from_pretrained(model, args.adapter_dir)
        model.eval()
        report["versions"]["peft"] = peft.__version__
        report["peft_load"] = peft_load_check(model, args.adapter_dir, torch)
        print("PEFTLOAD " + json.dumps(report["peft_load"]), flush=True)
    report["load_s"] = round(time.time() - t0, 1)
    print(f"LOADED {json.dumps({'class': report['class'], 's': report['load_s']})}", flush=True)
    items = [(name, i) for name in SETS for i in ids["check"][name]]
    prompts = {i: chat_prompt_ids(tok, outputs[ARM][i]["source"]) for _, i in items}
    for run in runs:
        mode, side = run.split(":")
        t1, rows = time.time(), []
        ctx = model.disable_adapter() if mode == "base" else contextlib.nullcontext()
        with ctx:
            for name, i in items:
                target = target_ids(tok, outputs[side][i]["hypothesis"])
                eq, n, first = agreement(model, prompts[i], target, torch)
                rows.append({"id": i, "set": name, "tokens": n, "equal": eq, "first_difference": first})
        s = summary(rows)
        s["per_set"] = {name: summary([r for r in rows if r["set"] == name]) for name in SETS}
        s["seconds"] = round(time.time() - t1, 1)
        report["runs"][run] = {**s, "rows": rows}
        print(f"CHECK {run} " + json.dumps({k: v for k, v in s.items() if k != "per_set"}), flush=True)
    if "adapter:arm" in report["runs"] and "base:arm" in report["runs"]:
        gap = report["runs"]["adapter:arm"]["agreement"] - report["runs"]["base:arm"]["agreement"]
        report["adapter_minus_base_on_arm"] = gap
        print(f"GAP {json.dumps({'adapter_minus_base_on_arm': gap})}", flush=True)
    failed = {r: x for r, x in require.items() if r not in report["runs"] or report["runs"][r]["agreement"] < x}
    report["passed"] = not failed
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"REQUIRE {json.dumps({'require': require, 'passed': not failed})}", flush=True)
    if failed:
        print(f"FAILED requirement(s): {json.dumps(failed)}", flush=True)
        sys.exit(3)


# ---------------------------------------------------------------------------------------------- example and smoke

def scores(rows, ref):
    """Per output: non-empty, Dutch by langid, chrF against G4-8-s140's MaxText output; and over all of them."""
    import langid
    import sacrebleu

    out = []
    for i, text in rows:
        text = text or ""
        out.append({"id": i, "chars": len(text), "nonempty": bool(text.strip()),
                    "lang": langid.classify(text)[0] if text.strip() else None,
                    "chrf": sacrebleu.sentence_chrf(text.strip(), [ref[i]]).score,
                    "exact": text.strip() == ref[i]})
    hyps = [t.strip() for _, t in rows]
    refs = [ref[i] for i, _ in rows]
    return {"n": len(out), "nonempty": sum(r["nonempty"] for r in out), "dutch": sum(r["lang"] == "nl" for r in out),
            "chrf_corpus": sacrebleu.corpus_chrf(hyps, [refs]).score,
            "chrf_mean": sum(r["chrf"] for r in out) / len(out), "exact": sum(r["exact"] for r in out), "rows": out}


def card_code(card):
    """The first ```python block of a model card."""
    m = re.search(r"```python\n(.*?)```", Path(card).read_text(encoding="utf-8"), re.S)
    if not m:
        raise SystemExit(f"{card}: no ```python block")
    return m.group(1)


def natural_v1_template():
    """natural-v1's user turn (translate_models.yml) for English -> Dutch with {text} left as the placeholder."""
    import translate

    return translate.user_text(translate.PROMPTS["natural-v1"], "{text}", "en", "nl")


def card_prompts(card):
    """Whether a card's prompts are natural-v1 letter for letter: PROMPT in its first ```python block (a string
    literal), and its ```text block with {text}; None where the card has none."""
    import ast

    text = Path(card).read_text(encoding="utf-8")
    want = natural_v1_template()
    out = {"python": None, "text": None}
    m = re.search(r"```python\n(.*?)```", text, re.S)
    if m:
        for node in ast.walk(ast.parse(m.group(1))):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROMPT" for t in node.targets):
                out["python"] = ast.literal_eval(node.value) == want
    blocks = [b for b in re.findall(r"```text\n(.*?)\n```", text, re.S) if "{text}" in b]
    if blocks:
        out["text"] = all(b == want for b in blocks)
    return out


def cmd_cards(args):
    result = {Path(c).name: card_prompts(c) for c in args.cards}
    print("CARDS " + json.dumps(result), flush=True)
    if any(v is False for r in result.values() for v in r.values()):
        sys.exit(3)


def cmd_example(args):
    ids = json.loads(Path(args.ids).read_text())["smoke"][: args.n]
    arm = read_pairs(args.arm)
    code = card_code(args.card)
    for spec in args.replace:
        old, new = spec.split("=", 1)
        if code.count(f'"{old}"') != 1:
            raise SystemExit(f"the card's code names \"{old}\" {code.count(chr(34) + old + chr(34))} times, not once")
        code = code.replace(f'"{old}"', json.dumps(new))
    namespace = {"__name__": "__card__"}
    sink = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(sink):  # the card prints its example's translation: kept from the log
        exec(compile(code, "card-example", "exec"), namespace)  # noqa: S102 - the card's own code
    if "translate" not in namespace:
        raise SystemExit("the card's code defines no translate()")
    printed = sink.getvalue().strip()
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):
        for i in ids:
            rows.append((i, namespace["translate"](arm[i]["source"])))
    result = scores(rows, {i: arm[i]["hypothesis"].strip() for i in ids})
    result.update(card_printed_chars=len(printed), seconds=round(time.time() - t0, 1), replaced=len(args.replace))
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("EXAMPLE " + json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    if not result["nonempty"] == result["dutch"] == result["n"] or not printed:
        sys.exit(3)


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cmd_smoke(args):
    ids = json.loads(Path(args.ids).read_text())["smoke"]
    arm = read_pairs(args.arm)
    t0, rows, template_equal, reasoning = time.time(), [], 0, 0
    for i in ids:
        messages = [{"role": "user", "content": user_turn(arm[i]["source"])}]
        rendered = post(f"{args.server}/apply-template", {"messages": messages})["prompt"]
        template_equal += rendered.removeprefix("<bos>") == prompt_text(arm[i]["source"]).removeprefix("<bos>")
        r = post(f"{args.server}/v1/chat/completions",
                 {"messages": messages, "temperature": 0, "max_tokens": 2 * arm[i]["tokens_out"] + 32})
        msg = r["choices"][0]["message"]
        reasoning += bool(msg.get("reasoning_content"))
        rows.append((i, msg.get("content") or ""))
    result = scores(rows, {i: arm[i]["hypothesis"].strip() for i in ids})
    result.update(quant=args.quant, template_equal=template_equal, reasoning_content=reasoning,
                  seconds=round(time.time() - t0, 1))
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("SMOKE " + json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="checks of G4-8-s140's Hugging Face release")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("--arm", required=True, help="G4-8-s140's evaluation (pairs/)")
    p.add_argument("--base", required=True, help="G4-0's evaluation (pairs/)")
    p.add_argument("--per-set", type=int, default=10)
    p.add_argument("--smoke", type=int, default=8)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--model-dir", required=True, help="the base model (with --adapter-dir) or a merged model")
    p.add_argument("--adapter-dir")
    p.add_argument("--arm", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--runs", required=True, help="e.g. adapter:arm,base:base,base:arm or model:arm")
    p.add_argument("--require", action="append", default=[], help="RUN>=X, e.g. adapter:arm>=0.99")
    p.add_argument("--threads", type=int, default=96)
    p.add_argument("--out", required=True)
    p = sub.add_parser("example")
    p.add_argument("--card", required=True)
    p.add_argument("--replace", action="append", default=[], help="MODEL_ID=LOCAL_DIR in the card's code")
    p.add_argument("--arm", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--server", required=True, help="llama-server's URL, e.g. http://127.0.0.1:8090")
    p.add_argument("--quant", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("cards")
    p.add_argument("cards", nargs="+", help="model cards whose prompts must be natural-v1")
    args = parser.parse_args(argv)
    {"select": cmd_select, "check": cmd_check, "example": cmd_example, "smoke": cmd_smoke,
     "cards": cmd_cards}[args.command](args)


if __name__ == "__main__":
    main()
