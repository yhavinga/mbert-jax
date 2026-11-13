#!/usr/bin/env python3
"""The smaller Gemma 4 instruction models as en->nl translators, with the setup of the 31B's run (bt-natural):
E2B, E4B, 12B and 26B-A4B (Yeb, 13 Nov). run_g4_sizes.sh drives it on a v6e-8.

    g4_sizes.py convert --model gemma4-12b --hf DIR --out DIR     MaxText checkpoint of a snapshot (to_maxtext)
    g4_sizes.py check --model gemma4-12b --hf DIR --tokenizer DIR --segments F --shards DIR --out F
                                                                 MaxText's greedy output against Hugging Face's own
                                                                 model on the CPU (teacher forcing)
    g4_sizes.py report --eval SIZE=evalcompare.json ... --g4 eval-gemma4.json --refqe refqe.json --out F.md

convert: MaxText's own converter. The 12B has no MaxText config: its text model has exactly the 31B's tensors and
decoder layer, so it converts as gemma4-31b with the 12B's sizes (translate_models.yml maxtext_overrides) and the
snapshot's own config.json in place of MaxText's stored 31B config (which sets the number of layers to map).

check: a wrong conversion or a decode path that differs from the model (cache, sliding window, shared KV, experts)
gives text that Hugging Face's model would not have produced. For a few translated segments the prompt and MaxText's
output go through the Hugging Face model in one forward pass; its greedy choice must equal MaxText's next token at
nearly every position (bf16 on both sides; a near-tie may flip a token, after which teacher forcing still follows
MaxText's text). Below --min-agreement it stops with an error.
"""
import argparse
import json
import re
import runpy
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "flan-nl"))

SIZES = {"e2b": "gemma4-e2b", "e4b": "gemma4-e4b", "12b": "gemma4-12b", "26b": "gemma4-26b"}
LABELS = {"e2b": "Gemma 4 E2B-it", "e4b": "Gemma 4 E4B-it", "12b": "Gemma 4 12B-it", "26b": "Gemma 4 26B-A4B-it"}
PROMPT = "natural-v1"
# (a1), Yeb 13 Nov: mean MetricX-QE not above the human reference's (bt-natural refqe.json, the 31B's report §0)
A1_SETS = ("wmt24pp", "flores", "ntrex")
# The pilot's check for the prompt in the output (mbert-ops/bt-natural/remote/leak.sh); not one of collect's checks
LEAK = re.compile(r"Vertaal de volgende|idiomatisch Nederlands|native speaker")


def arm(size):
    return f"G4-{size}"


# ---------------------------------------------------------------------------------------------------- convert

class StoredConfig:
    """What to_maxtext reads of an entry of HF_MODEL_CONFIGS: to_dict()."""

    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


def convert_argv(model, hf, out, run_name):
    """to_maxtext's command line for a model of translate_models.yml: unscanned, text only, bf16, lazy loading, as
    en-nl-judge's setup.sh converted the 31B; the model's maxtext_overrides as key=value."""
    argv = ["to_maxtext", f"model_name={model['maxtext_model']}", "scan_layers=false", "use_multimodal=false",
            "skip_jax_distributed_system=True", "hardware=cpu", f"base_output_directory={out}", f"run_name={run_name}"]
    for key, value in (model.get("maxtext_overrides") or {}).items():
        argv.append(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
    return argv + [f"--hf_model_path={hf}", "--lazy_load_tensors=True", "--save_dtype=bfloat16"]


def convert(args):
    import translate as tr

    model = tr.MODELS[args.model]
    if model.get("maxtext_overrides"):
        from maxtext.checkpoint_conversion.utils import hf_model_configs

        config = json.loads((Path(args.hf) / "config.json").read_text())
        hf_model_configs.HF_MODEL_CONFIGS[model["maxtext_model"]] = StoredConfig(config)
        print(f"CONVERT {args.model}: {model['maxtext_model']} with {model['maxtext_overrides']} and "
              f"{args.hf}/config.json ({config['text_config']['num_hidden_layers']} layers)", flush=True)
    sys.argv = convert_argv(model, args.hf, args.out, f"{args.model}_conversion")
    runpy.run_module("maxtext.checkpoint_conversion.to_maxtext", run_name="__main__", alter_sys=True)


# ---------------------------------------------------------------------------------------------------- check

def agreement(predicted, target):
    """Share of positions where the reference model's greedy token is the decoded output's token."""
    if not target:
        return None
    return sum(int(p) == int(t) for p, t in zip(predicted, target)) / len(target)


def pick(rows, n):
    """n translated segments for the check: ended, not empty, in id order"""
    chosen = [i for i in sorted(rows) if rows[i]["ended"] and rows[i]["text_nl"].strip()]
    return chosen[:n]


def check(args):
    import bt_natural as bn
    import pyarrow.parquet as pq
    import torch
    import transformers
    import translate as tr

    model = tr.with_prompt(tr.MODELS[args.model], PROMPT)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    codec = tr.ModelPrompts(tokenizer, model, "en", "nl")
    texts = {r["id"]: r["text"] for r in pq.read_table(args.segments).to_pylist()}
    rows = bn.read_shard_rows(args.shards)
    ids = pick(rows, args.n)
    if len(ids) < args.n:
        raise SystemExit(f"{args.shards}: {len(ids)} ended, non-empty translations, need {args.n}")
    config = json.loads((Path(args.hf) / "config.json").read_text())
    started = time.time()
    hf_class = getattr(transformers, config["architectures"][0])
    hf = hf_class.from_pretrained(args.hf, dtype=torch.bfloat16)
    hf.eval()
    loaded = time.time() - started
    out = []
    for i in ids:
        prompt = codec.encode(texts[i])
        target = tokenizer(rows[i]["text_nl"], add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            logits = hf(input_ids=torch.tensor([prompt + target])).logits[0]
        predicted = logits[len(prompt) - 1:-1].argmax(-1).tolist()
        out.append({"id": i, "prompt_tokens": len(prompt), "output_tokens": len(target),
                    "agreement": agreement(predicted, target)})
        print(f"CHECK {json.dumps(out[-1])}", flush=True)
    shares = [r["agreement"] for r in out]
    result = {"model": args.model, "hf": config["architectures"][0], "segments": out,
              "mean": statistics.mean(shares), "min": min(shares), "min_agreement": args.min_agreement,
              "load_seconds": round(loaded), "seconds": round(time.time() - started)}
    result["holds"] = result["mean"] >= args.min_agreement
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"CHECK_RESULT {json.dumps({k: v for k, v in result.items() if k != 'segments'})}", flush=True)
    if not result["holds"]:
        raise SystemExit(f"{args.model}: MaxText's output agrees with Hugging Face's greedy choice at "
                         f"{result['mean']:.3f} of the positions, below {args.min_agreement}")


# ---------------------------------------------------------------------------------------------------- report

def a1(sets, refqe):
    """(a1) per set and in all: the model's mean MetricX-QE not above the human reference's"""
    out = {s: {"model": sets[s]["gemma4"]["metricx_qe"]["mean"], "human": refqe[s]["ref_mean"]} for s in A1_SETS}
    for v in out.values():
        v["holds"] = v["model"] <= v["human"]
    return {"sets": out, "holds": all(v["holds"] for v in out.values())}


def nl(x, digits=3, signed=False):
    if x is None:
        return "–"
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace(".", ",")


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%".replace(".", ",")


def decode_cost(throughput):
    """engine loads (s) and output tokens per second over all classes of the run's logs"""
    loads, tokens, seconds = [], 0, 0.0
    for t in throughput.values():
        loads += t["engine_seconds"]
        for c in t["classes"].values():
            tokens += c["tokens_out"]
            seconds += c["seconds"]
    return {"engine_seconds": loads, "tokens_out": tokens, "seconds": seconds,
            "out_per_s": tokens / seconds if seconds else None}


def table(columns, refqe):
    """The comparison table: one column per (label, evalcompare result, extra), TG and the human reference last"""
    tg = columns[0][1]["sets"]
    rows = []

    def row(name, cells, tg_cell="–", human="–"):
        rows.append(f"| {name} | " + " | ".join(cells) + f" | {tg_cell} | {human} |")

    for s, name in (("wmt24pp", "WMT24++ (960)"), ("flores", "FLORES"), ("ntrex", "NTREX"), ("europarl", "Europarl")):
        row(f"MetricX-QE {name}", [nl(c[1]["sets"][s]["gemma4"]["metricx_qe"]["mean"]) for c in columns],
            nl(tg[s]["tg"]["metricx_qe"]["mean"]), nl(refqe[s]["ref_mean"]))
        row("idem, verschil met TG", [nl(c[1]["sets"][s]["gemma4"]["metricx_qe"]["mean"]
                                         - c[1]["sets"][s]["tg"]["metricx_qe"]["mean"], 2, True) for c in columns],
            "–", nl(refqe[s]["delta"], 2, True))
    row("(a1): QE niet hoger dan de mens (WMT / FLORES / NTREX)",
        ["**ja**" if c[2]["a1"]["holds"] else "nee (" + ", ".join(
            s for s, v in c[2]["a1"]["sets"].items() if not v["holds"]) + ")" for c in columns])
    for s, name in (("wmt24pp", "WMT24++ (960)"), ("flores", "FLORES"), ("ntrex", "NTREX"), ("europarl", "Europarl")):
        row(f"MetricX-ref {name}", [nl(c[1]["sets"][s]["gemma4"]["metricx_ref"]["mean"]) for c in columns],
            nl(tg[s]["tg"]["metricx_ref"]["mean"]))
    for s, name in (("wmt24pp", "WMT24++ (960)"), ("flores", "FLORES"), ("ntrex", "NTREX"), ("europarl", "Europarl")):
        row(f"COMET-22 {name}", [nl(c[1]["sets"][s]["gemma4"]["comet22"]["mean"]) for c in columns],
            nl(tg[s]["tg"]["comet22"]["mean"]))
    row("MetricX-QE FLAN (1.000)", [nl(c[1]["flan"]["qe_mean"], 2) for c in columns], "2,12")
    sets4 = ("wmt24pp", "flores", "ntrex", "europarl")
    row("uitvoer/referentie, woorden, mediaan (WMT / FLORES / NTREX / Europarl)",
        [" / ".join(nl((c[1]["sets"][s]["gemma4"]["length"] or {}).get("out_per_ref")) for s in sets4) for c in columns],
        " / ".join(nl((tg[s]["tg"]["length"] or {}).get("out_per_ref")) for s in sets4), "1")
    row("< 0,7 × referentie (WMT / FLORES / NTREX / Europarl)",
        [" / ".join(pct((c[1]["sets"][s]["gemma4"]["length"] or {}).get("under_0_7_of_ref")) for s in sets4)
         for c in columns],
        " / ".join(pct((tg[s]["tg"]["length"] or {}).get("under_0_7_of_ref")) for s in sets4))
    row("lengte tegen TG, mediaan per segment (WMT / FLORES / NTREX / Europarl)",
        [" / ".join(nl(c[1]["sets"][s]["per_segment"]["words_vs_tg"]) for s in sets4) for c in columns], "1")
    row("niet af / lus / Engels over / leeg, vier sets",
        [" / ".join(str(sum(c[1]["sets"][s]["gemma4"][k] for s in sets4))
                    for k in ("not_ended", "decode_loop", "english_left", "empty")) for c in columns])
    row("idem, FLAN", [" / ".join(str(c[1]["flan"][k]) for k in ("not_ended", "decode_loop", "english_left", "empty"))
                       for c in columns])
    row("prompt in de uitvoer (regex van de pilot), vier sets / FLAN",
        [c[2].get("leak", "–") for c in columns])
    row("chips per proces", [str(c[2]["chips"]) for c in columns])
    row("engine laden (s, per lengteklasse)", [c[2]["engine"] for c in columns])
    row("uitvoertokens/s per proces (vier sets + FLAN)", [nl(c[2]["cost"]["out_per_s"], 0) for c in columns])
    return rows


def report(args):
    refqe = json.loads(Path(args.refqe).read_text())
    g4 = json.loads(Path(args.g4).read_text())
    columns = []
    for spec in args.eval:
        size, path = spec.split("=", 1)
        result = json.loads(Path(path).read_text())
        extra = json.loads(Path(path).with_name(f"extra-{size}.json").read_text()) if Path(path).with_name(
            f"extra-{size}.json").exists() else {}
        cost = decode_cost(result["throughput"])
        engine = cost["engine_seconds"]
        columns.append((LABELS[size], result, {
            "a1": a1(result["sets"], refqe), "cost": cost, "chips": extra.get("chips", "–"),
            "engine": f"{min(engine)}–{max(engine)}" if engine else "–", "leak": extra.get("leak", "–")}))
    cost31 = decode_cost(g4["throughput"])
    engine31 = cost31["engine_seconds"]
    columns.append(("Gemma 4 31B-it", g4, {"a1": a1(g4["sets"], refqe), "cost": cost31, "chips": 4,
                                           "engine": f"{min(engine31)}–{max(engine31)}", "leak": "1 / 0"}))
    columns.sort(key=lambda c: list(LABELS.values()).index(c[0]) if c[0] in LABELS.values() else len(LABELS))
    header = "| maat | " + " | ".join(c[0] for c in columns) + " | TG 27B | menselijke referentie |"
    sep = "|---|" + "---:|" * (len(columns) + 2)
    lines = [header, sep] + table(columns, refqe)
    summary = {c[0]: {"a1": c[2]["a1"], "cost": c[2]["cost"]} for c in columns}
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"REPORT {args.out}", flush=True)


def leak_counts(pairs_four, pairs_flan):
    """'n / m': outputs of the four sets and of FLAN matched by the pilot's prompt regex"""
    import bt_natural as bn

    def n(path):
        return sum(bool(LEAK.search(r["hypothesis"] or "")) for r in bn.read_rows(path))

    return f"{n(pairs_four)} / {n(pairs_flan)}"


def extra(args):
    """extra-<size>.json beside the evalcompare result: chips per process and the prompt-leak counts"""
    import translate as tr

    out = {"model": SIZES[args.size], "chips": tr.MODELS[SIZES[args.size]]["chips"],
           "leak": leak_counts(args.pairs_four, args.pairs_flan)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"EXTRA {json.dumps(out)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("convert")
    p.add_argument("--model", required=True, choices=sorted(SIZES.values()))
    p.add_argument("--hf", required=True, help="Local snapshot of the model's Hugging Face repo")
    p.add_argument("--out", required=True, help="Local base_output_directory; the checkpoint lands in <out>/0/items")
    p = sub.add_parser("check")
    p.add_argument("--model", required=True, choices=sorted(SIZES.values()))
    p.add_argument("--hf", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--segments", required=True, help="segments-w0.parquet the shards translated")
    p.add_argument("--shards", required=True, help="Local copy of translate.py's shards")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--min-agreement", type=float, default=0.9)
    p.add_argument("--out", required=True)
    p = sub.add_parser("extra")
    p.add_argument("--size", required=True, choices=sorted(SIZES))
    p.add_argument("--pairs-four", required=True)
    p.add_argument("--pairs-flan", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--eval", nargs="+", required=True, help="SIZE=evalcompare JSON (extra-SIZE.json beside it)")
    p.add_argument("--g4", required=True, help="The 31B's eval-gemma4.json")
    p.add_argument("--refqe", required=True, help="refqe.json of the human references")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"convert": convert, "check": check, "extra": extra, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
