#!/bin/bash
# G4-6's CPU proof (draft amendment G4-6.6) on a worker's CPU, the TPU untouched, nothing written to the bucket:
# sft_lora.py --dpo in the SFT venv (MaxText 0.2.5, Tunix, qwix) with a tiny Gemma 4 (6 layers, random weights,
# float32) on 4 simulated CPU devices, rows of 4,096, on pairs of build6 packed again at 4 rows per batch:
#   (a) --dpo, 8 steps, LR 1e-2, evals at 0, 4, 8 on books and own (SFT files) and pref_btval (packed pairs)
#   (b) the same tiny model as plain SFT (MaxText's own loss with num_vocab_tiling=8), its evals at step 0
# and checks: (a) ran 8 steps on the packed file's first 8 batches (tokens per step = the file's); at step 0 the
# policy is the reference (pref loss log 2, margin 0, each NLL the reference's); (a)'s step-0 losses of books and own =
# (b)'s (the preference loss's per-token log-probabilities = MaxText's cross entropy); at step 8 the reference is still
# the base model (its NLL the same as at step 0) while the policy moved; (a)'s reference was computed before the first
# step over the 8 trained batches and pref_btval (amendment G4-6.4a).
# With BAKE=1 (amendment G4-6b.6) also, with (a)'s adapter as the adapter to merge:
#   (c) --dpo with --base-adapter (the adapter merged into the base kernels), 4 steps, evals at 0 and 4
#   (d) plain SFT with --init-adapter (the same adapter as LoRA in the forward pass), its evals at step 0
# and checks: (c)'s step-0 losses of books and own = (d)'s (the merged model is the adapter's model) and not (b)'s (the
# merge changed the model); at step 0 (c)'s policy is its reference (|pref loss - log 2| and |margin| <= 1e-4, G4-6.6's
# bound as in addendum G4-6.6a; each NLL the reference's) and that reference is not (a)'s; at step 4 (c)'s reference is unchanged.
# With SPEC=DPO6C and C6=1 (amendment G4-6c.6: G4-6c's loss, the turn's end outside the log-ratio, kept adapters) also:
#   (a) keeps its adapter after step 4 (--save-steps 4); the real pairs packed here end every segment in <turn|>\n
#   (e) plain SFT with (a)'s adapter of step 4 as --init-adapter, its evals at step 0
# and checks: (e)'s step-0 losses of books and own = (a)'s at step 4 (the kept adapter is the model of that step);
# every eval of pref_btval counts 2 tokens per side fewer in the log-ratio than in the NLL; (a)'s loss at step 1 = its
# DPO term + nll_alpha x the chosen NLL over every target token; (a)'s spec is G4-6c's.
# With REFIN=1 (addendum G4-6c.6e: the reference of the training batches from a run of its own; needs BAKE=1) also:
#   (f) --ref-only with (c)'s merged adapter over (c)'s 4 training batches (ref-train.npy, ref-train.json)
#   (g) (c) again with --ref-in (f)'s file: no reference pass over its training batches
#   (h) --ref-in (f)'s file in a run without the merged adapter (another model)
# and checks: (f) is the reference alone (4 batches, no adapter); (f)'s reference = (c)'s, computed in (c)'s own process
# before its first step, for every pair's chosen and rejected sum (within REF_MAX_DIFF6E relative, bitwise reported);
# (g) read it and computed no pass over its training batches; (g)'s training = (c)'s (every step's preference measures,
# every eval, the final adapter; within REF_MAX_DIFF6E); (h) is refused before its model is set up.
# Started by run_g4arm.sh cpuproof6 (w1) with G, G6, H, PY (the SFT venv's python), TOK (Gemma 4's tokenizer), CODE
# (the bundle) and X (the work directory) in the environment. Ends in CPU_PROOF6_OK or CPU_PROOF6_FAILED.
set -euo pipefail
: "${G:?}" "${G6:?}" "${H:?}" "${PY:?}" "${TOK:?}" "${CODE:?}" "${X:?}"
BAKE=${BAKE:-0}
SPEC=${SPEC:-DPO6}
C6=${C6:-0}
REFIN=${REFIN:-0}
[ "$REFIN" = 0 ] || [ "$BAKE" = 1 ] || { echo "REFIN=1 needs BAKE=1 (the reference of (c))"; exit 1; }
[ ! -e "$X" ] || { echo "$X exists"; exit 1; }
P=$X/cpu
mkdir -p $X/in $P
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
echo "$(date +%T) inputs"
gsutil -m -q cp $G6/pairs.parquet $G6/report.json $G6/template.jinja $X/in/
gsutil -m -q cp $H/eval-books.parquet $H/eval-own.parquet $X/in/
JAX_PLATFORMS=cpu $PY - <<EOF
import json, sys
sys.path.insert(0, "$CODE")
import g4arm, g4dpo
import pyarrow as pa, pyarrow.parquet as pq
report = json.load(open("$X/in/report.json"))
pairs = pq.read_table("$X/in/pairs.parquet").to_pylist()
train = sorted((p for p in pairs if p["set"] == "train"), key=lambda p: p["order"])[:200]
pref = sorted((p for p in pairs if p["set"] == "pref_btval"), key=lambda p: p["id"])[:24]
counter = g4arm.TokenCounter("$TOK")
pad, bos = report["pad_id"], report["bos_id"]
for name, part in (("train", train), ("pref_btval", pref)):
    tok = g4arm.tokenized6(part, counter)
    rows = g4dpo.pack_rows([{"id": p["id"], "prompt": p["prompt"], "chosen": p["chosen_ids"], "rejected": p["rejected_ids"]}
                            for p in tok], 4096, 4, pad_id=pad, bos_id=bos)
    g4dpo.write_rows(rows, f"$P/{name}.parquet")
    print("PACKED", name, len(rows), "rows", 1 + max(r["batch"] for r in rows), "batches", flush=True)
    if "$C6" == "1":
        # amendment G4-6c.6: in the real pairs every segment ends in the turn's end, the tokens the log-ratio leaves out
        end_ids = counter.ids([g4arm.END6C])[0]
        ends = g4dpo.check_ends(f"$P/{name}.parquet", g4arm.DPO6C["end_tokens"], end_ids)
        print("ENDS_REAL", name, json.dumps({"end_ids": end_ids, **ends}), flush=True)
        json.dump({"end_ids": end_ids, **ends}, open(f"$P/ends-{name}.json", "w"))
assert 1 + max(r["batch"] for r in pq.read_table("$P/train.parquet").to_pylist()) >= 8
pq.write_table(pa.Table.from_pylist([g4arm.messages_row(p["en"], p["chosen"]) for p in train]), "$P/train-chosen.parquet")
for name in ("books", "own"):
    rows = pq.read_table(f"$X/in/eval-{name}.parquet").to_pylist()[:20]
    pq.write_table(pa.Table.from_pylist(rows), f"$P/eval-{name}.parquet")
json.dump(g4dpo.spec(**getattr(g4arm, "$SPEC")), open("$P/dpo.json", "w"))
EOF
cp $X/in/template.jinja $P/template.jinja

export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE=$P/hf-datasets
MX=(model_name=gemma4-31b override_model_config=true base_num_decoder_layers=6 base_emb_dim=128 base_num_query_heads=4
  base_num_kv_heads=2 head_dim=32 global_head_dim=64 global_num_kv_heads=1 base_mlp_dim=256 vocab_size=262144
  use_multimodal=false convert_checkpoint_if_possible=false tokenizer_type=huggingface tokenizer_path=$TOK hf_path=parquet
  train_split=train chat_template_path=$P/template.jinja enable_data_shuffling=false
  max_target_length=4096 packing=true scan_layers=false lora.enable_lora=true lora.lora_rank=4 lora.lora_alpha=2.0
  "lora.lora_module_path=decoder/layers_[0-9]+/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))"
  learning_rate=1e-2 lr_schedule_type=wsd wsd_decay_style=linear
  learning_rate_final_fraction=0.1 adam_b2=0.999 adam_weight_decay=0.01 gradient_clipping_threshold=1.0 eval_interval=-1
  checkpoint_period=100000 enable_goodput_recording=false monitor_goodput=false log_period=1
  skip_jax_distributed_system=true num_vocab_tiling=8 jax_cache_dir=$P/jax_cache per_device_batch_size=1
  weight_dtype=float32 dtype=float32 attention=dot_product hf_train_files=$P/train-chosen.parquet)
EV=(--eval books=$P/eval-books.parquet --eval own=$P/eval-own.parquet)
SAVE=()
[ "$C6" = 1 ] && SAVE=(--save-steps 4)
echo "$(date +%T) (a) --dpo, 8 steps"
$PY -u $CODE/sft_lora.py --out $P/a --steps 8 "${EV[@]}" --eval-every 4 --warmup-steps 2 --decay-fraction 0 \
  --dpo $P/dpo.json --pref-train $P/train.parquet --pref-eval pref_btval=$P/pref_btval.parquet "${SAVE[@]}" \
  -- run_name=g4a6c_a "${MX[@]}"
echo "$(date +%T) (b) plain SFT, the evals at step 0"
$PY -u $CODE/sft_lora.py --out $P/b --steps 1 "${EV[@]}" --eval-every 1 -- run_name=g4a6c_b "${MX[@]}"
if [ "$BAKE" = 1 ]; then
  echo "$(date +%T) (c) --dpo with (a)'s adapter merged, 4 steps"
  $PY -u $CODE/sft_lora.py --out $P/c --steps 4 "${EV[@]}" --eval-every 4 --warmup-steps 1 --decay-fraction 0 \
    --dpo $P/dpo.json --pref-train $P/train.parquet --pref-eval pref_btval=$P/pref_btval.parquet \
    --base-adapter tiny=$P/a/adapter.npz -- run_name=g4a6c_c "${MX[@]}"
  echo "$(date +%T) (d) plain SFT with (a)'s adapter as LoRA, the evals at step 0"
  $PY -u $CODE/sft_lora.py --out $P/d --steps 1 "${EV[@]}" --eval-every 1 --init-adapter $P/a/adapter.npz \
    -- run_name=g4a6c_d "${MX[@]}"
fi
if [ "$REFIN" = 1 ]; then
  echo "$(date +%T) (f) --ref-only with (a)'s adapter merged, (c)'s 4 training batches"
  $PY -u $CODE/sft_lora.py --out $P/f --steps 4 --warmup-steps 1 --decay-fraction 0 --dpo $P/dpo.json \
    --pref-train $P/train.parquet --ref-only --base-adapter tiny=$P/a/adapter.npz -- run_name=g4a6c_f "${MX[@]}"
  echo "$(date +%T) (g) (c) with --ref-in (f)'s reference"
  $PY -u $CODE/sft_lora.py --out $P/g --steps 4 "${EV[@]}" --eval-every 4 --warmup-steps 1 --decay-fraction 0 \
    --dpo $P/dpo.json --pref-train $P/train.parquet --pref-eval pref_btval=$P/pref_btval.parquet \
    --base-adapter tiny=$P/a/adapter.npz --ref-in $P/f/ref-train.npy -- run_name=g4a6c_g "${MX[@]}"
  echo "$(date +%T) (h) --ref-in (f)'s reference without the merged adapter: refused"
  if $PY -u $CODE/sft_lora.py --out $P/h --steps 4 "${EV[@]}" --eval-every 4 --warmup-steps 1 --decay-fraction 0 \
    --dpo $P/dpo.json --pref-train $P/train.parquet --pref-eval pref_btval=$P/pref_btval.parquet \
    --ref-in $P/f/ref-train.npy -- run_name=g4a6c_h "${MX[@]}" > $P/h.log 2>&1; then echo "H_EXIT 0"; else echo "H_EXIT $?"; fi
  tail -n 3 $P/h.log
fi
if [ "$C6" = 1 ]; then
  echo "$(date +%T) (e) plain SFT with (a)'s adapter of step 4 as LoRA, the evals at step 0"
  $PY -u $CODE/sft_lora.py --out $P/e --steps 1 "${EV[@]}" --eval-every 1 --init-adapter $P/a/adapter-step4.npz \
    -- run_name=g4a6c_e "${MX[@]}"
fi
$PY - <<EOF
import json, math, sys
sys.path.insert(0, "$CODE")
import g4arm, g4dpo
import numpy as np
a, b = (json.load(open(f"$P/{r}/summary.json")) for r in "ab")
ev = lambda s, name, step: next(e for e in s["eval"] if e["set"] == name and e["step"] == step)
print("A_EVAL", json.dumps(a["eval"]))
print("A_DPO", json.dumps(a["dpo"]["steps"]))
print("B_EVAL", json.dumps(b["eval"]))
bad = []
def need(ok, what):
    print(("OK   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        bad.append(what)
need(a["steps_done"] == 8 and len(a["dpo"]["steps"]) == 8, "a: 8 steps, 8 preference records")
need(all(s["pairs"] > 0 and math.isfinite(s["loss"]) for s in a["dpo"]["steps"]), "a: every step's pairs and loss")
arrays, batch = g4dpo.read_packed("$P/train.parquet")
want = [int(np.count_nonzero(arrays["inputs_segmentation"][batch == k])) for k in range(8)]
need(a["tokens"]["per_step"] == want, f"a: the steps' tokens are the packed file's batches {a['tokens']['per_step']} {want}")
p0, p8 = ev(a, "pref_btval", 0), ev(a, "pref_btval", 8)
need(abs(p0["loss"] - math.log(2)) <= 1e-4 and abs(p0["margin"]) <= 1e-4, f"a: step 0 loss log 2, margin 0 {p0}")  # G4-6.6's step-0 bound (addendum G4-6.6a)
need(abs(p0["chosen_nll"] - p0["ref_chosen_nll"]) <= 1e-6 and abs(p0["rejected_nll"] - p0["ref_rejected_nll"]) <= 1e-6,
     "a: step 0 policy NLL = reference NLL")
for name in ("books", "own"):
    x, y = ev(a, name, 0)["loss"], ev(b, name, 0)["loss"]
    need(abs(x - y) <= 1e-4 * max(1.0, abs(y)), f"a: {name} at step 0 = MaxText's own loss ({x} vs {y})")
need(abs(p8["ref_chosen_nll"] - p0["ref_chosen_nll"]) <= 1e-5 and abs(p8["ref_rejected_nll"] - p0["ref_rejected_nll"]) <= 1e-5,
     f"a: the reference is still the base at step 8 ({p0['ref_chosen_nll']} {p8['ref_chosen_nll']})")
need(abs(p8["margin"]) > 1e-4 or abs(p8["chosen_nll"] - p0["chosen_nll"]) > 1e-4, f"a: the policy moved by step 8 {p8}")
need(all(sorted(e["step"] for e in a["eval"] if e["set"] == n) == [0, 4, 8] for n in ("books", "own", "pref_btval")),
     "a: evals at 0, 4, 8")
ref = a["dpo"].get("ref") or {}
need(set(ref) == {"train", "pref_btval"} and ref["train"]["batches"] == 8 and ref["pref_btval"]["batches"] == p0["batches"],
     f"a: the reference computed before the first step, the trained batches and pref_btval (amendment G4-6.4a) {ref}")
if "$BAKE" == "1":
    c, d = (json.load(open(f"$P/{r}/summary.json")) for r in "cd")
    print("C_EVAL", json.dumps(c["eval"]))
    print("C_BASE_ADAPTER", json.dumps(c["base_adapter"]))
    print("D_EVAL", json.dumps(d["eval"]))
    need(c["steps_done"] == 4 and len(c["dpo"]["steps"]) == 4, "c: 4 steps, 4 preference records")
    need((c["base_adapter"] or {}).get("modules", 0) > 0 and c["base_adapter"]["name"] == "tiny", f"c: the adapter merged {c['base_adapter']}")
    for name in ("books", "own"):
        x, y, z = ev(c, name, 0)["loss"], ev(d, name, 0)["loss"], ev(b, name, 0)["loss"]
        need(abs(x - y) <= 1e-4 * max(1.0, abs(y)), f"c: {name} at step 0 = the adapter as LoRA ({x} vs {y})")
        need(abs(x - z) > 1e-4, f"c: {name} at step 0 is not the base model's ({x} vs {z})")
    q0, q4 = ev(c, "pref_btval", 0), ev(c, "pref_btval", 4)
    need(abs(q0["loss"] - math.log(2)) <= 1e-4 and abs(q0["margin"]) <= 1e-4, f"c: step 0 loss log 2, margin 0 {q0}")  # G4-6.6a
    need(abs(q0["chosen_nll"] - q0["ref_chosen_nll"]) <= 1e-6 and abs(q0["rejected_nll"] - q0["ref_rejected_nll"]) <= 1e-6,
         "c: step 0 policy NLL = reference NLL")
    need(abs(q0["ref_chosen_nll"] - p0["ref_chosen_nll"]) > 1e-4, f"c: the reference is the merged model, not (a)'s ({q0['ref_chosen_nll']} vs {p0['ref_chosen_nll']})")
    need(abs(q4["ref_chosen_nll"] - q0["ref_chosen_nll"]) <= 1e-5 and abs(q4["ref_rejected_nll"] - q0["ref_rejected_nll"]) <= 1e-5,
         f"c: the reference is still the merged model at step 4 ({q0['ref_chosen_nll']} {q4['ref_chosen_nll']})")
    need(set(c["dpo"].get("ref") or {}) == {"train", "pref_btval"} and c["dpo"]["ref"]["train"]["batches"] == 4,
         f"c: the reference (the merged model) computed before the first step (amendment G4-6.4a) {c['dpo'].get('ref')}")
if "$C6" == "1":
    spec = g4dpo.spec(**g4arm.DPO6C)
    need(a["dpo"]["spec"] == spec, f"c6: (a)'s spec is G4-6c's {a['dpo']['spec']}")
    e = json.load(open("$P/e/summary.json"))
    print("E_EVAL", json.dumps(e["eval"]))
    meta = json.load(open("$P/a/adapter-step4.json"))
    kept = (a.get("saved_adapters") or {}).get("4") or {}
    need(meta["steps"] == 4 and (meta["rank"], meta["alpha"]) == (4, 2.0) and kept.get("sum_abs", 0) > 0
         and kept["sum_abs"] != a.get("adapter_sum_abs"), f"c6: (a) kept its adapter of step 4 {meta} {kept}")
    for name in ("books", "own"):
        x, y = ev(e, name, 0)["loss"], ev(a, name, 4)["loss"]
        need(abs(x - y) <= 1e-4 * max(1.0, abs(y)), f"c6: {name}: the kept adapter of step 4 = (a) at step 4 ({x} vs {y})")
    n_end = spec["end_tokens"]
    for step in (0, 4, 8):
        q = ev(a, "pref_btval", step)
        need(q["chosen_tokens_all"] - q["chosen_tokens"] == n_end * q["pairs"] and
             q["rejected_tokens_all"] - q["rejected_tokens"] == n_end * q["pairs"],
             f"c6: pref_btval at step {step}: the log-ratio {n_end} tokens per side short of the NLL {q}")
    s1, l1 = a["dpo"]["steps"][0], a["loss"][0][1]
    want = s1["loss"] + spec["nll_alpha"] * s1["chosen_nll_all"]
    need(abs(l1 - want) <= g4arm.LOSS_MAX_DIFF6C * max(1.0, abs(l1)),
         f"c6: step 1's loss = DPO term + {spec['nll_alpha']} x the chosen NLL over every target token ({l1} vs {want})")
    for name in ("train", "pref_btval"):
        ends = json.load(open(f"$P/ends-{name}.json"))
        need(ends["segments"] > 0 and len(ends["end_ids"]) == n_end, f"c6: the real pairs of {name} end in <turn|>\\n {ends}")
if "$REFIN" == "1":
    import os

    import pyarrow.parquet as pq

    bound = g4arm.REF_MAX_DIFF6E
    f, g = (json.load(open(f"$P/{r}/summary.json")) for r in "fg")
    print("F_REF", json.dumps(f["dpo"]["ref"]))
    print("G_REF", json.dumps(g["dpo"]["ref"]))
    print("G_EVAL", json.dumps(g["eval"]))
    need(f.get("ref_only") is True and set(f["dpo"]["ref"]) == {"train"} and f["dpo"]["ref"]["train"]["batches"] == 4
         and not os.path.exists("$P/f/adapter.npz"), f"e6: (f) is the reference alone, 4 batches, no adapter {f['dpo']['ref']}")
    ref_f, meta_f = np.load("$P/f/ref-train.npy"), json.load(open("$P/f/ref-train.json"))
    # (c)'s reference rows (dpo_ref.parquet: every pair's chosen and rejected sum, from (c)'s own pass) against the
    # same rows made from (f)'s file
    rows_c = [{k: v for k, v in r.items() if k != "set"}
              for r in pq.read_table("$P/c/dpo_ref.parquet").to_pylist() if r["set"] == "train"]
    rows_f = g4dpo.reference_rows("$P/train.parquet", ref_f, 4, 4)
    ids = lambda rows: [(r["file"], r["batch"], r["row"], r["pair"]) for r in rows]
    vals = [(a[k], b[k]) for a, b in zip(rows_c, rows_f) for k in ("ref_chosen_logp", "ref_rejected_logp")]
    diff = max(abs(a - b) / max(1.0, abs(a)) for a, b in vals) if vals else math.inf
    bitwise = bool(vals) and all(a == b for a, b in vals)
    print("E6_REF", json.dumps({"pairs": len(rows_c), "values": len(vals), "max_rel_diff": diff, "bitwise": bitwise,
                                "bound": bound, "ref_sum_c": c["dpo"]["ref"]["train"]["ref_sum"],
                                "ref_sum_f": f["dpo"]["ref"]["train"]["ref_sum"]}))
    need(len(rows_c) > 0 and ids(rows_c) == ids(rows_f) and diff <= bound,
         f"e6: (f)'s reference = (c)'s from its own process, every pair ({len(rows_c)} pairs, max relative diff {diff}, bitwise {bitwise})")
    rg = (g["dpo"]["ref"] or {}).get("train") or {}
    need(rg.get("ref_in") == "$P/f/ref-train.npy" and rg.get("sha256") == meta_f["sha256"] and "seconds" not in rg
         and rg.get("ref_sum") == f["dpo"]["ref"]["train"]["ref_sum"] and set(g["dpo"]["ref"]) == {"train", "pref_btval"},
         f"e6: (g) read (f)'s reference and ran no pass over its training batches {rg}")
    rows_g = [{k: v for k, v in r.items() if k != "set"}
              for r in pq.read_table("$P/g/dpo_ref.parquet").to_pylist() if r["set"] == "train"]
    need(rows_g == rows_f, "e6: (g)'s reference rows = those of (f)'s file")
    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    pairs = [(f"step {a['step']} {k}", a[k], b[k]) for a, b in zip(c["dpo"]["steps"], g["dpo"]["steps"]) for k in a if num(a[k])]
    key = lambda e: (e["set"], e["step"])
    ec, eg = sorted(c["eval"], key=key), sorted(g["eval"], key=key)
    pairs += [(f"eval {a['set']} {a['step']} {k}", a[k], b.get(k)) for a, b in zip(ec, eg) for k in a
              if num(a[k]) and k != "seconds"]
    with np.load("$P/c/adapter.npz") as zc, np.load("$P/g/adapter.npz") as zg:
        same_keys = set(zc.files) == set(zg.files)
        ad = max(float(np.max(np.abs(zc[k] - zg[k]))) for k in zc.files) if same_keys else math.inf
    tdiff = max([abs(x - y) / max(1.0, abs(x)) if num(y) else math.inf for _, x, y in pairs] + [ad])
    worst = max(pairs, key=lambda t: abs(t[1] - t[2]) if num(t[2]) else math.inf, default=None)
    print("E6_TRAIN", json.dumps({"values": len(pairs), "max_rel_diff": tdiff, "adapter_max_abs_diff": ad,
                                  "bitwise": all(x == y for _, x, y in pairs) and ad == 0.0, "worst": worst}))
    need(len(c["dpo"]["steps"]) == len(g["dpo"]["steps"]) == 4 and [key(e) for e in ec] == [key(e) for e in eg]
         and tdiff <= bound, f"e6: (g)'s training = (c)'s (steps, evals, adapter; max relative diff {tdiff})")
    h_log = open("$P/h.log").read()
    need("not this run's reference" in h_log and "base_adapter" in h_log and not os.path.exists("$P/h/summary.json"),
         "e6: (h) refused (f)'s reference for a model without the merged adapter")
print("CPU_PROOF6_OK" if not bad else f"CPU_PROOF6_FAILED {bad}", flush=True)
raise SystemExit(1 if bad else 0)
EOF
echo "$(date +%T) done"
