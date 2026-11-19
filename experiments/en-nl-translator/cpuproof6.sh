#!/bin/bash
# G4-6's CPU proof (draft amendment G4-6.6) on a worker's CPU, the TPU untouched, nothing written to the bucket:
# sft_lora.py --dpo in the SFT venv (MaxText 0.2.5, Tunix, qwix) with a tiny Gemma 4 (6 layers, random weights,
# float32) on 4 simulated CPU devices, rows of 4,096, on pairs of build6 packed again at 4 rows per batch:
#   (a) --dpo, 8 steps, LR 1e-2, evals at 0, 4, 8 on books and own (SFT files) and pref_btval (packed pairs)
#   (b) the same tiny model as plain SFT (MaxText's own loss with num_vocab_tiling=8), its evals at step 0
# and checks: (a) ran 8 steps on the packed file's first 8 batches (tokens per step = the file's); at step 0 the
# policy is the reference (pref loss log 2, margin 0, each NLL the reference's); (a)'s step-0 losses of books and own =
# (b)'s (the preference loss's per-token log-probabilities = MaxText's cross entropy); at step 8 the reference is still
# the base model (its NLL the same as at step 0) while the policy moved.
# With BAKE=1 (amendment G4-6b.6) also, with (a)'s adapter as the adapter to merge:
#   (c) --dpo with --base-adapter (the adapter merged into the base kernels), 4 steps, evals at 0 and 4
#   (d) plain SFT with --init-adapter (the same adapter as LoRA in the forward pass), its evals at step 0
# and checks: (c)'s step-0 losses of books and own = (d)'s (the merged model is the adapter's model) and not (b)'s (the
# merge changed the model); at step 0 (c)'s policy is its reference (pref loss log 2, |margin| <= 1e-4 as the smoke
# test's bound, each NLL the reference's) and that reference is not (a)'s; at step 4 (c)'s reference is unchanged.
# Started by run_g4arm.sh cpuproof6 (w1) with G, G6, H, PY (the SFT venv's python), TOK (Gemma 4's tokenizer), CODE
# (the bundle) and X (the work directory) in the environment. Ends in CPU_PROOF6_OK or CPU_PROOF6_FAILED.
set -euo pipefail
: "${G:?}" "${G6:?}" "${H:?}" "${PY:?}" "${TOK:?}" "${CODE:?}" "${X:?}"
BAKE=${BAKE:-0}
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
assert 1 + max(r["batch"] for r in pq.read_table("$P/train.parquet").to_pylist()) >= 8
pq.write_table(pa.Table.from_pylist([g4arm.messages_row(p["en"], p["chosen"]) for p in train]), "$P/train-chosen.parquet")
for name in ("books", "own"):
    rows = pq.read_table(f"$X/in/eval-{name}.parquet").to_pylist()[:20]
    pq.write_table(pa.Table.from_pylist(rows), f"$P/eval-{name}.parquet")
json.dump(g4dpo.spec(**g4arm.DPO6), open("$P/dpo.json", "w"))
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
echo "$(date +%T) (a) --dpo, 8 steps"
$PY -u $CODE/sft_lora.py --out $P/a --steps 8 "${EV[@]}" --eval-every 4 --warmup-steps 2 --decay-fraction 0 \
  --dpo $P/dpo.json --pref-train $P/train.parquet --pref-eval pref_btval=$P/pref_btval.parquet \
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
$PY - <<EOF
import json, math, sys
sys.path.insert(0, "$CODE")
import g4dpo
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
need(abs(p0["loss"] - math.log(2)) <= 1e-5 and abs(p0["margin"]) <= 1e-6, f"a: step 0 loss log 2, margin 0 {p0}")
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
    need(abs(q0["loss"] - math.log(2)) <= 1e-5 and abs(q0["margin"]) <= 1e-4, f"c: step 0 loss log 2, margin 0 {q0}")
    need(abs(q0["chosen_nll"] - q0["ref_chosen_nll"]) <= 1e-6 and abs(q0["rejected_nll"] - q0["ref_rejected_nll"]) <= 1e-6,
         "c: step 0 policy NLL = reference NLL")
    need(abs(q0["ref_chosen_nll"] - p0["ref_chosen_nll"]) > 1e-4, f"c: the reference is the merged model, not (a)'s ({q0['ref_chosen_nll']} vs {p0['ref_chosen_nll']})")
    need(abs(q4["ref_chosen_nll"] - q0["ref_chosen_nll"]) <= 1e-5 and abs(q4["ref_rejected_nll"] - q0["ref_rejected_nll"]) <= 1e-5,
         f"c: the reference is still the merged model at step 4 ({q0['ref_chosen_nll']} {q4['ref_chosen_nll']})")
print("CPU_PROOF6_OK" if not bad else f"CPU_PROOF6_FAILED {bad}", flush=True)
raise SystemExit(1 if bad else 0)
EOF
echo "$(date +%T) done"
