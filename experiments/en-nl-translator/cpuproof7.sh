#!/bin/bash
# G4-7's CPU proof (spec translator-sft/g4-rl-spec.md §8.2, amendment G4-7.9) on a worker's CPU, the TPU untouched,
# nothing written to the bucket: sft_lora.py --grpo in the SFT venv (MaxText 0.2.5, Tunix, qwix) with a tiny Gemma 4
# (6 layers, random weights, float32) on 4 simulated CPU devices, on two synthetic iterations of 64 samples (8 sources x
# G = 8; Gemma 4's tokenizer; the sample and score files of amendment G4-7.4a; g4rl.py iter7: rewards, advantages, packed
# rows of 1,024 at 4 rows per batch, so that 64 samples give at least 4 batches):
#   (a)  iteration 1: --grpo on F1 (iteration 0's file), 4 steps, LR 1e-2, 2 warmup steps, then constant
#   (c)  iteration 2 in a new process: --grpo on F2 (iteration 1's file) with --resume-step 4 (a's Tunix checkpoint of
#        step 4 copied into c's checkpoint directory, as resume5 does), 4 steps (5-8)
#   (f)  --passes-only on F2, a fresh model (every LoRA B zero: the base model)
#   (g)  --passes-only on F2 with --init-adapter a's adapter (the adapter as LoRA in the forward pass)
#   (e0) --passes-only on F12 (F1's first 4 batches, then F2's first 4): the identity of e's passes
#   (e)  --grpo on F12, 8 steps, --passes-in e0's file with a's and c's passes in it: one uninterrupted run of 8 steps on
#        the same batches with the same old and reference log-probabilities
#   (d)  --grpo on F2 with --base-adapter a's adapter (merged into the base kernels: the start G4-4 of the spec), 2 steps
# and checks (spec §8.2):
#   (a) on step 1 rho = 1 (|log pi - log pi_old| <= 1e-5), KL = 0, the loss = minus the token mean of A over the batch
#       (<= 1e-4 relative); a's passes: old = ref (iteration 1, every LoRA B zero)
#   (b) the reference pass is the base model: c's (a LoRA restored) = f's (<= 1e-5); the old pass is the model with the
#       loaded adapter: c's (restored by Tunix) = g's (loaded by --init-adapter) (<= 1e-5), and not the base
#   (c) c's adapter after step 8 = e's (<= 1e-5), and every step's GRPO measures of a (1-4) and c (5-8) = e's: the
#       optimizer state continued over the new process
#   (d) with --base-adapter the reference is the merged model: d's reference = g's old (<= 1e-4: merged kernels against
#       LoRA in the forward pass, float32), not f's; on d's step 1 rho = 1, KL = 0, the loss minus the mean of A
# Ends in CPU_PROOF7_OK or CPU_PROOF7_FAILED. PREP_ONLY=1: only the synthetic iterations and iter7 (co1: no MaxText).
# Environment: PY (the SFT venv's python), TOK (Gemma 4's tokenizer directory), CODE (the bundle: g4rl.py, sft_lora.py
# and what they import), X (the work directory, new).
set -euo pipefail
: "${PY:?}" "${TOK:?}" "${CODE:?}" "${X:?}"
PREP_ONLY=${PREP_ONLY:-0}
[ ! -e "$X" ] || { echo "$X exists"; exit 1; }
P=$X/cpu
mkdir -p $P
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
echo "$(date +%T) the synthetic iterations"
JAX_PLATFORMS=cpu $PY - <<EOF
import hashlib, json, math, os, random, sys
sys.path.insert(0, "$CODE")
import g4rl
import g4arm, g4lora
import pyarrow as pa, pyarrow.parquet as pq
counter = g4arm.TokenCounter("$TOK")
stop = counter.special_id("<turn|>")
EN = ("the house by the river is old and the light falls softly on the water while the city sleeps under a grey sky "
      "she walked along the street and saw how the children played near the bridge in the late afternoon").split()
NL = ("het huis aan de rivier is oud en het licht valt zacht op het water terwijl de stad slaapt onder een grijze lucht "
      "zij liep door de straat en zag hoe de kinderen speelden bij de brug in de late middag").split()
rng = random.Random(7)
plan = []
for it in (0, 1):
    for k in range(8):
        en = " ".join(rng.choice(EN) for _ in range(rng.randint(30, 60)))
        plan.append({"id": f"i{it}s{k}", "part": ("flan", "novels", "bt")[k % 3], "stratum": "", "doc": f"d{it}{k}",
                     "en": en, "tokens": counter.lengths([en])[0], "iteration": it, "position": k, "qe0": 2.5,
                     "l0": 180, "logit_c2_0": 0.0, "hypothesis0": "x"})
pq.write_table(pa.Table.from_pylist(plan), "$P/plan.parquet")
for it in (0, 1):
    root = f"$X/iter/{it}"
    samples, scores = {}, {"metricx": {}, "c2": {}}
    for n, src in enumerate(r for r in plan if r["iteration"] == it):
        prompt = counter.ids([g4lora.prompt_text(src["en"])])[0]
        for j in range(g4rl.G7):
            hyp = " ".join(rng.choice(NL) for _ in range(rng.randint(110, 150)))
            ids = counter.ids([hyp])[0]
            ended = not (n == 2 and j == 5)                  # one sample hit the decode limit: a failure
            ids = ids + [stop] if ended else ids
            u = g4rl.uid(src["id"], j)
            samples.setdefault(f"w{n % 2}", []).append({
                "uid": u, "id": src["id"], "j": j, "source": src["en"], "hypothesis": hyp, "token_ids": ids,
                "sampler_logp": [-rng.uniform(0.01, 3.0) for _ in ids], "ended": ended, "decode_loop": False,
                "english_left": False, "empty": False, "tokens_in": len(prompt), "tokens_out": len(ids)})
            logit = rng.uniform(-3, 3)
            scores["metricx"][u] = {"uid": u, "qe": 8.0 if (n == 4 and j == 1) else rng.uniform(1.0, 4.0)}
            scores["c2"][u] = {"uid": u, "p_c2": 1 / (1 + math.exp(-logit)), "logit_c2": logit, "c2_tokens": len(ids),
                               "c2_overflow": False}
    for w, rows in samples.items():
        d = f"{root}/samples/{w}"
        os.makedirs(d)
        digests = {}
        for k in range(0, len(rows), 16):
            name = f"shard-{k // 16:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows[k:k + 16]), f"{d}/{name}")
            digests[name] = (len(rows[k:k + 16]), g4rl.file_sha256(f"{d}/{name}"))
        open(f"{d}/DONE", "w").write(json.dumps(g4rl.samples_done(digests)))
    # the judges' files as g4judge.py writes them: a score shard per sample shard, MetricX on the parity of sha256(uid)
    for role, judges in (("metricx", ("w0", "w2")), ("c2", ("w1",))):
        for m, judge in enumerate(judges):
            d = f"{root}/scores/{role}/{judge}"
            os.makedirs(d)
            for w, rows in samples.items():
                marker = {"rows": 0, "shards": {}}
                for k in range(0, len(rows), 16):
                    name = f"shard-{k // 16:05d}.parquet"
                    mine = [scores[role][r["uid"]] for r in rows[k:k + 16]
                            if role == "c2" or int(hashlib.sha256(r["uid"].encode()).hexdigest()[:8], 16) % 2 == m]
                    pq.write_table(pa.Table.from_pylist(mine), f"{d}/shard-{w[1:]}-{k // 16}.parquet")
                    marker["shards"][name] = {"rows": len(mine), "samples_sha256": g4rl.file_sha256(f"{root}/samples/{w}/{name}")}
                    marker["rows"] += len(mine)
                open(f"{d}/DONE-{w[1:]}", "w").write(json.dumps(marker))
open("$P/template.jinja", "w").write(g4lora.TEMPLATE)
pq.write_table(pa.Table.from_pylist([g4arm.messages_row(r["en"], "het huis") for r in plan]), "$P/train-messages.parquet")
json.dump(g4rl.grpo_spec(), open("$P/grpo.json", "w"))
print("SYNTHETIC", json.dumps({"sources": len(plan), "stop_id": stop}), flush=True)
EOF
for it in 0 1; do
  $PY -u $CODE/g4rl.py iter7 --plan $P/plan.parquet --iteration $it --samples $X/iter/$it/samples \
    --scores $X/iter/$it/scores --tokenizer $TOK --seq 1024 --rows 4 --out $P/it$it | cut -c1-600
done
$PY - <<EOF
import json, sys
sys.path.insert(0, "$CODE")
import g4rl
import pyarrow as pa, pyarrow.parquet as pq
parts = []
for it, file in ((0, "$P/it0/train.parquet"), (1, "$P/it1/train.parquet")):
    info = g4rl.packed_info7(file, 4, 1024)
    report = json.load(open(f"$P/it{it}/report.json"))
    assert info["batches"] >= 4, (file, info)
    assert report["fails"]["not_ended"] == 1 and report["fails"]["qe"] == 1, report["fails"]
    rows = [r for r in pq.read_table(file).to_pylist() if r["batch"] < 4]
    parts += [{**r, "batch": r["batch"] + 4 * it} for r in rows]
    print("FILE", it, json.dumps(info), json.dumps(report["train"]), flush=True)
g4rl.write_rows7(parts, "$P/f12.parquet")
print("F12", json.dumps(g4rl.packed_info7("$P/f12.parquet", 4, 1024)), flush=True)
EOF
if [ "$PREP_ONLY" = 1 ]; then echo "PREP_OK"; exit 0; fi
$PY - <<EOF || true
import importlib.metadata as m, json
for name in ("google-tunix", "tunix", "maxtext", "qwix", "jax", "flax"):
    try:
        d = m.distribution(name)
        print("STACK", name, d.version, json.loads(d.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id"))
    except m.PackageNotFoundError:
        pass
EOF

export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE=$P/hf-datasets
MX=(model_name=gemma4-31b override_model_config=true base_num_decoder_layers=6 base_emb_dim=128 base_num_query_heads=4
  base_num_kv_heads=2 head_dim=32 global_head_dim=64 global_num_kv_heads=1 base_mlp_dim=256 vocab_size=262144
  use_multimodal=false convert_checkpoint_if_possible=false tokenizer_type=huggingface tokenizer_path=$TOK hf_path=parquet
  train_split=train chat_template_path=$P/template.jinja enable_data_shuffling=false
  max_target_length=1024 packing=true scan_layers=false lora.enable_lora=true lora.lora_rank=4 lora.lora_alpha=2.0
  "lora.lora_module_path=decoder/layers_[0-9]+/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))"
  learning_rate=1e-2 lr_schedule_type=wsd wsd_decay_style=linear
  learning_rate_final_fraction=0.1 adam_b2=0.999 adam_weight_decay=0.01 gradient_clipping_threshold=1.0 eval_interval=-1
  checkpoint_period=100000 enable_goodput_recording=false monitor_goodput=false log_period=1
  skip_jax_distributed_system=true num_vocab_tiling=8 jax_cache_dir=$P/jax_cache per_device_batch_size=1
  weight_dtype=float32 dtype=float32 attention=dot_product hf_train_files=$P/train-messages.parquet)
GR=(--grpo $P/grpo.json)
WU=(--warmup-steps 2 --decay-fraction 0)
F1=$P/it0/train.parquet
F2=$P/it1/train.parquet
echo "$(date +%T) (a) iteration 1, 4 steps"
$PY -u $CODE/sft_lora.py --out $P/a --steps 4 "${WU[@]}" "${GR[@]}" --grpo-train $F1 -- run_name=g4a7_a "${MX[@]}"
CK=$P/a/maxtext/g4a7_a/checkpoints/4
[ -d $CK ] || { echo "CPU_PROOF7_FAILED: no Tunix checkpoint of step 4 in $CK"; ls -R $P/a/maxtext | head -20; exit 1; }
mkdir -p $P/c/maxtext/g4a7_c/checkpoints
cp -r $CK $P/c/maxtext/g4a7_c/checkpoints/
echo "$(date +%T) (c) iteration 2 in a new process, --resume-step 4, steps 5-8"
$PY -u $CODE/sft_lora.py --out $P/c --steps 4 "${WU[@]}" "${GR[@]}" --grpo-train $F2 --resume-step 4 \
  --resume-adapter $P/a/adapter.npz -- run_name=g4a7_c "${MX[@]}"
echo "$(date +%T) (f) the passes over F2, a fresh model"
$PY -u $CODE/sft_lora.py --out $P/f --steps 4 "${GR[@]}" --grpo-train $F2 --passes-only -- run_name=g4a7_f "${MX[@]}"
echo "$(date +%T) (g) the passes over F2 with a's adapter as --init-adapter"
$PY -u $CODE/sft_lora.py --out $P/g --steps 4 "${GR[@]}" --grpo-train $F2 --passes-only --init-adapter $P/a/adapter.npz \
  -- run_name=g4a7_g "${MX[@]}"
echo "$(date +%T) (e0) the identity of e's passes"
$PY -u $CODE/sft_lora.py --out $P/e0 --steps 8 "${GR[@]}" --grpo-train $P/f12.parquet --passes-only -- run_name=g4a7_e0 "${MX[@]}"
$PY - <<EOF
import json, sys
sys.path.insert(0, "$CODE")
import g4rl
import numpy as np
with np.load("$P/a/grpo-passes.npz") as a, np.load("$P/c/grpo-passes.npz") as c:
    passes = {k: np.concatenate([a[k], c[k]]) for k in g4rl.PASS_KEYS}
meta = json.load(open("$P/e0/grpo-passes.json"))
import os
os.makedirs("$P/e-in")
out = g4rl.write_passes("$P/e-in", passes, meta["identity"], {"from": ["a", "c"]})
print("E_IN", json.dumps({"shape": out["shape"], "sha256": out["sha256"]}), flush=True)
EOF
echo "$(date +%T) (e) one uninterrupted run of 8 steps over F1's and F2's batches, a's and c's passes"
$PY -u $CODE/sft_lora.py --out $P/e --steps 8 "${WU[@]}" "${GR[@]}" --grpo-train $P/f12.parquet \
  --passes-in $P/e-in/grpo-passes.npz -- run_name=g4a7_e "${MX[@]}"
echo "$(date +%T) (d) --base-adapter a's adapter (merged), 2 steps on F2"
$PY -u $CODE/sft_lora.py --out $P/d --steps 2 "${WU[@]}" "${GR[@]}" --grpo-train $F2 --base-adapter tiny=$P/a/adapter.npz \
  -- run_name=g4a7_d "${MX[@]}"
$PY - <<EOF
import json, math, sys
sys.path.insert(0, "$CODE")
import g4rl
import numpy as np
bad = []
def need(ok, what):
    print(("OK   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        bad.append(what)
S = {r: json.load(open(f"$P/{r}/summary.json")) for r in ("a", "c", "d", "e", "f", "g", "e0")}
def passes(r):
    with np.load(f"$P/{r}/grpo-passes.npz") as z:
        return {k: z[k] for k in g4rl.PASS_KEYS}
P = {r: passes(r) for r in ("a", "c", "d", "e0", "f", "g")}
def mask(path, batches):
    arrays, _ = g4rl.read_packed7(path)
    return arrays, arrays["targets_segmentation"][:batches * 4] != 0
A1, M1 = mask("$P/it0/train.parquet", 4)
A2, M2 = mask("$P/it1/train.parquet", 4)
def maxdiff(x, y, m):
    return float(np.abs(x[m].astype(np.float64) - y[m].astype(np.float64)).max())
for r in "acde":
    print(r.upper() + "_GRPO", json.dumps(S[r]["grpo"]["steps"]), flush=True)
    print(r.upper() + "_PASSES", json.dumps(S[r]["grpo"]["passes"]), flush=True)
# (a)
a1 = S["a"]["grpo"]["steps"][0]
want = -float(A1["adv"][:4][M1[:4]].mean())
need(S["a"]["steps_done"] == 4 and len(S["a"]["grpo"]["steps"]) == 4, "a: 4 steps, 4 GRPO records")
need(a1["log_ratio_abs_max"] <= 1e-5, f"a: step 1 rho = 1, max |log pi - log pi_old| {a1['log_ratio_abs_max']}")
need(abs(a1["kl"]) <= 1e-8, f"a: step 1 KL = 0 ({a1['kl']})")
need(abs(a1["loss"] - want) <= 1e-4 * max(1.0, abs(want)), f"a: step 1 loss {a1['loss']} = -token mean of A {want}")
need(abs(S["a"]["loss"][0][1] - want) <= 1e-4 * max(1.0, abs(want)), f"a: Tunix's loss of step 1 {S['a']['loss'][0][1]} = {want}")
d_a = maxdiff(P["a"]["old_logp"], P["a"]["ref_logp"], M1)
need(d_a <= 1e-5, f"a: iteration 1's old pass = its reference (max |diff| {d_a}, bitwise {S['a']['grpo']['passes']['old_is_ref']})")
need(all(math.isfinite(s["loss"]) for s in S["a"]["grpo"]["steps"] + S["c"]["grpo"]["steps"]), "a, c: every loss finite")
# (b)
b1 = maxdiff(P["c"]["ref_logp"], P["f"]["ref_logp"], M2)
b2 = maxdiff(P["c"]["old_logp"], P["g"]["old_logp"], M2)
b3 = maxdiff(P["c"]["old_logp"], P["c"]["ref_logp"], M2)
b4 = maxdiff(P["g"]["ref_logp"], P["f"]["ref_logp"], M2)
need(b1 <= 1e-5, f"b: c's reference pass (a LoRA restored) = the base model's (f) (max |diff| {b1})")
need(b4 <= 1e-5, f"b: g's reference pass (a LoRA loaded) = the base model's (f) (max |diff| {b4})")
need(b2 <= 1e-5, f"b: c's old pass (restored) = the model with the loaded adapter (g) (max |diff| {b2})")
need(b3 > 1e-4, f"b: c's old pass is not the base model (max |old - ref| {b3})")
need((S["c"].get("resume") or {}).get("step") == 4 and set(S["c"]["resume"]["counters"].values()) == {4},
     f"c: Tunix restored step 4 with the optimizer's counters {S['c'].get('resume')}")
# (c)
with np.load("$P/c/adapter.npz") as zc, np.load("$P/e/adapter.npz") as ze:
    same = set(zc.files) == set(ze.files)
    ad = max(float(np.abs(zc[k] - ze[k]).max()) for k in zc.files) if same else math.inf
    bitwise = same and all(np.array_equal(zc[k], ze[k]) for k in zc.files)
    moved = max(float(np.abs(zc[k]).max()) for k in zc.files if k.endswith("_lora_b"))
need(ad <= 1e-5 and moved > 0, f"c: the adapter after step 8 = the uninterrupted run's (max |diff| {ad}, bitwise {bitwise}, max |B| {moved})")
num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
pairs = [(f"step {x['step']} {k}", x[k], y[k]) for x, y in zip(S["a"]["grpo"]["steps"] + S["c"]["grpo"]["steps"],
                                                              S["e"]["grpo"]["steps"]) for k in x if num(x[k])]
steps_e = [s["step"] for s in S["e"]["grpo"]["steps"]]
diff = max(abs(x - y) / max(1.0, abs(x)) for _, x, y in pairs)
print("C_STEPS", json.dumps({"values": len(pairs), "max_rel_diff": diff, "steps_e": steps_e,
                             "steps_ac": [s["step"] for s in S["a"]["grpo"]["steps"] + S["c"]["grpo"]["steps"]]}), flush=True)
need(steps_e == list(range(1, 9)) and [s["step"] for s in S["c"]["grpo"]["steps"]] == [5, 6, 7, 8] and diff <= 1e-5,
     f"c: a's steps 1-4 and c's 5-8 = the uninterrupted run's (max relative diff {diff})")
need(S["e"]["grpo"]["passes"]["passes_in"] == "$P/e-in/grpo-passes.npz", "e: the passes read in, none computed")
# (d)
dd = S["d"]
d1 = dd["grpo"]["steps"][0]
want2 = -float(A2["adv"][:4][M2[:4]].mean())
r1 = maxdiff(P["d"]["ref_logp"], P["g"]["old_logp"][:8], M2[:8])
r2 = maxdiff(P["d"]["ref_logp"], P["f"]["ref_logp"][:8], M2[:8])
need((dd["base_adapter"] or {}).get("modules", 0) > 0, f"d: the adapter merged {dd['base_adapter']}")
need(r1 <= 1e-4, f"d: the reference is the merged model = the adapter as LoRA (g's old) (max |diff| {r1})")
need(r2 > 1e-4, f"d: the reference is not the base model (max |diff| {r2})")
need(d1["log_ratio_abs_max"] <= 1e-5 and abs(d1["kl"]) <= 1e-8, f"d: step 1 rho = 1, KL = 0 {d1}")
need(abs(d1["loss"] - want2) <= 1e-4 * max(1.0, abs(want2)), f"d: step 1 loss {d1['loss']} = -token mean of A {want2}")
print("CPU_PROOF7_OK" if not bad else f"CPU_PROOF7_FAILED {bad}", flush=True)
raise SystemExit(1 if bad else 0)
EOF
echo "$(date +%T) done"
