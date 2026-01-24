# Dutch Model Benchmark (DUMB) Fine-tuning Evaluation Plan

Multi-session plan for evaluating Dutch ModernBERT models on the DUMB benchmark.

---

## 1. Current Checkpoint State

### Source Checkpoints (GCS → local)

| Name | Dataset | LR | Packing | Steps | Status |
|------|---------|-----|---------|-------|--------|
| `22L-mc4-lowlr` | mc4_nl_cleaned | 1e-5 | ✗ | 2M | ✓ All formats |
| `22L-2-mc4-3e5` | mc4_nl_cleaned | 3e-5 | ✗ | 2M | ✓ Flax only |
| `22L-3-mc4-packed-fp32` | mc4_nl_cleaned | 3e-5 | ✓ | 1.4M | ✓ Flax only |
| `22L-4-culturax-packed-fp32` | CulturaX_nl | 3e-5 | ✓ | 2M | ✓ Flax only |

### Checkpoint Formats & Locations

```
experiments/checkpoints/
├── orbax/                        # Raw Orbax (from TPU training)
│   ├── 22L-mc4-lowlr/2000000/
│   └── ...
├── flax/                         # Flax pickle (params.pkl + config.json)
│   ├── 22L-mc4-lowlr/
│   ├── 22L-2-mc4-3e5/
│   ├── 22L-3-mc4-packed-fp32/
│   └── 22L-4-culturax-packed-fp32/
└── hf/                           # HuggingFace PyTorch (not needed for JAX approach)
    └── 22L-mc4-lowlr/
```

---

## 2. DUMB Benchmark Overview

**Goal:** Evaluate Dutch MLM quality via fine-tuning on 9 downstream tasks.
**Paper:** de Vries, Wieling, Nissim (EMNLP 2023)
**Repo:** https://github.com/wietsedv/dumb

### Tasks

| Task ID | Task | Type | Train Size | BERTje Baseline | Metric | Available |
|---------|------|------|------------|-----------------|--------|-----------|
| `dbrd` | Sentiment | Seq Class (2) | ~20K | 93.3% | Accuracy | ✅ Auto |
| `sicknl-nli` | NLI | Seq Class (3) | ~4.5K | 85.2% | Accuracy | ✅ Auto |
| `wicnl` | Word Sense | Span Binary | ~7K | 65.9% | Accuracy | ✅ Auto |
| `dpr` | Pronoun Resolution | Span Binary | ~786 | 65.8% | Accuracy | ✅ Auto |
| `copanl` | Causal Reasoning | Multiple Choice | 400 | 62.0% | Accuracy | ✅ Auto |
| `squadnl` | Question Answering | Extractive QA | ~large | 50.9 F1 | F1/EM | ✅ Auto |
| `lassy-pos` | POS Tagging | Token | ~65K sent | 96.3% | Accuracy | 🔒 TST |
| `sonar-ne` | NER (6-class) | Token | ~large | 88.3% | F1 | 🔒 TST |
| `dalc` | Abusive Language | Seq Class (3) | ~6.8K | 58.8% | Macro F1 | 🔒 Request |

### Evaluation Metric: RER (Relative Error Reduction)

```
RER = (error_BERTje - error_model) / error_BERTje × 100
```

- Positive RER = better than BERTje
- Normalizes difficulty across tasks (beating 96.3% POS by 0.5% is impressive)

---

## 3. Dataset Acquisition

### Phase 1: Freely Available (6 tasks) ✅

| Task | Source | License |
|------|--------|---------|
| DBRD | hebban.nl reviews | MIT |
| SICK-NL | Wijnholds & Moortgat | MIT |
| WiC-NL | DutchSemCor | CC BY 3.0 |
| COPA-NL | Translated COPA | BSD-2-Clause |
| SQuAD-NL | Translated SQuAD | CC-BY-SA 4.0 |
| DPR | SemEval-2010 derived | Custom |

```bash
# Clone DUMB repo and build Phase 1 datasets
git clone https://github.com/wietsedv/dumb.git
cd dumb
python build.py  # No flags needed for freely available datasets
```

### Phase 2: Restricted Datasets (3 tasks) 🔒

| Task | Source | Action | Why It Matters |
|------|--------|--------|----------------|
| **Lassy Small** | TST-Centrale | Register → download v6.0 | Structural task: tests Dutch morphology (218 POS tags) |
| **SoNaR** | TST-Centrale | Register → download v1.2.1 | Structural task: tests compound words, entity boundaries |
| **DALC** | GitHub (tommasoc80) | Email request for v2.0 CSVs | Domain shift: informal Twitter Dutch |

> **Note:** Lassy and SoNaR are where native Dutch pretraining should shine. Add these after Phase 1 results are in.

```bash
# Build with restricted datasets (after acquisition)
python build.py \
  --lassy=/path/to/lassy-small \
  --sonar=/path/to/sonar_corpus \
  --dalc=/path/to/dalc_v2_csvs
```

---

## 4. Approach: TPU v6e + JAX/Flax (Native)

**Decision:** Use JAX/Flax natively. Matches training infrastructure, avoids HF conversion, proven stable.

### Why This Approach

| Factor | Benefit |
|--------|---------|
| Same framework as pretraining | No conversion errors, identical numerics |
| Flax checkpoints ready | All 4 checkpoints already in Flax format |
| TPU-native | No PyTorch/XLA translation layer |
| Fast iteration | 2hr spot preemption is fine for fine-tuning |

### Infrastructure

```
infra/spot/                       # Spot VM orchestration (runs on Contabo)
├── orchestrator.py               # Main control loop
├── task_runner.py                # Fine-tuning on VM
├── vm_ops.py                     # gcloud wrappers
├── state.py                      # GCS state operations
└── config.py                     # Constants

scripts/dumb/                     # Fine-tuning logic (called by task_runner)
├── finetune.py                   # Flax training loop
├── heads.py                      # Classification heads
├── data.py                       # DUMB dataset loading
└── eval.py                       # RER computation
```

### Spot VM Orchestration

GCS is the source of truth. State files at `gs://mbert-checkpoints/dumb-eval/state/`.

```bash
# On Contabo VM
python infra/spot/orchestrator.py init      # Create task matrix
python infra/spot/orchestrator.py status    # Check all tasks
python infra/spot/orchestrator.py run       # Continuous: launch + monitor + resume
```

See detailed plan: `.claude/plans/synthetic-snacking-lecun.md`

### Fine-Tuning Heads to Implement

| Head Type | Tasks | Complexity |
|-----------|-------|------------|
| SequenceClassification | DBRD, SICK-NL, DALC | Simple: pool + linear |
| SpanClassification | WiC-NL, DPR | Medium: mark spans + classify |
| MultipleChoice | COPA-NL | Simple: pool each choice + score |
| TokenClassification | Lassy-POS, SoNaR-NER | Simple: linear per token |
| QuestionAnswering | SQuAD-NL | Medium: start/end logits |

---

## 5. Fine-Tuning Strategy

### Standard Hyperparameters (for DUMB leaderboard comparison)

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 (default), 1e-5 (low-resource) |
| Batch Size | 32 |
| Epochs | 3-5 (high-resource), 10-20 (low-resource) |
| Selection | Best dev checkpoint → test eval |

### Low-Resource Task Strategy

> For COPA-NL (400 examples) and DPR (~800 examples), use **intermediate fine-tuning**:
> 1. Fine-tune on SICK-NL first (4.5K examples, same reasoning domain)
> 2. Continue fine-tuning on COPA-NL/DPR

### Task Execution Order

| Priority | Tasks | Rationale |
|----------|-------|-----------|
| 1 | DBRD | High-resource, simple head, good smoke test |
| 2 | SICK-NL | Medium-resource, needed for intermediate fine-tuning |
| 3 | WiC-NL, DPR, COPA-NL | Reasoning tasks (use SICK-NL checkpoint for low-resource) |
| 4 | SQuAD-NL | QA head implementation |
| 5 | Lassy-POS, SoNaR-NER, DALC | After restricted data acquired |

---

## 6. Directory Structure

```
mbert-jax/
├── scripts/
│   ├── convert_gcs_checkpoint.py    # ✓ exists
│   ├── convert_flax_to_hf.py        # ✓ exists
│   ├── eval_pppl_torch.py           # ✓ exists
│   └── dumb/                        # NEW: DUMB fine-tuning
│       ├── finetune.py
│       ├── heads.py
│       ├── data.py
│       └── eval.py
├── experiments/
│   ├── checkpoints/                 # All checkpoint formats (gitignored)
│   └── dumb/                        # Fine-tuned models (gitignored)
│       └── {checkpoint}/{task}/
└── results/                         # Evaluation results (committed)
    └── dumb/
        ├── phase1_results.json      # 6-task results
        ├── phase2_results.json      # 9-task results (after TST data)
        └── report.md
```

---

## 7. Session Plan

### Session 1: Spot Orchestration Infrastructure

- [ ] Create `infra/spot/config.py` - GCS paths, zone, VM specs
- [ ] Create `infra/spot/state.py` - GCS read/write state files
- [ ] Create `infra/spot/vm_ops.py` - gcloud create/delete/list wrappers
- [ ] Create `infra/spot/orchestrator.py` - init, status, launch commands
- [ ] Test: `orchestrator.py init` generates task matrix
- [ ] Test: `orchestrator.py status` shows all pending tasks

### Session 2: Fine-Tuning Core

- [ ] Clone DUMB repo, run `build.py` for Phase 1 datasets
- [ ] Create `scripts/dumb/heads.py` - SequenceClassification head
- [ ] Create `scripts/dumb/finetune.py` - basic Flax training loop
- [ ] Create `scripts/dumb/data.py` - load DUMB JSONL datasets
- [ ] Create `infra/spot/task_runner.py` - wrapper with heartbeat

### Session 3: Integration Test

- [ ] Deploy to Contabo VM
- [ ] Test: launch one task (22L-mc4-lowlr on DBRD)
- [ ] Verify: heartbeat appears in GCS state file
- [ ] Verify: checkpoint saved to GCS
- [ ] Test: manually kill VM, verify preemption detected

### Session 4: Auto-Resume + Phase 1

- [ ] Implement preemption detection in orchestrator
- [ ] Implement auto-resume logic
- [ ] Run `orchestrator.py run` for Phase 1 (6 tasks × 4 checkpoints)
- [ ] Monitor until all 24 jobs complete
- [ ] Compute RER scores, generate `results/dumb/phase1_results.json`

### Session 5: Remaining Heads + Phase 2

- [ ] Implement SpanClassification, MultipleChoice, TokenClassification, QA heads
- [ ] Acquire Lassy, SoNaR, DALC datasets (parallel to running jobs)
- [ ] Run Phase 2 tasks after data acquired
- [ ] Generate final report with 9-task RER

---

## 8. Quick Reference Commands

### Orchestrator (run on Contabo VM)
```bash
# Initialize task matrix (36 jobs: 4 checkpoints × 9 tasks)
python infra/spot/orchestrator.py init

# Check status of all tasks
python infra/spot/orchestrator.py status

# Launch pending tasks (respects MAX_CONCURRENT=2)
python infra/spot/orchestrator.py launch

# Continuous operation: launch + monitor + auto-resume
python infra/spot/orchestrator.py run

# Collect results from completed tasks
python infra/spot/orchestrator.py results

# Emergency: delete all spot VMs
python infra/spot/orchestrator.py cleanup
```

### GCS State Files
```bash
# List all state files
gcloud storage ls gs://mbert-checkpoints/dumb-eval/state/

# Check a specific task state
gcloud storage cat gs://mbert-checkpoints/dumb-eval/state/22l-mc4--dbrd.json

# List training checkpoints
gcloud storage ls gs://mbert-checkpoints/dumb-eval/runs/
```

### Manual VM Operations
```bash
# Create spot TPU VM
gcloud compute tpus tpu-vm create dumb-22l-mc4-dbrd \
  --zone=europe-west4-a \
  --accelerator-type=v6e-8 \
  --version=tpu-ubuntu2204-base \
  --spot

# SSH to VM
gcloud compute tpus tpu-vm ssh dumb-22l-mc4-dbrd --zone=europe-west4-a

# Delete VM
gcloud compute tpus tpu-vm delete dumb-22l-mc4-dbrd --zone=europe-west4-a --quiet
```

### Deploy Code to Contabo
```bash
rsync -avz --exclude='*.pkl' --exclude='checkpoints/' \
  . user@contabo:~/mbert-jax/
```
