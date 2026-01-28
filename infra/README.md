# Spot TPU VM Infrastructure

Orchestrates DUMB benchmark evaluation across spot TPU v6e VMs on GCP.

## Files

```
infra/
├── spot/
│   ├── config.py       # Constants: GCS paths, TPU config, checkpoints, tasks
│   ├── orchestrator.py # Main CLI: init, run, status, cleanup
│   ├── state.py        # GCS-backed task state (JSON files)
│   ├── task_runner.py  # Runs on VM: install deps, train, heartbeat
│   └── vm_ops.py       # gcloud wrapper: create/delete/ssh TPU VMs
```

## State Management

All state lives in GCS at `gs://mbert-checkpoints/dumb-eval/`:

```
dumb-eval/
├── state/          # Task state JSON files
├── checkpoints/    # Flax params.pkl + config.json per model
├── runs/           # Training outputs per task
├── results/        # Final results.json per task
└── code/           # Code tarball deployed to VMs
```

## How It Works

1. **Orchestrator** (your machine) manages VM lifecycle
2. **Task runner** (on VM) does the actual training
3. **State files** in GCS sync status between them
4. **Heartbeats** detect preemptions (VM gone + stale heartbeat = preempted)

Flow:
```
pending → provisioning → running → completed
                ↓            ↓
              failed     preempted → pending (retry)
```

## Running an Evaluation

### 1. Prepare Checkpoints

Convert Orbax checkpoint to Flax format:

```bash
python scripts/convert_gcs_checkpoint.py \
  --checkpoint path/to/orbax/checkpoint \
  --output experiments/checkpoints/flax/my-checkpoint \
  --num-layers 22 --hidden-size 1024 --vocab-size 32128 \
  --tokenizer yhavinga/dutch-llama-tokenizer
```

Upload to GCS:
```bash
gcloud storage cp experiments/checkpoints/flax/my-checkpoint/params.pkl \
  gs://mbert-checkpoints/dumb-eval/checkpoints/MY-CHECKPOINT-NAME/
gcloud storage cp experiments/checkpoints/flax/my-checkpoint/config.json \
  gs://mbert-checkpoints/dumb-eval/checkpoints/MY-CHECKPOINT-NAME/
```

### 2. Configure Tasks

Edit `config.py`:

```python
CHECKPOINTS = [
    ("MY-CHECKPOINT-NAME", "my-short-name"),  # (GCS folder, task ID prefix)
    ...
]

TASKS_PHASE1 = ["dbrd", "sicknl"]  # Tasks to run
MAX_CONCURRENT_VMS = 4             # Parallel VMs
```

### 3. Initialize Task Matrix

```bash
python -m infra.spot.orchestrator init
```

Creates state files for all checkpoint×task combinations.

### 4. Run Orchestrator

```bash
python -m infra.spot.orchestrator run
```

This loops until all tasks complete:
- Launches pending tasks (up to MAX_CONCURRENT)
- Deploys code tarball to each VM
- Detects preemptions via heartbeat timeout
- Auto-retries preempted tasks
- Cleans up finished VMs

### 5. Monitor Progress

```bash
python -m infra.spot.orchestrator status -v
```

Or check GCS directly:
```bash
gcloud storage ls gs://mbert-checkpoints/dumb-eval/state/
gcloud storage cat gs://mbert-checkpoints/dumb-eval/state/my-short-name--dbrd.json
```

### 6. Collect Results

Results auto-upload to `gs://mbert-checkpoints/dumb-eval/results/`.

Generate report:
```bash
python experiments/dumb-eval/generate_report.py
```

## CLI Commands

```bash
# Initialize task matrix
python -m infra.spot.orchestrator init

# Show status (add -v for details)
python -m infra.spot.orchestrator status -v

# Launch pending tasks (one-shot)
python -m infra.spot.orchestrator launch

# Resume preempted tasks (one-shot)
python -m infra.spot.orchestrator resume

# Continuous loop (recommended)
python -m infra.spot.orchestrator run

# Show completed results
python -m infra.spot.orchestrator results

# Delete all VMs (add -f to skip confirmation)
python -m infra.spot.orchestrator cleanup -f

# Test VM lifecycle
python -m infra.spot.orchestrator test
```

## Preemption Handling

Spot VMs get preempted. The system handles this:

1. Heartbeat thread runs every 60s during training
2. Orchestrator checks: VM gone + heartbeat > 5min stale = preempted
3. Preempted tasks return to pending queue
4. VM creation retries 3x with backoff on capacity failures

## Adding New Tasks

1. Add data loader in `scripts/dumb/data.py`
2. Add task to `TASKS_PHASE1` (or PHASE2/3) in `config.py`
3. Run `init` to create new state files
4. Run orchestrator

## Troubleshooting

**VMs stuck in provisioning**: Orchestrator restart left tasks in provisioning state. `resume_provisioning()` handles this on next run.

**HBM OOM**: Reduce `per_device_batch_size` in `task_runner.py` for that task.

**VM creation keeps failing**: Spot capacity exhausted. Wait and retry, or try different zone in `config.py`.

**Check VM logs**:
```bash
gcloud compute tpus tpu-vm ssh dumb-my-short-name-dbrd \
  --zone=europe-west4-a \
  --command="tail -100 /tmp/task_runner.log"
```
