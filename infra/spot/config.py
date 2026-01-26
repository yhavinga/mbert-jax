"""Configuration constants for spot TPU VM orchestration."""

# GCS paths
GCS_BUCKET = "mbert-checkpoints"
GCS_PREFIX = "dumb-eval"
GCS_STATE_DIR = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/state"
GCS_RUNS_DIR = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/runs"
GCS_RESULTS_DIR = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/results"

# TPU VM configuration
PROJECT = "t5-dutch-english"
ZONE = "europe-west4-a"
TPU_TYPE = "v6e-8"
RUNTIME_VERSION = "v2-alpha-tpuv6e"  # Trillium-specific runtime

# Orchestration settings
MAX_CONCURRENT_VMS = 4
HEARTBEAT_INTERVAL = 60  # seconds
HEARTBEAT_TIMEOUT = 300  # 5 minutes = presumed dead
POLL_INTERVAL = 60  # seconds between orchestrator loops

# Checkpoint sources (Flax format)
CHECKPOINTS = [
    ("22L-mc4-lowlr", "22l-mc4"),
    ("22L-2-mc4-3e5", "22l-2-mc4"),
    ("22L-3-mc4-packed-fp32", "22l-3-packed"),
    ("22L-4-culturax-packed-fp32", "culturax"),
    ("22L-5-fineweb2-packed-fp32", "fineweb2"),
]

# DUMB tasks
TASKS_PHASE1 = ["dbrd", "sicknl"]  # Only implemented loaders
TASKS_PHASE2 = ["wicnl", "dpr", "copanl", "squadnl"]  # Need loaders
TASKS_PHASE3 = ["lassy", "sonar", "dalc"]  # After TST data acquired

# Remote directory on TPU VMs
REMOTE_DIR = "~/mbert-jax"

# Local checkpoint directory (relative to project root)
LOCAL_CHECKPOINT_DIR = "experiments/checkpoints/flax"
