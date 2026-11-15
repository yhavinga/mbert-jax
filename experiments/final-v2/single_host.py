#!/usr/bin/env python3
"""Run a script that imports scripts/train_minimal.py on one host of a multi-host TPU slice, as one process.

train_minimal.py calls jax.distributed.initialize() at import. On one worker of a v4-32 that auto-detects the slice
from the TPU metadata and waits for the other three hosts (measured on mbert-mix w0, 15 Nov 2026: no answer in 240 s).
Here the runtime is initialized first for exactly one process on this host, so train_minimal's import leaves it as it
is; the TPU environment (TPU_PROCESS_BOUNDS=1,1,1 ..., as eval_v2.sh sets it) keeps the 4 chips of this host.

    python experiments/final-v2/single_host.py scripts/eval_windows.py --checkpoint ... [its arguments]
"""
import os
import runpy
import sys

import jax

if len(sys.argv) < 2:
    sys.exit("usage: single_host.py <script.py> [arguments of the script]")
port = os.environ.get("SINGLE_HOST_PORT", "8492")
jax.distributed.initialize(coordinator_address=f"localhost:{port}", num_processes=1, process_id=0)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
