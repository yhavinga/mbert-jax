#!/usr/bin/env python3
"""Orchestrator for spot TPU VM fine-tuning jobs.

Commands:
    init     - Initialize task matrix (create state files for all jobs)
    status   - Show status of all tasks
    launch   - Launch pending tasks (respects MAX_CONCURRENT)
    resume   - Resume preempted tasks
    run      - Continuous loop: launch + monitor + resume
    results  - Collect results from completed tasks
    cleanup  - Delete all spot VMs
"""

import argparse
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import state as state_ops
from . import vm_ops
from .config import (
    CHECKPOINTS,
    GCS_RUNS_DIR,
    HEARTBEAT_TIMEOUT,
    MAX_CONCURRENT_VMS,
    POLL_INTERVAL,
    REMOTE_DIR,
    TASKS_PHASE1,
)


def make_task_id(ckpt_short: str, task: str) -> str:
    """Create a task ID from checkpoint short name and task."""
    return f"{ckpt_short}--{task}"


def make_vm_name(ckpt_short: str, task: str) -> str:
    """Create a VM name from checkpoint short name and task."""
    return f"dumb-{ckpt_short}-{task}"


def parse_task_id(task_id: str) -> tuple[str, str]:
    """Parse task_id into (ckpt_short, task)."""
    parts = task_id.split("--")
    return parts[0], parts[1]


def cmd_init(args):
    """Initialize task matrix - create state files for all pending jobs."""
    print("Initializing task matrix...")

    existing = {s["task_id"] for s in state_ops.list_states()}
    created = 0
    skipped = 0

    for ckpt_name, ckpt_short in CHECKPOINTS:
        for task in TASKS_PHASE1:
            task_id = make_task_id(ckpt_short, task)
            vm_name = make_vm_name(ckpt_short, task)

            if task_id in existing:
                print(f"  [skip] {task_id} (already exists)")
                skipped += 1
                continue

            initial_state = state_ops.create_initial_state(
                task_id=task_id,
                checkpoint=ckpt_name,
                task=task,
                vm_name=vm_name,
            )
            state_ops.put_state(task_id, initial_state)
            print(f"  [create] {task_id}")
            created += 1

    print(f"\nCreated {created} tasks, skipped {skipped} existing")


def cmd_status(args):
    """Show status of all tasks."""
    states = state_ops.list_states()
    if not states:
        print("No tasks found. Run 'init' first.")
        return

    # Get current VMs
    vms = {v["name"]: v["state"] for v in vm_ops.list_vms()}

    # Group by status
    by_status = {}
    for s in states:
        status = s["status"]
        by_status.setdefault(status, []).append(s)

    # Print summary
    print(f"{'Status':<12} {'Count':>5}")
    print("-" * 20)
    for status in ["completed", "running", "provisioning", "pending", "preempted", "failed"]:
        count = len(by_status.get(status, []))
        if count > 0:
            print(f"{status:<12} {count:>5}")

    print(f"\nTotal: {len(states)} tasks")
    print(f"Active VMs: {len(vms)}")

    # Detailed view if verbose
    if args.verbose:
        print("\n" + "=" * 60)
        for s in sorted(states, key=lambda x: (x["status"], x["task_id"])):
            vm_state = vms.get(s["vm_name"], "NONE")
            step_info = f"step {s['step']}" if s["step"] else "not started"
            score_info = f"score={s['best_dev_score']:.3f}" if s["best_dev_score"] else ""
            preempt_info = f"(preempted {s['preemption_count']}x)" if s["preemption_count"] else ""

            print(f"{s['task_id']:<25} {s['status']:<12} {vm_state:<10} {step_info} {score_info} {preempt_info}")


def get_running_count() -> int:
    """Count tasks that are running or provisioning."""
    states = state_ops.list_states()
    return sum(1 for s in states if s["status"] in ("running", "provisioning"))


def detect_preemptions():
    """Detect preempted VMs and update state."""
    now = datetime.now(timezone.utc)
    vms = {v["name"]: v["state"] for v in vm_ops.list_vms()}

    for s in state_ops.list_states():
        if s["status"] != "running":
            continue

        vm_name = s["vm_name"]
        vm_alive = vm_name in vms and vms[vm_name] == "READY"

        # Check heartbeat freshness
        heartbeat_fresh = False
        if s["last_heartbeat"]:
            last_hb = datetime.fromisoformat(s["last_heartbeat"])
            age = (now - last_hb).total_seconds()
            heartbeat_fresh = age < HEARTBEAT_TIMEOUT

        if not vm_alive and not heartbeat_fresh:
            print(f"[preempt] {s['task_id']} - VM gone, heartbeat stale")
            s["status"] = "preempted"
            s["preemption_count"] = s.get("preemption_count", 0) + 1
            state_ops.put_state(s["task_id"], s)


def deploy_code(vm_name: str) -> bool:
    """Deploy code to a VM by cloning from GCS tarball."""
    print("  Creating code tarball...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tarball_path = f.name

    # Create tarball with essential directories
    tar_cmd = [
        "tar", "-czf", tarball_path,
        "--exclude=*.pyc",
        "--exclude=__pycache__",
        "--exclude=.git",
        "--exclude=venv",
        "--exclude=checkpoints",
        "--exclude=experiments",
        "--exclude=artifacts",
        "--exclude=vendor/dumb",
        "--exclude=*.pkl",
        "--exclude=*.safetensors",
        "-C", str(Path(__file__).parent.parent.parent),
        "src", "scripts", "infra", "requirements.txt"
    ]
    result = subprocess.run(tar_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] Failed to create tarball: {result.stderr}")
        return False

    # Upload to GCS
    gcs_tarball = "gs://mbert-checkpoints/dumb-eval/code/mbert-jax.tar.gz"
    print("  Uploading to GCS...")
    result = subprocess.run(
        ["gcloud", "storage", "cp", tarball_path, gcs_tarball],
        capture_output=True, text=True
    )
    Path(tarball_path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"  [error] Failed to upload tarball: {result.stderr}")
        return False

    # Download and extract on VM
    print("  Extracting on VM...")
    setup_cmd = (
        f"mkdir -p {REMOTE_DIR} && "
        f"cd {REMOTE_DIR} && "
        f"gcloud storage cp {gcs_tarball} /tmp/code.tar.gz && "
        f"tar -xzf /tmp/code.tar.gz && "
        f"rm /tmp/code.tar.gz"
    )
    returncode, stdout, stderr = vm_ops.ssh_command(vm_name, setup_cmd)
    if returncode != 0:
        print(f"  [error] Failed to extract code: {stderr}")
        return False

    return True


def launch_task(s: dict) -> bool:
    """Launch a single task. Returns True if successful."""
    task_id = s["task_id"]
    vm_name = s["vm_name"]
    checkpoint = s["checkpoint"]
    task = s["task"]

    print(f"[launch] {task_id}")

    # Update state to provisioning
    s["status"] = "provisioning"
    s["started_at"] = datetime.now(timezone.utc).isoformat()
    state_ops.put_state(task_id, s)

    # Create VM
    print(f"  Creating VM {vm_name}...")
    if not vm_ops.create_vm(vm_name):
        print(f"  [error] Failed to create VM")
        s["status"] = "failed"
        s["error"] = "VM creation failed"
        state_ops.put_state(task_id, s)
        return False

    # Wait for VM to be ready
    print(f"  Waiting for VM to be ready...")
    for _ in range(30):  # 5 minutes max
        if vm_ops.vm_exists(vm_name):
            break
        time.sleep(10)
    else:
        print(f"  [error] VM not ready after 5 minutes")
        s["status"] = "failed"
        s["error"] = "VM not ready"
        state_ops.put_state(task_id, s)
        return False

    # Deploy code to VM
    print(f"  Deploying code...")
    if not deploy_code(vm_name):
        s["status"] = "failed"
        s["error"] = "Code deployment failed"
        state_ops.put_state(task_id, s)
        return False

    # Start task runner
    print(f"  Starting task runner...")
    gcs_checkpoint = f"gs://mbert-checkpoints/dumb-eval/checkpoints/{checkpoint}"
    gcs_output = f"{GCS_RUNS_DIR}/{task_id}"

    # Run task_runner in background via nohup
    # Need PYTHONPATH for src imports, and use python3 (system python with JAX)
    cmd = (
        f"cd {REMOTE_DIR} && "
        f"export PYTHONPATH=$PWD/src:$PYTHONPATH && "
        f"nohup python3 -m infra.spot.task_runner "
        f"--task-id {task_id} "
        f"--checkpoint {gcs_checkpoint} "
        f"--task {task} "
        f"--output {gcs_output} "
        f"--skip-install "
        f"> /tmp/task_runner.log 2>&1 &"
    )
    returncode, stdout, stderr = vm_ops.ssh_command(vm_name, cmd)
    if returncode != 0:
        print(f"  [error] Failed to start task runner: {stderr}")
        s["status"] = "failed"
        s["error"] = f"Task runner start failed: {stderr}"
        state_ops.put_state(task_id, s)
        return False

    # Update state to running
    s["status"] = "running"
    s["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    state_ops.put_state(task_id, s)

    print(f"  [ok] Task running")
    return True


def cmd_launch(args):
    """Launch pending tasks up to MAX_CONCURRENT."""
    running = get_running_count()
    available = MAX_CONCURRENT_VMS - running

    if available <= 0:
        print(f"Already at max concurrent VMs ({MAX_CONCURRENT_VMS})")
        return

    print(f"Launching up to {available} tasks...")

    # Get pending tasks
    pending = [s for s in state_ops.list_states() if s["status"] == "pending"]
    if not pending:
        print("No pending tasks")
        return

    launched = 0
    for s in pending[:available]:
        if launch_task(s):
            launched += 1

    print(f"\nLaunched {launched} tasks")


def cmd_resume(args):
    """Resume preempted tasks."""
    preempted = [s for s in state_ops.list_states() if s["status"] == "preempted"]
    if not preempted:
        print("No preempted tasks")
        return

    running = get_running_count()
    available = MAX_CONCURRENT_VMS - running

    print(f"Resuming up to {available} preempted tasks...")

    resumed = 0
    for s in preempted[:available]:
        # Reset to pending so launch picks it up
        s["status"] = "pending"
        state_ops.put_state(s["task_id"], s)
        if launch_task(s):
            resumed += 1

    print(f"\nResumed {resumed} tasks")


def cmd_run(args):
    """Continuous orchestration loop."""
    print("Starting orchestration loop (Ctrl+C to stop)...")

    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking...")

            # Detect preemptions
            detect_preemptions()

            # Resume preempted tasks
            preempted = [s for s in state_ops.list_states() if s["status"] == "preempted"]
            if preempted:
                running = get_running_count()
                available = MAX_CONCURRENT_VMS - running
                for s in preempted[:available]:
                    s["status"] = "pending"
                    state_ops.put_state(s["task_id"], s)
                    launch_task(s)

            # Launch pending tasks
            running = get_running_count()
            available = MAX_CONCURRENT_VMS - running
            if available > 0:
                pending = [s for s in state_ops.list_states() if s["status"] == "pending"]
                for s in pending[:available]:
                    launch_task(s)

            # Check if all done
            states = state_ops.list_states()
            not_done = [s for s in states if s["status"] not in ("completed", "failed")]
            if not not_done:
                print("\nAll tasks completed!")
                break

            # Status summary
            running = sum(1 for s in states if s["status"] == "running")
            pending = sum(1 for s in states if s["status"] == "pending")
            completed = sum(1 for s in states if s["status"] == "completed")
            print(f"  Running: {running}, Pending: {pending}, Completed: {completed}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped by user")


def cmd_results(args):
    """Collect results from completed tasks."""
    completed = [s for s in state_ops.list_states() if s["status"] == "completed"]
    if not completed:
        print("No completed tasks")
        return

    print(f"Results for {len(completed)} completed tasks:\n")
    print(f"{'Task ID':<25} {'Score':>8} {'Steps':>8} {'Preempts':>8}")
    print("-" * 55)

    for s in sorted(completed, key=lambda x: x["task_id"]):
        score = s.get("best_dev_score")
        score_str = f"{score:.4f}" if score else "N/A"
        print(f"{s['task_id']:<25} {score_str:>8} {s['step']:>8} {s['preemption_count']:>8}")


def cmd_cleanup(args):
    """Delete all spot VMs."""
    vms = vm_ops.list_vms()
    if not vms:
        print("No VMs to delete")
        return

    if not args.force:
        print(f"Will delete {len(vms)} VMs:")
        for v in vms:
            print(f"  - {v['name']} ({v['state']})")
        response = input("\nContinue? [y/N] ")
        if response.lower() != "y":
            print("Aborted")
            return

    for v in vms:
        print(f"Deleting {v['name']}...")
        vm_ops.delete_vm(v["name"])

    print(f"\nDeleted {len(vms)} VMs")


def cmd_test(args):
    """Test spot VM lifecycle: create, SSH, optional TPU check, delete."""
    vm_name = "dumb-test"
    success = True

    def cleanup():
        print(f"\n[cleanup] Deleting test VM {vm_name}...")
        if vm_ops.delete_vm(vm_name):
            print("[cleanup] Done")
        else:
            print("[cleanup] Failed (VM may not exist)")

    try:
        # Step 1: Create VM
        print(f"[1/5] Creating test VM {vm_name}...")
        if not vm_ops.create_vm(vm_name):
            print("[FAIL] VM creation failed")
            return False

        # Step 2: Wait for READY
        print("[2/5] Waiting for READY state...")
        for i in range(30):  # 5 minutes max
            state = vm_ops.vm_state(vm_name)
            if state == "READY":
                print(f"[OK] VM ready after ~{i * 10}s")
                break
            if state is None:
                print(f"[FAIL] VM disappeared during provisioning")
                return False
            print(f"  state={state}, waiting...")
            time.sleep(10)
        else:
            print("[FAIL] VM not ready after 5 minutes")
            cleanup()
            return False

        # Step 3: Test SSH
        print("[3/5] Testing SSH connectivity...")
        returncode, stdout, stderr = vm_ops.ssh_command(vm_name, "echo hello")
        if returncode != 0:
            print(f"[FAIL] SSH failed: {stderr}")
            cleanup()
            return False
        if "hello" not in stdout:
            print(f"[FAIL] SSH returned unexpected output: {stdout}")
            cleanup()
            return False
        print("[OK] SSH working")

        # Step 4: Test TPU (optional)
        if not args.skip_tpu:
            print("[4/5] Testing TPU devices...")
            returncode, stdout, stderr = vm_ops.ssh_command(
                vm_name,
                "python3 -c 'import jax; devs = jax.devices(); print(f\"Found {len(devs)} devices: {devs}\")'"
            )
            if returncode != 0:
                print(f"[WARN] TPU check failed: {stderr}")
                print("  (This may be expected if JAX is not installed)")
            else:
                print(f"[OK] {stdout.strip()}")
        else:
            print("[4/5] Skipping TPU check (--skip-tpu)")

        # Step 5: Cleanup
        print("[5/5] Deleting test VM...")
        if not vm_ops.delete_vm(vm_name):
            print("[WARN] Failed to delete VM")
            success = False
        else:
            print("[OK] VM deleted")

        if success:
            print("\nAll tests passed!")
        return success

    except KeyboardInterrupt:
        print("\n[interrupted]")
        cleanup()
        return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        cleanup()
        return False


def main():
    parser = argparse.ArgumentParser(description="Spot TPU VM orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    subparsers.add_parser("init", help="Initialize task matrix")

    # status
    status_parser = subparsers.add_parser("status", help="Show task status")
    status_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed status")

    # launch
    subparsers.add_parser("launch", help="Launch pending tasks")

    # resume
    subparsers.add_parser("resume", help="Resume preempted tasks")

    # run
    subparsers.add_parser("run", help="Continuous orchestration loop")

    # results
    subparsers.add_parser("results", help="Show results from completed tasks")

    # cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Delete all VMs")
    cleanup_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # test
    test_parser = subparsers.add_parser("test", help="Test spot VM lifecycle")
    test_parser.add_argument("--skip-tpu", action="store_true", help="Skip TPU device check")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "status": cmd_status,
        "launch": cmd_launch,
        "resume": cmd_resume,
        "run": cmd_run,
        "results": cmd_results,
        "cleanup": cmd_cleanup,
        "test": cmd_test,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
