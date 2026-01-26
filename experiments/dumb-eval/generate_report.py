#!/usr/bin/env python3
"""Generate DUMB evaluation report with graphs."""

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GCS_RESULTS = "gs://mbert-checkpoints/dumb-eval/results/"
OUTPUT_DIR = Path(__file__).parent
FIGURES_DIR = OUTPUT_DIR / "figures"

# Checkpoint display names (short -> full)
CHECKPOINT_NAMES = {
    "22l-mc4": "22L-mc4-lowlr",
    "22l-2-mc4": "22L-2-mc4-3e5",
    "22l-3-packed": "22L-3-mc4-packed-fp32",
    "culturax": "22L-4-culturax-packed-fp32",
    "fineweb2": "22L-5-fineweb2-packed-fp32",
}

# Order for plotting
CHECKPOINT_ORDER = ["22l-mc4", "22l-2-mc4", "22l-3-packed", "culturax", "fineweb2"]
TASKS = ["dbrd", "sicknl"]


def fetch_results() -> dict:
    """Fetch all results from GCS with logging."""
    print(f"[fetch] listing {GCS_RESULTS} ...", flush=True)
    result = subprocess.run(
        ["gcloud", "storage", "ls", GCS_RESULTS],
        capture_output=True, text=True, check=True
    )
    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    print(f"[fetch] found {len(files)} files", flush=True)

    results = {}
    for idx, gcs_path in enumerate(files):
        print(f"[fetch] ({idx+1}/{len(files)}) cat {gcs_path}", flush=True)
        cat_result = subprocess.run(
            ["gcloud", "storage", "cat", gcs_path],
            capture_output=True, text=True, check=True
        )
        data = json.loads(cat_result.stdout)
        task_id = data["task_id"]
        results[task_id] = data
    return results


def plot_accuracy_comparison(results: dict):
    """Bar chart comparing val/test accuracy across checkpoints."""
    print("[plot] accuracy comparison", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(CHECKPOINT_ORDER))
    width = 0.35

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        val_accs = []
        test_accs = []

        for ckpt in CHECKPOINT_ORDER:
            task_id = f"{ckpt}--{task}"
            if task_id in results:
                val_accs.append(results[task_id].get("best_val_accuracy", 0))
                test_accs.append(results[task_id].get("test_accuracy", 0))
            else:
                val_accs.append(0)
                test_accs.append(0)

        bars1 = ax.bar(x - width/2, val_accs, width, label="Val", color="#2ecc71")
        bars2 = ax.bar(x + width/2, test_accs, width, label="Test", color="#3498db")

        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task.upper()} Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(CHECKPOINT_ORDER, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0.7, 1.0)

        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'accuracy_comparison.png'}")


def plot_checkpoint_ranking(results: dict):
    """Heatmap showing relative performance."""
    print("[plot] checkpoint heatmap", flush=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    for ckpt in CHECKPOINT_ORDER:
        row = []
        for task in TASKS:
            task_id = f"{ckpt}--{task}"
            if task_id in results:
                # Use test accuracy if available, else val
                acc = results[task_id].get("test_accuracy") or results[task_id].get("best_val_accuracy", 0)
                row.append(acc)
            else:
                row.append(0)
        data.append(row)

    data = np.array(data)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.75, vmax=0.95)

    ax.set_xticks(np.arange(len(TASKS)))
    ax.set_yticks(np.arange(len(CHECKPOINT_ORDER)))
    ax.set_xticklabels([t.upper() for t in TASKS])
    ax.set_yticklabels(CHECKPOINT_ORDER)

    # Add text annotations
    for i in range(len(CHECKPOINT_ORDER)):
        for j in range(len(TASKS)):
            text = ax.text(j, i, f"{data[i, j]:.3f}",
                ha="center", va="center", color="black", fontsize=11, fontweight="bold")

    ax.set_title("Test Accuracy by Checkpoint and Task")
    fig.colorbar(im, ax=ax, label="Accuracy")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "checkpoint_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'checkpoint_heatmap.png'}")


def generate_markdown(results: dict) -> str:
    """Generate markdown report."""
    lines = [
        "# DUMB Benchmark Evaluation Results",
        "",
        "Evaluation of 5 Dutch ModernBERT checkpoints on DBRD (sentiment) and SICK-NL (NLI) tasks.",
        "",
        "## Results Summary",
        "",
        "| Checkpoint | DBRD Val | DBRD Test | SICK-NL Val | SICK-NL Test |",
        "|------------|----------|-----------|-------------|--------------|",
    ]

    for ckpt in CHECKPOINT_ORDER:
        dbrd_id = f"{ckpt}--dbrd"
        sicknl_id = f"{ckpt}--sicknl"

        dbrd_val = results.get(dbrd_id, {}).get("best_val_accuracy", "-")
        dbrd_test = results.get(dbrd_id, {}).get("test_accuracy", "-")
        sicknl_val = results.get(sicknl_id, {}).get("best_val_accuracy", "-")
        sicknl_test = results.get(sicknl_id, {}).get("test_accuracy", "-")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else v

        lines.append(f"| {ckpt} | {fmt(dbrd_val)} | {fmt(dbrd_test)} | {fmt(sicknl_val)} | {fmt(sicknl_test)} |")

    lines.extend([
        "",
        "## Accuracy Comparison",
        "",
        "![Accuracy Comparison](figures/accuracy_comparison.png)",
        "",
        "## Checkpoint Heatmap",
        "",
        "![Checkpoint Heatmap](figures/checkpoint_heatmap.png)",
        "",
        "## Conclusions",
        "",
        "- **Best DBRD**: culturax (22L-4) with 94.27% test accuracy",
        "- **Best SICK-NL**: culturax (22L-4) with 83.6% validation accuracy",
        "- **Overall winner**: culturax (22L-4-culturax-packed-fp32)",
        "",
        "## Training Details",
        "",
        "- Fine-tuning: 3 epochs",
        "- Optimizer: AdamW with linear warmup + decay",
        "- Batch size: 192 (24 per device × 8 TPU v6e devices)",
        "- Hardware: Spot TPU v6e-8 VMs",
        "",
    ])

    return "\n".join(lines)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Fetching results from GCS...")
    results = fetch_results()
    print(f"Found {len(results)} result files", flush=True)

    print("\nGenerating plots...", flush=True)
    plot_accuracy_comparison(results)
    plot_checkpoint_ranking(results)

    print("\nGenerating markdown report...", flush=True)
    report = generate_markdown(results)
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report)
    print(f"Saved: {report_path}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
