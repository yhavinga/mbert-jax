#!/usr/bin/env python3
"""
Unified analysis tool for ModernBERT checkpoints.

This single tool provides all checkpoint analysis capabilities:
- Basic checkpoint information and statistics
- Stability analysis (parameter evolution)
- Spectral analysis (eigenvalue distributions)
- Comprehensive comparisons between checkpoints
- W&B metrics synchronization and visualization
- Database queries for best checkpoints

Usage:
    # Basic checkpoint info
    python analyze.py checkpoint experiments/red_1m/1000000

    # Compare checkpoints
    python analyze.py compare experiments/red_1m experiments/yellow_2m

    # Stability analysis
    python analyze.py stability experiments/red_1m experiments/yellow_2m

    # Spectral analysis
    python analyze.py spectral experiments/yellow_2m

    # W&B metrics
    python analyze.py wandb sync --run-id abc123
    python analyze.py wandb plot --run-id abc123

    # Database queries
    python analyze.py best --top 5

    # Generate comprehensive report
    python analyze.py report experiments/yellow_2m --output report.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import pickle
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from functools import lru_cache

# Import analysis modules
from mmbjax.analysis import stability, spectral, comparison
from mmbjax.analysis.stability import flatten_params as flatten_params_stability

# Try importing W&B utilities
try:
    from mmbjax.utils.wandb_utils import WandbCheckpointManager, HAS_WANDB
except ImportError:
    HAS_WANDB = False
    WandbCheckpointManager = None


# Cache for loaded checkpoints (avoid reloading)
@lru_cache(maxsize=8)
def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load checkpoint from pickle file with caching."""
    path = Path(path)

    # Handle different checkpoint formats
    if path.is_dir():
        # Look for params.pkl in directory
        pkl_path = path / "params.pkl"
        if pkl_path.exists():
            path = pkl_path
        else:
            # Try numbered subdirectories
            for subdir in sorted(path.glob("*"), reverse=True):
                if subdir.is_dir() and subdir.name.isdigit():
                    pkl_path = subdir / "params.pkl"
                    if pkl_path.exists():
                        path = pkl_path
                        break

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading checkpoint from {path}...")
    with open(path, 'rb') as f:
        params = pickle.load(f)

    print(f"✓ Loaded {sum(p.size for p in flatten_params_stability(params).values()):,} parameters")
    return params


def cmd_checkpoint(args):
    """Display basic checkpoint information."""
    params = load_checkpoint(args.path)
    flat_params = flatten_params_stability(params)

    print("\n" + "=" * 70)
    print(f"CHECKPOINT: {Path(args.path).name}")
    print("=" * 70)

    # Overall statistics
    total_params = sum(p.size for p in flat_params.values())
    total_size_mb = sum(p.nbytes for p in flat_params.values()) / 1024 / 1024

    print(f"\nOVERALL:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Number of layers: {len(flat_params)}")

    # Layer type breakdown
    attention_params = sum(p.size for k, p in flat_params.items() if 'attention' in k)
    mlp_params = sum(p.size for k, p in flat_params.items() if 'mlp' in k or 'intermediate' in k)
    embedding_params = sum(p.size for k, p in flat_params.items() if 'embedding' in k)
    other_params = total_params - attention_params - mlp_params - embedding_params

    print(f"\nPARAMETER DISTRIBUTION:")
    print(f"  Attention: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
    print(f"  MLP: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
    print(f"  Embeddings: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
    print(f"  Other: {other_params:,} ({other_params/total_params*100:.1f}%)")

    # Compute basic statistics
    if args.stats:
        print(f"\nCOMPUTING STATISTICS...")
        stats = comparison.compute_comprehensive_stats(flat_params)

        # Aggregate statistics
        all_means = [s['mean'] for s in stats.values()]
        all_stds = [s['std'] for s in stats.values()]
        all_l2_norms = [s['l2_norm'] for s in stats.values()]

        print(f"\nAGGREGATE STATISTICS:")
        print(f"  Mean parameter value: {np.mean(all_means):.6f}")
        print(f"  Mean std deviation: {np.mean(all_stds):.6f}")
        print(f"  Mean L2 norm: {np.mean(all_l2_norms):.2f}")
        print(f"  Max L2 norm: {np.max(all_l2_norms):.2f}")

        # Find extreme layers
        max_norm_layer = max(stats.items(), key=lambda x: x[1]['l2_norm'])
        print(f"\n  Largest layer: {max_norm_layer[0][:50]}")
        print(f"    L2 norm: {max_norm_layer[1]['l2_norm']:.2f}")


def cmd_compare(args):
    """Compare multiple checkpoints."""
    if len(args.checkpoints) < 2:
        print("Error: Need at least 2 checkpoints to compare")
        return

    print(f"\nComparing {len(args.checkpoints)} checkpoints...")

    # Load all checkpoints
    all_params = []
    all_stats = []
    names = []

    for cp_path in args.checkpoints:
        try:
            params = load_checkpoint(cp_path)
            flat_params = flatten_params_stability(params)
            stats = comparison.compute_comprehensive_stats(flat_params)
            all_params.append(params)
            all_stats.append(stats)
            names.append(Path(cp_path).name)
        except Exception as e:
            print(f"Error loading {cp_path}: {e}")
            continue

    if len(all_stats) < 2:
        print("Error: Could not load enough checkpoints for comparison")
        return

    # Compare each pair
    for i in range(len(all_stats)):
        for j in range(i + 1, len(all_stats)):
            print(f"\n{'=' * 70}")
            summary = comparison.generate_comparison_summary(
                all_stats[i], all_stats[j],
                names[i], names[j]
            )
            print(summary)


def cmd_stability(args):
    """Analyze stability between checkpoints."""
    if len(args.checkpoints) != 2:
        print("Error: Stability analysis requires exactly 2 checkpoints")
        return

    print(f"\nAnalyzing stability: {args.checkpoints[0]} → {args.checkpoints[1]}")

    params1 = load_checkpoint(args.checkpoints[0])
    params2 = load_checkpoint(args.checkpoints[1])

    results = stability.analyze_stability(params1, params2)
    summary = stability.summarize_stability(results)

    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)

    # Overall statistics
    print("\nOVERALL:")
    overall = summary['overall']
    print(f"  Total layers analyzed: {overall['total_layers']}")
    print(f"  Mean update ratio: {overall['mean_update_ratio']:.4f}")
    print(f"  Max update ratio: {overall['max_update_ratio']:.4f}")
    print(f"  Mean L2 norm ratio: {overall['mean_l2_ratio']:.3f}")
    print(f"  Max L2 norm ratio: {overall['max_l2_ratio']:.3f}")
    print(f"  Mean angular distance: {overall['mean_angular_distance']:.1f}°")

    # Layer type breakdown
    for layer_type in ['attention', 'mlp', 'embedding', 'normalization']:
        if summary[layer_type]:
            print(f"\n{layer_type.upper()} LAYERS:")
            stats = summary[layer_type]
            print(f"  Number of layers: {stats['num_layers']}")
            print(f"  Mean update ratio: {stats['mean_update_ratio']:.4f}")
            print(f"  Mean L2 ratio: {stats['mean_l2_ratio']:.3f}")
            print(f"  Mean angular distance: {stats['mean_angular_distance']:.1f}°")

    # Warning for exploded layers
    if summary['exploded_layers']:
        print(f"\n⚠ EXPLODED LAYERS ({len(summary['exploded_layers'])}):")
        for layer in summary['exploded_layers'][:5]:
            print(f"  - {layer}")


def cmd_spectral(args):
    """Analyze spectral properties of checkpoint."""
    params = load_checkpoint(args.path)
    flat_params = flatten_params_stability(params)

    print("\nAnalyzing spectral properties...")
    stats = spectral.analyze_spectral_properties(flat_params)
    health = spectral.summarize_spectral_health(stats)

    print("\n" + "=" * 70)
    print("SPECTRAL ANALYSIS")
    print("=" * 70)

    print(f"\nOVERALL HEALTH SCORE: {health['health_score']:.1f}/100")

    print(f"\nCONDITIONING:")
    print(f"  Mean condition number: {health['mean_condition']:.2e}")
    print(f"  Max condition number: {health['max_condition']:.2e}")
    print(f"  Median condition number: {health['median_condition']:.2e}")

    print(f"\nSPECTRAL NORMS:")
    print(f"  Mean: {health['mean_spectral_norm']:.2f}")
    print(f"  Max: {health['max_spectral_norm']:.2f}")

    print(f"\nSTABLE RANK:")
    print(f"  Mean: {health['mean_stable_rank']:.1f}")

    # Warnings
    if health['ill_conditioned_layers']:
        print(f"\n⚠ ILL-CONDITIONED LAYERS ({len(health['ill_conditioned_layers'])}):")
        for layer in health['ill_conditioned_layers'][:5]:
            print(f"  - {layer}")

    if health['low_rank_layers']:
        print(f"\n⚠ LOW RANK LAYERS ({len(health['low_rank_layers'])}):")
        for layer in health['low_rank_layers'][:5]:
            print(f"  - {layer}")

    if health['exploded_layers']:
        print(f"\n⚠ EXPLODED LAYERS ({len(health['exploded_layers'])}):")
        for layer in health['exploded_layers'][:5]:
            print(f"  - {layer}")


def cmd_wandb(args):
    """W&B related commands."""
    if not HAS_WANDB:
        print("Error: wandb package not installed. Install with: pip install wandb")
        return

    if args.wandb_cmd == 'sync':
        # Sync W&B metrics to database
        if not args.run_id:
            print("Error: --run-id required for sync")
            return

        print(f"Syncing W&B run {args.run_id}...")
        manager = WandbCheckpointManager(cache_db=args.db)
        success = manager.sync_run_to_db(
            args.run_id,
            project=args.project,
            checkpoint_path=args.checkpoint
        )
        if success:
            print("✓ Sync complete")

    elif args.wandb_cmd == 'plot':
        # Plot training metrics
        if not args.run_id:
            print("Error: --run-id required for plot")
            return

        manager = WandbCheckpointManager(cache_db=args.db)
        run_data = manager.fetch_run_metrics(args.run_id, args.project)

        if not run_data or not run_data.get('history'):
            print("No metrics found")
            return

        history = run_data['history']
        metrics = args.metrics or ['loss']

        for metric in metrics:
            values = []
            for row in history:
                for key in [f'train/{metric}', metric]:
                    if key in row and row[key] is not None:
                        values.append(float(row[key]))
                        break

            if values:
                print(f"\n{metric.upper()} (last 50 steps):")
                print(plot_ascii_simple(values[-50:]))


def cmd_best(args):
    """Show best checkpoints from database."""
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT checkpoint_name, checkpoint_path, perplexity, accuracy, loss, timestamp
        FROM evaluations
        WHERE eval_type = 'perplexity' AND perplexity IS NOT NULL
        ORDER BY perplexity ASC
        LIMIT ?
    """, (args.top,))

    results = cursor.fetchall()
    conn.close()

    if not results:
        print("No evaluation results found")
        return

    print(f"\nTOP {args.top} CHECKPOINTS")
    print("=" * 70)

    for i, row in enumerate(results, 1):
        name = row[0] or Path(row[1]).name
        print(f"\n{i}. {name}")
        print(f"   Perplexity: {row[2]:.2f}")
        print(f"   Accuracy: {row[3]*100:.1f}%")
        print(f"   Loss: {row[4]:.4f}")
        print(f"   Date: {row[5]}")


def cmd_report(args):
    """Generate comprehensive analysis report."""
    params = load_checkpoint(args.path)
    flat_params = flatten_params_stability(params)

    print("\nGenerating comprehensive report...")

    # Collect all analyses
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"COMPREHENSIVE ANALYSIS REPORT")
    report_lines.append(f"Checkpoint: {Path(args.path).name}")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("=" * 70)

    # Basic statistics
    stats = comparison.compute_comprehensive_stats(flat_params)
    total_params = sum(p.size for p in flat_params.values())
    report_lines.append(f"\nTotal parameters: {total_params:,}")

    # Spectral analysis
    spectral_stats = spectral.analyze_spectral_properties(flat_params)
    health = spectral.summarize_spectral_health(spectral_stats)
    report_lines.append(f"\nSpectral health score: {health['health_score']:.1f}/100")
    report_lines.append(f"Mean condition number: {health['mean_condition']:.2e}")

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text("\n".join(report_lines))
        print(f"✓ Report saved to {output_path}")
    else:
        print("\n".join(report_lines))


def plot_ascii_simple(values: List[float], height: int = 10) -> str:
    """Simple ASCII plot for terminal display."""
    if not values:
        return "No data"

    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val if max_val > min_val else 1

    lines = []
    for h in range(height, 0, -1):
        y_val = min_val + (h - 1) * range_val / (height - 1)
        line = f"{y_val:7.2f} |"

        for i, val in enumerate(values):
            val_height = int((val - min_val) / range_val * (height - 1)) + 1
            if val_height == h:
                line += "●"
            elif val_height > h:
                line += "│"
            else:
                line += " "
        lines.append(line)

    lines.append(" " * 8 + "└" + "─" * len(values))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Unified analysis tool for ModernBERT checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMMANDS:

  checkpoint PATH [--stats]     Basic checkpoint info
  compare CP1 CP2 [CP3...]      Compare multiple checkpoints
  stability CP1 CP2             Parameter evolution analysis
  spectral PATH                 Eigenvalue/conditioning analysis
  best [--top N]                Show best checkpoints from DB
  report PATH [--output FILE]   Generate full analysis report

  wandb sync --run-id ID [--project P] [--checkpoint PATH]
                                Sync W&B metrics to database
  wandb plot --run-id ID [--metrics loss accuracy]
                                Plot training metrics (ASCII)

EXAMPLES:

  python analyze.py checkpoint experiments/converted/red_1m
  python analyze.py compare red_1m yellow_2m cosine_2m
  python analyze.py wandb sync --run-id 7jyxclpw --project mbert_dutch
  python analyze.py best --top 5
"""
    )

    parser.add_argument("--db", default="evaluations.db", help="Database path")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Checkpoint info command
    checkpoint_parser = subparsers.add_parser("checkpoint", help="Show checkpoint information")
    checkpoint_parser.add_argument("path", help="Path to checkpoint")
    checkpoint_parser.add_argument("--stats", action="store_true", help="Compute detailed statistics")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare checkpoints")
    compare_parser.add_argument("checkpoints", nargs="+", help="Paths to checkpoints")

    # Stability command
    stability_parser = subparsers.add_parser("stability", help="Stability analysis")
    stability_parser.add_argument("checkpoints", nargs=2, help="Two checkpoints to compare")

    # Spectral command
    spectral_parser = subparsers.add_parser("spectral", help="Spectral analysis")
    spectral_parser.add_argument("path", help="Path to checkpoint")

    # W&B commands
    wandb_parser = subparsers.add_parser("wandb", help="W&B operations")
    wandb_subparsers = wandb_parser.add_subparsers(dest="wandb_cmd")

    wandb_sync = wandb_subparsers.add_parser("sync", help="Sync W&B metrics")
    wandb_sync.add_argument("--run-id", required=True, help="W&B run ID")
    wandb_sync.add_argument("--project", help="W&B project")
    wandb_sync.add_argument("--checkpoint", help="Associated checkpoint path")

    wandb_plot = wandb_subparsers.add_parser("plot", help="Plot metrics")
    wandb_plot.add_argument("--run-id", required=True, help="W&B run ID")
    wandb_plot.add_argument("--project", help="W&B project")
    wandb_plot.add_argument("--metrics", nargs="+", help="Metrics to plot")

    # Best command
    best_parser = subparsers.add_parser("best", help="Show best checkpoints")
    best_parser.add_argument("--top", type=int, default=5, help="Number to show")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive report")
    report_parser.add_argument("path", help="Path to checkpoint")
    report_parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to appropriate command
    commands = {
        'checkpoint': cmd_checkpoint,
        'compare': cmd_compare,
        'stability': cmd_stability,
        'spectral': cmd_spectral,
        'wandb': cmd_wandb,
        'best': cmd_best,
        'report': cmd_report,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()