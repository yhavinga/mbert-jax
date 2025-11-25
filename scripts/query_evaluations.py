#!/usr/bin/env python3
"""
Query evaluation results from SQLite database.
Simple, pragmatic queries for finding best models.

Usage:
    python query_evaluations.py --best-perplexity
    python query_evaluations.py --checkpoint red_1m
    python query_evaluations.py --compare red_1m yellow_2m
    python query_evaluations.py --latest 10
"""

import argparse
import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


def connect_db(db_path: str) -> sqlite3.Connection:
    """Connect to database with row factory."""
    if not Path(db_path).exists():
        print(f"Database {db_path} not found. Run evaluations first.")
        exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def print_separator():
    """Print a visual separator."""
    print("=" * 60)


def format_timestamp(ts: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts


def query_best_perplexity(conn: sqlite3.Connection, limit: int = 10):
    """Find checkpoints with best perplexity scores."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT checkpoint_name, perplexity, accuracy, loss,
               wandb_run_name, timestamp, num_samples
        FROM evaluations
        WHERE eval_type = 'perplexity' AND perplexity IS NOT NULL
        ORDER BY perplexity ASC
        LIMIT ?
    """, (limit,))

    results = cursor.fetchall()

    print_separator()
    print("BEST PERPLEXITY SCORES")
    print_separator()

    for i, row in enumerate(results, 1):
        print(f"\n{i}. {row['checkpoint_name'] or 'Unknown'}")
        print(f"   Perplexity: {row['perplexity']:.2f}")
        print(f"   Accuracy: {row['accuracy']*100:.2f}%")
        print(f"   Loss: {row['loss']:.4f}")
        if row['wandb_run_name']:
            print(f"   W&B: {row['wandb_run_name']}")
        print(f"   Samples: {row['num_samples']}")
        print(f"   Date: {format_timestamp(row['timestamp'])}")


def query_checkpoint_history(conn: sqlite3.Connection, checkpoint: str):
    """Show evaluation history for a specific checkpoint."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT eval_type, perplexity, accuracy, loss,
               timestamp, duration_seconds, num_samples
        FROM evaluations
        WHERE checkpoint_name = ? OR checkpoint_path LIKE ?
        ORDER BY timestamp DESC
    """, (checkpoint, f"%{checkpoint}%"))

    results = cursor.fetchall()

    print_separator()
    print(f"EVALUATION HISTORY: {checkpoint}")
    print_separator()

    if not results:
        print("No evaluations found for this checkpoint.")
        return

    for row in results:
        print(f"\n{format_timestamp(row['timestamp'])}")
        print(f"  Type: {row['eval_type']}")

        if row['eval_type'] == 'perplexity':
            print(f"  Perplexity: {row['perplexity']:.2f}")
            print(f"  Accuracy: {row['accuracy']*100:.2f}%")
            print(f"  Loss: {row['loss']:.4f}")

        print(f"  Samples: {row['num_samples']}")
        print(f"  Duration: {row['duration_seconds']:.1f}s")


def compare_checkpoints(conn: sqlite3.Connection, checkpoints: List[str]):
    """Compare multiple checkpoints side by side."""
    cursor = conn.cursor()

    # Get latest perplexity evaluation for each checkpoint
    results = {}
    for checkpoint in checkpoints:
        cursor.execute("""
            SELECT checkpoint_name, perplexity, accuracy, loss,
                   wandb_run_name, training_step
            FROM evaluations
            WHERE (checkpoint_name = ? OR checkpoint_path LIKE ?)
                  AND eval_type = 'perplexity'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (checkpoint, f"%{checkpoint}%"))

        row = cursor.fetchone()
        if row:
            results[checkpoint] = row

    print_separator()
    print("CHECKPOINT COMPARISON")
    print_separator()

    if not results:
        print("No evaluations found for specified checkpoints.")
        return

    # Print comparison table
    print("\n| Checkpoint | Perplexity | Accuracy | Loss |")
    print("|------------|------------|----------|------|")

    for checkpoint, row in results.items():
        name = row['checkpoint_name'] or checkpoint
        ppl = f"{float(row['perplexity']):.2f}" if row['perplexity'] else "N/A"
        acc = f"{float(row['accuracy'])*100:.1f}%" if row['accuracy'] else "N/A"
        loss = f"{float(row['loss']):.3f}" if row['loss'] else "N/A"
        print(f"| {name:10} | {ppl:10} | {acc:8} | {loss:4} |")

    # Determine winner
    if len(results) > 1:
        best = min(
            results.items(),
            key=lambda x: float(x[1]['perplexity']) if x[1]['perplexity'] else float('inf')
        )
        print(f"\n✓ Best: {best[0]} (perplexity: {float(best[1]['perplexity']):.2f})")


def query_latest(conn: sqlite3.Connection, limit: int = 10):
    """Show latest evaluation runs."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT checkpoint_name, eval_type, perplexity, accuracy,
               timestamp, duration_seconds
        FROM evaluations
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    results = cursor.fetchall()

    print_separator()
    print(f"LATEST {limit} EVALUATIONS")
    print_separator()

    for row in results:
        print(f"\n{format_timestamp(row['timestamp'])}")
        print(f"  Checkpoint: {row['checkpoint_name']}")
        print(f"  Type: {row['eval_type']}")

        if row['eval_type'] == 'perplexity' and row['perplexity']:
            print(f"  Perplexity: {row['perplexity']:.2f}")
            if row['accuracy']:
                print(f"  Accuracy: {row['accuracy']*100:.2f}%")

        print(f"  Duration: {row['duration_seconds']:.1f}s")


def export_to_csv(conn: sqlite3.Connection, output_path: str):
    """Export all evaluations to CSV."""
    import csv

    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM evaluations
        ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    if not rows:
        print("No data to export.")
        return

    # Get column names
    columns = [description[0] for description in cursor.description]

    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    print(f"✓ Exported {len(rows)} evaluations to {output_path}")


def print_summary_stats(conn: sqlite3.Connection):
    """Print database summary statistics."""
    cursor = conn.cursor()

    # Total evaluations
    cursor.execute("SELECT COUNT(*) as total FROM evaluations")
    total = cursor.fetchone()['total']

    # Unique checkpoints
    cursor.execute("SELECT COUNT(DISTINCT checkpoint_path) as unique_checkpoints FROM evaluations")
    unique = cursor.fetchone()['unique_checkpoints']

    # By type
    cursor.execute("""
        SELECT eval_type, COUNT(*) as count
        FROM evaluations
        GROUP BY eval_type
    """)
    by_type = cursor.fetchall()

    print_separator()
    print("DATABASE SUMMARY")
    print_separator()
    print(f"\nTotal evaluations: {total}")
    print(f"Unique checkpoints: {unique}")
    print("\nEvaluations by type:")
    for row in by_type:
        print(f"  {row['eval_type']}: {row['count']}")


def print_usage_examples():
    """Print usage examples when called without arguments."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    Evaluation Results Query Tool                     ║
╚══════════════════════════════════════════════════════════════════════╝

Query and analyze checkpoint evaluation results from SQLite database.

USAGE EXAMPLES:

1. Show database summary (default):
   python query_evaluations.py

2. Find best performing checkpoints:
   python query_evaluations.py --best-perplexity

3. Compare specific checkpoints:
   python query_evaluations.py --compare red_1m yellow_2m

4. View evaluation history for a checkpoint:
   python query_evaluations.py --checkpoint red_1m

5. Show latest evaluations:
   python query_evaluations.py --latest 10

6. Export all results to CSV:
   python query_evaluations.py --export-csv results.csv

7. Use custom database:
   python query_evaluations.py --db my_evaluations.db --summary

COMMON WORKFLOWS:

Finding the best model:
   python query_evaluations.py --best-perplexity
   → Shows checkpoints ranked by perplexity

Comparing candidates:
   python query_evaluations.py --compare red_1m yellow_2m blue_500k
   → Side-by-side comparison with winner

Tracking progress:
   python query_evaluations.py --checkpoint yellow_2m
   → Shows all evaluations for that checkpoint

Data export for analysis:
   python query_evaluations.py --export-csv analysis.csv
   → Export for Excel, pandas, etc.

For detailed options: python query_evaluations.py --help
""")


def main():
    # Custom help if no arguments
    if len(sys.argv) == 1:
        print_usage_examples()
        print("\nShowing database summary...\n")

    parser = argparse.ArgumentParser(
        description="Query evaluation results from database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Find best models
  %(prog)s --best-perplexity

  # Compare checkpoints
  %(prog)s --compare red_1m yellow_2m

  # View checkpoint history
  %(prog)s --checkpoint red_1m

  # Export to CSV
  %(prog)s --export-csv results.csv
"""
    )
    parser.add_argument(
        "--db",
        type=str,
        default="evaluations.db",
        help="Database path (default: evaluations.db)"
    )
    parser.add_argument(
        "--best-perplexity",
        action="store_true",
        help="Show checkpoints with best perplexity"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Show evaluation history for specific checkpoint"
    )
    parser.add_argument(
        "--compare",
        nargs='+',
        help="Compare multiple checkpoints"
    )
    parser.add_argument(
        "--latest",
        type=int,
        metavar="N",
        help="Show latest N evaluations"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        metavar="PATH",
        help="Export all results to CSV"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show database summary statistics"
    )

    args = parser.parse_args()

    # Connect to database
    conn = connect_db(args.db)

    # Execute requested query
    query_executed = False

    if args.best_perplexity:
        query_best_perplexity(conn)
        query_executed = True

    if args.checkpoint:
        query_checkpoint_history(conn, args.checkpoint)
        query_executed = True

    if args.compare:
        compare_checkpoints(conn, args.compare)
        query_executed = True

    if args.latest:
        query_latest(conn, args.latest)
        query_executed = True

    if args.export_csv:
        export_to_csv(conn, args.export_csv)
        query_executed = True

    if args.summary:
        print_summary_stats(conn)
        query_executed = True

    if not query_executed:
        # Default: show summary
        print_summary_stats(conn)
        print("\nUse --help to see available queries.")

    conn.close()


if __name__ == "__main__":
    main()