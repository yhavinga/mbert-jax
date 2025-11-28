#!/usr/bin/env python3
"""
Setup extended SQLite database schema for training and evaluation tracking.

This script creates or upgrades the database to include:
- Training runs table (W&B metadata, configs)
- Training metrics table (time-series training data)
- Checkpoint metadata table (links checkpoints to runs)
- Enhanced evaluations table (existing)

Usage:
    python setup_training_db.py
    python setup_training_db.py --db custom_path.db
"""

import argparse
import sqlite3
from pathlib import Path


def setup_database(db_path: str, verbose: bool = True):
    """Create or upgrade database schema with training tables."""
    db_path = Path(db_path)
    is_new = not db_path.exists()

    if verbose:
        action = "Creating new" if is_new else "Upgrading"
        print(f"{action} database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")

    # 1. Training runs table - stores W&B run metadata and configs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wandb_run_id TEXT UNIQUE,
            wandb_project TEXT,
            wandb_url TEXT,
            wandb_name TEXT,  -- Human-readable run name

            -- Training configuration (JSON)
            config TEXT,
            model_config TEXT,
            optimizer_config TEXT,

            -- Run metadata
            start_time DATETIME,
            end_time DATETIME,
            total_steps INTEGER,
            checkpoint_dir TEXT,

            -- Final metrics
            final_loss REAL,
            final_accuracy REAL,
            final_perplexity REAL,

            -- Hardware info
            hardware TEXT,  -- TPU/GPU type
            num_devices INTEGER,

            -- Status
            status TEXT DEFAULT 'running',  -- running, completed, failed, interrupted
            notes TEXT,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Training metrics table - time-series data from training
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,

            -- Step info
            step INTEGER NOT NULL,
            epoch INTEGER,

            -- Core metrics
            loss REAL,
            accuracy REAL,
            perplexity REAL,

            -- Learning rates
            learning_rate REAL,
            layer_norm_lr REAL,  -- If using separate LR for layer norms

            -- Gradient metrics
            grad_norm REAL,
            param_norm REAL,
            update_ratio REAL,  -- Ratio of update to parameter norms

            -- Performance metrics
            tokens_per_sec REAL,
            samples_per_sec REAL,
            steps_per_sec REAL,

            -- Training details
            n_masked_tokens INTEGER,
            masking_percentage REAL,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

            -- Composite index for efficient queries
            UNIQUE(run_id, step)
        )
    """)

    # 3. Checkpoint metadata table - links checkpoints to training runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_path TEXT UNIQUE NOT NULL,
            run_id INTEGER REFERENCES training_runs(id),

            -- Checkpoint info
            step INTEGER NOT NULL,
            wandb_run_id TEXT,  -- Stored in checkpoint

            -- Checkpoint stats
            file_size_mb REAL,
            param_count INTEGER,

            -- Conversion info (if converted from Orbax)
            original_format TEXT,  -- 'orbax', 'pkl', etc.
            converted_at DATETIME,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 4. Per-layer gradient norms table (optional detailed tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS layer_gradients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,

            -- Layer identification
            layer_name TEXT NOT NULL,
            layer_type TEXT,  -- 'attention', 'mlp', 'embedding', etc.

            -- Gradient statistics
            grad_norm REAL,
            grad_mean REAL,
            grad_std REAL,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 5. Update evaluations table if it exists (backward compatible)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_path TEXT NOT NULL,
            checkpoint_name TEXT,
            training_step INTEGER,
            wandb_run_name TEXT,

            -- Link to training run
            run_id INTEGER REFERENCES training_runs(id),

            eval_type TEXT NOT NULL,
            dataset TEXT,
            dataset_config TEXT,
            split TEXT,
            num_samples INTEGER,

            -- Metrics
            loss REAL,
            accuracy REAL,
            perplexity REAL,

            -- Metadata
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            duration_seconds REAL,
            config TEXT,
            predictions TEXT
        )
    """)

    # Add run_id column to existing evaluations table if missing
    cursor.execute("PRAGMA table_info(evaluations)")
    columns = [col[1] for col in cursor.fetchall()]
    if "run_id" not in columns:
        try:
            cursor.execute("ALTER TABLE evaluations ADD COLUMN run_id INTEGER REFERENCES training_runs(id)")
        except sqlite3.OperationalError:
            pass  # Column might already exist

    # Create indices for common queries
    indices = [
        # Training runs indices
        "CREATE INDEX IF NOT EXISTS idx_wandb_run_id ON training_runs(wandb_run_id)",
        "CREATE INDEX IF NOT EXISTS idx_run_status ON training_runs(status)",
        "CREATE INDEX IF NOT EXISTS idx_run_project ON training_runs(wandb_project)",

        # Training metrics indices
        "CREATE INDEX IF NOT EXISTS idx_metrics_run ON training_metrics(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_metrics_step ON training_metrics(run_id, step)",
        "CREATE INDEX IF NOT EXISTS idx_metrics_loss ON training_metrics(loss)",

        # Checkpoint indices
        "CREATE INDEX IF NOT EXISTS idx_checkpoint_wandb ON checkpoint_metadata(wandb_run_id)",
        "CREATE INDEX IF NOT EXISTS idx_checkpoint_step ON checkpoint_metadata(step)",
        "CREATE INDEX IF NOT EXISTS idx_checkpoint_run ON checkpoint_metadata(run_id)",

        # Layer gradients indices
        "CREATE INDEX IF NOT EXISTS idx_layer_step ON layer_gradients(run_id, step)",

        # Evaluation indices
        "CREATE INDEX IF NOT EXISTS idx_eval_checkpoint ON evaluations(checkpoint_path)",
        "CREATE INDEX IF NOT EXISTS idx_eval_type ON evaluations(eval_type)",
        "CREATE INDEX IF NOT EXISTS idx_eval_perplexity ON evaluations(perplexity)",
        "CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluations(run_id)",
    ]

    for idx_sql in indices:
        try:
            cursor.execute(idx_sql)
        except sqlite3.OperationalError as e:
            # Skip indices for columns that don't exist
            if "no such column" in str(e):
                continue
            else:
                raise

    # Create views for common queries
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS v_run_summary AS
        SELECT
            r.wandb_run_id,
            r.wandb_name,
            r.wandb_project,
            r.status,
            r.total_steps,
            r.final_loss,
            r.final_perplexity,
            COUNT(DISTINCT c.id) as num_checkpoints,
            COUNT(DISTINCT e.id) as num_evaluations,
            MIN(m.loss) as best_loss,
            MIN(e.perplexity) as best_eval_perplexity
        FROM training_runs r
        LEFT JOIN checkpoint_metadata c ON r.id = c.run_id
        LEFT JOIN evaluations e ON r.id = e.run_id
        LEFT JOIN training_metrics m ON r.id = m.run_id
        GROUP BY r.id
    """)

    cursor.execute("""
        CREATE VIEW IF NOT EXISTS v_checkpoint_quality AS
        SELECT
            c.checkpoint_path,
            c.step,
            r.wandb_name,
            r.wandb_project,
            e.perplexity,
            e.accuracy,
            e.loss,
            e.timestamp as eval_date
        FROM checkpoint_metadata c
        LEFT JOIN training_runs r ON c.run_id = r.id
        LEFT JOIN evaluations e ON c.checkpoint_path = e.checkpoint_path
        WHERE e.eval_type = 'perplexity'
        ORDER BY e.perplexity ASC
    """)

    conn.commit()

    if verbose:
        # Report what was created/exists
        cursor.execute("""
            SELECT name, type FROM sqlite_master
            WHERE type IN ('table', 'view')
            AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
        """)
        objects = cursor.fetchall()

        print(f"\nDatabase objects:")
        current_type = None
        for name, obj_type in objects:
            if obj_type != current_type:
                current_type = obj_type
                print(f"\n{obj_type.capitalize()}s:")
            print(f"  - {name}")

        # Check if we need to migrate existing data
        cursor.execute("""
            SELECT COUNT(*) FROM evaluations
            WHERE run_id IS NULL AND wandb_run_name IS NOT NULL
        """)
        unmigrated = cursor.fetchone()[0]
        if unmigrated > 0:
            print(f"\n⚠ Found {unmigrated} evaluations that could be linked to training runs")
            print("  Run 'python migrate_evaluations.py' to link them")

    conn.close()

    if verbose:
        print(f"\n✓ Database setup complete: {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup extended database schema for training tracking"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="evaluations.db",
        help="Database path (default: evaluations.db)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    setup_database(args.db, verbose=not args.quiet)


if __name__ == "__main__":
    main()