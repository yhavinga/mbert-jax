"""
W&B utilities for fetching training metrics and managing checkpoints.

This module provides efficient access to W&B training data with caching
to minimize API calls, following Noam Shazeer's efficiency principles.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class WandbCheckpointManager:
    """Manages W&B metadata and metrics for checkpoints with smart caching."""

    def __init__(self, api_key: Optional[str] = None, cache_db: Optional[str] = None):
        """
        Initialize W&B checkpoint manager.

        Args:
            api_key: Optional W&B API key. If None, uses environment variable.
            cache_db: Optional path to SQLite cache database. If None, uses evaluations.db.
        """
        if not HAS_WANDB:
            raise ImportError("wandb package required. Install with: pip install wandb")

        self.api = wandb.Api(api_key=api_key or os.environ.get("WANDB_API_KEY"))
        self.cache_db = cache_db or "evaluations.db"
        self._ensure_cache_tables()

    def _ensure_cache_tables(self):
        """Ensure cache tables exist in the database."""
        # Uses the schema from setup_training_db.py
        # This just checks they exist, doesn't recreate
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        # Check if training_runs table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='training_runs'
        """)
        if not cursor.fetchone():
            print(f"Warning: Training tables not found in {self.cache_db}")
            print("Run 'python setup_training_db.py' to create them")

        conn.close()

    def fetch_run_metrics(
        self,
        run_id: str,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        force_refresh: bool = False,
        include_history: bool = False,
        history_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Fetch metrics from a W&B run with caching.

        Args:
            run_id: W&B run ID
            project: W&B project name (optional if run_id is fully qualified)
            entity: W&B entity/username (optional)
            force_refresh: Force refresh from W&B API even if cached
            include_history: Whether to fetch training history (slow for long runs)
            history_samples: If include_history, sample this many points (default 1000)

        Returns:
            Dictionary with run data including config and summary
        """
        # Check cache first unless force refresh
        if not force_refresh:
            cached = self._get_cached_metrics(run_id)
            if cached:
                return cached

        # Construct run path
        if "/" in run_id:
            # Fully qualified path
            run_path = run_id
        elif entity and project:
            run_path = f"{entity}/{project}/{run_id}"
        elif project:
            run_path = f"{project}/{run_id}"
        else:
            raise ValueError("Need either fully qualified run_id or project name")

        # Fetch from W&B API
        try:
            run = self.api.run(run_path)
        except Exception as e:
            print(f"Error fetching run {run_path}: {e}")
            return {}

        # Extract run data (fast - just metadata)
        run_data = {
            "run_id": run.id,
            "name": run.name,
            "project": run.project,
            "entity": run.entity,
            "url": run.url,
            "state": run.state,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "created_at": run.created_at,
            "tags": run.tags,
        }

        # Optionally fetch history (can be very slow for long runs)
        if include_history:
            print(f"Fetching training history (sampling {history_samples} points)...")
            history = []
            # Use sampling to avoid fetching millions of rows
            for row in run.scan_history(keys=["_step", "train/loss", "train/accuracy",
                                               "train/perplexity", "train/learning_rate",
                                               "train/grad_norm", "train/param_norm"],
                                        page_size=history_samples):
                history.append(dict(row))
                if len(history) >= history_samples:
                    break
            run_data["history"] = history
            print(f"  Fetched {len(history)} history points")
        else:
            run_data["history"] = []

        # Cache the data
        self._cache_metrics(run_data)

        return run_data

    def _get_cached_metrics(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get cached metrics from database."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        try:
            # Get run metadata
            cursor.execute("""
                SELECT wandb_project, wandb_url, wandb_name, config,
                       start_time, end_time, total_steps, status
                FROM training_runs
                WHERE wandb_run_id = ?
            """, (run_id,))

            row = cursor.fetchone()
            if not row:
                return None

            run_data = {
                "run_id": run_id,
                "project": row[0],
                "url": row[1],
                "name": row[2],
                "config": json.loads(row[3]) if row[3] else {},
                "start_time": row[4],
                "end_time": row[5],
                "total_steps": row[6],
                "state": row[7],
            }

            # Get training metrics history
            cursor.execute("""
                SELECT step, loss, accuracy, perplexity, learning_rate,
                       grad_norm, param_norm, update_ratio, tokens_per_sec
                FROM training_metrics
                WHERE run_id = (SELECT id FROM training_runs WHERE wandb_run_id = ?)
                ORDER BY step
            """, (run_id,))

            history = []
            for row in cursor.fetchall():
                history.append({
                    "_step": row[0],
                    "train/loss": row[1],
                    "train/accuracy": row[2],
                    "train/perplexity": row[3],
                    "train/learning_rate": row[4],
                    "train/grad_norm": row[5],
                    "train/param_norm": row[6],
                    "train/update_ratio": row[7],
                    "performance/tokens_per_sec": row[8],
                })

            run_data["history"] = history

            # Return data if we have the run (even without history)
            return run_data

        finally:
            conn.close()

    def _cache_metrics(self, run_data: Dict[str, Any]):
        """Cache metrics to database for faster retrieval."""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        try:
            # Extract final metrics from summary
            summary = run_data.get("summary", {})
            final_loss = summary.get("final_loss") or summary.get("train/loss")
            final_accuracy = summary.get("final_accuracy") or summary.get("train/accuracy")
            final_perplexity = summary.get("final_perplexity") or summary.get("train/perplexity")
            total_steps = summary.get("total_steps") or summary.get("_step", 0)

            # Insert or update run metadata
            cursor.execute("""
                INSERT OR REPLACE INTO training_runs
                (wandb_run_id, wandb_project, wandb_url, wandb_name,
                 config, status, start_time, total_steps,
                 final_loss, final_accuracy, final_perplexity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_data["run_id"],
                run_data.get("project"),
                run_data.get("url"),
                run_data.get("name"),
                json.dumps(run_data.get("config", {})),
                run_data.get("state", "unknown"),
                run_data.get("created_at"),
                total_steps,
                final_loss,
                final_accuracy,
                final_perplexity,
            ))

            run_db_id = cursor.lastrowid

            # Insert metrics history (batch insert for efficiency)
            if "history" in run_data and run_data["history"]:
                metrics_data = []
                for row in run_data["history"]:
                    step = row.get("_step", row.get("step", 0))
                    metrics_data.append((
                        run_db_id,
                        step,
                        row.get("train/loss"),
                        row.get("train/accuracy"),
                        row.get("train/perplexity"),
                        row.get("train/learning_rate"),
                        row.get("train/grad_norm"),
                        row.get("train/param_norm"),
                        row.get("train/update_ratio"),
                        row.get("performance/tokens_per_sec"),
                    ))

                # Clear old metrics first
                cursor.execute("DELETE FROM training_metrics WHERE run_id = ?", (run_db_id,))

                # Batch insert new metrics
                cursor.executemany("""
                    INSERT INTO training_metrics
                    (run_id, step, loss, accuracy, perplexity, learning_rate,
                     grad_norm, param_norm, update_ratio, tokens_per_sec)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, metrics_data)

            conn.commit()

        except Exception as e:
            print(f"Error caching metrics: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_run_config(self, run_id: str, project: Optional[str] = None) -> Dict[str, Any]:
        """Get training configuration from W&B run."""
        run_data = self.fetch_run_metrics(run_id, project)
        return run_data.get("config", {})

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
        project: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple runs side-by-side.

        Args:
            run_ids: List of W&B run IDs to compare
            metrics: List of metric names to compare (default: loss, accuracy, perplexity)
            project: W&B project name

        Returns:
            Comparison dictionary with statistics for each run
        """
        if metrics is None:
            metrics = ["loss", "accuracy", "perplexity"]

        comparison = {}
        for run_id in run_ids:
            run_data = self.fetch_run_metrics(run_id, project)
            if not run_data or not run_data.get("history"):
                continue

            # Extract final and best metrics
            history = run_data["history"]
            run_metrics = {
                "name": run_data.get("name", run_id),
                "url": run_data.get("url"),
                "total_steps": len(history),
                "final": {},
                "best": {},
                "mean": {},
            }

            # Calculate statistics for each metric
            for metric in metrics:
                # Handle different metric name formats
                metric_keys = [
                    f"train/{metric}",
                    metric,
                    f"eval/{metric}",
                ]

                values = []
                for row in history:
                    for key in metric_keys:
                        if key in row and row[key] is not None:
                            values.append(float(row[key]))
                            break

                if values:
                    run_metrics["final"][metric] = values[-1]
                    if metric in ["loss", "perplexity"]:
                        run_metrics["best"][metric] = min(values)
                    else:
                        run_metrics["best"][metric] = max(values)
                    run_metrics["mean"][metric] = np.mean(values)

            comparison[run_id] = run_metrics

        return comparison

    def find_checkpoint_for_run(
        self,
        run_id: str,
        checkpoint_dir: str = "experiments",
        step: Optional[int] = None
    ) -> Optional[Path]:
        """
        Find checkpoint file for a W&B run.

        Args:
            run_id: W&B run ID
            checkpoint_dir: Base directory to search for checkpoints
            step: Specific step number (if None, finds latest)

        Returns:
            Path to checkpoint or None if not found
        """
        base_path = Path(checkpoint_dir)
        if not base_path.exists():
            return None

        # First check database for known associations
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        if step:
            cursor.execute("""
                SELECT checkpoint_path FROM checkpoint_metadata
                WHERE wandb_run_id = ? AND step = ?
                ORDER BY created_at DESC LIMIT 1
            """, (run_id, step))
        else:
            cursor.execute("""
                SELECT checkpoint_path FROM checkpoint_metadata
                WHERE wandb_run_id = ?
                ORDER BY step DESC LIMIT 1
            """, (run_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            path = Path(result[0])
            if path.exists():
                return path

        # If not in database, search filesystem
        # This is slower but handles cases where DB is out of sync
        for checkpoint_path in base_path.rglob("**/checkpoint_*"):
            if checkpoint_path.is_dir():
                # Check for W&B metadata in checkpoint
                # This would require loading the checkpoint which is expensive
                # Skip for now - rely on database
                pass

        return None

    def sync_run_to_db(
        self,
        run_id: str,
        project: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> bool:
        """
        Sync a W&B run to the local database.

        Args:
            run_id: W&B run ID
            project: W&B project name
            checkpoint_path: Optional path to associated checkpoint

        Returns:
            True if successful
        """
        try:
            # Fetch and cache metrics
            run_data = self.fetch_run_metrics(run_id, project, force_refresh=True)

            if not run_data:
                return False

            # If checkpoint path provided, update association
            if checkpoint_path:
                conn = sqlite3.connect(self.cache_db)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO checkpoint_metadata
                    (checkpoint_path, wandb_run_id, step)
                    VALUES (?, ?, ?)
                """, (str(checkpoint_path), run_id, run_data.get("total_steps", 0)))

                conn.commit()
                conn.close()

            print(f"✓ Synced W&B run {run_id} to database")
            return True

        except Exception as e:
            print(f"Error syncing run {run_id}: {e}")
            return False


def get_latest_metrics(run_id: str, project: Optional[str] = None) -> Dict[str, float]:
    """
    Quick function to get latest metrics for a run.

    Args:
        run_id: W&B run ID
        project: W&B project name

    Returns:
        Dictionary of latest metric values
    """
    manager = WandbCheckpointManager()
    run_data = manager.fetch_run_metrics(run_id, project)

    if not run_data or not run_data.get("history"):
        return {}

    # Return last row of history
    return run_data["history"][-1] if run_data["history"] else {}


def download_checkpoint_for_run(
    run_id: str,
    project: Optional[str] = None,
    output_dir: str = "checkpoints"
) -> Optional[Path]:
    """
    Download checkpoint artifacts from W&B for a run.

    Args:
        run_id: W&B run ID
        project: W&B project name
        output_dir: Directory to save checkpoint

    Returns:
        Path to downloaded checkpoint or None
    """
    if not HAS_WANDB:
        print("wandb not available for downloading artifacts")
        return None

    try:
        api = wandb.Api()
        run_path = f"{project}/{run_id}" if project else run_id
        run = api.run(run_path)

        # Look for checkpoint artifacts
        for artifact in run.logged_artifacts():
            if "checkpoint" in artifact.name.lower():
                # Download artifact
                output_path = Path(output_dir) / run_id / artifact.name
                artifact.download(root=str(output_path))
                print(f"✓ Downloaded checkpoint to {output_path}")
                return output_path

        print(f"No checkpoint artifacts found for run {run_id}")
        return None

    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        return None