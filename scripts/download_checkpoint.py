#!/usr/bin/env python3
"""
Download checkpoints from TPU VM to local machine.
Uses SSH configuration from tpu.sh for consistency.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download checkpoints from TPU VM"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Specific checkpoint step to download (e.g., 1000000)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available checkpoints"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Download only the latest checkpoint"
    )
    parser.add_argument(
        "--output-dir", "--local-dir",
        type=str,
        default="checkpoints",
        help="Local directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        default="~/mbert-jax/checkpoints",
        help="Remote checkpoint directory on TPU VM (default: ~/mbert-jax/checkpoints)"
    )
    parser.add_argument(
        "--tpu-name",
        type=str,
        default="tpu-v4-modernbert",
        help="TPU name from ssh_config (default: tpu-v4-modernbert)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    return parser.parse_args()


class CheckpointDownloader:
    def __init__(self, args):
        self.args = args
        self.ssh_config_path = Path("ssh_config")

        # Check SSH config exists
        if not self.ssh_config_path.exists():
            print("ERROR: ssh_config not found. Run './tpu.sh status' first to generate it.")
            sys.exit(1)

    def run_remote_command(self, command: str) -> str:
        """Execute command on TPU VM and return output."""
        cmd = [
            "ssh", "-F", str(self.ssh_config_path),
            self.args.tpu_name,
            command
        ]

        if self.args.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"ERROR running remote command: {e.stderr}")
            sys.exit(1)

    def list_remote_checkpoints(self) -> List[int]:
        """List available checkpoint steps on TPU VM."""
        # List directories in checkpoint path
        cmd = f"ls -1d {self.args.remote_dir}/*/  2>/dev/null | while read dir; do basename \"$dir\"; done | grep -E '^[0-9]+$' | sort -n"
        output = self.run_remote_command(cmd)

        if not output:
            print(f"No checkpoints found in {self.args.remote_dir}")
            return []

        steps = [int(line) for line in output.split('\n') if line]
        return steps

    def get_checkpoint_info(self, step: int) -> dict:
        """Get information about a specific checkpoint."""
        checkpoint_path = f"{self.args.remote_dir}/{step}"

        # Check if checkpoint exists
        cmd = f"if [ -d {checkpoint_path} ]; then echo 'exists'; fi"
        exists = self.run_remote_command(cmd)
        if not exists:
            return None

        # Get size
        cmd = f"du -sh {checkpoint_path} | cut -f1"
        size = self.run_remote_command(cmd)

        # Count files
        cmd = f"find {checkpoint_path} -type f | wc -l"
        file_count = self.run_remote_command(cmd)

        return {
            "step": step,
            "path": checkpoint_path,
            "size": size,
            "file_count": int(file_count)
        }

    def download_checkpoint(self, step: int):
        """Download a specific checkpoint using rsync."""
        remote_path = f"{self.args.remote_dir}/{step}/"
        local_path = Path(self.args.output_dir) / str(step)

        # Create local directory
        local_path.mkdir(parents=True, exist_ok=True)

        # Rsync command with compression and progress
        rsync_cmd = [
            "rsync",
            "-avz",  # archive, verbose, compress
            "--progress",
            f"--rsh=ssh -F {self.ssh_config_path}",
        ]

        if self.args.dry_run:
            rsync_cmd.append("--dry-run")

        rsync_cmd.extend([
            f"{self.args.tpu_name}:{remote_path}",
            str(local_path) + "/"
        ])

        print(f"\n{'[DRY RUN] ' if self.args.dry_run else ''}Downloading checkpoint {step}...")
        print(f"  From: {remote_path}")
        print(f"  To:   {local_path}")

        if self.args.verbose:
            print(f"  Command: {' '.join(rsync_cmd)}")

        try:
            subprocess.run(rsync_cmd, check=True)
            if not self.args.dry_run:
                print(f"✓ Successfully downloaded checkpoint {step}")

                # Verify Orbax structure
                if (local_path / "default").exists():
                    print(f"✓ Verified Orbax checkpoint structure")
                else:
                    print(f"⚠ Warning: Checkpoint may not have expected Orbax structure")

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download checkpoint {step}")
            sys.exit(1)

    def run(self):
        """Main execution logic."""
        print("=" * 60)
        print("CHECKPOINT DOWNLOADER")
        print("=" * 60)

        # List available checkpoints
        print(f"\nQuerying TPU VM for available checkpoints...")
        available_steps = self.list_remote_checkpoints()

        if not available_steps:
            print("No checkpoints found on TPU VM")
            sys.exit(0)

        print(f"\nFound {len(available_steps)} checkpoint(s):")
        for step in available_steps:
            info = self.get_checkpoint_info(step)
            if info:
                print(f"  • Step {step:8d}: {info['size']:>8s} ({info['file_count']} files)")

        # Determine which checkpoints to download
        steps_to_download = []

        if self.args.all:
            steps_to_download = available_steps
            print(f"\nWill download ALL {len(steps_to_download)} checkpoints")
        elif self.args.latest:
            steps_to_download = [available_steps[-1]]
            print(f"\nWill download latest checkpoint: {steps_to_download[0]}")
        elif self.args.step:
            if self.args.step in available_steps:
                steps_to_download = [self.args.step]
                print(f"\nWill download checkpoint: {self.args.step}")
            else:
                print(f"\nERROR: Checkpoint {self.args.step} not found")
                print(f"Available steps: {available_steps}")
                sys.exit(1)
        else:
            print("\nNo checkpoint specified. Use --step, --latest, or --all")
            print("\nExample usage:")
            print(f"  python {sys.argv[0]} --latest")
            print(f"  python {sys.argv[0]} --step 1000000")
            print(f"  python {sys.argv[0]} --all")
            sys.exit(0)

        # Download checkpoints
        print(f"\nOutput directory: {Path(self.args.output_dir).absolute()}")

        for i, step in enumerate(steps_to_download, 1):
            print(f"\n[{i}/{len(steps_to_download)}] Processing checkpoint {step}")
            self.download_checkpoint(step)

        if not self.args.dry_run:
            print("\n" + "=" * 60)
            print(f"✓ Downloaded {len(steps_to_download)} checkpoint(s) successfully")
            print(f"✓ Saved to: {Path(self.args.output_dir).absolute()}")

            # Show how to use the checkpoints
            print("\nNext steps:")
            print("1. Evaluate a checkpoint:")
            print(f"   python scripts/eval_checkpoint.py --checkpoint-dir {self.args.output_dir}/{steps_to_download[-1]}")
            print("\n2. Resume training from checkpoint:")
            print(f"   python scripts/train_minimal.py --checkpoint-dir {self.args.output_dir}")
        else:
            print("\n[DRY RUN] No files were actually downloaded")


def main():
    args = parse_args()
    downloader = CheckpointDownloader(args)
    downloader.run()


if __name__ == "__main__":
    main()