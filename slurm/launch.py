#!/usr/bin/env python3
"""SLURM job launcher for DDE-FNO project.

Reads a .sbatch template, fills {template variables} from CLI arguments,
optionally submits the job via sbatch, and prints the job ID.

Usage examples:
    # Generate data for the hutch family
    python slurm/launch.py --task generate --family hutch --n_train 2000

    # Train single-GPU
    python slurm/launch.py --task train --family hutch --config_path configs/train_hutch.yaml

    # Train multi-GPU with DDP
    python slurm/launch.py --task train --family hutch --n_gpus 4 --config_path configs/train_hutch.yaml

    # Evaluate a checkpoint
    python slurm/launch.py --task eval --family hutch --checkpoint_path outputs/hutch/best.pt

    # Dry run (prints filled template without submitting)
    python slurm/launch.py --task train --family hutch --config_path configs/train_hutch.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Resolve directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # slurm/
PROJECT_ROOT = SCRIPT_DIR.parent                      # dde-fno/
TEMPLATE_DIR = SCRIPT_DIR / "templates"

# Map task names to template filenames
TASK_TEMPLATES = {
    "generate": "generate_data.sbatch",
    "train":    None,  # resolved dynamically based on n_gpus
    "eval":     "eval.sbatch",
}


def resolve_train_template(n_gpus: int) -> str:
    """Pick single-GPU or DDP template based on GPU count."""
    if n_gpus <= 1:
        return "train_single.sbatch"
    return "train_ddp.sbatch"


def load_template(task: str, n_gpus: int) -> str:
    """Load the raw sbatch template string for *task*."""
    if task == "train":
        filename = resolve_train_template(n_gpus)
    else:
        filename = TASK_TEMPLATES[task]
    path = TEMPLATE_DIR / filename
    if not path.exists():
        print(f"Error: template not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text()


def fill_template(template: str, variables: dict[str, str]) -> str:
    """Replace every ``{key}`` in *template* with the corresponding value.

    Uses simple str.replace so that no special escaping is needed for SLURM
    environment variables like ``$SLURM_SUBMIT_DIR``.
    """
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def find_unfilled(filled: str) -> list[str]:
    """Return any remaining {placeholders} that were not filled."""
    # Match {word} but NOT ${word} (bash variables like $SLURM_SUBMIT_DIR)
    return re.findall(r"(?<!\$)\{(\w+)\}", filled)


def ensure_slurm_logs_dir() -> None:
    """Create the slurm_logs/ directory under the project root if needed."""
    log_dir = PROJECT_ROOT / "slurm_logs"
    log_dir.mkdir(exist_ok=True)


def submit_job(script_content: str) -> str | None:
    """Write *script_content* to a temp file, submit via sbatch, return job ID."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sbatch", dir=str(PROJECT_ROOT), delete=False
    ) as f:
        f.write(script_content)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", tmp_path],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"sbatch failed (rc={result.returncode}):", file=sys.stderr)
            print(result.stderr.strip(), file=sys.stderr)
            sys.exit(1)

        # sbatch prints: "Submitted batch job 12345"
        stdout = result.stdout.strip()
        print(stdout)
        parts = stdout.split()
        job_id = parts[-1] if parts else None
        return job_id
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch SLURM jobs for the DDE-FNO project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--task",
        required=True,
        choices=["generate", "train", "eval"],
        help="Which job type to launch.",
    )
    p.add_argument(
        "--family",
        required=True,
        help="DDE/PDE family name (e.g. hutch, vdp, burgers).",
    )

    # Resource / scheduling
    p.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="Number of GPUs (>1 selects the DDP template). Default: 1.",
    )
    p.add_argument(
        "--walltime",
        default=None,
        help="Wall-clock limit (HH:MM:SS). Defaults: generate=02:00:00, "
             "train=12:00:00, eval=00:30:00.",
    )

    # Data generation
    p.add_argument("--n_train", type=int, default=2000,  help="Training samples.   Default: 2000.")
    p.add_argument("--n_val",   type=int, default=500,   help="Validation samples. Default: 500.")
    p.add_argument("--n_test",  type=int, default=500,   help="Test samples.       Default: 500.")
    p.add_argument("--data_dir", default="data",         help="Root data directory. Default: data.")

    # Training
    p.add_argument("--config_path", default=None,        help="Path to training YAML config.")
    p.add_argument("--output_dir",  default="outputs",   help="Model output directory. Default: outputs.")

    # Evaluation
    p.add_argument("--checkpoint_path", default=None,    help="Path to model checkpoint for eval.")
    p.add_argument("--eval_dir",        default="reports", help="Evaluation output directory. Default: reports.")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the filled sbatch script without submitting.",
    )

    return p


WALLTIME_DEFAULTS = {
    "generate": "02:00:00",
    "train":    "12:00:00",
    "eval":     "00:30:00",
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Determine walltime
    walltime = args.walltime or WALLTIME_DEFAULTS[args.task]

    # Build variable dict for template filling
    variables: dict[str, str] = {
        "family":          args.family,
        "walltime":        walltime,
        "seed":            str(args.seed),
        "data_dir":        args.data_dir,
        "n_gpus":          str(args.n_gpus),
        # Data generation
        "n_train":         str(args.n_train),
        "n_val":           str(args.n_val),
        "n_test":          str(args.n_test),
        # Training
        "config_path":     args.config_path or "",
        "output_dir":      args.output_dir,
        # Evaluation
        "checkpoint_path": args.checkpoint_path or "",
        "eval_dir":        args.eval_dir,
    }

    # Validate task-specific requirements
    if args.task == "train" and not args.config_path:
        parser.error("--config_path is required for training jobs.")
    if args.task == "eval" and not args.checkpoint_path:
        parser.error("--checkpoint_path is required for evaluation jobs.")

    # Load and fill template
    template = load_template(args.task, args.n_gpus)
    filled = fill_template(template, variables)

    # Warn about any unfilled placeholders
    unfilled = find_unfilled(filled)
    if unfilled:
        print(
            f"Warning: unfilled placeholders remain: {unfilled}",
            file=sys.stderr,
        )

    if args.dry_run:
        print("=" * 72)
        print("DRY RUN -- would submit the following script:")
        print("=" * 72)
        print(filled)
        print("=" * 72)
        return

    # Ensure log directory exists
    ensure_slurm_logs_dir()

    # Submit
    job_id = submit_job(filled)
    if job_id:
        print(f"Job ID: {job_id}")


if __name__ == "__main__":
    main()
