#!/usr/bin/env python3
"""SLURM sweep launcher for DDE-FNO hyperparameter / benchmark sweeps.

Reads a sweep YAML config (e.g. configs/sweep_hard_benchmark.yaml) and
generates one SLURM job per combination of (family, data_scale, seed,
model_config).  Supports SLURM ``--array`` for parameter sweeps and a
``--dry-run`` flag that prints every generated script without submitting.

Usage examples:
    # Launch the full hard-benchmark sweep
    python slurm/sweep.py configs/sweep_hard_benchmark.yaml

    # Dry-run: inspect generated scripts
    python slurm/sweep.py configs/sweep_hard_benchmark.yaml --dry-run

    # Filter to a single family
    python slurm/sweep.py configs/sweep_hard_benchmark.yaml --filter-family hutch

    # Only generate data (skip training)
    python slurm/sweep.py configs/sweep_hard_benchmark.yaml --stage generate

    # Only train (assume data already exists)
    python slurm/sweep.py configs/sweep_hard_benchmark.yaml --stage train

    # Only evaluate
    python slurm/sweep.py configs/sweep_hard_benchmark.yaml --stage eval
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def load_template(name: str) -> str:
    path = TEMPLATE_DIR / name
    if not path.exists():
        print(f"Error: template not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text()


def fill_template(template: str, variables: dict[str, str]) -> str:
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


# ---------------------------------------------------------------------------
# Sweep config loader
# ---------------------------------------------------------------------------

def load_sweep_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Validate required keys
    required = ["sweep_name", "families", "data_scales", "seeds",
                "model_configs", "training", "slurm"]
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"Error: sweep config missing keys: {missing}", file=sys.stderr)
        sys.exit(1)
    return cfg


def gather_families(cfg: dict[str, Any]) -> list[str]:
    """Flatten the families dict into a single list."""
    families: list[str] = []
    for _category, fam_list in cfg["families"].items():
        if isinstance(fam_list, list):
            families.extend(fam_list)
        else:
            families.append(str(fam_list))
    return families


# ---------------------------------------------------------------------------
# Job generation
# ---------------------------------------------------------------------------

def make_config_path(sweep_name: str, family: str, model_name: str,
                     n_train: int, seed: int) -> str:
    """Return a path for an auto-generated training config."""
    return (
        f"configs/auto/{sweep_name}/{family}_{model_name}_n{n_train}_s{seed}.yaml"
    )


def make_output_dir(sweep_name: str, family: str, model_name: str,
                    n_train: int, seed: int) -> str:
    return f"outputs/{sweep_name}/{family}/{model_name}_n{n_train}_s{seed}"


def make_eval_dir(sweep_name: str, family: str, model_name: str,
                  n_train: int, seed: int) -> str:
    return f"reports/{sweep_name}/{family}/{model_name}_n{n_train}_s{seed}"


def write_auto_config(path: str, family: str, model_cfg: dict,
                      training_cfg: dict, data_dir: str, n_train: int,
                      seed: int) -> None:
    """Write a per-run YAML training config for use by train_fno_sharded.py."""
    config = {
        "family": family,
        "data_dir": data_dir,
        "seed": seed,
        "n_train": n_train,
        "model": dict(model_cfg),
        "training": dict(training_cfg),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def submit_script(script_content: str, dry_run: bool = False,
                  extra_sbatch_args: list[str] | None = None) -> str | None:
    """Submit an sbatch script.  Returns the job ID or None on dry-run."""
    if dry_run:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sbatch", dir=str(PROJECT_ROOT), delete=False
    ) as f:
        f.write(script_content)
        tmp_path = f.name

    try:
        cmd = ["sbatch"]
        if extra_sbatch_args:
            cmd.extend(extra_sbatch_args)
        cmd.append(tmp_path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            print(f"  sbatch failed: {result.stderr.strip()}", file=sys.stderr)
            return None

        stdout = result.stdout.strip()
        parts = stdout.split()
        return parts[-1] if parts else None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def run_sweep(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    sweep_name = cfg["sweep_name"]
    families = gather_families(cfg)
    data_scales = cfg["data_scales"]
    seeds = cfg["seeds"]
    model_configs = cfg["model_configs"]
    training_cfg = cfg["training"]
    slurm_cfg = cfg["slurm"]

    walltime_gen = slurm_cfg.get("walltime_generate", "02:00:00")
    walltime_train = slurm_cfg.get("walltime_train", "12:00:00")
    n_gpus = slurm_cfg.get("n_gpus", 1)
    data_dir = args.data_dir

    # Apply family filter if set
    if args.filter_family:
        families = [f for f in families if f in args.filter_family]
        if not families:
            print("No families matched the filter.", file=sys.stderr)
            sys.exit(1)

    # Apply model filter if set
    if args.filter_model:
        model_configs = {
            k: v for k, v in model_configs.items() if k in args.filter_model
        }
        if not model_configs:
            print("No model configs matched the filter.", file=sys.stderr)
            sys.exit(1)

    # Ensure slurm_logs exists
    ensure_dir(str(PROJECT_ROOT / "slurm_logs"))

    # Track submitted job counts
    gen_jobs: list[str] = []
    train_jobs: list[str] = []
    eval_jobs: list[str] = []

    stage = args.stage  # "all", "generate", "train", "eval"

    # -----------------------------------------------------------------
    # STAGE 1: Data generation  (one job per family x data_scale x seed)
    # -----------------------------------------------------------------
    if stage in ("all", "generate"):
        gen_template = load_template("generate_data.sbatch")
        combos = list(itertools.product(families, data_scales, seeds))
        print(f"\n=== Data generation: {len(combos)} jobs ===")

        for family, n_train, seed in combos:
            n_val = max(n_train // 4, 100)
            n_test = max(n_train // 4, 100)

            variables = {
                "family":   family,
                "walltime": walltime_gen,
                "n_train":  str(n_train),
                "n_val":    str(n_val),
                "n_test":   str(n_test),
                "data_dir": data_dir,
                "seed":     str(seed),
            }
            filled = fill_template(gen_template, variables)

            if args.dry_run:
                print(f"\n--- generate: {family} n={n_train} seed={seed} ---")
                print(filled)
            else:
                jid = submit_script(filled)
                if jid:
                    gen_jobs.append(jid)
                    print(f"  Submitted generate {family} n={n_train} seed={seed} -> {jid}")

    # -----------------------------------------------------------------
    # STAGE 2: Training (one job per family x data_scale x model x seed)
    # -----------------------------------------------------------------
    if stage in ("all", "train"):
        if n_gpus > 1:
            train_template = load_template("train_ddp.sbatch")
        else:
            train_template = load_template("train_single.sbatch")

        combos = list(itertools.product(
            families, data_scales, model_configs.items(), seeds
        ))
        print(f"\n=== Training: {len(combos)} jobs ===")

        for family, n_train, (model_name, model_cfg), seed in combos:
            config_path = make_config_path(
                sweep_name, family, model_name, n_train, seed
            )
            output_dir = make_output_dir(
                sweep_name, family, model_name, n_train, seed
            )

            # Write the auto-generated per-run config
            if not args.dry_run:
                write_auto_config(
                    str(PROJECT_ROOT / config_path),
                    family, model_cfg, training_cfg,
                    f"{data_dir}/{family}", n_train, seed,
                )

            variables = {
                "family":      family,
                "walltime":    walltime_train,
                "n_gpus":      str(n_gpus),
                "config_path": config_path,
                "data_dir":    f"{data_dir}/{family}",
                "output_dir":  output_dir,
                "seed":        str(seed),
            }
            filled = fill_template(train_template, variables)

            # If data generation jobs were submitted, add dependency
            extra_args: list[str] | None = None
            if gen_jobs and stage == "all":
                dep_str = ":".join(gen_jobs)
                extra_args = [f"--dependency=afterok:{dep_str}"]

            if args.dry_run:
                print(f"\n--- train: {family} {model_name} n={n_train} seed={seed} ---")
                print(filled)
            else:
                jid = submit_script(filled, extra_sbatch_args=extra_args)
                if jid:
                    train_jobs.append(jid)
                    print(
                        f"  Submitted train {family} {model_name} "
                        f"n={n_train} seed={seed} -> {jid}"
                    )

    # -----------------------------------------------------------------
    # STAGE 3: Evaluation (one job per family x data_scale x model x seed)
    # -----------------------------------------------------------------
    if stage in ("all", "eval"):
        eval_template = load_template("eval.sbatch")

        combos = list(itertools.product(
            families, data_scales, model_configs.items(), seeds
        ))
        print(f"\n=== Evaluation: {len(combos)} jobs ===")

        for family, n_train, (model_name, model_cfg), seed in combos:
            output_dir = make_output_dir(
                sweep_name, family, model_name, n_train, seed
            )
            eval_dir = make_eval_dir(
                sweep_name, family, model_name, n_train, seed
            )
            checkpoint_path = f"{output_dir}/best.pt"

            variables = {
                "family":          family,
                "data_dir":        f"{data_dir}/{family}",
                "checkpoint_path": checkpoint_path,
                "eval_dir":        eval_dir,
            }
            filled = fill_template(eval_template, variables)

            # If training jobs were submitted, add dependency
            extra_args = None
            if train_jobs and stage == "all":
                dep_str = ":".join(train_jobs)
                extra_args = [f"--dependency=afterok:{dep_str}"]

            if args.dry_run:
                print(f"\n--- eval: {family} {model_name} n={n_train} seed={seed} ---")
                print(filled)
            else:
                jid = submit_script(filled, extra_sbatch_args=extra_args)
                if jid:
                    eval_jobs.append(jid)
                    print(
                        f"  Submitted eval {family} {model_name} "
                        f"n={n_train} seed={seed} -> {jid}"
                    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    total = len(gen_jobs) + len(train_jobs) + len(eval_jobs)
    if args.dry_run:
        print(f"\n[DRY RUN] Would submit jobs across stages.")
    else:
        print(f"\nSweep '{sweep_name}' submitted {total} jobs total:")
        print(f"  Generate: {len(gen_jobs)}")
        print(f"  Train:    {len(train_jobs)}")
        print(f"  Eval:     {len(eval_jobs)}")

    # Write a manifest for tracking
    if not args.dry_run and total > 0:
        manifest = {
            "sweep_name": sweep_name,
            "generate_jobs": gen_jobs,
            "train_jobs": train_jobs,
            "eval_jobs": eval_jobs,
        }
        manifest_path = PROJECT_ROOT / f"slurm_logs/sweep_manifest_{sweep_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch a SLURM sweep from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "sweep_config",
        help="Path to sweep YAML config (e.g. configs/sweep_hard_benchmark.yaml).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated scripts without submitting.",
    )
    p.add_argument(
        "--stage",
        choices=["all", "generate", "train", "eval"],
        default="all",
        help="Which stage(s) to run.  Default: all.",
    )
    p.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory.  Default: data.",
    )
    p.add_argument(
        "--filter-family",
        nargs="+",
        default=None,
        help="Only sweep over these families.",
    )
    p.add_argument(
        "--filter-model",
        nargs="+",
        default=None,
        help="Only sweep over these model configs (e.g. small medium).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_sweep_config(args.sweep_config)
    run_sweep(cfg, args)


if __name__ == "__main__":
    main()
