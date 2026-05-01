#!/usr/bin/env python3
"""Phase B sweep orchestrator.

Drives the core-DDE benchmark from EXPERIMENTATION_PLAN.txt §4 / §8:
    11 models × 7 families × ~4 splits × 3 seeds ≈ 700 runs.

Consumes `configs/sweep_phase_b.yaml`. Differs from the generic
`sweep.py` in three ways:

  1. The matrix axis is (family, split, model, seed) — not
     (family, data_scale, model, seed) — because Phase B uses a single
     data scale and stratifies over OOD splits instead.

  2. Each `model_configs` entry bundles a `model_class` token plus
     per-class kwargs; the generated per-run YAML is consumed by
     `src/train/build_model.py::build_model`, which dispatches on that
     token.

  3. `split_applicability` gates which splits run on which family
     (plan §4: continuous-τ only for DistExp/DistUniform/Hutchinson;
     history-amplitude only for VdP/MG/PP).  Combos whose data
     directory is absent are SKIPPED with a warning so the sweep can
     be launched incrementally as MG/PP/continuous-τ data lands.

Usage:
    # Launch on ready families only
    python slurm/sweep_phase_b.py configs/sweep_phase_b.yaml \\
        --data_dir data_baseline_v2

    # Dry run to see what would submit
    python slurm/sweep_phase_b.py configs/sweep_phase_b.yaml --dry-run

    # Only train step (assume data generated)
    python slurm/sweep_phase_b.py configs/sweep_phase_b.yaml --stage train
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
import subprocess
import os
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


# ---------------------------------------------------------------------------
# Helpers (copied from sweep.py to avoid cross-import coupling)
# ---------------------------------------------------------------------------

def load_template(name: str) -> str:
    path = TEMPLATE_DIR / name
    if not path.exists():
        sys.exit(f"template not found: {path}")
    return path.read_text()


def fill_template(tpl: str, vars_: dict[str, str]) -> str:
    for k, v in vars_.items():
        tpl = tpl.replace(f"{{{k}}}", str(v))
    return tpl


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def submit_script(script: str, dry: bool, extra: list[str] | None = None) -> str | None:
    if dry:
        return None
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sbatch", dir=str(PROJECT_ROOT), delete=False,
    ) as f:
        f.write(script)
        tmp = f.name
    try:
        cmd = ["sbatch"] + (extra or []) + [tmp]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            print(f"  sbatch failed: {r.stderr.strip()}", file=sys.stderr)
            return None
        parts = r.stdout.strip().split()
        return parts[-1] if parts else None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Phase B matrix
# ---------------------------------------------------------------------------

def gather_families(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for _cat, fam_list in cfg["families"].items():
        out.extend(fam_list if isinstance(fam_list, list) else [fam_list])
    return out


def data_dir_for(root: str, family: str, split: str,
                 dir_map: dict[str, str], seed: int,
                 per_seed: bool = False) -> str:
    """Return the data directory for (family, split).

    By default the layout is seed-agnostic, matching the frozen-baseline
    convention.  `dir_map[split]` gives the full relative path to the
    split's data root; an empty entry for `id` means "use `root` itself"
    (so the CLI's `--data_dir` controls the ID training directory while
    OOD dirs are pinned by the config).

    With `per_seed=True`, the legacy per-seed suffix from
    `sweep.py::make_data_dir` is appended (`..._s{seed}`). This is useful
    only when regenerating data per seed for concurrent-write safety.
    """
    entry = dir_map.get(split, "")
    base = root if entry == "" else entry
    return f"{base}_s{seed}" if per_seed else base


def matrix(cfg: dict[str, Any], data_dir_root: str,
           per_seed: bool = False) -> list[dict]:
    families = gather_families(cfg)
    splits = cfg["splits"]
    split_app = cfg.get("split_applicability", {})
    dir_map = cfg.get("split_data_dir", cfg.get("split_data_suffix", {}))
    model_configs = cfg["model_configs"]
    seeds = cfg["seeds"]
    data_scales = cfg.get("data_scales", [10000])

    combos: list[dict] = []
    for family, split, (model_name, model_cfg), seed, n_train in itertools.product(
        families, splits, model_configs.items(), seeds, data_scales,
    ):
        if family in split_app and split not in split_app[family]:
            continue
        ddir = data_dir_for(data_dir_root, family, split, dir_map, seed,
                            per_seed=per_seed)
        combos.append({
            "family": family, "split": split, "model_name": model_name,
            "model_cfg": model_cfg, "seed": seed, "n_train": n_train,
            "data_dir": ddir,
        })
    return combos


def write_auto_config(path: Path, combo: dict, training: dict,
                      overrides: dict) -> None:
    # `model_class`, `shift_aug_p`, `shift_aug_m` are top-level concerns
    # (dispatch token + training-loop augmentation). `sigma` and all
    # model-architecture hyperparams stay inside the `model:` block so
    # `create_lemo` / research-baseline factories pick them up.
    model_cfg = dict(combo["model_cfg"])
    model_class = model_cfg.pop("model_class")
    shift_aug_p = model_cfg.pop("shift_aug_p", 0.0)
    shift_aug_m = model_cfg.pop("shift_aug_m", None)

    per_class_train = overrides.get(model_class, {})
    training_flat = {**training, **per_class_train}

    cfg_out = {
        "family":       combo["family"],
        "split":        combo["split"],
        "seed":         combo["seed"],
        "n_train":      combo["n_train"],
        "model_class":  model_class,
        "model":        model_cfg,       # sigma (if any) lives in here.
        "data_dir":     combo["data_dir"],
        "use_residual": True,
        "shift_aug_p":  shift_aug_p,
        **training_flat,
    }
    if shift_aug_m is not None:
        cfg_out["shift_aug_m"] = shift_aug_m
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    sweep_name = cfg["sweep_name"]
    combos = matrix(cfg, args.data_dir, per_seed=args.per_seed)
    training = cfg["training"]
    overrides = cfg.get("training_overrides", {})
    slurm_cfg = cfg["slurm"]
    walltime = slurm_cfg.get("walltime_train", "12:00:00")
    n_gpus = slurm_cfg.get("n_gpus", 1)

    ensure_dir(PROJECT_ROOT / "slurm_logs")
    tpl = load_template("train_ddp.sbatch" if n_gpus > 1 else "train_single.sbatch")

    # Classify combos: ready vs blocked on missing data. Data layout is
    # {data_dir}/{family}/manifest.json (ID) or
    # {data_dir}/{family}/test_ood/ (OOD shards directly under family).
    ready, blocked = [], []
    for c in combos:
        dd = Path(c["data_dir"])
        base = dd if dd.is_absolute() else (PROJECT_ROOT / dd)
        fam_dir = base / c["family"]
        has_manifest = (fam_dir / "manifest.json").exists()
        has_testood = (fam_dir / "test_ood").is_dir()
        if has_manifest or has_testood:
            ready.append(c)
        else:
            blocked.append(c)

    if blocked:
        print(f"\n[skipped — data_dir missing] {len(blocked)} combos")
        by_split: dict[str, int] = {}
        for c in blocked:
            by_split[c["split"]] = by_split.get(c["split"], 0) + 1
        for split, n in sorted(by_split.items()):
            print(f"  split={split:<20} {n} combos skipped")

    print(f"\n=== Phase B training: {len(ready)} jobs ===")

    submitted: list[str] = []
    for c in ready:
        cfg_path = (
            PROJECT_ROOT / "configs" / "auto" / sweep_name /
            f"{c['family']}_{c['split']}_{c['model_name']}_s{c['seed']}.yaml"
        )
        out_dir = (
            f"outputs/{sweep_name}/{c['family']}/{c['split']}/"
            f"{c['model_name']}_s{c['seed']}"
        )

        if not args.dry_run:
            write_auto_config(cfg_path, c, training, overrides)

        filled = fill_template(tpl, {
            "family":      c["family"],
            "walltime":    walltime,
            "n_gpus":      str(n_gpus),
            "config_path": str(cfg_path.relative_to(PROJECT_ROOT)),
            "data_dir":    c["data_dir"],
            "output_dir":  out_dir,
            "seed":        str(c["seed"]),
        })

        if args.dry_run:
            print(f"-- train: {c['family']}/{c['split']} "
                  f"{c['model_name']} seed={c['seed']}")
        else:
            jid = submit_script(filled, dry=False)
            if jid:
                submitted.append(jid)
                print(f"  Submitted {c['family']}/{c['split']} "
                      f"{c['model_name']} seed={c['seed']} -> {jid}")

    if not args.dry_run and submitted:
        manifest_path = (
            PROJECT_ROOT / "slurm_logs" / f"sweep_manifest_{sweep_name}.json"
        )
        with open(manifest_path, "w") as f:
            json.dump({
                "sweep_name": sweep_name,
                "train_jobs": submitted,
                "n_ready": len(ready),
                "n_blocked": len(blocked),
            }, f, indent=2)
        print(f"\nmanifest: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sweep_config", help="Path to configs/sweep_phase_b.yaml")
    p.add_argument("--data_dir", default="data_baseline_v2",
                   help="Data root (before per-split suffix and per-seed suffix).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stage", choices=["train", "eval", "all"], default="train",
                   help="Currently only 'train' is implemented.")
    p.add_argument("--per-seed", action="store_true",
                   help="Use per-seed data dirs ({base}_s{seed}). "
                        "Default: seed-agnostic paths (matches frozen baseline).")
    args = p.parse_args()

    with open(args.sweep_config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
