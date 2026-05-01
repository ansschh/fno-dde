#!/usr/bin/env python3
"""
Phase B runner.

Adapts scripts/run_phase_a.py to the Phase B sweep matrix:
    families x splits x models x seeds, with per-family split_applicability
    and per-model training_overrides.

Reads configs/sweep_phase_b.yaml. Skips:
  - (family, split) pairs not in split_applicability[family]
  - (family, split) pairs whose data dir is missing
  - already-completed runs (test_results.json present in output_dir)

Dispatches `--n_workers` parallel workers across `--n_gpus` H100s with
CUDA_VISIBLE_DEVICES pinning. Each worker pulls from a shared queue
and runs `src/train/train_fno_sharded.py` as a subprocess.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def family_list(cfg: dict[str, Any]) -> list[str]:
    fams = cfg["families"]
    if isinstance(fams, dict):
        out: list[str] = []
        for v in fams.values():
            out.extend(v)
        return out
    return list(fams)


def split_data_dir_for(family: str, split: str, cfg: dict[str, Any],
                        id_data_dir: str) -> str | None:
    """Return the absolute / repo-relative dir for (family, split), or
    None if the data is not present (signals 'skip this combo')."""
    split_data_dir = cfg.get("split_data_dir", {})
    if split == "id":
        # id uses the family's manifest under the given --data_dir base.
        path = Path(id_data_dir) / family / "manifest.json"
        return id_data_dir if path.exists() else None
    base = split_data_dir.get(split, "")
    if not base:
        return None
    # OOD splits live under base/family/test_ood/manifest.json typically;
    # here we pass `base` to the trainer and let the dataset loader pick
    # the family + test_ood subdir up via the family arg. Skip if base/family
    # is missing.
    fam_dir = Path(base) / family
    return base if fam_dir.exists() else None


def enumerate_matrix(cfg: dict[str, Any], id_data_dir: str) -> list[dict]:
    fams = family_list(cfg)
    seeds = cfg["seeds"]
    splits = cfg["splits"]
    applic = cfg.get("split_applicability", {})
    models = cfg["model_configs"]

    combos: list[dict] = []
    for family in fams:
        applicable = applic.get(family, splits)
        for split in splits:
            if split not in applicable:
                continue
            data_dir = split_data_dir_for(family, split, cfg, id_data_dir)
            if data_dir is None:
                continue
            for seed in seeds:
                for model_name, model_cfg in models.items():
                    combos.append({
                        "family":     family,
                        "split":      split,
                        "data_dir":   data_dir,
                        "seed":       seed,
                        "model_name": model_name,
                        "model_cfg":  model_cfg,
                    })
    return combos


def write_auto_config(
    path: Path, combo: dict, training: dict, training_overrides: dict,
    sweep_name: str,
) -> None:
    model_cfg = dict(combo["model_cfg"])
    model_class = model_cfg.pop("model_class")
    shift_aug_p = model_cfg.pop("shift_aug_p", 0.0)
    shift_aug_m = model_cfg.pop("shift_aug_m", None)

    # Apply per-model-class training overrides (e.g. mfno: lr=5e-4).
    eff_training = dict(training)
    if model_class in training_overrides:
        eff_training.update(training_overrides[model_class])

    cfg_out = {
        "family":       combo["family"],
        "split":        combo["split"],
        "seed":         combo["seed"],
        "model_class":  model_class,
        "model":        model_cfg,
        "data_dir":     combo["data_dir"],
        "use_residual": True,
        "shift_aug_p":  shift_aug_p,
        **eff_training,
    }
    if shift_aug_m is not None:
        cfg_out["shift_aug_m"] = shift_aug_m
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)


def worker(worker_id: int, gpu_id: int, job_queue: Queue, result_queue: Queue,
           project_root: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        job = job_queue.get()
        if job is None:
            break
        config_path = job["config_path"]
        output_dir = job["output_dir"]
        log_path = job["log_path"]

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        t0 = time.time()
        cmd = [
            "python3", "src/train/train_fno_sharded.py",
            f"--config={config_path}",
            f"--data_dir={job['data_dir']}",
            f"--output_dir={output_dir}",
            "--device=cuda",
            f"--seed={job['seed']}",
        ]
        with open(log_path, "w") as f:
            f.write(f"[worker {worker_id} gpu {gpu_id}] $ {' '.join(cmd)}\n")
            f.flush()
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                     cwd=project_root, env=env)
        elapsed = time.time() - t0
        result_queue.put({
            "worker_id": worker_id, "gpu_id": gpu_id,
            "run_name": job["run_name"], "rc": result.returncode,
            "wall_s": elapsed, "log_path": log_path,
        })


def run(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    sweep_name = cfg["sweep_name"]
    training = cfg["training"]
    training_overrides = cfg.get("training_overrides", {})
    combos = enumerate_matrix(cfg, args.id_data_dir)

    # Build job list.
    jobs: list[dict] = []
    for c in combos:
        run_name = (f"{c['family']}_{c['split']}_{c['model_name']}_s{c['seed']}")
        config_path = (
            PROJECT_ROOT / "configs" / "auto" / sweep_name / f"{run_name}.yaml"
        )
        output_dir = (
            f"outputs/{sweep_name}/{c['family']}/{c['split']}/"
            f"{c['model_name']}_s{c['seed']}"
        )
        log_path = (
            PROJECT_ROOT / "outputs" / sweep_name / "logs" / f"{run_name}.log"
        )
        if not args.dry_run:
            write_auto_config(config_path, c, training, training_overrides,
                              sweep_name)
        jobs.append({
            "run_name":    run_name,
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "data_dir":    c["data_dir"],
            "output_dir":  output_dir,
            "seed":        c["seed"],
            "log_path":    str(log_path),
        })

    # Skip already-completed runs.
    if not args.no_skip:
        before = len(jobs)
        jobs = [j for j in jobs
                if not list(Path(j["output_dir"]).rglob("test_results.json"))]
        print(f"  skipping {before - len(jobs)} already-completed runs "
              f"({len(jobs)} remain)")

    print(f"=== Phase B: {len(jobs)} runs across {args.n_workers} workers ===")
    if args.dry_run:
        for j in jobs[:8]:
            print(f"  {j['run_name']}  config={j['config_path']}")
        if len(jobs) > 8:
            print(f"  ... ({len(jobs)} jobs total)")
        return

    job_queue: Queue = Queue()
    result_queue: Queue = Queue()
    for j in jobs:
        job_queue.put(j)
    for _ in range(args.n_workers):
        job_queue.put(None)

    workers = []
    for i in range(args.n_workers):
        gpu_id = i % args.n_gpus
        p = Process(target=worker, args=(i, gpu_id, job_queue, result_queue,
                                          str(PROJECT_ROOT)))
        p.start()
        workers.append(p)

    results = []
    n_ok = 0
    n_fail = 0
    for k in range(len(jobs)):
        r = result_queue.get()
        results.append(r)
        ok = r["rc"] == 0
        n_ok += int(ok)
        n_fail += int(not ok)
        status = "OK" if ok else f"FAIL(rc={r['rc']})"
        print(f"  [{k+1:>4d}/{len(jobs)}] [gpu {r['gpu_id']}] "
              f"{r['run_name']:<60} {status}  {r['wall_s']:.1f}s   "
              f"(ok={n_ok} fail={n_fail})", flush=True)

    for p in workers:
        p.join()

    manifest_path = (
        PROJECT_ROOT / "outputs" / sweep_name / "run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "sweep_name": sweep_name,
            "total_runs": len(jobs),
            "ok":         n_ok,
            "failed":     n_fail,
            "results":    results,
        }, f, indent=2)
    print(f"\nmanifest: {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep_config", help="Path to configs/sweep_phase_b.yaml")
    p.add_argument("--id_data_dir", default="data_baseline_v2",
                   help="Base directory holding id-split data, e.g. "
                        "data_baseline_v2/{family}/manifest.json.")
    p.add_argument("--n_workers", type=int, default=32,
                   help="Parallel training workers. Default: 32.")
    p.add_argument("--n_gpus", type=int, default=8,
                   help="Number of GPUs on the pod. Default: 8.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip", action="store_true",
                   help="Re-run completed jobs.")
    args = p.parse_args()
    cfg = load_cfg(args.sweep_config)
    run(cfg, args)


if __name__ == "__main__":
    main()
