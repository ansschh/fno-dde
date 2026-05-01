#!/usr/bin/env python3
"""
Phase A runner.

Enumerates the phase_a_theorem_suite_v1 matrix (3 datasets × 8 models
× 5 seeds = 120 runs), auto-generates per-run configs, and dispatches
them across the pod's GPUs with `CUDA_VISIBLE_DEVICES` pinning.
N_WORKERS-way parallelism, each worker pulls from a shared queue and
runs `src/train/train_fno_sharded.py` as a subprocess.

The pod has SLURM binaries installed but no controller daemon running,
so we bypass SLURM entirely and rely on subprocess + multiprocessing.

Usage:
    python3 scripts/run_phase_a.py configs/sweep_phase_a.yaml \\
        --n_workers 8 --device cuda

    # Dry run (print configs, do not execute):
    python3 scripts/run_phase_a.py configs/sweep_phase_a.yaml --dry-run
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


def enumerate_matrix(cfg: dict[str, Any]) -> list[dict]:
    datasets = cfg["datasets"]
    seeds = cfg["seeds"]
    models = cfg["model_configs"]
    combos: list[dict] = []
    for (ds_key, ds_info), seed, (model_name, model_cfg) in itertools.product(
        datasets.items(), seeds, models.items()
    ):
        combos.append({
            "dataset_key": ds_key,
            "family":      ds_info["family"],
            "data_dir":    ds_info["data_dir"],
            "seed":        seed,
            "model_name":  model_name,
            "model_cfg":   model_cfg,
        })
    return combos


def write_auto_config(
    path: Path, combo: dict, training: dict, sweep_name: str,
) -> None:
    # `model_class`, `shift_aug_p`, `shift_aug_m` are top-level concerns
    # (dispatch token + training-loop augmentation). `sigma` and all
    # model-architecture hyperparams stay inside the `model:` block so
    # `create_lemo` / research-baseline factories see them.
    model_cfg = dict(combo["model_cfg"])
    model_class = model_cfg.pop("model_class")
    shift_aug_p = model_cfg.pop("shift_aug_p", 0.0)
    shift_aug_m = model_cfg.pop("shift_aug_m", None)

    cfg_out = {
        "family":       combo["family"],
        "seed":         combo["seed"],
        "model_class":  model_class,
        "model":        model_cfg,       # sigma (if any) lives in here.
        "data_dir":     combo["data_dir"],
        "use_residual": True,
        "shift_aug_p":  shift_aug_p,
        **training,
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
    combos = enumerate_matrix(cfg)

    # Build job list.
    jobs: list[dict] = []
    for c in combos:
        run_name = f"{c['dataset_key']}_{c['model_name']}_s{c['seed']}"
        config_path = (
            PROJECT_ROOT / "configs" / "auto" / sweep_name / f"{run_name}.yaml"
        )
        output_dir = f"outputs/{sweep_name}/{c['dataset_key']}/{c['model_name']}_s{c['seed']}"
        log_path = (
            PROJECT_ROOT / "outputs" / sweep_name / "logs" / f"{run_name}.log"
        )
        if not args.dry_run:
            write_auto_config(config_path, c, training, sweep_name)
        jobs.append({
            "run_name":    run_name,
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "data_dir":    c["data_dir"],
            "output_dir":  output_dir,
            "seed":        c["seed"],
            "log_path":    str(log_path),
        })

    print(f"=== Phase A: {len(jobs)} runs across {args.n_workers} workers ===")
    if args.dry_run:
        for j in jobs[:5]:
            print(f"  {j['run_name']}  config={j['config_path']}")
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
    for _ in range(len(jobs)):
        r = result_queue.get()
        results.append(r)
        status = "OK" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  [gpu {r['gpu_id']}] {r['run_name']:<40} {status}  {r['wall_s']:.1f}s")

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
            "ok":         sum(1 for r in results if r["rc"] == 0),
            "failed":     sum(1 for r in results if r["rc"] != 0),
            "results":    results,
        }, f, indent=2)
    print(f"\nmanifest: {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep_config", help="Path to configs/sweep_phase_a.yaml")
    p.add_argument("--n_workers", type=int, default=8,
                   help="Parallel training workers. Default: 8.")
    p.add_argument("--n_gpus", type=int, default=8,
                   help="Number of GPUs on the pod. Default: 8.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_cfg(args.sweep_config)
    run(cfg, args)


if __name__ == "__main__":
    main()
