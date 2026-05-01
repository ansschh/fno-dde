"""
APEBench multi-dataset, multi-model sweep launcher.

Spawns N parallel workers, each running one (dataset, model, regime, seed)
cell of the sweep matrix.  Distributes across `--n_gpus` GPUs.

Usage:
    python3 scripts/run_apebench_sweep.py \\
        --datasets kolmogorov_2d,burgers_1d,burgers_3d,gray_scott_3d,decaying_turbulence_2d \\
        --models lemo_pc_nd,fno_nd \\
        --regimes clean \\
        --seeds 42,123,456 \\
        --n_workers 8 --n_gpus 8 --epochs 200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from multiprocessing import Process, Queue
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def worker(worker_id, gpu_id, job_queue, result_queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Critical: cap thread spawning so 8 parallel workers don't thrash the CPU.
    # Each worker still uses GPU for the heavy lifting; CPU is just data prep.
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    env["TORCH_NUM_THREADS"] = "2"
    while True:
        job = job_queue.get()
        if job is None:
            break
        cmd = job["cmd"]
        log_path = job["log"]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        t0 = time.time()
        with open(log_path, "w") as f:
            f.write(f"[worker {worker_id} gpu {gpu_id}] $ {' '.join(cmd)}\n")
            f.flush()
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        result_queue.put({
            "worker_id": worker_id, "gpu_id": gpu_id, "name": job["name"],
            "rc": r.returncode, "wall_s": time.time() - t0, "log": log_path,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="Comma-separated APEBench dataset names")
    ap.add_argument("--models", required=True,
                    help="Comma-separated model names")
    ap.add_argument("--regimes", default="clean",
                    help="Comma-separated regimes: clean, lowres, noisy")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--lag_modes", type=int, default=12)
    ap.add_argument("--spatial_modes", type=int, default=12)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--downsample_factor", type=int, default=2)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--n_gpus", type=int, default=8)
    ap.add_argument("--data_dir", default="data_apebench")
    ap.add_argument("--output_dir", default="outputs/sweep_apebench")
    ap.add_argument("--no_skip", action="store_true")
    ap.add_argument("--sigma", type=str, default="none")
    ap.add_argument("--num_train_samples", type=int, default=None)
    ap.add_argument("--residual_anchor", action="store_true",
                    help="Enable residual-anchor data presentation in trainer.")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    models = args.models.split(",")
    regimes = args.regimes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    for fam in datasets:
        for model in models:
            for regime in regimes:
                for seed in seeds:
                    name = f"{fam}_{model}_{regime}_s{seed}"
                    out_dir = Path(args.output_dir) / fam / regime / model / f"s{seed}"
                    if (not args.no_skip) and (out_dir / "test_results.json").exists():
                        print(f"  skip {name} (already done)")
                        continue
                    cmd = [
                        "python3", "-u", "scripts/train_apebench_smoke.py",
                        "--family", fam, "--model", model,
                        "--regime", regime,
                        "--noise_std", str(args.noise_std),
                        "--downsample_factor", str(args.downsample_factor),
                        "--epochs", str(args.epochs),
                        "--batch_size", str(args.batch_size),
                        "--width", str(args.width),
                        "--n_layers", str(args.n_layers),
                        "--lag_modes", str(args.lag_modes),
                        "--spatial_modes", str(args.spatial_modes),
                        "--seed", str(seed),
                        "--data_dir", args.data_dir,
                        "--output_dir", str(Path(args.output_dir) / "raw"),
                    ]
                    if args.residual_anchor:
                        cmd.append("--residual_anchor")
                    if args.num_train_samples is not None:
                        cmd.extend(["--num_train_samples", str(args.num_train_samples)])
                    if args.sigma.lower() not in ("none","null",""):
                        cmd.extend(["--sigma", args.sigma])
                    log = str(Path(args.output_dir) / "logs" / f"{name}.log")
                    jobs.append({
                        "name": name, "cmd": cmd, "log": log,
                        "regime": regime,
                    })

    print(f"=== APEBench sweep: {len(jobs)} jobs across {args.n_workers} workers ===")
    if not jobs:
        return

    job_q, res_q = Queue(), Queue()
    for j in jobs:
        job_q.put(j)
    for _ in range(args.n_workers):
        job_q.put(None)
    workers = []
    for i in range(args.n_workers):
        gpu = i % args.n_gpus
        p = Process(target=worker, args=(i, gpu, job_q, res_q))
        p.start()
        workers.append(p)
    n_ok = n_fail = 0
    for k in range(len(jobs)):
        r = res_q.get()
        ok = r["rc"] == 0
        n_ok += int(ok); n_fail += int(not ok)
        tag = "OK" if ok else f"FAIL(rc={r['rc']})"
        print(f"  [{k+1:>4d}/{len(jobs)}] [gpu {r['gpu_id']}] {r['name']:<60s} {tag}  {r['wall_s']:.0f}s  (ok={n_ok} fail={n_fail})")
    for p in workers:
        p.join()


if __name__ == "__main__":
    main()
