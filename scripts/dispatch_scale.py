"""Manual GPU-pinned dispatcher for LEMO 3D scaling investigation.

Launches 9 cells (3 widths x 3 seeds) on burgers_3d with residual_anchor,
distributing across 8 GPUs (cell 8 shares GPU 0 with cell 0 via natural
process scheduling).
"""
import os, subprocess, time, sys

WIDTHS = [48, 64, 96]
SEEDS = [42, 123, 456]
launched = []

for w_idx, width in enumerate(WIDTHS):
    for s_idx, seed in enumerate(SEEDS):
        cell_idx = w_idx * len(SEEDS) + s_idx
        gpu = cell_idx % 8
        outdir = f"outputs/sweep_lemo_scale/raw_w{width}"
        log = f"outputs/sweep_lemo_scale/logs/burgers3d_lemo_w{width}_s{seed}.log"
        os.makedirs(os.path.dirname(log), exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        cmd = [
            "python3", "-u", "scripts/train_apebench_smoke.py",
            "--family", "burgers_3d", "--model", "lemo_pc_nd",
            "--regime", "clean", "--epochs", "200",
            "--batch_size", "4", "--width", str(width),
            "--n_layers", "3", "--lag_modes", "12",
            "--spatial_modes", "12", "--seed", str(seed),
            "--residual_anchor", "--output_dir", outdir,
        ]
        with open(log, "w") as f:
            f.write("# " + " ".join(cmd) + "\n")
            f.write(f"# GPU={gpu}\n")
        log_fd = open(log, "a", buffering=1)
        p = subprocess.Popen(
            cmd, stdout=log_fd, stderr=subprocess.STDOUT,
            env=env, close_fds=True,
        )
        launched.append((p.pid, width, seed, gpu))
        print(f"launched: PID={p.pid} w={width} s={seed} gpu={gpu}", flush=True)
        time.sleep(2)

print(f"--- {len(launched)} jobs launched ---", flush=True)
