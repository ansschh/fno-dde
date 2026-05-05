"""Unified per-GPU worker — runs all post-hoc evals on a shard of cells.

For each checkpoint in shard:
  1. capture --minimal: per_frame.json + viz_samples.npz + kernel_snapshot.npz + residuals.npz
  2. eval_w1_empirical_lipschitz: empirical_lipschitz.json
  3. eval_equivariance_dense: equivariance_dense.json
  4. eval_adversarial_dense: adversarial_dense.json
  5. eval_noise_dense: noise_dense.json

Each step is idempotent — skips if output JSON already exists.

Usage:
  CUDA_VISIBLE_DEVICES=$g python3 scripts/_pod_unified_worker.py \\
    --shard shard_$g.txt --data_dir data_dde_pde --gpu $g
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def parse_path(ckpt_path):
    """Extract (family, regime, model, seed) from path."""
    parts = ckpt_path.parts
    try:
        idx = parts.index("raw")
    except ValueError:
        return None
    if idx + 4 >= len(parts):
        return None
    fam, reg, mdl, seed_str = parts[idx + 1: idx + 5]
    if not seed_str.startswith("s"):
        return None
    return fam, reg, mdl, int(seed_str[1:])


def run_capture_minimal(ckpt, data_dir, fam, device):
    from capture_paper_artifacts import process_cell
    return process_cell(ckpt, data_dir, fam, device, n_viz=4, minimal=True)


def run_w1_lipschitz(ckpt, data_dir, fam, device, n_pairs=100):
    """Run eval_w1_empirical_lipschitz logic inline for one cell."""
    out_path = ckpt.parent / "empirical_lipschitz.json"
    if out_path.exists():
        return "skip (already done)"
    from eval_w1_empirical_lipschitz import (load_cell, empirical_lipschitz,
                                              get_sigma_target)
    meta = parse_path(ckpt)
    if meta is None:
        return "FAIL bad path"
    fam, reg, mdl, seed = meta
    try:
        t0 = time.time()
        model, test_loader, cfg = load_cell(ckpt, data_dir, fam, device)
        n_layers = int(cfg.get("model", {}).get("n_layers",
                       cfg.get("n_layers", 3)))
        sigma = get_sigma_target(cfg)
        eta_cert = sigma ** (n_layers + 1) if sigma is not None else None
        stats = empirical_lipschitz(model, test_loader, device, n_pairs=n_pairs)
        if stats is None:
            return "FAIL no data"
        stats.update({
            "family": fam, "regime": reg, "model": mdl, "seed": seed,
            "D": n_layers, "sigma_target": sigma,
            "eta_certified": eta_cert,
            "tightness_ratio": (stats["L_emp_p95"] / eta_cert
                                if eta_cert and eta_cert > 0 else None),
            "elapsed_s": time.time() - t0,
        })
        out_path.write_text(json.dumps(stats, indent=2))
        return f"ok L_p95={stats['L_emp_p95']:.4f} eta_c={eta_cert} t={stats['elapsed_s']:.1f}s"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {e}"


def run_dense_eval(eval_module_name, output_filename, ckpt, data_dir, device,
                   load_cell_fn, eval_fn, **eval_kwargs):
    out_path = ckpt.parent / output_filename
    if out_path.exists():
        return "skip"
    meta = parse_path(ckpt)
    if meta is None:
        return "FAIL bad path"
    fam, reg, mdl, seed = meta
    try:
        t0 = time.time()
        model, test_loader, cfg = load_cell_fn(ckpt, data_dir, fam, device)
        result = eval_fn(model, test_loader, device, **eval_kwargs)
        if result is None:
            return "FAIL no data"
        if isinstance(result, dict):
            payload = {**result,
                       "family": fam, "regime": reg, "model": mdl,
                       "seed": seed, "elapsed_s": time.time() - t0}
            out_path.write_text(json.dumps(payload, indent=2))
        return f"ok t={time.time()-t0:.1f}s"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {e}"


def run_equivariance(ckpt, data_dir, device):
    out_path = ckpt.parent / "equivariance_dense.json"
    if out_path.exists():
        return "skip"
    meta = parse_path(ckpt)
    if meta is None:
        return "FAIL bad path"
    fam, reg, mdl, seed = meta
    try:
        t0 = time.time()
        from eval_w1_empirical_lipschitz import load_cell
        from eval_equivariance_dense import compute_e_orbit
        model, test_loader, cfg = load_cell(ckpt, data_dir, fam, device)
        n_state = test_loader.dataset[0]["target"].shape[-1] if hasattr(test_loader.dataset, '__getitem__') else 1
        shifts = [1, 2, 4, 8, 16, 32, 64]
        result = compute_e_orbit(model, test_loader, device, shifts,
                                  n_state_channels=n_state, n_batches=8)
        out = {"family": fam, "regime": reg, "model": mdl, "seed": seed,
               "shifts": shifts, "e_per_shift": result,
               "elapsed_s": time.time() - t0}
        out_path.write_text(json.dumps(out, indent=2, default=str))
        return f"ok t={time.time()-t0:.1f}s"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {e}"


def run_per_frame(ckpt, data_dir, device):
    out_path = ckpt.parent / "per_frame.json"
    if out_path.exists():
        return "skip (already done by capture)"
    meta = parse_path(ckpt)
    if meta is None:
        return "FAIL bad path"
    fam, reg, mdl, seed = meta
    try:
        t0 = time.time()
        from eval_per_frame_dense import evaluate_checkpoint
        result = evaluate_checkpoint(ckpt, data_dir, device, n_max_samples=64)
        if result is None:
            return "FAIL no data"
        out_path.write_text(json.dumps(result, indent=2))
        return f"ok t={time.time()-t0:.1f}s"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="Text file with one ckpt path per line")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0, help="Logical GPU id (for log prefix)")
    ap.add_argument("--steps", default="capture,lipschitz,equivariance",
                    help="Comma-separated steps to run")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(f"[gpu {args.gpu}] WARNING: CUDA not visible, running on CPU", flush=True)

    steps = set(args.steps.split(","))
    paths = [Path(line.strip()) for line in open(args.shard) if line.strip()]
    print(f"[gpu {args.gpu}] {len(paths)} cells in shard, steps={sorted(steps)}", flush=True)

    n_done = n_skip = n_fail = 0
    for i, ckpt in enumerate(paths, 1):
        meta = parse_path(ckpt)
        if meta is None:
            print(f"[gpu {args.gpu}] [{i}/{len(paths)}] SKIP unparseable: {ckpt}", flush=True)
            continue
        fam, reg, mdl, seed = meta
        cell_label = f"{fam}/{reg}/{mdl}/s{seed}"

        # Step 1: capture --minimal (per_frame.json + viz_samples.npz)
        if "capture" in steps:
            msg = run_capture_minimal(ckpt, args.data_dir, fam, device)
            print(f"[gpu {args.gpu}] [{i}/{len(paths)}] capture {cell_label}: {msg}", flush=True)
            if "FAIL" in msg:
                n_fail += 1
                continue
            elif "skip" not in msg:
                n_done += 1
            else:
                n_skip += 1

        # Step 2: empirical Lipschitz (W1-E2)
        if "lipschitz" in steps:
            msg = run_w1_lipschitz(ckpt, args.data_dir, fam, device, n_pairs=80)
            print(f"[gpu {args.gpu}] [{i}/{len(paths)}] lipschitz {cell_label}: {msg}", flush=True)

        # Step 3: equivariance dense
        if "equivariance" in steps:
            msg = run_equivariance(ckpt, args.data_dir, device)
            print(f"[gpu {args.gpu}] [{i}/{len(paths)}] equivariance {cell_label}: {msg}", flush=True)

        # Step 4: per_frame dense (only if capture didn't already do it)
        if "per_frame" in steps:
            msg = run_per_frame(ckpt, args.data_dir, device)
            print(f"[gpu {args.gpu}] [{i}/{len(paths)}] per_frame {cell_label}: {msg}", flush=True)

    print(f"[gpu {args.gpu}] DONE done={n_done} skip={n_skip} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
