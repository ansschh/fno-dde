"""W1-E2: Empirical Euclidean Lipschitz constant of trained Φ.

For each B4 σ-sweep checkpoint (or any model checkpoint), sample K random
input pairs (H, H̃) from the test loader, compute the ratio
    L_emp = max_{(H, H̃)} ‖Φ(H) - Φ(H̃)‖_2 / ‖H - H̃‖_2

and compare to the certified bound η = σ^{D+1}.

Writes `empirical_lipschitz.json` next to each best_model.pt with:
  - L_emp_max, L_emp_p95, L_emp_p99, L_emp_mean over K random pairs
  - L_emp_finite_diff: 0-th order central-difference Lipschitz at K random
    interior points (additional check)
  - sigma_target: the σ from cell config (None if unconstrained)
  - eta_certified: σ^{D+1} where D = n_layers
  - tightness_ratio: L_emp_p95 / eta_certified

Usage:
  python scripts/eval_w1_empirical_lipschitz.py \\
    --layer_root extracted/pod_pulls_2026_05_03_final/Pod1_h100/outputs/sigma_0.5_runpod \\
    --data_dir extracted/244_FULL_pull/workspace/dde-fno/data_dde_pde \\
    --n_pairs 100 --device cuda
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def load_cell(ckpt_path: Path, data_dir: str, family: str, device: str):
    """Load model + test loader from checkpoint."""
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    parts = ckpt_path.parts
    regime = cfg.get("regime", parts[-4] if len(parts) >= 4 else "clean")
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    ra = bool(cfg.get("residual_anchor", False))
    _, _, test_loader = create_apebench_dataloaders(
        data_dir, family, batch_size=4,
        regime=regime, noise_std=noise_std,
        downsample_factor=downsample_factor,
        residual_anchor=ra, seed=42,
    )
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, test_loader, cfg


def empirical_lipschitz(model, test_loader, device, n_pairs=100):
    """Estimate empirical Lipschitz of Φ in Euclidean norm via random pairs."""
    model.eval()
    ratios = []
    finite_diff_norms = []
    inputs_pool = []
    with torch.no_grad():
        for batch in test_loader:
            inputs_pool.append(batch["input"].to(device).float())
            if sum(x.shape[0] for x in inputs_pool) >= n_pairs * 2:
                break
    if not inputs_pool:
        return None
    H = torch.cat(inputs_pool, dim=0)  # (N, T, *spatial, C)
    N = H.shape[0]
    pair_idx = torch.randint(0, N, (n_pairs, 2), device=device)
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    if pair_idx.shape[0] == 0:
        return None
    with torch.no_grad():
        for i, j in pair_idx:
            h1 = H[i:i+1]
            h2 = H[j:j+1]
            phi1 = model(h1)
            phi2 = model(h2)
            num = torch.linalg.vector_norm((phi1 - phi2).reshape(1, -1), dim=1)
            den = torch.linalg.vector_norm((h1 - h2).reshape(1, -1), dim=1)
            ratios.append(float(num / den.clamp_min(1e-12)))
        # Finite-difference Lipschitz at random interior points: small perturbations.
        n_fd = min(50, N)
        idx = torch.randperm(N, device=device)[:n_fd]
        for k in idx:
            h = H[k:k+1]
            eps = 1e-3 * torch.randn_like(h)
            phi_h = model(h)
            phi_h_eps = model(h + eps)
            num = torch.linalg.vector_norm((phi_h_eps - phi_h).reshape(1, -1), dim=1)
            den = torch.linalg.vector_norm(eps.reshape(1, -1), dim=1)
            finite_diff_norms.append(float(num / den.clamp_min(1e-12)))
    arr = np.array(ratios, dtype=np.float64)
    fd_arr = np.array(finite_diff_norms, dtype=np.float64) if finite_diff_norms else np.array([np.nan])
    return {
        "n_pairs": int(arr.size),
        "L_emp_max": float(arr.max()),
        "L_emp_p99": float(np.percentile(arr, 99)),
        "L_emp_p95": float(np.percentile(arr, 95)),
        "L_emp_mean": float(arr.mean()),
        "L_emp_median": float(np.median(arr)),
        "L_emp_min": float(arr.min()),
        "n_finite_diff": int(fd_arr.size),
        "L_fd_max": float(np.nanmax(fd_arr)),
        "L_fd_p95": float(np.nanpercentile(fd_arr, 95)),
        "L_fd_mean": float(np.nanmean(fd_arr)),
        "ratios": arr.tolist(),
    }


def parse_path(ckpt_path: Path):
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


def get_sigma_target(cfg: dict) -> float | None:
    """Extract σ target from cell config, if any."""
    sigma = cfg.get("sigma")
    if sigma is not None:
        return float(sigma)
    sigma = cfg.get("model", {}).get("sigma")
    return float(sigma) if sigma is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_root", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"[w1-e2] cuda unavailable, falling back to {device}")

    layer_root = Path(args.layer_root)
    if not layer_root.is_absolute():
        layer_root = REPO / layer_root
    ckpts = sorted(layer_root.glob("**/best_model.pt"))
    print(f"[w1-e2] {len(ckpts)} ckpts under {layer_root}")

    n_done = n_skip = n_fail = 0
    for ckpt in ckpts:
        out_path = ckpt.parent / "empirical_lipschitz.json"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        meta = parse_path(ckpt)
        if meta is None:
            print(f"  SKIP unparseable: {ckpt}")
            continue
        fam, reg, mdl, seed = meta
        try:
            t0 = time.time()
            model, test_loader, cfg = load_cell(ckpt, args.data_dir, fam, device)
            n_layers = int(cfg.get("model", {}).get("n_layers", cfg.get("n_layers", 3)))
            D = n_layers
            sigma = get_sigma_target(cfg)
            eta_cert = sigma ** (D + 1) if sigma is not None else None
            stats = empirical_lipschitz(model, test_loader, device, n_pairs=args.n_pairs)
            if stats is None:
                print(f"  SKIP no data: {ckpt}")
                n_fail += 1
                continue
            stats["family"] = fam
            stats["regime"] = reg
            stats["model"] = mdl
            stats["seed"] = seed
            stats["D"] = D
            stats["sigma_target"] = sigma
            stats["eta_certified"] = eta_cert
            stats["tightness_ratio"] = (
                stats["L_emp_p95"] / eta_cert if eta_cert and eta_cert > 0 else None
            )
            stats["elapsed_s"] = time.time() - t0
            with open(out_path, "w") as fh:
                json.dump(stats, fh, indent=2)
            n_done += 1
            print(f"  OK {fam}/{reg}/{mdl}/s{seed} σ={sigma} "
                  f"L_emp_p95={stats['L_emp_p95']:.4f} "
                  f"η_cert={eta_cert} tight={stats['tightness_ratio']} "
                  f"t={stats['elapsed_s']:.1f}s")
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {ckpt}: {e}")

    print(f"[w1-e2] done={n_done} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
