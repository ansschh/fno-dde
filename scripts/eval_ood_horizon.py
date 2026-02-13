#!/usr/bin/env python3
"""
OOD-horizon evaluation: Test models trained on T=20 with T=40 data.
"""
import numpy as np
import torch
import yaml
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dde.families import HutchDDE, Linear2DDE, get_family
from dde.solve_python.dde_solver import solve_dde
from models.fno1d import FNO1d
from datasets.sharded_dataset import ShardedDDEDataset
from torch.utils.data import DataLoader


def generate_extended_horizon_samples(family, n_samples, T_extended, seed=42):
    """Generate samples with extended time horizon."""
    rng = np.random.default_rng(seed)
    
    tau_max = family.config.tau_max
    n_hist = 256
    dt_out = 0.05
    n_points = int(T_extended / dt_out) + 1
    
    t_hist = np.linspace(-tau_max, 0, n_hist)
    
    samples = []
    pbar = tqdm(total=n_samples, desc=f"Generating T={T_extended} samples")
    attempts = 0
    
    while len(samples) < n_samples and attempts < n_samples * 20:
        attempts += 1
        params = family.sample_params(rng)
        phi = family.sample_history(rng, t_hist)
        
        sol = solve_dde(family, params, phi, t_hist, T_extended, n_points=n_points)
        
        if sol.success and np.all(np.isfinite(sol.y)) and np.max(np.abs(sol.y)) <= 100.0:
            if family.config.requires_positive and np.min(sol.y) < -1e-6:
                continue
            samples.append({
                "params": params,
                "phi": phi,
                "y": sol.y,
                "t": sol.t,
            })
            pbar.update(1)
    
    pbar.close()
    return samples, t_hist


def evaluate_on_horizon(model, samples, t_hist, T_train, T_test, device, family_name, in_channels=6):
    """Evaluate model on different horizon windows."""
    model.eval()
    
    # Get input format from model
    n_hist_points = 64
    n_future_points = 192
    dt_out = 0.05
    
    # Original horizon indices (T=20)
    t_train_end_idx = int(T_train / dt_out)
    
    # Extended horizon indices
    t_test_end_idx = int(T_test / dt_out)
    
    results = {"id": [], "ood": []}
    
    for sample in tqdm(samples, desc="Evaluating"):
        phi = sample["phi"]
        y_full = sample["y"]
        t_full = sample["t"]
        
        # Downsample history
        hist_indices = np.linspace(0, len(t_hist)-1, n_hist_points, dtype=int)
        phi_down = phi[hist_indices]
        
        # ID evaluation: predict t=[0, T_train]
        out_indices_id = np.linspace(0, t_train_end_idx, n_future_points, dtype=int)
        y_id = y_full[out_indices_id]
        t_id = t_full[out_indices_id]
        
        # Build input for ID
        n_total = n_hist_points + n_future_points
        t_combined_id = np.concatenate([t_hist[hist_indices], t_id])
        
        # Simplified input matching training format
        # Channels: [phi/future_zeros, t, history_mask, ...params...]
        x_id = np.zeros((n_total, in_channels))
        x_id[:n_hist_points, 0] = phi_down.flatten()
        x_id[:, 1] = t_combined_id
        x_id[:n_hist_points, 2] = 1.0  # history mask
        
        target_id = np.zeros((n_total, 1))
        target_id[:n_hist_points, 0] = phi_down.flatten()
        target_id[n_hist_points:, 0] = y_id.flatten()
        
        # OOD evaluation: predict t=[T_train, T_test]
        out_indices_ood = np.linspace(t_train_end_idx, min(t_test_end_idx, len(y_full)-1), 
                                      n_future_points, dtype=int)
        y_ood = y_full[out_indices_ood]
        t_ood = t_full[out_indices_ood]
        
        # For OOD, use the solution at T_train as "history"
        # This tests extrapolation beyond training horizon
        hist_start_idx = max(0, t_train_end_idx - int(2.0/dt_out))  # 2s history window
        hist_indices_ood = np.linspace(hist_start_idx, t_train_end_idx, n_hist_points, dtype=int)
        phi_ood = y_full[hist_indices_ood]
        t_hist_ood = t_full[hist_indices_ood] - t_full[t_train_end_idx]  # Shift to [-2, 0]
        
        t_combined_ood = np.concatenate([t_hist_ood, t_ood - t_full[t_train_end_idx]])
        
        x_ood = np.zeros((n_total, in_channels))
        x_ood[:n_hist_points, 0] = phi_ood.flatten()
        x_ood[:, 1] = t_combined_ood
        x_ood[:n_hist_points, 2] = 1.0
        
        target_ood = np.zeros((n_total, 1))
        target_ood[:n_hist_points, 0] = phi_ood.flatten()
        target_ood[n_hist_points:, 0] = y_ood.flatten()
        
        # Evaluate
        with torch.no_grad():
            x_id_t = torch.tensor(x_id, dtype=torch.float32).unsqueeze(0).to(device)
            target_id_t = torch.tensor(target_id, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred_id = model(x_id_t)
            diff_id = pred_id[:, n_hist_points:] - target_id_t[:, n_hist_points:]
            l2_diff_id = torch.sqrt(torch.sum(diff_id**2))
            l2_target_id = torch.sqrt(torch.sum(target_id_t[:, n_hist_points:]**2))
            rel_l2_id = (l2_diff_id / (l2_target_id + 1e-8)).item()
            results["id"].append(rel_l2_id)
            
            x_ood_t = torch.tensor(x_ood, dtype=torch.float32).unsqueeze(0).to(device)
            target_ood_t = torch.tensor(target_ood, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred_ood = model(x_ood_t)
            diff_ood = pred_ood[:, n_hist_points:] - target_ood_t[:, n_hist_points:]
            l2_diff_ood = torch.sqrt(torch.sum(diff_ood**2))
            l2_target_ood = torch.sqrt(torch.sum(target_ood_t[:, n_hist_points:]**2))
            rel_l2_ood = (l2_diff_ood / (l2_target_ood + 1e-8)).item()
            results["ood"].append(rel_l2_ood)
    
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["hutch", "linear2"])
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--T_test", type=float, default=40.0)
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    
    T_train = 20.0
    
    print(f"\n{'='*60}")
    print(f"OOD-Horizon Evaluation: T={T_train} → T={args.T_test}")
    print(f"{'='*60}")
    
    # Get family
    family = get_family(args.family)
    
    # Generate extended samples
    samples, t_hist = generate_extended_horizon_samples(
        family, args.n_samples, args.T_test, seed=456
    )
    
    # Load model
    model_dir = Path(args.model_dir)
    with open(model_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    
    ckpt = torch.load(model_dir / "best_model.pt", map_location=args.device)
    
    # Detect input channels from checkpoint
    in_channels = ckpt["model_state_dict"]["lift.weight"].shape[1]
    
    model = FNO1d(
        modes=model_cfg["modes"],
        width=model_cfg["width"],
        in_channels=in_channels,
        out_channels=1,
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg.get("dropout", 0.0),
    ).to(args.device)
    
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    # Evaluate
    results = evaluate_on_horizon(
        model, samples, t_hist, T_train, args.T_test, args.device, args.family, in_channels
    )
    
    id_arr = np.array(results["id"])
    ood_arr = np.array(results["ood"])
    
    print(f"\nID (t ∈ [0, {T_train}]):")
    print(f"  median: {np.median(id_arr):.4f}")
    print(f"  p95:    {np.percentile(id_arr, 95):.4f}")
    
    print(f"\nOOD (t ∈ [{T_train}, {args.T_test}]):")
    print(f"  median: {np.median(ood_arr):.4f}")
    print(f"  p95:    {np.percentile(ood_arr, 95):.4f}")
    
    print(f"\nOOD/ID ratio: {np.median(ood_arr)/np.median(id_arr):.2f}x")


if __name__ == "__main__":
    main()
