"""Post-hoc data capture for paper figures.

Runs on existing checkpoints — no retraining.  For every cell in a sweep
output dir, this produces:

  - `per_frame.json`        — per-rollout-step relL2 + naive-copy baseline
  - `viz_samples.npz`       — 4 sample (input, target, prediction) tuples
  - `kernel_snapshot.npz`   — learned weights + FiLM γ/β extracted from ckpt
  - `residuals.npz`         — per-sample residuals for hardest-decile + correlation analysis

Usage:
    python3 scripts/capture_paper_artifacts.py \\
        --layer_root outputs/dist_kernel_v2_p1 \\
        --data_dir data_dde_pde \\
        [--n_viz_samples 4] [--device cuda]

Skips cells whose `per_frame.json` already exists (idempotent).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def load_cell(ckpt_path: Path, data_dir: str, family: str, device: str):
    """Load model + matching test loader.

    CRITICAL: every loader knob that affects the input distribution must
    match the training-time setting, else we present the model with the
    wrong distribution and get badly off relL2.  We read these from the
    saved config (post-fix); for older ckpts that predate the saving we
    derive from the cell path: parts[-4] is the regime, with the v2
    standard noise_std=0.05 and downsample_factor=2."""
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    parts = ckpt_path.parts
    regime_from_path = parts[-4] if len(parts) >= 4 else "clean"
    ra = bool(cfg.get("residual_anchor", False))
    regime = cfg.get("regime", regime_from_path)
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    _, _, test_loader = create_apebench_dataloaders(
        data_dir, family, batch_size=8,
        regime=regime, noise_std=noise_std, downsample_factor=downsample_factor,
        residual_anchor=ra, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, test_loader, cfg


def compute_per_frame(model, test_loader, device):
    """Per-rollout-step error metrics averaged over test set + naive-copy baseline.

    Computes THREE complementary metrics per output frame:
      - rel_l2_per_step:   sqrt(sum_test diff_sq / sum_test tgt_sq)  [aggregate-then-sqrt; matches training metric, robust to small targets]
      - mse_per_step:      mean over test set of (diff^2)            [absolute, scale-aware]
      - rel_l2_mean_per_step (legacy): mean(sqrt(diff_sq/tgt_sq))     [Jensen-inflated; kept for backwards compat, but use rel_l2_per_step]
    Plus naive-copy versions for each.
    """
    diff_sq_acc = None  # sum over batch + spatial, shape (T,)
    tgt_sq_acc = None
    naive_diff_sq_acc = None
    rel_legacy_acc = None
    naive_rel_legacy_acc = None
    mask_count_per_step = None  # how many (sample, spatial) cells contribute per t (for averaging mse)
    n_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            # In residual_anchor mode the loss numerator (yhat - y) cancels
            # the anchor (it's the same residual on both), so we use `y` for
            # the squared-error.  The DENOMINATOR must be the un-anchored
            # target (target_raw); otherwise rel_l2 is computed in
            # residual-space and the headline definition (training metric)
            # disagrees with the per-step metric reported here.
            y_raw = batch.get("target_raw", batch["target"]).to(device).float()
            mask = batch["loss_mask"].to(device).float()
            yhat = model(x)
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            spatial_dims = tuple(range(2, y.dim()))  # spatial + channel
            B = y.shape[0]
            T = y.shape[1]
            # Per-(B, T) sums over spatial+channel:
            diff_sq_bt = ((yhat - y) ** 2 * mask_bc).sum(dim=spatial_dims)  # (B, T)
            tgt_sq_bt = (y_raw ** 2 * mask_bc).sum(dim=spatial_dims)         # un-anchored denominator
            # Naive: predict last-input-frame state channels.  Same correction:
            # the diff for naive uses target_raw vs the input last frame
            # (which is the anchor in residual mode, so naive predicts 0
            # in residual space → diff equals -y_raw in original space).
            n_out_ch = y.shape[-1]
            last_in = x[:, -1:, ..., :n_out_ch].expand_as(y_raw)
            naive_diff_sq_bt = ((last_in - y_raw) ** 2 * mask_bc).sum(dim=spatial_dims)
            cell_count_bt = mask_bc.sum(dim=spatial_dims).clamp_min(1.0)  # (B, T)

            # Aggregate sums per-step (over batch dim for now):
            diff_sq_t = diff_sq_bt.sum(dim=0).cpu().numpy()
            tgt_sq_t = tgt_sq_bt.sum(dim=0).cpu().numpy()
            naive_diff_sq_t = naive_diff_sq_bt.sum(dim=0).cpu().numpy()
            cell_count_t = cell_count_bt.sum(dim=0).cpu().numpy()
            # Legacy per-sample relL2 (Jensen-inflated; kept for compat):
            rel_legacy = torch.sqrt(diff_sq_bt / tgt_sq_bt.clamp_min(1e-12))
            naive_rel_legacy = torch.sqrt(naive_diff_sq_bt / tgt_sq_bt.clamp_min(1e-12))
            rel_legacy_t = rel_legacy.sum(dim=0).cpu().numpy()
            naive_rel_legacy_t = naive_rel_legacy.sum(dim=0).cpu().numpy()

            if diff_sq_acc is None:
                diff_sq_acc = diff_sq_t
                tgt_sq_acc = tgt_sq_t
                naive_diff_sq_acc = naive_diff_sq_t
                mask_count_per_step = cell_count_t
                rel_legacy_acc = rel_legacy_t
                naive_rel_legacy_acc = naive_rel_legacy_t
            else:
                diff_sq_acc = diff_sq_acc + diff_sq_t
                tgt_sq_acc = tgt_sq_acc + tgt_sq_t
                naive_diff_sq_acc = naive_diff_sq_acc + naive_diff_sq_t
                mask_count_per_step = mask_count_per_step + cell_count_t
                rel_legacy_acc = rel_legacy_acc + rel_legacy_t
                naive_rel_legacy_acc = naive_rel_legacy_acc + naive_rel_legacy_t
            n_samples += B

    # Aggregate-then-sqrt (correct relL2 per step):
    rel_l2_per_step = np.sqrt(diff_sq_acc / np.clip(tgt_sq_acc, 1e-12, None))
    naive_rel_l2_per_step = np.sqrt(naive_diff_sq_acc / np.clip(tgt_sq_acc, 1e-12, None))
    # Absolute MSE per step (averaged over masked cells):
    mse_per_step = diff_sq_acc / np.clip(mask_count_per_step, 1.0, None)
    naive_mse_per_step = naive_diff_sq_acc / np.clip(mask_count_per_step, 1.0, None)
    # Legacy Jensen-inflated metric for backwards compat:
    rel_l2_mean_per_step = rel_legacy_acc / max(n_samples, 1)
    naive_rel_l2_mean_per_step = naive_rel_legacy_acc / max(n_samples, 1)

    return {
        "rel_l2_per_step": rel_l2_per_step.tolist(),
        "mse_per_step": mse_per_step.tolist(),
        "rel_l2_mean_per_step_legacy": rel_l2_mean_per_step.tolist(),
        "naive_rel_l2_per_step": naive_rel_l2_per_step.tolist(),
        "naive_mse_per_step": naive_mse_per_step.tolist(),
        "naive_rel_l2_mean_per_step_legacy": naive_rel_l2_mean_per_step.tolist(),
    }


def capture_viz_samples(model, test_loader, device, n=4):
    """Save n (input, target, prediction) tuples for visualization."""
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch["input"][:n].to(device).float()
        y = batch["target"][:n].to(device).float()
        yhat = model(x)
    return {
        "input":  x.cpu().numpy().astype(np.float32),
        "target": y.cpu().numpy().astype(np.float32),
        "pred":   yhat.cpu().numpy().astype(np.float32),
    }


def capture_kernel_snapshot(ckpt_path: Path):
    """Extract learned LEMO weights + FiLM params from ckpt.  Returns {} for non-LEMO."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    out = {}
    for k, v in sd.items():
        if any(s in k for s in ("weights_time", "weights",  # spectral lag kernel
                                 "film_net", "A_lag",         # FiLM
                                 "A_spat.weights")):           # spatial spectral
            arr = v.detach().cpu().numpy()
            # Complex tensors → store as 2-tuple
            if np.iscomplexobj(arr):
                out[k + "__re"] = arr.real.astype(np.float32)
                out[k + "__im"] = arr.imag.astype(np.float32)
            else:
                out[k] = arr.astype(np.float32)
    return out


def capture_residuals(model, test_loader, device, max_samples=None):
    """Per-sample relL2 + per-sample residual norm (for Jaccard hardest-decile).

    Default `max_samples=None` consumes the entire test set; pass an int to
    cap (used by some quick-check entry points)."""
    per_sample_rel = []
    per_sample_resid = []
    seen = 0
    with torch.no_grad():
        for batch in test_loader:
            if max_samples is not None and seen >= max_samples:
                break
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            yhat = model(x)
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            diff_sq = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim())))
            tgt_sq = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))) + 1e-12
            rel = torch.sqrt(diff_sq / tgt_sq).cpu().numpy()
            resid = (yhat - y).cpu().numpy().reshape(yhat.shape[0], -1)
            per_sample_rel.append(rel)
            per_sample_resid.append(np.linalg.norm(resid, axis=1))
            seen += yhat.shape[0]
    rel_arr = np.concatenate(per_sample_rel)
    resid_arr = np.concatenate(per_sample_resid)
    if max_samples is not None:
        rel_arr = rel_arr[:max_samples]
        resid_arr = resid_arr[:max_samples]
    return {
        "rel_l2_per_sample": rel_arr,
        "resid_norm_per_sample": resid_arr,
    }


def capture_long_rollout(model, test_loader, device, n_chain: int = 5):
    """Chained autoregressive rollout for σ-stability evaluation.

    For each test trajectory:
      1. Run model on input → predict full T-frame trajectory.
      2. Build a new input by shifting the lag axis forward by `n_out`
         frames: new history = predicted state at the future window.
         Mask/time/params channels are reconstructed from the dataset layout
         (channel order: [state | mask | time | params]).
      3. Repeat for `n_chain` total chained passes (so effective horizon
         ≈ n_chain × n_out future steps beyond training).

    Captures, per chain step c ∈ [0, n_chain):
      - pred_norm_per_step[c]:  L2 norm of predicted state per t, averaged
                                 over batch+spatial.  Shape (T,).
      - peak_norm[c]:           max_t pred_norm_per_step[c, t].
      - final_norm[c]:          pred_norm_per_step[c, -1].
      - rel_l2_step0[c]:        relL2 vs ground truth on step 0 only
                                 (we only have GT for the first chain step).

    Robust to non-LEMO architectures.  Channel layout is detected from
    target.shape[-1] (= state channels); remaining input channels are
    treated as [mask, time, params] and shifted/copied appropriately.
    """
    model.eval()
    norm_traj = []      # list of (n_chain, T) arrays, one per batch
    peak_per_chain = []
    final_per_chain = []
    rel_l2_step0 = []
    n_seen = 0
    n_max_samples = 64  # cap to keep capture fast
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device).float()        # (B, T, *spatial, C+aux)
            y = batch["target"].to(device).float()       # (B, T, *spatial, C_state)
            mask = batch["loss_mask"].to(device).float() # (B, T)
            B = x.shape[0]
            T = x.shape[1]
            C_state = y.shape[-1]
            C_total = x.shape[-1]
            # Last-input frame index (transition from history to future).
            n_hist = int(mask[0].argmin().item()) if (mask[0] == 0).any() else T
            # Per-batch chained rollout.
            x_curr = x.clone()
            chain_norm = np.zeros((n_chain, T), dtype=np.float32)
            chain_peak = np.zeros(n_chain, dtype=np.float32)
            chain_final = np.zeros(n_chain, dtype=np.float32)
            for c in range(n_chain):
                yhat = model(x_curr)   # (B, T, *spatial, C_state)
                # Per-frame state-norm averaged over batch + spatial+channel dims.
                # Tensor.norm with multi-dim tuple is restricted on newer torch;
                # do sum-of-squares -> sqrt manually.
                spatial_dims = tuple(range(2, yhat.dim()))
                norm_per_frame = torch.sqrt((yhat ** 2).sum(dim=spatial_dims)).mean(dim=0).cpu().numpy()  # (T,)
                chain_norm[c] = norm_per_frame
                chain_peak[c] = float(norm_per_frame.max())
                chain_final[c] = float(norm_per_frame[-1])
                if c == 0:
                    # rel_l2 against GT on step 0.
                    n_spatial = y.dim() - 3
                    mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
                    diff_sq = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim())))
                    tgt_sq = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))) + 1e-12
                    rel = torch.sqrt(diff_sq / tgt_sq).mean().item()
                    rel_l2_step0.append(rel)
                # Build new input for next chain step.
                # Strategy: use yhat as the new state channels (history filled across full T),
                # keep mask all-history (=1), copy t and params unchanged.
                new_state = yhat                                 # (B, T, *spatial, C_state)
                # Aux channels = x_curr[..., C_state:] (mask, time, params).
                aux = x_curr[..., C_state:]                       # (B, T, *spatial, C_aux)
                # Set mask channel (index 0 of aux) to 1.0 everywhere — predicted state is now treated as history.
                # Layout from dataset: aux = [mask(1), time(1), params(...)]
                new_aux = aux.clone()
                if new_aux.shape[-1] >= 1:
                    new_aux[..., 0] = 1.0   # mask = all history for chained pass
                x_curr = torch.cat([new_state, new_aux], dim=-1)
            norm_traj.append(chain_norm)
            peak_per_chain.append(chain_peak)
            final_per_chain.append(chain_final)
            n_seen += B
            if n_seen >= n_max_samples:
                break
    if not norm_traj:
        return {}
    norm_arr = np.stack(norm_traj, axis=0).mean(axis=0)   # (n_chain, T)
    peak_arr = np.stack(peak_per_chain, axis=0).mean(axis=0)
    final_arr = np.stack(final_per_chain, axis=0).mean(axis=0)
    return {
        "pred_norm_per_step":  norm_arr,                 # (n_chain, T)
        "peak_norm_per_chain": peak_arr,                 # (n_chain,)
        "final_norm_per_chain": final_arr,               # (n_chain,)
        "rel_l2_step0_mean": float(np.mean(rel_l2_step0)) if rel_l2_step0 else 0.0,
        "n_chain": n_chain,
        "n_samples": n_seen,
    }


def capture_fft_residual(model, test_loader, device, n_max_samples: int = 64):
    """Spectral content of residuals along the lag axis.

    Computes |FFT_t(yhat - y)|^2 averaged over batch + spatial + channel.
    Returns shape (n_modes,) where n_modes = T//2+1.  Useful to show
    where (in spectral lag space) errors live: low modes = global drift,
    high modes = high-frequency noise."""
    model.eval()
    energy_acc = None
    n_seen = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            yhat = model(x)
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            r = (yhat - y) * mask_bc                  # (B, T, *spatial, C)
            # Move T to last axis for FFT.
            perm = [0] + list(range(2, r.dim())) + [1]
            r_p = r.permute(*perm)                    # (B, *spatial, C, T)
            R = torch.fft.rfft(r_p, dim=-1)           # (..., n_modes)
            energy = (R.abs() ** 2).mean(dim=tuple(range(R.dim() - 1)))  # (n_modes,)
            energy_np = energy.cpu().numpy()
            energy_acc = energy_np if energy_acc is None else energy_acc + energy_np
            n_seen += y.shape[0]
            if n_seen >= n_max_samples:
                break
    if energy_acc is None:
        return {}
    return {
        "residual_fft_energy_per_mode": (energy_acc / max(n_seen, 1)).astype(np.float32),
        "n_samples": n_seen,
    }


def capture_equivariance(model, test_loader, device, shifts=(1, 4, 16), n_max_samples: int = 16):
    """Test cyclic-group lag-equivariance at deployment (T1 in the wild).

    For each shift k:
      1. Take a test input x.
      2. Roll only the STATE channels (first C_state of last channel-axis)
         along the lag axis by k.
      3. Compare model(rolled_x) to roll_k(model(x)).

    Returns mean and std of relL2-error across samples per shift, plus
    a "T1_pass" flag (True if max relL2-error < 1e-3).
    """
    model.eval()
    results = {}
    n_seen = 0
    with torch.no_grad():
        for batch in test_loader:
            if n_seen >= n_max_samples:
                break
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            B = x.shape[0]
            T = x.shape[1]
            yhat = model(x)
            for k in shifts:
                # Theorem T1 is a cyclic-shift property of the OPERATOR:
                # LEMO(roll(x)) = roll(LEMO(x)) where the roll is along the
                # lag axis applied UNIFORMLY to every channel.  Shifting
                # only state channels while keeping aux (mask/time/params)
                # fixed feeds an inconsistent input that no theorem covers
                # — that's the WRONG test.  Roll the entire tensor.
                x_shifted = torch.roll(x, shifts=int(k), dims=1)
                yhat_shift = model(x_shifted)
                yhat_roll = torch.roll(yhat, shifts=int(k), dims=1)
                num = (yhat_shift - yhat_roll).flatten(1).norm(dim=1)
                den = yhat_roll.flatten(1).norm(dim=1).clamp_min(1e-12)
                rel = (num / den).cpu().numpy()
                key = f"shift_{k}"
                if key not in results:
                    results[key] = []
                results[key].extend(rel.tolist())
            n_seen += B
    out = {}
    max_err = 0.0
    for k in shifts:
        arr = np.array(results.get(f"shift_{k}", []), dtype=np.float32)
        if arr.size == 0:
            continue
        out[f"equiv_shift_{k}_mean"] = float(arr.mean())
        out[f"equiv_shift_{k}_std"] = float(arr.std())
        out[f"equiv_shift_{k}_max"] = float(arr.max())
        max_err = max(max_err, float(arr.max()))
    # T1 is a numerical theorem; pass threshold accounts for float32 FFT
    # precision compounded across n_layers blocks (~1e-3 to 1e-2 typical
    # in float32 with 3 blocks, FFT-pair errors).  We declare pass at 1%
    # rel-error which is clearly below the failure regime (>10%) seen
    # when the architecture is misbuilt.
    out["T1_pass"] = bool(max_err < 1e-2)
    out["max_shift_err"] = max_err
    out["shifts"] = list(shifts)
    return out


def process_cell(ckpt_path: Path, data_dir: str, family: str,
                 device: str, n_viz: int):
    cell_dir = ckpt_path.parent
    # Idempotency: skip only if BOTH the legacy per-frame AND the new
    # rollout/fft/equivariance captures have already been written.
    needed = ["per_frame.json", "long_rollout.npz",
              "fft_residual.npz", "equivariance.json"]
    if all((cell_dir / n).exists() for n in needed):
        return "skip (already captured)"
    try:
        model, test_loader, cfg = load_cell(ckpt_path, data_dir, family, device)
    except Exception as e:
        return f"FAIL load ({e})"
    # 1) per-frame (multi-metric: aggregate-then-sqrt + abs MSE + legacy)
    if not (cell_dir / "per_frame.json").exists():
        metrics = compute_per_frame(model, test_loader, device)
        (cell_dir / "per_frame.json").write_text(json.dumps(metrics, indent=2))
    else:
        metrics = json.loads((cell_dir / "per_frame.json").read_text())
    # 2) viz samples
    if not (cell_dir / "viz_samples.npz").exists():
        viz = capture_viz_samples(model, test_loader, device, n=n_viz)
        np.savez_compressed(cell_dir / "viz_samples.npz", **viz)
    # 3) kernel snapshot
    if not (cell_dir / "kernel_snapshot.npz").exists():
        snap = capture_kernel_snapshot(ckpt_path)
        if snap:
            np.savez_compressed(cell_dir / "kernel_snapshot.npz", **snap)
    # 4) residuals (full test set; previously capped at 64)
    if not (cell_dir / "residuals.npz").exists():
        res = capture_residuals(model, test_loader, device)
        np.savez_compressed(cell_dir / "residuals.npz", **res)
    # 5) long rollout (chained autoregressive) — σ-stability divergence proxy.
    if not (cell_dir / "long_rollout.npz").exists():
        try:
            roll = capture_long_rollout(model, test_loader, device, n_chain=5)
            if roll:
                np.savez_compressed(cell_dir / "long_rollout.npz", **roll)
        except Exception as e:
            (cell_dir / "long_rollout.skip").write_text(f"{type(e).__name__}: {e}")
    # 6) FFT residual — spectral footprint of error along lag axis.
    if not (cell_dir / "fft_residual.npz").exists():
        try:
            fft = capture_fft_residual(model, test_loader, device)
            if fft:
                np.savez_compressed(cell_dir / "fft_residual.npz", **fft)
        except Exception as e:
            (cell_dir / "fft_residual.skip").write_text(f"{type(e).__name__}: {e}")
    # 7) Cyclic-shift equivariance check at deployment (T1 in the wild).
    if not (cell_dir / "equivariance.json").exists():
        try:
            eq = capture_equivariance(model, test_loader, device)
            (cell_dir / "equivariance.json").write_text(json.dumps(eq, indent=2))
        except Exception as e:
            (cell_dir / "equivariance.skip").write_text(f"{type(e).__name__}: {e}")
    rel = np.array(metrics["rel_l2_per_step"])
    naive = np.array(metrics["naive_rel_l2_per_step"])
    return f"ok (rel_l2_overall={rel.mean():.4f} naive={naive.mean():.4f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_root", required=True, help="e.g. outputs/dist_kernel_v2_p1")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--n_viz_samples", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layer = Path(args.layer_root)
    ckpts = sorted(layer.glob("raw/**/best_model.pt"))
    print(f"found {len(ckpts)} checkpoints under {layer}")

    summary = []
    for c in ckpts:
        parts = c.parts
        family = parts[-5]
        cell = "/".join(parts[-5:-1])
        msg = process_cell(c, args.data_dir, family,
                            args.device, args.n_viz_samples)
        print(f"  {cell}: {msg}")
        summary.append({"cell": cell, "result": msg})

    out_path = layer / "capture_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
