"""
D3 + D4: per-sample residual extraction.

Loads `best_model.pt` for each (run_dir), runs inference on the test
set, and saves per-sample arrays:

  rel_l2_per_sample  : (N,)            per-sample relative L2 (original units)
  amplitude_err      : (N, n_modes)    |fft(target) - fft(pred)| magnitude
  phase_err          : (N, n_modes)    angle(fft(target) * conj(fft(pred)))
  energy_target      : (N, n_modes)    |fft(target)|^2 per mode (for weighting)

Saved to <run_dir>/residuals.npz.

Usage:
    python3 scripts/extract_residuals.py <output_dir>
where output_dir is the parent like outputs/phase_b_core_dde_v1/linear2/id/tcn_s42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train.build_model import build_model
from datasets.sharded_dataset import create_sharded_dataloaders


def find_run_subdir(run_dir: Path) -> Path:
    subs = [p for p in run_dir.iterdir() if p.is_dir() and (p / "best_model.pt").exists()]
    if not subs:
        raise FileNotFoundError(f"no best_model.pt under {run_dir}")
    return subs[0]


def load_model(run_subdir: Path, device: torch.device, length: int,
                in_channels: int, out_channels: int) -> torch.nn.Module:
    cfg = yaml.safe_load(open(run_subdir.parent.parent.parent.parent /
                                "configs" / "auto" /
                                run_subdir.parent.parent.parent.parent.name /
                                ""))  # not used; we pull config from saved checkpoint cfg
    # Actually: load config from the run's config.yaml saved alongside the model
    config_path = run_subdir / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(open(config_path))
    else:
        # Fallback: use the auto config used to launch the run
        raise FileNotFoundError(f"no config.yaml in {run_subdir}")
    model = build_model(config, in_channels=in_channels,
                         out_channels=out_channels, length=length)
    ckpt = torch.load(run_subdir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="e.g. outputs/phase_b_core_dde_v1/linear2/id/tcn_s42")
    p.add_argument("--data_dir", default="data_baseline_v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_modes", type=int, default=32,
                   help="Number of FFT modes to keep for amplitude/phase residuals.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_subdir = find_run_subdir(run_dir)
    family = run_subdir.parts[-4]
    split = run_subdir.parts[-3]
    print(f"family={family}  split={split}  run={run_subdir.name}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build dataloaders to get test data + dimensions.
    family_for_loader = family
    if split != "id":
        # OOD splits live under a different data dir
        # Get actual data path from the launch config
        cfg_path = run_subdir / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(open(cfg_path))
            data_dir = cfg.get("data_dir", args.data_dir)
        else:
            data_dir = args.data_dir
    else:
        data_dir = args.data_dir

    train_loader, val_loader, test_loader = create_sharded_dataloaders(
        data_dir=data_dir, family=family_for_loader, batch_size=64,
        num_workers=2, streaming=False,
    )

    # Get dimensions
    sample = next(iter(test_loader))
    in_channels = sample["input"].shape[-1]
    out_channels = sample["target"].shape[-1]
    length = sample["input"].shape[1]

    cfg = yaml.safe_load(open(run_subdir / "config.yaml"))
    model = build_model(cfg, in_channels=in_channels, out_channels=out_channels,
                        length=length).to(device)
    ckpt = torch.load(run_subdir / "best_model.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference + collect per-sample residuals
    rel_l2_list = []
    amp_err_list = []
    phase_err_list = []
    energy_t_list = []

    # For original-unit denormalization, get y_mean/y_std from the test_loader's dataset
    test_ds = test_loader.dataset
    y_mean = torch.from_numpy(np.asarray(test_ds.y_mean)).to(device).float()
    y_std = torch.from_numpy(np.asarray(test_ds.y_std)).to(device).float()

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device).float()
            tgt = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            pred = model(x)
            # Denormalize to original units
            pred_o = pred * y_std + y_mean
            tgt_o = tgt * y_std + y_mean
            # Apply mask along time axis
            mask_t = mask.unsqueeze(-1)
            diff = (pred_o - tgt_o) * mask_t
            tgt_m = tgt_o * mask_t
            # Per-sample relL2
            num = torch.sqrt((diff ** 2).sum(dim=(1, 2)) + 1e-10)
            den = torch.sqrt((tgt_m ** 2).sum(dim=(1, 2)) + 1e-10)
            rel = num / den
            rel_l2_list.append(rel.cpu().numpy())

            # FFT analysis on the masked region (the future part).
            # Take the last n_out points (where loss_mask==1).
            n_hist = (mask[0] == 0).sum().item()
            future_pred = pred_o[:, n_hist:, :]   # (B, n_out, C)
            future_tgt  = tgt_o[:,  n_hist:, :]   # (B, n_out, C)
            # FFT along time axis, average over channels
            fft_p = torch.fft.rfft(future_pred, dim=1)  # (B, n_modes_full, C)
            fft_t = torch.fft.rfft(future_tgt,  dim=1)
            n_keep = min(args.n_modes, fft_p.shape[1])
            fp = fft_p[:, :n_keep, :].mean(dim=-1)   # (B, n_modes)
            ft = fft_t[:, :n_keep, :].mean(dim=-1)
            amp_err = (ft.abs() - fp.abs()).abs().cpu().numpy()
            phase_err = (ft * fp.conj()).angle().abs().cpu().numpy()
            energy_t = (ft.abs() ** 2).cpu().numpy()
            amp_err_list.append(amp_err)
            phase_err_list.append(phase_err)
            energy_t_list.append(energy_t)

    rel_l2 = np.concatenate(rel_l2_list)
    amp_err = np.concatenate(amp_err_list)
    phase_err = np.concatenate(phase_err_list)
    energy_t = np.concatenate(energy_t_list)

    out = run_subdir / "residuals.npz"
    np.savez(out,
              rel_l2=rel_l2,
              amplitude_err=amp_err,
              phase_err=phase_err,
              energy_target=energy_t)
    print(f"  saved {out}: N={len(rel_l2)}  mean_relL2={rel_l2.mean():.4f}  "
          f"median={np.median(rel_l2):.4f}  p95={np.percentile(rel_l2, 95):.4f}")


if __name__ == "__main__":
    main()
