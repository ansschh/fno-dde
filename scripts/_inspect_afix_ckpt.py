"""Diagnostic: inspect A-fix LEMO-PC checkpoint to figure out why
equivariance error is not at FP32 floor."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from train.build_model import build_model

CKPT = REPO / "outputs/a_fix_runpod/raw/dist_exp_rd_2d/clean/lemo_pc_nd/s42/best_model.pt"

ckpt = torch.load(CKPT, weights_only=False, map_location="cpu")
cfg = ckpt["config"]
mcfg = cfg.get("model", cfg)
print("config zero_beta_above_dc:", mcfg.get("zero_beta_above_dc", "KEY_NOT_PRESENT"))
print("config model_class:", cfg.get("model_class", "?"))
print("model_cfg top keys:", list(mcfg.keys())[:15])

m = build_model(cfg, in_channels=8, out_channels=1, length=128)
print("block0 A_lag.zero_beta_above_dc:", m.blocks[0].A_lag.zero_beta_above_dc)
print("top-level zero_beta_above_dc:", m.zero_beta_above_dc)
m.load_state_dict(ckpt["model_state_dict"])

# Equivariance test using REAL test data
from datasets.apebench_dataset import create_apebench_dataloaders
_, _, test_loader = create_apebench_dataloaders(
    "/workspace/dde-fno/data_dde_pde", "dist_exp_rd_2d",
    batch_size=2, residual_anchor=True, seed=42)
batch = next(iter(test_loader))
x = batch["input"]
print("real x shape:", x.shape)

m.eval()
with torch.no_grad():
    y = m(x)
    print("y shape:", y.shape, "y norm:", y.norm().item())
    print("--- TEST: state-only shift (eval_equivariance_dense convention) ---")
    for k in [1, 4, 16, 64]:
        x_shift = torch.cat([torch.roll(x[..., :1], shifts=k, dims=1), x[..., 1:]], dim=-1)
        y_shift = m(x_shift)
        y_rolled = torch.roll(y, shifts=k, dims=1)
        err = (y_shift - y_rolled).norm() / y.norm()
        print(f"  state-only k={k}: equiv_err={err.item():.6e}")

    print("--- TEST: FULL roll (rolls all input channels) ---")
    for k in [1, 4, 16, 64]:
        x_shift = torch.roll(x, shifts=k, dims=1)
        y_shift = m(x_shift)
        y_rolled = torch.roll(y, shifts=k, dims=1)
        err = (y_shift - y_rolled).norm() / y.norm()
        print(f"  full-roll k={k}: equiv_err={err.item():.6e}")

    print("--- TEST: untrained random model w/ zero_beta_above_dc=True ---")
    m_rand = build_model(cfg, in_channels=8, out_channels=1, length=128)
    m_rand.eval()
    y_rand = m_rand(x)
    for k in [1, 4, 16, 64]:
        x_shift = torch.roll(x, shifts=k, dims=1)
        y_shift = m_rand(x_shift)
        y_rolled = torch.roll(y_rand, shifts=k, dims=1)
        err = (y_shift - y_rolled).norm() / y_rand.norm()
        print(f"  random+full-roll k={k}: equiv_err={err.item():.6e}")
