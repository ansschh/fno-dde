"""
Layer A T4 — Quadrature resolution sweep (verifies the W_1 quantization theorem).

Continuous-tau target with fine-grid resolution G fixed at the data side.
We vary the model's internal lag-grid resolution (modes/grid) to test:

  Empirical claim: test MSE decays as O(1/G) when the kernel is
  parameterized at G grid points and evaluated on the same target.

Theoretical claim (Thm 12, W_1 ≤ L/(2G)):
  the discretization error of a continuous lag measure on G grid points
  is bounded by L/(2G).

Sweep G in {16, 32, 64, 128, 256, 512} for LEMOContinuous (no sigma)
and FNO+Aug as baseline.  Plot MSE vs 1/G; expect a roughly linear
trend confirming the O(1/G) rate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from models.lemo_continuous import LEMOContinuousScalar


# Reuse the same task as scripts/sweep_continuous_tau.py
L = 16.0
W_TEMPLATE = 1.5
N_TRAIN = 512
N_TEST = 1024
EPOCHS = 1000
LR = 3e-3


def make_K_template(G: int):
    fine_grid = np.linspace(0.0, L, G, endpoint=False, dtype=np.float32)
    ds = L / G
    def K_template(delta):
        d = np.minimum(delta % L, L - (delta % L))
        out = np.exp(-0.5 * (d / W_TEMPLATE) ** 2).astype(np.float32)
        out = out / (out.sum() * ds)
        return out
    K_FINE = K_template(fine_grid)
    return fine_grid, ds, K_template, K_FINE


def target_y(h, tau, fine_grid, K_template, ds):
    delta = (tau[:, None] - fine_grid[None, :]) % L
    Kt = K_template(delta)
    return (Kt * h).sum(axis=1) * ds


def gen(n, sampler, rng, fine_grid, K_template, ds):
    G = len(fine_grid)
    h = rng.standard_normal((n, G)).astype(np.float32)
    tau = sampler(n, rng).astype(np.float32)
    y = target_y(h, tau, fine_grid, K_template, ds).astype(np.float32)
    return h, tau, y


def continuous_sampler(n, rng):
    return rng.uniform(0.0, L, size=n)


def train_eval(model, data_train, data_test, device, epochs=EPOCHS, lr=LR):
    h_tr, tau_tr, y_tr = (torch.from_numpy(x).to(device) for x in data_train)
    h_te, tau_te, y_te = (torch.from_numpy(x).to(device) for x in data_test)
    h_tr_b, h_te_b = h_tr.unsqueeze(-1), h_te.unsqueeze(-1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-3)
    model.train()
    for ep in range(epochs):
        y_pred = model(h_tr_b, tau_tr)
        if y_pred.ndim == 3: y_pred = y_pred.mean(dim=(1, 2))
        loss = F.mse_loss(y_pred, y_tr)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(h_te_b, tau_te)
        if y_pred.ndim == 3: y_pred = y_pred.mean(dim=(1, 2))
        return F.mse_loss(y_pred, y_te).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G_values", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", default="outputs/sweep_t4_resolution.json")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results = {}
    for G in args.G_values:
        fine_grid, ds, K_template, K_FINE = make_K_template(G)
        per_seed = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            data_test = gen(N_TEST, continuous_sampler, rng, fine_grid, K_template, ds)
            data_train = gen(N_TRAIN, continuous_sampler, rng, fine_grid, K_template, ds)
            torch.manual_seed(seed)
            model = LEMOContinuousScalar(
                in_channels=1, out_channels=1, length=G, L_phys=L,
                width=48, n_layers=3, kernel_hidden=64,
                sigma=0.99, activation="relu",
            ).to(device)
            t0 = time.time()
            mse = train_eval(model, data_train, data_test, device)
            per_seed.append(mse)
            print(f"  G={G:>4d}  seed={seed}  test MSE={mse:.3e}  ({time.time()-t0:.0f}s)")
        results[G] = {"mean": float(np.mean(per_seed)),
                       "std":  float(np.std(per_seed)),
                       "vals": [float(v) for v in per_seed]}

    print("\n=== T4 — quadrature resolution sweep ===")
    print(f"{'G':>5}  {'mean MSE':>14}  {'std':>14}  {'1/G':>10}")
    for G in args.G_values:
        r = results[G]
        print(f"{G:>5d}  {r['mean']:>14.3e}  {r['std']:>14.3e}  {1.0/G:>10.6f}")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args),
        "L": L,
        "results": results,
    }, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
