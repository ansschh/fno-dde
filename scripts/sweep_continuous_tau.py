"""
SOTA sweep: continuous-tau OOD transfer.

Reproduces and scales up `experiments/continuous_lag_transfer.py`:
    - Train tau drawn from a discrete grid of size m (m in {4, 8, 16, 32, 64, 128})
    - Test tau continuous in [0, L]
    - Compare:
        (a) LEMOContinuousScalar  (the new architecture; this paper)
        (b) LEMOContinuousScalar with sigma in {0.99, 0.9}
        (c) MLP + one-hot grid augmentation (the original baseline)
        (d) FNOLagAug — FNO1d on (h, tau) with one-hot grid aug
        (e) ContinuousLagEquiv (the original paper's local model, as a
            single-layer baseline that already achieves SOTA)

Goal: LEMOContinuous (both unconstrained and sigma-constrained variants)
matches or beats the original ContinuousLagEquiv (3e-7) and beats MLP+aug
by ~10^3x at every m.
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

from models.lemo_continuous import LEMOContinuousScalar, count_parameters


# -----------------------------------------------------------------------------
# Task setup
# -----------------------------------------------------------------------------

L = 16.0
G = 64
W_TEMPLATE = 1.5
N_TRAIN = 512
N_TEST = 1024


def K_template(delta: np.ndarray) -> np.ndarray:
    d = np.minimum(delta % L, L - (delta % L))
    out = np.exp(-0.5 * (d / W_TEMPLATE) ** 2).astype(np.float32)
    out = out / (out.sum() * (L / G))
    return out


FINE_GRID = np.linspace(0.0, L, G, endpoint=False, dtype=np.float32)
DS = L / G
K_FINE = K_template(FINE_GRID)


def target_y_batch(h: np.ndarray, tau: np.ndarray) -> np.ndarray:
    delta = (tau[:, None] - FINE_GRID[None, :]) % L
    Kt = K_template(delta)
    return (Kt * h).sum(axis=1) * DS


def generate(n: int, tau_sampler, rng):
    h = rng.standard_normal((n, G)).astype(np.float32)
    tau = tau_sampler(n, rng).astype(np.float32)
    y = target_y_batch(h, tau).astype(np.float32)
    return h, tau, y


def coarse_grid_sampler(m: int):
    grid = np.linspace(0.0, L, m, endpoint=False)
    def sample(n, rng):
        return rng.choice(grid, size=n)
    return sample


def continuous_sampler(n, rng):
    return rng.uniform(0.0, L, size=n)


# -----------------------------------------------------------------------------
# Baselines (mirroring the original experiment)
# -----------------------------------------------------------------------------

class MLPGridAug(nn.Module):
    def __init__(self, fine: int, m: int, hidden: int = 64) -> None:
        super().__init__()
        self.fine = fine
        self.m = m
        self.fc1 = nn.Linear(fine + m, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def tau_to_index(self, tau: torch.Tensor) -> torch.Tensor:
        return (torch.round(tau / L * self.m).long()) % self.m

    def forward(self, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        idx = self.tau_to_index(tau)
        tau_oh = F.one_hot(idx, num_classes=self.m).float()
        x = torch.cat([h, tau_oh], dim=1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))).squeeze(-1)


class ContinuousLagEquiv(nn.Module):
    """The original local model (single-MLP kernel, scalar output)."""
    def __init__(self, fine: int, hidden: int = 64) -> None:
        super().__init__()
        self.fine = fine
        self.k_net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("fine_grid", torch.from_numpy(FINE_GRID))
        self.register_buffer("ds", torch.tensor(DS))

    def forward(self, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        delta = (tau[:, None] - self.fine_grid[None, :]) % L
        delta_signed = torch.where(delta > L / 2, delta - L, delta)
        k_vals = self.k_net(delta_signed.unsqueeze(-1)).squeeze(-1)
        return (k_vals * h).sum(dim=1) * self.ds


class FNOTauAug(nn.Module):
    """FNO1d-style spectral conv on h, with one-hot tau augmentation
    concatenated to the lifted features.  Non-equivariant in tau."""
    def __init__(self, fine: int, m: int, modes: int = 16,
                  width: int = 48, n_layers: int = 3) -> None:
        super().__init__()
        from models.fno1d import SpectralConv1d
        self.fine = fine
        self.m = m
        self.width = width
        # Lift: 1 channel (h) + m channels (one-hot tau, broadcast across length)
        # = (1 + m) channels in.
        self.lift = nn.Linear(1 + m, width)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes)
                                    for _ in range(n_layers)])
        self.point = nn.ModuleList([nn.Conv1d(width, width, 1)
                                     for _ in range(n_layers)])
        self.act = nn.GELU()
        self.head = nn.Linear(width, 1)

    def tau_to_index(self, tau: torch.Tensor) -> torch.Tensor:
        return (torch.round(tau / L * self.m).long()) % self.m

    def forward(self, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # h: (B, fine).  Reshape and concat tau one-hot as channels.
        idx = self.tau_to_index(tau)                                 # (B,)
        tau_oh = F.one_hot(idx, num_classes=self.m).float()           # (B, m)
        h_ch = h.unsqueeze(-1)                                        # (B, fine, 1)
        tau_ch = tau_oh.unsqueeze(1).expand(-1, self.fine, -1)        # (B, fine, m)
        x = torch.cat([h_ch, tau_ch], dim=-1)                         # (B, fine, 1+m)
        x = self.lift(x)                                              # (B, fine, width)
        x = x.permute(0, 2, 1)                                        # (B, width, fine)
        for spec, point in zip(self.spec, self.point):
            x = self.act(spec(x) + point(x))
        x = x.permute(0, 2, 1)                                        # (B, fine, width)
        # Mean-pool over length, then head.
        return self.head(x.mean(dim=1)).squeeze(-1)


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def train_eval(model, data_train, data_test, epochs: int, lr: float,
               device: torch.device, takes_tau: bool = True) -> tuple:
    h_tr, tau_tr, y_tr = (torch.from_numpy(x).to(device) for x in data_train)
    h_te, tau_te, y_te = (torch.from_numpy(x).to(device) for x in data_test)
    # For LEMOContinuousScalar we need (B, L, C=1) shape; for the others (B, L) is fine.
    if isinstance(model, LEMOContinuousScalar):
        h_tr_b = h_tr.unsqueeze(-1)
        h_te_b = h_te.unsqueeze(-1)
    else:
        h_tr_b = h_tr
        h_te_b = h_te
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 1e-3)
    model.train()
    losses = []
    for ep in range(epochs):
        if takes_tau:
            y_pred = model(h_tr_b, tau_tr)
        else:
            y_pred = model(h_tr_b)
        # If model output has shape (B, L, C) (sequence), pool to scalar.
        if y_pred.ndim == 3:
            y_pred = y_pred.mean(dim=(1, 2))
        elif y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(-1)
        loss = F.mse_loss(y_pred, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        if takes_tau:
            y_pred = model(h_te_b, tau_te)
        else:
            y_pred = model(h_te_b)
        if y_pred.ndim == 3:
            y_pred = y_pred.mean(dim=(1, 2))
        elif y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(-1)
        test_mse = F.mse_loss(y_pred, y_te).item()
    return test_mse, losses[-1]


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m_values", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--kernel_hidden", type=int, default=64)
    ap.add_argument("--out", type=str, default="outputs/sweep_continuous_tau.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}  # results[(model_name, m)] -> list of test MSE per seed
    all_runs = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        # Test set is always continuous (independent of m).
        data_test = generate(N_TEST, continuous_sampler, rng)
        for m in args.m_values:
            # Independent training data per m.
            data_train = generate(N_TRAIN, coarse_grid_sampler(m), rng)

            for model_name in ["mlp_aug", "fno_aug", "cont_orig",
                                "lemo_cont", "lemo_cont_sigma_099",
                                "lemo_cont_sigma_09"]:
                torch.manual_seed(seed)
                if model_name == "mlp_aug":
                    model = MLPGridAug(fine=G, m=m).to(device)
                elif model_name == "fno_aug":
                    model = FNOTauAug(fine=G, m=m,
                                       modes=16, width=args.width,
                                       n_layers=args.n_layers).to(device)
                elif model_name == "cont_orig":
                    model = ContinuousLagEquiv(fine=G).to(device)
                elif model_name == "lemo_cont":
                    model = LEMOContinuousScalar(
                        in_channels=1, out_channels=1, length=G,
                        L_phys=L, width=args.width,
                        n_layers=args.n_layers,
                        kernel_hidden=args.kernel_hidden,
                        sigma=None, activation="gelu",
                    ).to(device)
                elif model_name == "lemo_cont_sigma_099":
                    model = LEMOContinuousScalar(
                        in_channels=1, out_channels=1, length=G,
                        L_phys=L, width=args.width,
                        n_layers=args.n_layers,
                        kernel_hidden=args.kernel_hidden,
                        sigma=0.99, activation="relu",
                    ).to(device)
                elif model_name == "lemo_cont_sigma_09":
                    model = LEMOContinuousScalar(
                        in_channels=1, out_channels=1, length=G,
                        L_phys=L, width=args.width,
                        n_layers=args.n_layers,
                        kernel_hidden=args.kernel_hidden,
                        sigma=0.9, activation="relu",
                    ).to(device)
                else:
                    continue

                t0 = time.time()
                try:
                    test_mse, train_loss = train_eval(
                        model, data_train, data_test,
                        epochs=args.epochs, lr=args.lr, device=device,
                    )
                    elapsed = time.time() - t0
                    print(f"  seed={seed}  m={m:>3d}  {model_name:<25} "
                          f"test_mse={test_mse:.3e}  train_loss={train_loss:.3e}  "
                          f"params={count_parameters(model):,}  {elapsed:.1f}s")
                except Exception as e:
                    print(f"  seed={seed}  m={m:>3d}  {model_name:<25} FAILED: {e}")
                    test_mse = float("nan")
                    train_loss = float("nan")
                    elapsed = -1

                results.setdefault((model_name, m), []).append(test_mse)
                all_runs.append({
                    "model": model_name, "m": m, "seed": seed,
                    "test_mse": test_mse, "train_loss": train_loss,
                    "params": count_parameters(model), "wall_s": elapsed,
                })

    # Aggregate.
    print("\n" + "=" * 92)
    print(f"{'model':<25} | " +
          "  ".join(f"m={m:>3d}" for m in args.m_values))
    print("-" * 92)
    model_names = ["mlp_aug", "fno_aug", "cont_orig",
                   "lemo_cont", "lemo_cont_sigma_099", "lemo_cont_sigma_09"]
    summary = {}
    for name in model_names:
        row = [name]
        summary[name] = {}
        for m in args.m_values:
            vals = results.get((name, m), [])
            if not vals or all(np.isnan(v) for v in vals):
                row.append("  --   ")
                summary[name][m] = float("nan")
                continue
            mean = float(np.mean(vals))
            row.append(f"{mean:.2e}")
            summary[name][m] = mean
        print(f"{row[0]:<25} | " + "  ".join(f"{r:>10s}" for r in row[1:]))

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args),
        "summary": {n: {str(m): v for m, v in mv.items()}
                    for n, mv in summary.items()},
        "all_runs": all_runs,
    }, indent=2))
    print(f"\nResults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
