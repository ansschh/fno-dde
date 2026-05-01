"""Empirical verification of Theorem T1 (cyclic-shift lag-equivariance) on
the LEMOPCND PyTorch implementation, on real (random) data.

The Lean proof at A:/dde research/lean/LEMO/ proves T1 for the abstract
architecture. Here we check the PyTorch realization is faithful.

Definition (T1):
    Let S_k denote cyclic shift by k along the lag (length) axis applied
    only to the *state* channels (NOT the per-sample params, NOT spatial
    coords). For F = LEMOPCND, T1 claims:

        F(S_k x) == S_k F(x)   for every k ∈ Z_L

If the architecture truly realizes T1, the relative error
    ||F(S_k x) - S_k F(x)|| / ||F(x)||
should be at the level of float32 round-off (~1e-6 to 1e-5) for all k.

Run:
    python scripts/verify_T1_empirical.py

Cap a few hundred million flops; runs CPU-only in seconds. No GPU needed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from models.lemo_pc_nd import LEMOPCND, create_causal_lemo_pc_nd  # noqa: E402


# Float32 round-off tolerance for "equivariant".
PASS_TOL = 1e-4
FAIL_TOL = 1e-2


def _shift_state_only(x: torch.Tensor, k: int, state_channels: int) -> torch.Tensor:
    """Roll only the first `state_channels` along lag (axis 1).

    x has shape (B, L, *spatial, C). We split C into [state | aux] where
    aux holds mask / time-coord / params / spatial coords. Aux must NOT
    be shifted (params are constant in time, t-coord encodes absolute
    position, etc.).  Equivariance holds only on the state subspace.
    """
    state = x[..., :state_channels]
    aux = x[..., state_channels:]
    state_shift = torch.roll(state, shifts=k, dims=1)
    return torch.cat([state_shift, aux], dim=-1)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).reshape(-1).norm().item()
    den = a.reshape(-1).norm().item() + 1e-30
    return num / den


def _check_one(model: torch.nn.Module, x: torch.Tensor, state_channels: int,
               shifts, label: str) -> dict:
    model.eval()
    with torch.no_grad():
        Fx = model(x)
        out = {}
        for k in shifts:
            x_sh = _shift_state_only(x, k, state_channels)
            F_xs = model(x_sh)
            target = torch.roll(Fx, shifts=k, dims=1)
            err = _rel_err(F_xs, target)
            out[k] = err
    print(f"\n=== {label} ===")
    for k, err in out.items():
        verdict = "PASS" if err < PASS_TOL else ("WARN" if err < FAIL_TOL else "FAIL")
        print(f"  shift k={k:>3}: ||F(S_k x) - S_k F(x)|| / ||F(x)|| = {err:.3e}  [{verdict}]")
    return out


def _build(in_channels: int, out_channels: int, length: int,
           spatial_shape, lag_modes: int, n_layers: int, width: int,
           sigma, params_dim: int, causal: bool = False) -> torch.nn.Module:
    if causal:
        cfg = {"model": {
            "spatial_shape": spatial_shape,
            "lag_modes": lag_modes, "width": width, "n_layers": n_layers,
            "params_dim": params_dim, "extract_params": True,
            "sigma": sigma,
        }}
        return create_causal_lemo_pc_nd(in_channels, out_channels, cfg, length=length)
    return LEMOPCND(
        in_channels=in_channels, out_channels=out_channels,
        length=length, params_dim=params_dim,
        spatial_shape=spatial_shape,
        lag_modes=lag_modes, width=width, n_layers=n_layers,
        sigma=sigma, extract_params=True, causal=False,
    )


def main() -> None:
    torch.manual_seed(0)

    # Spec from prompt: B=2, L=64, spatial=16x16, in_channels=4
    # state(1) + mask(1) + t(1) + 1 param  ⇒  state_channels=1
    B, L, H, W, C = 2, 64, 16, 16, 4
    state_channels = 1
    params_dim = 1
    shifts = [1, 2, 4, 8, 16, 32]

    print(f"input shape (B, L, H, W, C) = ({B}, {L}, {H}, {W}, {C})")
    print(f"state_channels = {state_channels}, shifts to test: {shifts}")
    print(f"PASS tolerance: {PASS_TOL}  |  FAIL tolerance: {FAIL_TOL}")

    # Build a single random input. Re-used across all model variants so
    # results are directly comparable.
    x = torch.randn(B, L, H, W, C)

    # ---- Variant 1: σ=None, n_layers=2, width=32 (prompt spec) ----
    m1 = _build(C, 1, L, (H, W), lag_modes=16, n_layers=2, width=32,
                sigma=None, params_dim=params_dim)
    _check_one(m1, x, state_channels, shifts, "LEMOPCND  σ=None  L=2  width=32  (2D 16x16)")

    # ---- Variant 2: σ=0.5 with SVD projection ----
    m2 = _build(C, 1, L, (H, W), lag_modes=16, n_layers=2, width=32,
                sigma=0.5, params_dim=params_dim)
    _check_one(m2, x, state_channels, shifts, "LEMOPCND  σ=0.5 (SVD)  L=2  width=32")

    # ---- Variant 3: causal LEMO_PC_ND ----
    # Strictly causal FIR (right-padded). Equivariant only over the
    # cyclic group restricted to k that don't drag the support past L —
    # for a length-L cyclic FFT, the kernel is implemented via cyclic
    # convolution so cyclic shifts STILL commute on the state subspace.
    m3 = _build(C, 1, L, (H, W), lag_modes=16, n_layers=2, width=32,
                sigma=None, params_dim=params_dim, causal=True)
    _check_one(m3, x, state_channels, shifts, "Causal LEMOPCND  σ=None  L=2  width=32")

    # ---- Edge cases ----
    print("\n--- Edge cases ---")

    # 1D (spatial=())
    x1d = torch.randn(B, L, C)
    m1d = _build(C, 1, L, (), lag_modes=16, n_layers=2, width=32,
                 sigma=None, params_dim=params_dim)
    _check_one(m1d, x1d, state_channels, [1, 4, 16], "1D LEMOPCND (spatial=())")

    # batch=1
    xb1 = torch.randn(1, L, H, W, C)
    _check_one(m1, xb1, state_channels, [1, 8], "B=1  (re-using m1)")

    # Different lag_modes (8 and 32)
    m_lm8 = _build(C, 1, L, (H, W), lag_modes=8, n_layers=2, width=32,
                   sigma=None, params_dim=params_dim)
    _check_one(m_lm8, x, state_channels, [1, 4, 16], "lag_modes=8")

    m_lm32 = _build(C, 1, L, (H, W), lag_modes=32, n_layers=2, width=32,
                    sigma=None, params_dim=params_dim)
    _check_one(m_lm32, x, state_channels, [1, 4, 16], "lag_modes=32")

    # Different L (32 and 128) — re-randomize x
    x_L32 = torch.randn(B, 32, H, W, C)
    m_L32 = _build(C, 1, 32, (H, W), lag_modes=8, n_layers=2, width=32,
                   sigma=None, params_dim=params_dim)
    _check_one(m_L32, x_L32, state_channels, [1, 4, 16], "L=32")

    x_L128 = torch.randn(B, 128, H, W, C)
    m_L128 = _build(C, 1, 128, (H, W), lag_modes=16, n_layers=2, width=32,
                    sigma=None, params_dim=params_dim)
    _check_one(m_L128, x_L128, state_channels, [1, 4, 16, 64], "L=128")

    # n_layers=1 and n_layers=4 — composition of equivariant layers must
    # remain equivariant; check the depth-invariance empirically.
    m_l1 = _build(C, 1, L, (H, W), lag_modes=16, n_layers=1, width=32,
                  sigma=None, params_dim=params_dim)
    _check_one(m_l1, x, state_channels, [1, 8], "n_layers=1")
    m_l4 = _build(C, 1, L, (H, W), lag_modes=16, n_layers=4, width=32,
                  sigma=None, params_dim=params_dim)
    _check_one(m_l4, x, state_channels, [1, 8], "n_layers=4")

    print("\nDone. Any 'FAIL' line above means the PyTorch implementation "
          "violates Theorem T1 and the architecture is broken; "
          "any 'WARN' line means equivariance holds but with worse-than-"
          "round-off precision (worth investigating).")


if __name__ == "__main__":
    main()
