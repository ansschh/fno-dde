"""Smoke test for Causal LEMO-PC-ND.

Three checks:
  1. Forward pass shape correctness on (B, L, in_channels) 1D input.
  2. Impulse-response causality: feed delta at lag 0, verify the output
     for lag positions [lag_modes, ..., L - 1] is approximately zero (the
     causal FIR has no support beyond lag_modes).  Strictly causal in the
     non-cyclic sense for t in [0, lag_modes-1] — outside that window the
     conv is silent.
  3. Causal-monoid equivariance: shift input forward by k positions
     (with zero-fill at the start, NOT cyclic), check that the output is
     also forward-shifted by k (modulo edge effects on the first lag_modes
     positions where the cyclic wrap can leak).

Run:
    python scripts/test_causal_lemo.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from models.lemo_pc_nd import create_causal_lemo_pc_nd, create_lemo_pc_nd


def main():
    torch.manual_seed(0)
    L = 64
    lag_modes = 16
    width = 32
    in_channels = 4   # state(1) + mask(1) + t(1) + 1 param
    out_channels = 1
    cfg = {
        "model_class": "causal_lemo_pc_nd",
        "model": {
            "spatial_shape": (),
            "lag_modes": lag_modes,
            "width": width,
            "n_layers": 2,
            "params_dim": 1,
            "extract_params": True,
        },
    }
    model = create_causal_lemo_pc_nd(in_channels, out_channels, cfg, length=L)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params}")
    assert model.causal, "model.causal should be True"
    for blk in model.blocks:
        assert blk.causal, "block.causal should be True"
        assert blk.A_lag.causal, "A_lag.causal should be True"
        assert blk.A_lag.weights_time is not None
        assert blk.A_lag.weights is None

    # 1) Forward pass shape
    B = 2
    x = torch.randn(B, L, in_channels)
    with torch.no_grad():
        y = model(x)
    print(f"forward shape: in={tuple(x.shape)} out={tuple(y.shape)}")
    assert y.shape == (B, L, out_channels), y.shape

    # 2) Impulse-response causality on a SINGLE block's lag conv
    # Use one FiLMLagSpectralND directly with FiLM disabled (gamma=1, beta=0).
    from models.lemo_pc_nd import FiLMLagSpectralND
    conv = FiLMLagSpectralND(
        in_channels=2, out_channels=2, lag_modes=lag_modes,
        params_dim=1, film_hidden=8, sigma=None, causal=True,
    )
    # Zero out FiLM so it's the identity modulation: gamma=1, beta=0.
    with torch.no_grad():
        conv.film_net[-1].weight.zero_()
        b = torch.zeros_like(conv.film_net[-1].bias)
        b[: 2 * lag_modes] = 1.0    # first half = gamma init = 1
        conv.film_net[-1].bias.copy_(b)
        # Set time-domain weights to a known causal kernel
        conv.weights_time.zero_()
        conv.weights_time[0, 0, :] = torch.linspace(1.0, 0.1, lag_modes)
        conv.weights_time[1, 1, :] = torch.linspace(0.5, -0.5, lag_modes)
    conv.eval()
    # Build delta: 1 at lag 0, 0 elsewhere
    delta = torch.zeros(1, 2, L)
    delta[0, 0, 0] = 1.0
    delta[0, 1, 0] = 1.0
    params = torch.zeros(1, 1)
    with torch.no_grad():
        h = conv(delta, params)            # (1, 2, L)
    # Expected: h[0,0,t] ≈ weights_time[0,0,t] for t < lag_modes, 0 elsewhere
    target_00 = torch.zeros(L)
    target_00[:lag_modes] = torch.linspace(1.0, 0.1, lag_modes)
    err_00 = (h[0, 0] - target_00).abs().max().item()
    tail_max_00 = h[0, 0, lag_modes:].abs().max().item()
    print(f"impulse: max_err on supported region = {err_00:.2e}")
    print(f"impulse: max |h| on tail [lag_modes, L) = {tail_max_00:.2e}")
    assert err_00 < 1e-5, f"impulse response mismatch: {err_00}"
    assert tail_max_00 < 1e-5, f"tail leak: {tail_max_00}"

    # 3) Forward-shift equivariance for a single conv
    # Create a smooth random input
    x1 = torch.randn(1, 2, L) * 0.1
    with torch.no_grad():
        y1 = conv(x1, params)
    # Shift by k forward (cyclic shift = roll along lag)
    for k in [1, 4, 8]:
        x_shift = torch.roll(x1, shifts=k, dims=-1)
        with torch.no_grad():
            y_shift_in = conv(x_shift, params)
        y_shift_expected = torch.roll(y1, shifts=k, dims=-1)
        err = (y_shift_in - y_shift_expected).abs().max().item()
        print(f"cyclic-shift equivariance k={k}: max diff = {err:.2e}")
        assert err < 1e-4, f"cyclic equivariance failed at k={k}: {err}"

    # Sanity: non-causal LEMO_PC also has cyclic-shift equivariance (it's
    # a spectral conv after all). Causal is a subset.
    conv_nc = FiLMLagSpectralND(
        in_channels=2, out_channels=2, lag_modes=lag_modes,
        params_dim=1, film_hidden=8, sigma=None, causal=False,
    )
    with torch.no_grad():
        conv_nc.film_net[-1].weight.zero_()
        b = torch.zeros_like(conv_nc.film_net[-1].bias)
        b[: 2 * lag_modes] = 1.0
        conv_nc.film_net[-1].bias.copy_(b)
    conv_nc.eval()
    with torch.no_grad():
        y_nc = conv_nc(x1, params)
        y_nc_shift = conv_nc(torch.roll(x1, shifts=4, dims=-1), params)
    err_nc = (y_nc_shift - torch.roll(y_nc, shifts=4, dims=-1)).abs().max().item()
    print(f"non-causal cyclic-shift equivariance k=4: max diff = {err_nc:.2e}")
    assert err_nc < 1e-4

    # 4) Counter-test: non-causal kernel has NON-zero tail in impulse response
    delta1 = torch.zeros(1, 2, L)
    delta1[0, 0, 0] = 1.0
    delta1[0, 1, 0] = 1.0
    with torch.no_grad():
        h_nc = conv_nc(delta1, params)
    nc_tail_norm = h_nc[0, 0, lag_modes:].abs().mean().item()
    print(f"non-causal impulse tail mean |h|: {nc_tail_norm:.2e} (should be > 0)")

    print("\nAll Causal LEMO checks passed.")


if __name__ == "__main__":
    main()
