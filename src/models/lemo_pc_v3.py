"""
LEMO-PC v3 — per-(out_channel, mode) FiLM in the spectral domain.

Round 2.20 follow-up to lemo_pc_v2 (Round 2.19, per-channel FiLM).
v2 fixed the 28.9M-param explosion but lost expressivity on tasks
where different spectral modes need different per-sample weighting:
linear2 (coupled oscillators with varying coupling strength, where
sample-specific coupling shifts dominant frequency) and
predator_prey (param-dependent nonlinear dynamics where the
relevant modes shift with the params).

v3 design: FiLM operates in the **spectral domain**, with
γ, β shape `(B, out_channels, modes)`.  This gives per-sample
per-mode reweighting of the spectral output, while keeping the
parameter count modest (~830k for the full model at width=48,
modes=16, n_layers=3 — vs 141k for v2 and 28.9M for the broken
original).

Architecture per layer:
    K[in, out, modes]                          (learned, shared)
    γ, β  shape (B, out, modes), from film_net(params)
    out_ft[b, o, k] = γ[b, o, k] * (Σ_i K[i,o,k] x_hat[b,i,k])
                    + β[b, o, k]               (real-valued FiLM)
    y[b, o, l]      = irfft(out_ft)[b, o, l]

For σ-constrained variants the base kernel K is normalized to
operator norm σ (elementwise max-DFT-magnitude bound), and γ is
squashed through tanh per element so |γ[b, o, k]| ≤ 1, giving the
per-sample per-mode bound  |γ| · ||K||_op ≤ σ.  β is additive and
does not enter the op norm.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMSpectralConv1dV3(nn.Module):
    """1D spectral conv with per-(out, mode) FiLM in the spectral domain."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        params_dim: int,
        film_hidden: int = 64,
        sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.params_dim = params_dim
        self.sigma = sigma
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes,
                                dtype=torch.cfloat)
        )
        # Per-(out, mode) FiLM: real-valued (γ, β) for each spectral coefficient.
        self.film_net = nn.Sequential(
            nn.Linear(params_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * out_channels * modes),
        )
        # Init: γ = 1 (identity), β = 0.
        with torch.no_grad():
            self.film_net[-1].weight.mul_(0.01)
            b = torch.zeros(2 * out_channels * modes)
            b[:out_channels * modes] = 1.0
            self.film_net[-1].bias.copy_(b)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, in, L). params: (B, params_dim). Returns (B, out, L)."""
        B, _, L = x.shape
        x_hat = torch.fft.rfft(x, dim=-1)
        if self.sigma is not None:
            max_mag = self.weights.abs().max().clamp_min(1e-10)
            K = self.weights * (self.sigma / max_mag)
        else:
            K = self.weights
        out_modes = L // 2 + 1
        eff_modes = min(self.modes, out_modes)
        # Spectral conv at the truncated mode count (out-of-place build).
        active = torch.einsum(
            "iom,bim->bom",
            K[:, :, :eff_modes],
            x_hat[:, :, :eff_modes],
        )                                              # (B, out, eff_modes)
        # Per-(out, mode) FiLM in spectral domain.
        film = self.film_net(params)
        OC, M = self.out_channels, self.modes
        gamma = film[:, :OC * M].view(B, OC, M)        # (B, out, modes), real
        beta  = film[:, OC * M:].view(B, OC, M)        # (B, out, modes), real
        if self.sigma is not None:
            gamma = torch.tanh(gamma)                  # |γ| ≤ 1 per (b, o, m)
        eff = min(eff_modes, M)
        active = (gamma[:, :, :eff].to(torch.cfloat) * active[:, :, :eff]
                  + beta[:, :, :eff].to(torch.cfloat))
        # Pad to full rfft length with zeros (out-of-place).
        if out_modes > eff:
            pad = torch.zeros(B, self.out_channels, out_modes - eff,
                              dtype=torch.cfloat, device=x.device)
            out_ft = torch.cat([active, pad], dim=-1)
        else:
            out_ft = active
        return torch.fft.irfft(out_ft, n=L, dim=-1)


class LEMOPCv3Block(nn.Module):
    """B + A_FiLM_v3 + activation + LayerNorm.  Same skeleton as v2."""

    def __init__(
        self,
        width: int,
        modes: int,
        params_dim: int,
        film_hidden: int = 64,
        sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.B = nn.Conv1d(width, width, 1)
        self.A = FiLMSpectralConv1dV3(width, width, modes, params_dim,
                                       film_hidden=film_hidden, sigma=sigma)
        self.act = nn.ReLU() if sigma is not None else nn.GELU()
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        Bx = self.B(x)
        Ax = self.A(x, params)
        y = self.act(Bx + Ax)
        return self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)


class LEMOPCv3(nn.Module):
    """Drop-in replacement for LEMOPCv2 with spectral-domain per-mode FiLM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        params_dim: int,
        modes: int = 16,
        width: int = 48,
        n_layers: int = 3,
        film_hidden: int = 64,
        sigma: Optional[float] = None,
        extract_params: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.params_dim = params_dim
        self.extract_params = extract_params
        self.modes = modes
        self.width = width
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            LEMOPCv3Block(width, modes, params_dim, film_hidden, sigma)
            for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.ReLU() if sigma is not None else nn.GELU()

    def _split_x_params(self, x: torch.Tensor) -> tuple:
        if not self.extract_params or self.params_dim == 0:
            B = x.shape[0]
            return x, torch.zeros(B, self.params_dim, device=x.device)
        params = x[:, 0, -self.params_dim:]
        return x, params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_full, params = self._split_x_params(x)
        x_lift = self.lift(x_full)
        x_chan = x_lift.permute(0, 2, 1)
        for blk in self.blocks:
            x_chan = blk(x_chan, params)
        x_seq = x_chan.permute(0, 2, 1)
        h = self.head_act(self.head1(x_seq))
        return self.head2(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo_pc_v3(in_channels: int, out_channels: int, config: dict,
                      length: Optional[int] = None) -> nn.Module:
    """Factory; mirrors create_lemo_pc_v2 signature."""
    model_cfg = config.get("model", config)
    if length is None:
        length = model_cfg.get("length", config.get("length", 64))
    sigma = model_cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    params_dim = model_cfg.get("params_dim", max(in_channels - 4, 1))
    return LEMOPCv3(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        params_dim=params_dim,
        modes=model_cfg.get("modes", 16),
        width=model_cfg.get("width", 48),
        n_layers=model_cfg.get("n_layers", 3),
        film_hidden=model_cfg.get("film_hidden", 64),
        sigma=sigma,
        extract_params=model_cfg.get("extract_params", True),
    )
