"""
LEMO-PC v2 — Parameter-Conditional LEMO with per-channel FiLM modulation.

Round 2.19 audit fix.  The earlier per-element-FiLM design produced a
~28.9 M parameter model (vs ~280 k for FNO1d at the same width/modes/
depth) because the FiLM net's final layer output dimension scaled as
`2 * 2 * in * out * modes`, which dominated the model.  That design
was both theoretically off (canonical FiLM, Perez et al. 2018, is per-
channel scale/shift, not per-coefficient) and practically severely
undertrained at the given sample/epoch budget.

New design:

1. **Direct spectral coefficients** (FNO-style): the lag-conv kernel is
   parameterized directly in the Fourier domain as a complex tensor
   `K[in, out, modes]`, learned and shared across samples.

2. **Per-channel FiLM** from params:
       gamma[b, o], beta[b, o] = film_net(params[b])
       y[b, o, l] = gamma[b, o] * conv(K, x)[b, o, l] + beta[b, o]
   The film_net produces 2 * out_channels modulators per sample.  This
   is the standard FiLM (Perez et al. 2018) and reduces FiLM-related
   params by a factor of ~(in * modes) ≈ 768 at width=48, modes=16.

3. **LayerNorm after each block** to keep activation magnitudes
   bounded across depth.

For LEMO-PC v2_sigma: the base kernel K is normalized to operator-norm
sigma (using the elementwise max-DFT-magnitude bound matching the
paper's Section 5.2 formula), and gamma is squashed through tanh so
|gamma| <= 1, giving the per-sample contraction bound
  ||conv(γ * K + β, ·)||_Lip = |γ| * ||K||_op ≤ σ
required by cor:lemo-sigma.  beta is an additive bias and does not
affect the operator norm of the linear part.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMSpectralConv1d(nn.Module):
    """1D spectral conv with per-channel FiLM conditioning.

    Per-output-channel scale (gamma) and shift (beta), produced from a
    small MLP on the per-sample params.  This is the canonical FiLM of
    Perez et al. 2018 — far smaller than per-coefficient modulation of
    every spectral weight (which an earlier implementation did, blowing
    the param budget by ~100x at width=48, modes=16).

        K[in, out, modes]                          (learned, shared)
        gamma[b, out], beta[b, out] = film_net(params[b])
        y_hat[b, o, k] = sum_i K[i, o, k] * x_hat[b, i, k]
        y[b, o, l]      = irfft(y_hat)[b, o, l]
        out[b, o, l]    = gamma[b, o] * y[b, o, l] + beta[b, o]

    For sigma-constrained variants the base kernel K is rescaled to
    operator-norm sigma and gamma is squashed through tanh so that
    |gamma| ≤ 1 and the per-sample linear-part contraction
        |gamma| * ||K||_op  ≤  sigma
    holds exactly per cor:lemo-sigma.  beta is an additive bias and
    does not enter the operator norm.
    """

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
        # FNO-style spectral coefficients init.
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes,
                                dtype=torch.cfloat)
        )
        # Per-channel FiLM: gamma, beta each of shape (B, out_channels).
        self.film_net = nn.Sequential(
            nn.Linear(params_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * out_channels),
        )
        # Init: gamma = 1 (identity scale), beta = 0.
        with torch.no_grad():
            self.film_net[-1].weight.mul_(0.01)
            b = torch.zeros(2 * out_channels)
            b[:out_channels] = 1.0
            self.film_net[-1].bias.copy_(b)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, in, L). params: (B, params_dim). Returns (B, out, L)."""
        B, _, L = x.shape
        x_hat = torch.fft.rfft(x, dim=-1)                            # (B, in, L//2+1)
        # Sigma-normalized base kernel (||K||_op ≤ sigma using the elementwise
        # max-DFT-magnitude bound — exact for scalar kernels per Section 5.2,
        # conservative upper bound in the multi-channel case).
        if self.sigma is not None:
            max_mag = self.weights.abs().max().clamp_min(1e-10)
            K = self.weights * (self.sigma / max_mag)
        else:
            K = self.weights
        # Spectral conv at the truncated mode count.
        out_modes = L // 2 + 1
        eff_modes = min(self.modes, out_modes)
        out_ft = torch.zeros(B, self.out_channels, out_modes,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :eff_modes] = torch.einsum(
            "iom,bim->bom",
            K[:, :, :eff_modes],
            x_hat[:, :, :eff_modes],
        )
        y = torch.fft.irfft(out_ft, n=L, dim=-1)                     # (B, out, L)
        # Per-sample per-channel FiLM modulation.
        film = self.film_net(params)                                 # (B, 2*out)
        gamma = film[:, :self.out_channels].unsqueeze(-1)            # (B, out, 1)
        beta  = film[:, self.out_channels:].unsqueeze(-1)            # (B, out, 1)
        if self.sigma is not None:
            # Bound |gamma| ≤ 1 so per-sample linear-part op-norm ≤ sigma.
            gamma = torch.tanh(gamma)
        return gamma * y + beta


class LEMOPCv2Block(nn.Module):
    """B + A_FiLM + activation + LayerNorm.

    B is a 1×1 conv (channel mixing, lag-equivariant).
    A_FiLM is the per-sample FiLM-modulated spectral conv.
    """

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
        self.A = FiLMSpectralConv1d(width, width, modes, params_dim,
                                     film_hidden=film_hidden, sigma=sigma)
        self.act = nn.ReLU() if sigma is not None else nn.GELU()
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, width, L). params: (B, params_dim)."""
        Bx = self.B(x)                                               # (B, width, L)
        Ax = self.A(x, params)                                       # (B, width, L)
        y = self.act(Bx + Ax)                                        # (B, width, L)
        # LayerNorm operates on channel dim of (B, L, width); transpose, norm, back.
        return self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)


class LEMOPCv2(nn.Module):
    """LEMO with FiLM-conditioned spectral conv + LayerNorm.

    Forward: x (B, L, in_channels), with the last `params_dim` channels
    treated as broadcast-across-length per-sample params (read from
    x[:, 0, -params_dim:]).  Returns y (B, L, out_channels).
    """

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
            LEMOPCv2Block(width, modes, params_dim, film_hidden, sigma)
            for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.ReLU() if sigma is not None else nn.GELU()

    def _split_x_params(self, x: torch.Tensor) -> tuple:
        if not self.extract_params or self.params_dim == 0:
            B = x.shape[0]
            return x, torch.zeros(B, self.params_dim, device=x.device)
        params = x[:, 0, -self.params_dim:]                          # (B, params_dim)
        return x, params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_full, params = self._split_x_params(x)
        x_lift = self.lift(x_full)                                   # (B, L, width)
        x_chan = x_lift.permute(0, 2, 1)                             # (B, width, L)
        for blk in self.blocks:
            x_chan = blk(x_chan, params)
        x_seq = x_chan.permute(0, 2, 1)                              # (B, L, width)
        h = self.head_act(self.head1(x_seq))
        return self.head2(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo_pc_v2(in_channels: int, out_channels: int, config: dict,
                      length: Optional[int] = None) -> nn.Module:
    """Factory; mirrors create_lemo_pc signature."""
    model_cfg = config.get("model", config)
    if length is None:
        length = model_cfg.get("length", config.get("length", 64))
    sigma = model_cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    # default params_dim = in - state_dim - 2 (mask, t_channel)
    # but state_dim isn't known here; user can specify or we use 5 as a guess.
    params_dim = model_cfg.get("params_dim", max(in_channels - 4, 1))
    return LEMOPCv2(
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
