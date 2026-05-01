"""
LEMO-PC — Parameter-Conditional Lag-Equivariant Memory Operator.

Generalizes LEMOv2 (`src/models/lemo_v2.py`) to per-sample parametric
operators by making the kernel `kernel_net(offset, params)` rather than
just `kernel_net(offset)`.

Diagnostic finding (Round 2.15, sweep_phase_b_v1): both LEMO and LNO
(shared-kernel architectures) trivially fail (rel L2 ≈ 1.0) on every
parametric DDE family in Phase B because they cannot represent
per-sample operator variation. FNO1d trains because its
`pointwise_conv` reads the param channels and modulates the residual
path. LEMO's pointwise B in v2 has the same access but can only do
channel mixing; it cannot rescale the spectral kernel.

LEMO-PC fixes this by giving the kernel_net the per-sample params:

    K[b, g, out, in] = kernel_net(concat(offset[g], params[b]))

so that each sample in the batch sees a sample-specific kernel. The
spectral conv then becomes a per-sample cyclic conv:

    y_hat[b, c_out, l] = sum_{c_in}  dft_K[b, c_out, c_in, l] * dft_x[b, c_in, l].

For LEMO-PC_σ, the operator-norm constraint is applied per-sample
(each sample's kernel is normalized to <= sigma).

Lag-equivariance is preserved (the kernel is still a function of
the relative lag offset, just modulated by per-sample params).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParamConditionalLagConv1d(nn.Module):
    """Cyclic lag conv with kernel `K(offset, params)` — per-sample conditional."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        params_dim: int,
        kernel_hidden: int = 64,
        sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.params_dim = params_dim
        self.sigma = sigma
        # Input to kernel_net: 1 (offset) + params_dim
        in_dim = 1 + params_dim
        self.kernel_net = nn.Sequential(
            nn.Linear(in_dim, kernel_hidden),
            nn.ReLU(),
            nn.Linear(kernel_hidden, kernel_hidden),
            nn.ReLU(),
            nn.Linear(kernel_hidden, out_channels * in_channels),
        )
        # Calibrated init: per-element kernel std ~ 1/sqrt(L*in)
        # so the cyclic conv preserves activation magnitude (same
        # principle as LEMOv2 / FNO1d's spectral coeff scaling).
        target_std = 1.0 / math.sqrt(max(length * in_channels, 1))
        with torch.no_grad():
            final = self.kernel_net[-1]
            nn.init.normal_(final.weight, mean=0.0,
                            std=target_std / math.sqrt(kernel_hidden))
            nn.init.zeros_(final.bias)

    def _offsets(self, length: int, device: torch.device) -> torch.Tensor:
        """Signed normalized lag offsets in [-1/2, 1/2)."""
        idx = torch.arange(length, device=device, dtype=torch.float32)
        signed = torch.where(idx > length / 2, idx - length, idx)
        return (signed / length).unsqueeze(-1)              # (length, 1)

    def compute_kernel(self, params: torch.Tensor) -> torch.Tensor:
        """Per-sample kernel `K[b, out, in, length]`.

        params: (B, params_dim).  Returns (B, out, in, L).
        """
        B = params.shape[0]
        device = params.device
        offsets = self._offsets(self.length, device)        # (L, 1)
        # Build (B, L, 1+params_dim): offset broadcast across batch,
        # params broadcast across length.
        off = offsets.unsqueeze(0).expand(B, -1, -1)        # (B, L, 1)
        par = params.unsqueeze(1).expand(-1, self.length, -1)  # (B, L, params_dim)
        net_in = torch.cat([off, par], dim=-1)              # (B, L, 1+params_dim)
        K_flat = self.kernel_net(net_in)                    # (B, L, out*in)
        K = K_flat.view(B, self.length,
                         self.out_channels, self.in_channels)  # (B, L, out, in)
        K = K.permute(0, 2, 3, 1).contiguous()              # (B, out, in, L)
        if self.sigma is None:
            return K
        # Per-sample sigma normalization via Frobenius upper bound on |K_hat|.
        K_hat = torch.fft.rfft(K, dim=-1)                   # (B, out, in, L//2+1)
        frob_per_freq = K_hat.abs().pow(2).sum(dim=(1, 2)).sqrt()  # (B, L//2+1)
        max_frob = frob_per_freq.max(dim=-1).values.clamp_min(1e-10)  # (B,)
        scale = (self.sigma / max_frob).view(-1, 1, 1, 1)   # (B, 1, 1, 1)
        return K * scale

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, in, L), params: (B, params_dim)."""
        B, _, L = x.shape
        K = self.compute_kernel(params)                     # (B, out, in, L)
        # Per-sample cyclic conv via FFT.
        K_hat = torch.fft.rfft(K, dim=-1)                   # (B, out, in, L//2+1)
        x_hat = torch.fft.rfft(x, dim=-1)                   # (B, in, L//2+1)
        y_hat = torch.einsum("boil,bil->bol", K_hat, x_hat)
        y = torch.fft.irfft(y_hat, n=L, dim=-1)             # (B, out, L)
        return y


class PointwiseNorm(nn.Module):
    """sigma-constrained pointwise linear via Frobenius upper bound."""
    def __init__(self, linear: nn.Linear, sigma: Optional[float] = None) -> None:
        super().__init__()
        self.linear = linear
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma is None:
            return self.linear(x)
        w = self.linear.weight
        frob = w.norm().clamp_min(1e-10)
        w_scaled = w / frob * self.sigma
        return F.linear(x, w_scaled, self.linear.bias)


class LEMOPCBlock(nn.Module):
    """B + A_PC + activation.  A_PC is the parameter-conditional lag conv."""

    def __init__(
        self,
        width: int,
        length: int,
        params_dim: int,
        kernel_hidden: int,
        sigma: Optional[float],
    ) -> None:
        super().__init__()
        self.width = width
        self.sigma = sigma
        self._B = nn.Linear(width, width)
        self.B = PointwiseNorm(self._B, sigma)
        self.A = ParamConditionalLagConv1d(
            width, width, length, params_dim,
            kernel_hidden=kernel_hidden, sigma=sigma,
        )
        # ReLU is required when sigma is set (1-Lipschitz);
        # use GELU otherwise for better optimization.
        self.act = nn.ReLU() if sigma is not None else nn.GELU()

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, width, L), params: (B, params_dim)."""
        x_chan_last = x.permute(0, 2, 1)                    # (B, L, width)
        Bx = self.B(x_chan_last).permute(0, 2, 1)           # (B, width, L)
        Ax = self.A(x, params)                              # (B, width, L)
        return self.act(Bx + Ax)


class LEMOPC(nn.Module):
    """LEMO with per-sample parameter-conditional kernels.

    Forward:
        x:      (B, L, in_channels)   -- data channels (history + ...)
        params: (B, params_dim)       -- per-sample family parameters
        y:      (B, L, out_channels)  -- predicted sequence

    For drop-in compatibility with the train_fno_sharded pipeline (which
    feeds a single concatenated input tensor of shape (B, L, in_channels)
    where the last `params_dim` channels are the broadcast params),
    `forward(x)` extracts params from the last `params_dim` channels of x
    and routes them to the conditional kernel.  See `extract_params=True`
    in __init__.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        params_dim: int,
        width: int = 48,
        n_layers: int = 3,
        kernel_hidden: int = 64,
        sigma: Optional[float] = None,
        extract_params: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.params_dim = params_dim
        self.extract_params = extract_params
        self.width = width
        self.sigma = sigma

        # Lift takes the FULL input (data + params channels broadcast).
        self._lift = nn.Linear(in_channels, width)
        self.lift = PointwiseNorm(self._lift, sigma)

        self.blocks = nn.ModuleList([
            LEMOPCBlock(width, length, params_dim, kernel_hidden, sigma)
            for _ in range(n_layers)
        ])

        self._head1 = nn.Linear(width, width)
        self._head2 = nn.Linear(width, out_channels)
        self.head1 = PointwiseNorm(self._head1, sigma)
        self.head2 = PointwiseNorm(self._head2, sigma)
        self.head_act = nn.ReLU() if sigma is not None else nn.GELU()

    def _split_x_params(self, x: torch.Tensor) -> tuple:
        """If `extract_params`, split the last `params_dim` channels of x as params.
        params are broadcast across length in the input; we read them at length=0."""
        if not self.extract_params or self.params_dim == 0:
            B = x.shape[0]
            params = torch.zeros(B, self.params_dim, device=x.device)
            return x, params
        params = x[:, 0, -self.params_dim:]                 # (B, params_dim)
        return x, params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, in_channels). Returns (B, L, out_channels)."""
        x_full, params = self._split_x_params(x)
        x_lift = self.lift(x_full)                          # (B, L, width)
        x_chan = x_lift.permute(0, 2, 1)                    # (B, width, L)
        for blk in self.blocks:
            x_chan = blk(x_chan, params)                    # (B, width, L)
        x_seq = x_chan.permute(0, 2, 1)                     # (B, L, width)
        h = self.head_act(self.head1(x_seq))
        return self.head2(h)                                # (B, L, out_channels)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo_pc(in_channels: int, out_channels: int, config: dict,
                    length: Optional[int] = None) -> nn.Module:
    """Factory.  Reads sigma, params_dim, etc. from config['model'].
    `length` arg, when given, overrides any length in config (the trainer
    passes seq_length from the data shape)."""
    model_cfg = config.get("model", config)
    if length is None:
        length = model_cfg.get("length", config.get("length", 64))
    sigma = model_cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    # params_dim: defaults to in_channels - state_dim (rest are param channels).
    # We can't know state_dim from the trainer signature, so trust the config.
    # If unset, default to the entire input minus 1 channel for the history.
    params_dim = model_cfg.get("params_dim", max(in_channels - 1, 0))
    return LEMOPC(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        params_dim=params_dim,
        width=model_cfg.get("width", 48),
        n_layers=model_cfg.get("n_layers", 3),
        kernel_hidden=model_cfg.get("kernel_hidden", 64),
        sigma=sigma,
        extract_params=model_cfg.get("extract_params", True),
    )
