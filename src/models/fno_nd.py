"""
FNO-ND — N-dimensional Fourier Neural Operator baseline for ND PDEs.

Round 2.23.  A single FNO that handles 1D, 2D, 3D, and arbitrary higher-dim
operator-learning tasks via `torch.fft.rfftn` over a configurable set of
"physical" axes.  Recovers the existing `fno1d.py` numerically when the
spatial shape is empty (G2 regression).

Layout: input/output `(batch, *physical_shape, in_channels)` with
`physical_shape = (length,)` for 1D, `(length, H, W)` for 2D-with-time,
`(length, H, W, D)` for 3D-with-time, etc.

Architecture: lift -> N FNO blocks (spectral conv + 1×1 channel mix) -> proj.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvND(nn.Module):
    """ND spectral conv via rfftn over the physical axes.

    Input:  (batch, in_channels, *physical_shape)
    Output: (batch, out_channels, *physical_shape)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        physical_shape: Sequence[int],
        modes: Sequence[int],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.physical_shape = tuple(physical_shape)
        self.n_phys = len(self.physical_shape)
        max_modes = []
        for a, n in enumerate(self.physical_shape):
            if a == self.n_phys - 1:
                max_modes.append(n // 2 + 1)
            else:
                max_modes.append(n)
        self.modes = tuple(min(m, mm) for m, mm in zip(modes, max_modes))
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, *self.modes,
                                dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_phys == 0:
            return x
        phys_dims = tuple(range(2, 2 + self.n_phys))
        # Use the actual input shape, not the configured one — FNO is
        # resolution-agnostic on the input axes.
        actual_shape = tuple(x.shape[d] for d in phys_dims)
        x_hat = torch.fft.rfftn(x, dim=phys_dims)
        out_shape = list(x_hat.shape)
        out_shape[1] = self.out_channels
        out_ft = torch.zeros(*out_shape, dtype=torch.cfloat, device=x.device)
        # Cap modes by what's available in the actual rfft output
        # (last axis is rfft so its size is shape[-1]//2 + 1).
        eff_modes = []
        for i, m in enumerate(self.modes):
            avail = x_hat.shape[2 + i]
            eff_modes.append(min(m, avail))
        slc = [slice(None), slice(None)] + [slice(0, m) for m in eff_modes]
        x_block = x_hat[tuple(slc)]
        # Slice spectral weights to active modes (in case input is smaller than configured).
        w_slc = [slice(None), slice(None)] + [slice(0, m) for m in eff_modes]
        weights_block = self.weights[tuple(w_slc)]
        ax = "pqrstuv"[:self.n_phys]
        eq = f"io{ax},bi{ax}->bo{ax}"
        y_block = torch.einsum(eq, weights_block, x_block)
        out_ft[tuple(slc)] = y_block
        return torch.fft.irfftn(out_ft, s=actual_shape, dim=phys_dims)


class FNONDBlock(nn.Module):
    """Standard FNO block: spectral conv + 1x1 channel mix + activation."""

    def __init__(
        self,
        width: int,
        physical_shape: Sequence[int],
        modes: Sequence[int],
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.A = SpectralConvND(width, width, physical_shape, modes)
        # 1x1 channel-mix implemented as a Linear (dim-agnostic).
        self.W = nn.Linear(width, width)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, width, *physical). Returns same shape."""
        Ax = self.A(x)
        # 1x1 channel mix: move width to last, Linear, move back.
        x_chan_last = x.movedim(1, -1)
        Wx = self.W(x_chan_last).movedim(-1, 1)
        return self.dropout(self.act(Ax + Wx))


class FNOND(nn.Module):
    """N-dimensional Fourier Neural Operator.

    Input:  (batch, *physical_shape, in_channels)
    Output: (batch, *physical_shape, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        physical_shape: Sequence[int],
        modes: Optional[Sequence[int]] = None,
        width: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.physical_shape = tuple(physical_shape)
        self.n_phys = len(self.physical_shape)
        self.width = width
        if modes is None:
            modes = tuple(min(s // 2, 16) for s in self.physical_shape)
        self.modes = tuple(modes)

        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            FNONDBlock(width, self.physical_shape, self.modes,
                        activation=activation, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.proj1 = nn.Linear(width, width * 2)
        self.proj2 = nn.Linear(width * 2, out_channels)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, *physical, in_channels)."""
        x = self.lift(x)                                    # (B, *physical, width)
        # Channels-second.
        if self.n_phys == 0:
            x = x.unsqueeze(1)
        else:
            perm = [0, x.dim() - 1] + list(range(1, x.dim() - 1))
            x = x.permute(*perm)                              # (B, width, *physical)
        for blk in self.blocks:
            x = blk(x)
        # Channels-last for projection.
        if self.n_phys == 0:
            x = x.squeeze(1)
        else:
            inv = [0] + list(range(2, 2 + self.n_phys)) + [1]
            x = x.permute(*inv)                               # (B, *physical, width)
        x = self.act(self.proj1(x))
        return self.proj2(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_fno_nd(in_channels: int, out_channels: int, config: dict) -> nn.Module:
    """Factory.  Reads `model.physical_shape` and `model.modes`."""
    model_cfg = config.get("model", config)
    physical_shape = tuple(model_cfg.get("physical_shape", ()))
    if not physical_shape:
        # Fallback: 1D length-only.
        physical_shape = (model_cfg.get("length", 64),)
    modes = model_cfg.get("modes", None)
    if modes is not None and not isinstance(modes, (list, tuple)):
        modes = (modes,)
    return FNOND(
        in_channels=in_channels,
        out_channels=out_channels,
        physical_shape=physical_shape,
        modes=tuple(modes) if modes is not None else None,
        width=model_cfg.get("width", 32),
        n_layers=model_cfg.get("n_layers", 4),
        activation=model_cfg.get("activation", "gelu"),
        dropout=model_cfg.get("dropout", 0.0),
    )
