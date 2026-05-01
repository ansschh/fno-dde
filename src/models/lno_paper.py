"""
Laplace Neural Operator (LNO), vendored from the authors' reference
implementation.

Paper:  Cao, Goswami, Karniadakis, "Laplace neural operator for solving
        differential equations," Nature Machine Intelligence 6, 631-640 (2024).
Source: https://github.com/qianyingcao/Laplace-Neural-Operator
        Path: 1D_Duffing_c0/main.py
        Commit: main branch HEAD at time of vendoring
Attribution: original author Qianying Cao (qianying_cao@brown.edu)
License: see dde-fno/third_party/lno_cao/LICENSE

Modifications compared to the upstream file (minimal; surgical only):
  1. The global `grid_x_train.cuda()` access inside `PR.forward` is replaced
     by an explicit `t` argument computed from the input length at forward
     time. This is a trivial refactor; the Laplace-layer math is byte-
     identical (same pole/residue parameterization, same FFT/IFFT flow,
     same einsum contractions, same transient + steady-state decomposition).
  2. The `LNO1d.__init__` signature accepts `in_channels` and `out_channels`
     (the upstream version hard-codes 1 for both). All other hyperparameters
     (width=4, modes=16, sin activation in the projection head) are kept at
     the paper's 1D Duffing defaults.
  3. Script-level training/data/saving code was not vendored; only the
     model classes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PR(nn.Module):
    """Pole-Residue Laplace layer. Verbatim from the upstream code except
    that the time grid `t` is passed explicitly instead of being pulled
    from the global `grid_x_train`.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights_pole = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )
        self.weights_residue = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    def output_PR(self, lambda1, alpha, weights_pole, weights_residue):
        term1 = torch.div(1, torch.sub(lambda1, weights_pole))
        Hw = weights_residue * term1
        output_residue1 = torch.einsum("bix,xiok->box", alpha, Hw)
        output_residue2 = torch.einsum("bix,xiok->bok", alpha, -Hw)
        return output_residue1, output_residue2

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        # t: (length,) uniformly-spaced time grid
        dt = (t[1] - t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0 = torch.fft.fftfreq(t.shape[0], dt) * 2 * np.pi * 1j
        lambda1 = lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(alpha.device)

        output_residue1, output_residue2 = self.output_PR(
            lambda1, alpha, self.weights_pole, self.weights_residue
        )

        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)

        term1 = torch.einsum(
            "bix,kz->bixz",
            self.weights_pole,
            t.type(torch.complex64).reshape(1, -1).to(x.device),
        )
        term2 = torch.exp(term1)
        x2 = torch.einsum("bix,ioxz->boz", output_residue2, term2)
        x2 = torch.real(x2)
        x2 = x2 / x.size(-1)
        return x1 + x2


class LNO1d(nn.Module):
    """Laplace Neural Operator, 1D, author-faithful.

    Default hyperparameters match the paper's 1D Duffing configuration:
    ``width=4``, ``modes=16``, sin activation in the projection head.
    The ``fc0``/``fc2`` linear maps are generalized to arbitrary input and
    output channel counts; the rest matches the upstream file byte-for-byte.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 4,
        modes: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes

        self.fc0 = nn.Linear(in_channels, self.width)
        self.conv0 = PR(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, length, in_channels)
        length = x.shape[1]
        t = torch.linspace(0.0, 1.0, length, device=x.device, dtype=torch.float32)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # (batch, width, length)

        x1 = self.conv0(x, t)
        x2 = self.w0(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)  # (batch, length, width)
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)
        return x
