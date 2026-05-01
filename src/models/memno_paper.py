"""
Memory Neural Operator (MemNO), author-faithful architectural pattern.

Paper:  Buitrago, Marwah, Gu, Risteski, "On the Benefits of Memory for
        Modeling Time-Dependent PDEs," ICLR 2025 (arXiv:2409.02313).
Source: architecture code from the paper's supplementary material (not
        on a public GitHub as of April 2026).

The paper defines MemNO as

    (B, T, S, H)  --rearrange 'b t s h -> (b t) s h'-->
       first `markovian_layers[:memory_position+1]`
    --rearrange '(b t) s h -> (b s) t h'-->
       `memory_layer`  (causal sequential, paper uses S4 / Gu 2022)
    --rearrange '(b s) t h -> (b t) s h'-->
       remaining `markovian_layers[memory_position+1:]`
    --rearrange '(b t) s h -> b t s h'-->  output

For our 1D time series benchmark we have no genuine spatial axis, so we
set `S = 1` and `H = hidden`. With `S = 1` every `rearrange` between
`(b t) s h` and `b t s h` is a no-op, and the architecture reduces to

    pointwise lift  ->  N_pre pointwise MLPs  ->  S4D causal memory  ->
    N_post pointwise MLPs  ->  pointwise projection

which matches the author's pattern exactly in the degenerate S=1 case.

Memory backend: we include a minimal diagonal S4 (S4D; Gu 2022, "On the
Parameterization and Initialization of Diagonal State Space Models").
S4D is algebraically a strict subset of S4: diagonal A instead of DPLR.
For the MemNO paper's claimed scaling this is a faithful substitute for
small-to-medium `d_state`. Swapping in the full S4 or a Mamba backend
is a one-line change.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4D(nn.Module):
    """Diagonal State Space Model (S4D, Gu 2022).

    Independent per-channel diagonal state-space kernels; FFT-based causal
    convolution in O(L log L). Initialization makes A stable by
    construction (negative real part via ``-exp(log_A_real)``), and the
    imaginary parts of A are initialized on a uniform spacing on the
    imaginary axis (matching the S4D-Lin initialization from Gu 2022).

    Input:  (batch, length, d_model)
    Output: (batch, length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Learnable time step per channel, log-uniform init in [dt_min, dt_max]
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A = -exp(log_A_real) + i * A_imag  — diagonal, stable by construction.
        # S4D-Lin init: real parts ~ 0.5 (log = log 0.5), imag parts on pi * n.
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(d_state).float().expand(d_model, -1).contiguous())

        # B and C as complex, initialized by a Gaussian (standard S4D init).
        self.B_real = nn.Parameter(torch.randn(d_model, d_state))
        self.B_imag = nn.Parameter(torch.randn(d_model, d_state))
        self.C_real = nn.Parameter(torch.randn(d_model, d_state))
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state))

        # Skip connection coefficient (per channel).
        self.D = nn.Parameter(torch.randn(d_model))

    def kernel(self, L: int) -> torch.Tensor:
        """Causal convolution kernel of length L, shape (d_model, L)."""
        dt = torch.exp(self.log_dt)                               # (d_model,)
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)  # (d_model, d_state)
        B = torch.complex(self.B_real, self.B_imag)
        C = torch.complex(self.C_real, self.C_imag)
        dtA = A * dt.unsqueeze(-1)                                 # (d_model, d_state)

        l_idx = torch.arange(L, device=dt.device, dtype=torch.float32)
        # exp(l * dtA)  ->  shape (d_model, d_state, L)
        exp_term = torch.exp(dtA.unsqueeze(-1) * l_idx)
        # k[m, l] = sum_n (C_mn * B_mn) * exp(l * dtA_mn)
        CB = (C * B).unsqueeze(-1)
        k = (CB * exp_term).sum(dim=1).real                        # (d_model, L)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        L = x.shape[1]
        k = self.kernel(L)                       # (d_model, L)
        x_c = x.transpose(1, 2)                  # (B, d_model, L)
        x_f = torch.fft.rfft(x_c, n=2 * L, dim=-1)
        k_f = torch.fft.rfft(k, n=2 * L, dim=-1)
        y_f = x_f * k_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=2 * L, dim=-1)[..., :L]
        y = y.transpose(1, 2)                    # (B, L, d_model)
        return y + x * self.D.view(1, 1, -1)


class MemNO(nn.Module):
    """Memory Neural Operator, author-faithful architectural pattern.

    Matches the supplementary code in the paper: a list of pointwise
    ("markovian") layers with a causal sequential ("memory") layer
    interleaved at position ``memory_position``. For our 1D time-series
    benchmark the spatial axis is trivial (S=1), so the markovian layers
    are pointwise channel MLPs and the memory layer is an S4D over the
    length dimension.

    Parameters mirror the paper's names: ``hidden`` is H, ``n_markovian``
    is the total number of markovian layers (split around the memory
    layer at ``memory_position``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        hidden: int = 64,
        n_markovian: int = 4,
        memory_position: int = 2,
        s4_state: int = 64,
        **_ignored,
    ) -> None:
        super().__init__()
        if memory_position < 0 or memory_position >= n_markovian:
            raise ValueError(
                f"memory_position={memory_position} out of range for n_markovian={n_markovian}"
            )
        self.lift = nn.Linear(in_channels, hidden)
        self.markovian_layers = nn.ModuleList(
            [self._make_markovian(hidden) for _ in range(n_markovian)]
        )
        self.memory_layer = S4D(hidden, d_state=s4_state)
        self.memory_position = memory_position
        self.proj = nn.Linear(hidden, out_channels)

    @staticmethod
    def _make_markovian(hidden: int) -> nn.Module:
        # Pointwise MLP: a 2-layer block applied independently at each
        # time step. In the paper's original S != 1 setting this would
        # be a spatial FNO block; for S=1 it collapses to channel mixing.
        return nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_channels). Treat as (B, T=L, S=1, H=hidden).
        h = self.lift(x)  # (B, L, hidden)
        # First half of markovian layers (pointwise on channels; S=1 collapses
        # the 'b t s h -> (b t) s h' rearrange to identity).
        for layer in self.markovian_layers[: self.memory_position + 1]:
            h = layer(h)
        # Memory layer: causal sequential over time (the '(b t) s h -> (b s) t h'
        # rearrange with S=1 also collapses to identity).
        h = self.memory_layer(h)
        # Remaining markovian layers.
        for layer in self.markovian_layers[self.memory_position + 1 :]:
            h = layer(h)
        return self.proj(h)
