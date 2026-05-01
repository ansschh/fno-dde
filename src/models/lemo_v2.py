"""
LEMO v2 — Lag-Equivariant Memory Operator with the missing pointwise term.

Diagnostic finding (scripts/diagnose_lemo_init.py): the original
src/models/lemo.py is missing the `B_ell U(j)` pointwise channel-mixing
term that the paper explicitly defines in
sections/proposed_framework_method_theory.tex:36-46:

    (L_ell U)(j) = B_ell(z,p) U(j)                  <-- MISSING
                 + sum_r A_{ell,r}(z,p) U(j-r)      <-- the spectral conv
                 + b_ell(z,p).

Without B, the ONLY path through each block is the spectral conv. Forward
magnitudes either explode (default init: ~5x per layer) or collapse
(post-1/length patch: ~0.1x per layer). FNO1d trains because it has
exactly the same B + A structure (`pointwise_conv + spectral_conv`).

This file adds the B term and uses FNO-style init scaling. Lag
equivariance is preserved because B is a 1x1 conv along the channel
axis (pointwise across lag) and A is the cyclic conv.

For the LEMO_sigma subclass (Cor 16), B is constrained the same way as
the existing lift/proj: spectral norm <= sigma via Frobenius upper bound.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Lag-convolution primitives
# -----------------------------------------------------------------------------

class ContinuousLagConv1dV2(nn.Module):
    """Cyclic lag convolution with the kernel parameterized as an MLP of the
    (signed, normalized) lag offset.  Identical to v1 except for the
    final-layer init scaling.

    Init: the kernel_net's final Linear is scaled so the per-element kernel
    magnitude is ~1/sqrt(L * in_channels).  This matches FNO1d's
    `scale = 1 / (in*out)` principle: the spectral coefficients of a fresh
    cyclic-conv layer are O(1/sqrt(L * in)), small enough that a single layer
    leaves the activation magnitude approximately preserved on i.i.d. inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        kernel_hidden: int = 64,
        sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.sigma = sigma
        self.kernel_net = nn.Sequential(
            nn.Linear(1, kernel_hidden),
            nn.ReLU(),
            nn.Linear(kernel_hidden, kernel_hidden),
            nn.ReLU(),
            nn.Linear(kernel_hidden, out_channels * in_channels),
        )
        # Calibrated init: target per-element kernel magnitude ~1/sqrt(L*in).
        # The final Linear has fan_in = kernel_hidden; standard Kaiming gives
        # output magnitude ~1.  We rescale to get magnitude 1/sqrt(L*in).
        target_std = 1.0 / math.sqrt(max(length * in_channels, 1))
        with torch.no_grad():
            final = self.kernel_net[-1]
            # Reinit with calibrated std.
            nn.init.normal_(final.weight, mean=0.0,
                            std=target_std / math.sqrt(kernel_hidden))
            nn.init.zeros_(final.bias)

    def _offsets(self, length: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(length, device=device, dtype=torch.float32)
        signed = torch.where(idx > length / 2, idx - length, idx)
        return (signed / length).unsqueeze(-1)

    def compute_kernel(self, length: Optional[int] = None) -> torch.Tensor:
        length = length or self.length
        device = next(self.parameters()).device
        offsets = self._offsets(length, device)
        K_flat = self.kernel_net(offsets)
        K = K_flat.view(length, self.out_channels, self.in_channels).permute(1, 2, 0)
        if self.sigma is None:
            return K
        K_hat = torch.fft.rfft(K, dim=-1)
        # Use Frobenius upper bound (Cor 16's caveat: any upper bound suffices
        # for the contraction certificate; Frobenius is conservative but easy.)
        frob = K_hat.abs().pow(2).sum(dim=(0, 1)).sqrt()
        max_frob = frob.max().clamp_min(1e-10)
        return K / max_frob * self.sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, length)
        length = x.shape[-1]
        K = self.compute_kernel(length)
        K_hat = torch.fft.rfft(K, dim=-1)
        x_hat = torch.fft.rfft(x, dim=-1)
        y_hat = torch.einsum("oil,bil->bol", K_hat, x_hat)
        y = torch.fft.irfft(y_hat, n=length, dim=-1)
        return y


class PointwiseNorm(nn.Module):
    """Spectral-norm-constrained pointwise linear (Frobenius upper bound)."""
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


class LEMOBlock(nn.Module):
    """A single LEMO block: pointwise B + spectral conv A + activation.

    Implements the paper's
        (L_ell U)(j) = B_ell U(j) + sum_r A_{ell,r} U(j-r) + b_ell.

    Both B and A preserve lag-equivariance:
      - B is a 1x1 conv along channels, applied identically at every lag j.
      - A is a cyclic lag-conv with the kernel shared across absolute lag.
    The activation is pointwise (and 1-Lipschitz when ReLU).
    """

    def __init__(
        self,
        width: int,
        length: int,
        kernel_hidden: int,
        sigma: Optional[float],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.width = width
        self.sigma = sigma
        # B: pointwise channel mixing (a 1x1 Conv along the lag axis is the
        # same as a Linear applied at each lag point).
        self._B_linear = nn.Linear(width, width)
        self.B = PointwiseNorm(self._B_linear, sigma)
        # A: spectral lag-conv (the equivariant memory term).
        self.A = ContinuousLagConv1dV2(width, width, length,
                                        kernel_hidden=kernel_hidden,
                                        sigma=sigma)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            if sigma is not None:
                raise ValueError(
                    "LEMO_sigma requires activation='relu' (1-Lipschitz)."
                )
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, width, length).  Returns (batch, width, length)."""
        # B is pointwise across lag: apply Linear on channel axis, which is
        # equivalent to permuting -> Linear -> permuting back, OR using a
        # 1x1 conv along length.  Using transpose:
        x_chan_last = x.permute(0, 2, 1)             # (batch, length, width)
        Bx = self.B(x_chan_last).permute(0, 2, 1)    # (batch, width, length)
        Ax = self.A(x)                                # (batch, width, length)
        y = self.act(Bx + Ax)
        return y


# -----------------------------------------------------------------------------
# LEMO v2 model
# -----------------------------------------------------------------------------

class LEMOv2(nn.Module):
    """Lag-Equivariant Memory Operator, v2 (with pointwise B term).

    Architecture:
        x : (batch, length, in_channels)
          -> Linear lift to width channels                                (LIFT)
          -> N blocks of (B + A + ReLU)                                   (BODY)
          -> Linear projection to width then to out_channels (with ReLU)  (HEAD)
        y : (batch, length, out_channels)

    LEMO_sigma subclass: every Linear (lift, B, proj1, proj2) and every
    spectral kernel A is normalized to operator norm <= sigma via the
    Frobenius upper bound.  With sigma < 1 and ReLU, the model is
    contractive in Euclidean norm by Cor 16.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        width: int = 48,
        n_layers: int = 3,
        kernel_hidden: int = 64,
        sigma: Optional[float] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.width = width
        self.n_layers = n_layers
        self.sigma = sigma

        self._lift_linear = nn.Linear(in_channels, width)
        self.lift = PointwiseNorm(self._lift_linear, sigma)

        self.blocks = nn.ModuleList([
            LEMOBlock(width, length, kernel_hidden, sigma, activation)
            for _ in range(n_layers)
        ])

        # Two-layer projection head with activation in between (mirrors FNO).
        self._proj1_linear = nn.Linear(width, width)
        self._proj2_linear = nn.Linear(width, out_channels)
        self.proj1 = PointwiseNorm(self._proj1_linear, sigma)
        self.proj2 = PointwiseNorm(self._proj2_linear, sigma)
        if activation == "relu":
            self.proj_act = nn.ReLU()
        else:
            if sigma is not None:
                raise ValueError(
                    "LEMO_sigma requires activation='relu' (1-Lipschitz)."
                )
            self.proj_act = nn.GELU()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, in_channels).  Returns (batch, length, out_channels)."""
        x = self.lift(x)                  # (batch, length, width)
        x = x.permute(0, 2, 1)            # (batch, width, length)
        for blk in self.blocks:
            x = blk(x)
            x = self.dropout(x)
        x = x.permute(0, 2, 1)            # (batch, length, width)
        x = self.proj1(x)
        x = self.proj_act(x)
        x = self.proj2(x)                 # (batch, length, out_channels)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo_v2(
    in_channels: int,
    out_channels: int,
    config: dict,
) -> nn.Module:
    """Factory; mirrors create_lemo / create_fno1d."""
    model_cfg = config.get("model", config)
    length = model_cfg.get("length", config.get("length", 64))
    sigma = model_cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    return LEMOv2(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        width=model_cfg.get("width", 48),
        n_layers=model_cfg.get("n_layers", 3),
        kernel_hidden=model_cfg.get("kernel_hidden", 64),
        sigma=sigma,
        activation=model_cfg.get("activation", "relu"),
        dropout=model_cfg.get("dropout", 0.0),
    )
