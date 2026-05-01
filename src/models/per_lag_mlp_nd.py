"""
Per-LAG-position MLP — non-equivariant baseline for the orbit-OOD experiment.

Round 2.26 (T2 orbit OOD).  This is the explicit non-equivariant counterpart
to LEMO-PC-ND for the lag-shift orbit experiment described in
``scripts/orbit_ood_design.md``.

Design:
  Input/output layout: ``(B, T, *spatial, in_channels)`` -> ``(B, T, *spatial,
  out_channels)`` — identical interface to ``LEMOPCND`` so dispatch in
  ``train.build_model`` is uniform.

  CRUCIAL: NO cyclic-shift weight sharing across the lag axis.  Each absolute
  lag position ``t in {0, ..., T-1}`` owns its own dense map and 1x1 spatial
  block.  This realizes the unrestricted hypothesis class ``F`` of
  ``Cref{thm:ood-gap}`` part (ii): no equivariance constraint.

  Per-lag dense channel mix
  -------------------------
  We keep a learnable 3D weight tensor ``W[t, c_in, c_out]`` and bias
  ``b[t, c_out]``.  The channel mix at lag t is a separate ``Linear`` per t
  (equivalent to ``W[t]`` applied to the channel axis).  In tensor
  shorthand::

      output[b, t, ..., c_out] = sum_{c_in} W[t, c_in, c_out] * input[b, t, ..., c_in]
                               + b[t, c_out]

  Per-lag 1x1 spatial conv
  ------------------------
  An additional ``W_spatial[t, c_in, c_out]`` (also independent across t)
  applied as a 1x1 spatial conv at each lag.  This is a second learnable
  channel mix; combined with the first via a residual + GELU gives a width-
  preserving block.  The 1x1 spatial nonlinear block contributes comparable
  expressive power to LEMO-PC's spectral lag conv + spatial FNO without
  sharing weights across lags.

  Lift / project heads use ordinary ``nn.Linear`` (shared across all lags)
  on the channel axis — matching LEMO-PC's lift / proj.  Equivariance
  breaking is concentrated in the per-lag block weights.

Parameter count
---------------
  width=64, n_layers=3, length=64, in_channels=4, out_channels=1::

      lift:        4 -> 64                         = 4*64 + 64           = 320
      block (per-lag W):  64*64*length             = 64*64*64            ~= 262K
      block (per-lag b):  64*length                = 64*64               = 4096
      block (per-lag spatial): 64*64*length        = 64*64*64            ~= 262K
      block (per-lag spatial b): 64*length         = 64*64               = 4096
      n_layers blocks: 3 * (262K + 4K + 262K + 4K) = ~1.6M
      head: 64 -> 64 -> 1                          = ~4K
      total: ~1.6M
  Comparable to LEMO-PC-ND (~2.7M with FiLM, lag spectral conv, spatial FNO).
  Knob: increase ``width`` to push parity higher if needed (handled by the
  factory which honors ``config['model']['width']``).

ND-correctness note
-------------------
  We support 0D / 1D / 2D / 3D spatial via the input rank.  Spatial axes are
  treated as a flat batch over which the per-lag 1x1 channel mix is applied
  uniformly.  The "1x1 spatial" conv is itself just a per-lag channel mix
  applied at every spatial location — i.e. there is no actual spatial
  receptive field in the baseline.  This is intentional: the per-lag MLP is
  meant to demonstrate that a lag-non-equivariant operator without spatial
  inductive bias still trains on the in-orbit shifts but fails to transfer
  to held-out shifts in ``S_test``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerLagBlock(nn.Module):
    """One per-lag-position block: per-lag dense channel mix + per-lag 1x1
    spatial channel mix + GELU + LayerNorm + residual.

    No weights are shared across the lag axis: every block holds two 3D
    weight tensors of shape ``(length, width, width)`` (one for the
    "channel mix" branch, one for the "1x1 spatial" branch) and two 2D
    bias tensors of shape ``(length, width)``.

    Input/output: ``(B, T, *spatial, width)``.
    """

    def __init__(self, length: int, width: int) -> None:
        super().__init__()
        self.length = length
        self.width = width
        # Per-lag dense channel mix (W: (T, in, out)).  Init: small uniform
        # scaled by 1/sqrt(width) like nn.Linear's default Kaiming-uniform.
        bound = 1.0 / (width ** 0.5)
        self.W_chan = nn.Parameter(
            torch.empty(length, width, width).uniform_(-bound, bound)
        )
        self.b_chan = nn.Parameter(torch.zeros(length, width))
        # Per-lag 1x1 spatial channel mix (independent across t).
        self.W_spat = nn.Parameter(
            torch.empty(length, width, width).uniform_(-bound, bound)
        )
        self.b_spat = nn.Parameter(torch.zeros(length, width))
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(B, T, *spatial, width)``."""
        # Branch 1: per-lag channel mix.
        # einsum over channel axis with per-lag weights.
        # x: (B, T, *spatial, in_ch); W_chan: (T, in_ch, out_ch).
        # Build the einsum dynamically to handle any spatial rank.
        n_spatial = x.dim() - 3
        sp_letters = "pqrstuv"[:n_spatial]
        eq = f"bt{sp_letters}i,tio->bt{sp_letters}o"
        h_chan = torch.einsum(eq, x, self.W_chan)
        # Add per-lag bias (broadcast over batch + spatial).
        view_b = [1, self.length] + [1] * n_spatial + [self.width]
        h_chan = h_chan + self.b_chan.view(*view_b)

        # Branch 2: per-lag 1x1 spatial conv.  In effect another channel mix
        # with independent weights — kept as a separate term to mirror the
        # lag + spatial split in LEMO-PC and double the expressive power.
        h_spat = torch.einsum(eq, x, self.W_spat)
        h_spat = h_spat + self.b_spat.view(*view_b)

        # Combine: residual + GELU + LayerNorm.
        y = x + self.act(h_chan + h_spat)
        return self.norm(y)


class PerLagMLPND(nn.Module):
    """Per-LAG-position MLP, ND.  Non-equivariant baseline for the orbit-OOD
    experiment.

    Same interface as ``LEMOPCND``::

        in:  (B, T, *spatial, in_channels)
        out: (B, T, *spatial, out_channels)

    Differs from LEMO-PC-ND in that there is NO weight sharing across the
    lag axis: every absolute lag position t has its own per-block weights.
    By construction this model is NOT cyclic-shift-equivariant on the lag
    axis.

    Notes
    -----
    * ``params_dim`` and ``sigma`` are accepted in the factory for config
      uniformity but ignored — the per-lag MLP has no FiLM modulation and
      no operator-norm constraint.
    * ``spatial_modes`` and ``lag_modes`` are likewise accepted and ignored;
      the per-lag MLP has no spectral component.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        width: int = 64,
        n_layers: int = 3,
        spatial_shape: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.width = width
        self.n_layers = n_layers
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)

        # Lift / project: shared across lag (channel-axis ops only).
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            PerLagBlock(length=length, width=width) for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(B, T, *spatial, in_channels)``.

        Returns ``(B, T, *spatial, out_channels)``.
        """
        if x.shape[1] != self.length:
            # Defensive — the per-lag weight tensor is sized for the
            # build-time ``length``; truncate or pad if needed (we just
            # error since the trainer always passes a fixed length).
            raise ValueError(
                f"PerLagMLPND: expected lag length {self.length}, got "
                f"{x.shape[1]} (per-lag weights are not shared across t).")
        h = self.lift(x)                     # (B, T, *spatial, width)
        for blk in self.blocks:
            h = blk(h)
        h = self.head_act(self.head1(h))
        return self.head2(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_per_lag_mlp_nd(in_channels: int, out_channels: int, config: dict,
                          length: Optional[int] = None) -> nn.Module:
    """Factory for ``PerLagMLPND``; mirrors ``create_lemo_pc_nd`` signature.

    Reads from ``config['model']``::

        length:        passed positionally (or read from cfg if None)
        width:         hidden width per lag (default 64)
        n_layers:      stack depth (default 3)
        spatial_shape: pass-through; not used in computation but kept for
                       interface compatibility

    ``params_dim``, ``sigma``, ``lag_modes``, ``spatial_modes`` are accepted
    but ignored (this model has no FiLM, no operator-norm projection, and
    no spectral component).
    """
    model_cfg = config.get("model", config)
    if length is None:
        length = model_cfg.get("length", config.get("length", 64))
    spatial_shape = tuple(model_cfg.get("spatial_shape", ()))
    width = model_cfg.get("width", 64)
    n_layers = model_cfg.get("n_layers", 3)
    return PerLagMLPND(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        width=width,
        n_layers=n_layers,
        spatial_shape=spatial_shape,
    )
