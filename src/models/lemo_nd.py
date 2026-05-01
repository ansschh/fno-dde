"""
LEMO-ND — N-dimensional Lag-Equivariant Memory Operator.

Round 2.23 (2D/3D upgrade).  Generic version of `lemo_v2.py` that
admits arbitrary spatial state-space dimensions while keeping the lag
axis 1D.

Input/output layout:  `(batch, L_lag, *spatial_shape, channels)` where
`spatial_shape` is `()` for the 1D scalar case (recovering the
existing `lemo_v2.py`), `(H, W)` for 2D fields, or `(H, W, D)` for 3D.

Architecture per block:
    (B U)(j, x)  +  (A_lag U)(j, x)  +  (A_spat U)(j, x)  +  bias
where:
    B          : pointwise channel mix (Linear on the channel axis)
    A_lag      : cyclic conv along lag axis with kernel shared
                  across spatial positions (the "lag-equivariant" term)
    A_spat     : ND spectral (FNO) conv along the spatial axes per lag

The lag-equivariance theorem (Abstract.affineLayer_shift in
LEMO/AbstractCore.lean) holds when the coefficients depend only on
relative lag offset, never on absolute lag index; the spatial operator
is applied identically at every lag, so the composition is
lag-equivariant for any choice of A_spat.

For LEMO_sigma: the lag kernel and the lift/B/proj linears are
operator-norm-bounded by sigma; ReLU is the only activation allowed.
The bound proof is in `LEMO/NDDFTConvolution.lean:lagConvCG_l2_bound`.

Setting `spatial_shape=()` recovers the 1D LEMOv2 numerically (regression
gate G2).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Helpers: dim handling for arbitrary spatial shapes
# ---------------------------------------------------------------------

def _spatial_axes_count(spatial_shape: Sequence[int]) -> int:
    return len(spatial_shape)


def _move_lag_to_last(x: torch.Tensor, n_spatial: int) -> torch.Tensor:
    """(batch, channels, lag, *spatial) -> (batch, channels, *spatial, lag).

    Used to apply rfft-along-lag with spatial axes batched.
    """
    perm = [0, 1] + list(range(3, 3 + n_spatial)) + [2]
    return x.permute(*perm)


def _move_lag_back(x: torch.Tensor, n_spatial: int) -> torch.Tensor:
    """Inverse of `_move_lag_to_last`."""
    perm = [0, 1, 2 + n_spatial] + list(range(2, 2 + n_spatial))
    return x.permute(*perm)


# ---------------------------------------------------------------------
# Lag convolution: 1D cyclic, shared across spatial positions
# ---------------------------------------------------------------------

class LagConvND(nn.Module):
    """Cyclic lag-conv along axis 1 with kernel shared across spatial axes.

    Kernel is parameterized as an MLP of the (signed, normalized) lag
    offset (continuous-tau parameterization, matching `lemo_v2.py`).

    Input:  (batch, in_channels, lag, *spatial)
    Output: (batch, out_channels, lag, *spatial)
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
        target_std = 1.0 / math.sqrt(max(length * in_channels, 1))
        with torch.no_grad():
            final = self.kernel_net[-1]
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
        # Frobenius upper bound on operator norm — conservative but valid (Cor 16).
        frob = K_hat.abs().pow(2).sum(dim=(0, 1)).sqrt()
        max_frob = frob.max().clamp_min(1e-10)
        return K / max_frob * self.sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, lag, *spatial). lag = axis 2."""
        n_spatial = x.dim() - 3
        length = x.shape[2]
        K = self.compute_kernel(length)                 # (out, in, lag)
        K_hat = torch.fft.rfft(K, dim=-1)
        # Move lag to last axis; spatial axes batched.
        x_moved = _move_lag_to_last(x, n_spatial)        # (B, in, *spatial, lag)
        x_hat = torch.fft.rfft(x_moved, dim=-1)
        # Einsum: contract over `in` channel dim. Use a generic ellipsis form.
        #   K_hat: (out, in, lag);  x_hat: (B, in, *spatial, lag)
        #   y_hat: (B, out, *spatial, lag)
        y_hat = torch.einsum("oil,b...il->b...ol",
                              K_hat,
                              x_hat.movedim(1, -2))      # bring `in` next to lag
        # Above einsum signature: x rearranged to (B, *spatial, in, lag).
        # Output shape: (B, *spatial, out, lag). Move out back to axis 1.
        y = torch.fft.irfft(y_hat, n=length, dim=-1)     # (B, *spatial, out, lag)
        y = y.movedim(-2, 1)                              # (B, out, *spatial, lag)
        return _move_lag_back(y, n_spatial)


# ---------------------------------------------------------------------
# Spatial spectral conv (FNO-style, ND, applied per lag)
# ---------------------------------------------------------------------

class SpatialSpectralConvND(nn.Module):
    """ND spectral conv along the spatial axes only.  Per-lag-position, the
    same kernel is applied independently.  Internally uses rfftn over the
    spatial axes.

    Input:  (batch, in_channels, lag, *spatial)
    Output: (batch, out_channels, lag, *spatial)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: Sequence[int],
        modes: Sequence[int],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(spatial_shape)
        # Truncate `modes` per axis to <= floor(n_a / 2) + 1 for rfft.
        # Non-rfft axes can use up to n_a/2 modes (by symmetry).
        max_modes = []
        for a, n in enumerate(spatial_shape):
            if a == self.n_spatial - 1:
                max_modes.append(n // 2 + 1)  # rfft axis
            else:
                max_modes.append(n)
        self.modes = tuple(min(m, mm) for m, mm in zip(modes, max_modes))
        # Spectral weights: complex tensor of shape (out, in, *modes).
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, *self.modes,
                                dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, lag, *spatial). FFT over spatial axes."""
        n_spatial = self.n_spatial
        if n_spatial == 0:
            # Degenerate case: no spatial axes, just return identity over a
            # 1x1 spatial. Used in 1D regression so the API is uniform.
            return x
        # Spatial axes are 3..3+n_spatial-1.  rfftn over them.
        spatial_dims = tuple(range(3, 3 + n_spatial))
        x_hat = torch.fft.rfftn(x, dim=spatial_dims)
        # Build output spectrum, fill in low-mode block, apply weights.
        out_shape = list(x_hat.shape)
        out_shape[1] = self.out_channels
        out_ft = torch.zeros(*out_shape, dtype=torch.cfloat, device=x.device)
        # Slice both x_hat and out_ft to the low-mode block.
        slc = [slice(None), slice(None), slice(None)]            # B, C, lag
        for i in range(n_spatial):
            slc.append(slice(0, self.modes[i]))
        x_block = x_hat[tuple(slc)]   # (B, in, lag, *modes)
        # Einsum: contract over `in`.
        # K (in, out, *modes) × x (B, in, lag, *modes) → y (B, out, lag, *modes)
        # Build dynamic einsum signature using letters that don't collide
        # with `b` (batch), `i`/`o` (channels), or `l` (lag).
        ax_letters = "pqrstuv"[:n_spatial]
        eq = f"io{ax_letters},bil{ax_letters}->bol{ax_letters}"
        y_block = torch.einsum(eq, self.weights, x_block)
        out_ft[tuple(slc[:2] + [slc[2]] + [slice(0, m) for m in self.modes])] = y_block
        # Inverse rfft over spatial axes.
        y = torch.fft.irfftn(out_ft, s=tuple(self.spatial_shape), dim=spatial_dims)
        return y


# ---------------------------------------------------------------------
# Pointwise channel-mix with optional spectral-norm constraint
# ---------------------------------------------------------------------

class PointwiseChannelMix(nn.Module):
    """1×1 channel-mixing convolution applied identically at every position.

    Spectral-norm constraint via Frobenius upper bound (matches `lemo_v2.py`).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 sigma: Optional[float] = None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_channels). Returns (..., out_channels)."""
        if self.sigma is None:
            return self.linear(x)
        w = self.linear.weight
        frob = w.norm().clamp_min(1e-10)
        w_scaled = w / frob * self.sigma
        return F.linear(x, w_scaled, self.linear.bias)


# ---------------------------------------------------------------------
# LEMO-ND block and full model
# ---------------------------------------------------------------------

class LEMONDBlock(nn.Module):
    """One LEMO-ND block: B (channel-mix) + A_lag + (optional) A_spat + ReLU.

    Both B and A_lag preserve lag-equivariance; A_spat acts only on spatial
    axes and is applied identically at every lag, so it is also
    lag-equivariant.
    """

    def __init__(
        self,
        width: int,
        length: int,
        spatial_shape: Sequence[int],
        spatial_modes: Sequence[int],
        kernel_hidden: int = 64,
        sigma: Optional[float] = None,
        activation: str = "relu",
        use_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.use_spatial = use_spatial and len(spatial_shape) > 0

        self.B = PointwiseChannelMix(width, width, sigma=sigma)
        self.A_lag = LagConvND(width, width, length,
                                kernel_hidden=kernel_hidden, sigma=sigma)
        if self.use_spatial:
            self.A_spat = SpatialSpectralConvND(width, width,
                                                  spatial_shape=spatial_shape,
                                                  modes=spatial_modes)
        else:
            self.A_spat = None

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            if sigma is not None:
                raise ValueError("LEMO_sigma requires activation='relu'.")
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, width, lag, *spatial). Returns same shape."""
        # B is per-position channel-mix. Apply via channel-last.
        x_chan_last = x.movedim(1, -1)                  # (B, lag, *spatial, width)
        Bx = self.B(x_chan_last).movedim(-1, 1)          # (B, width, lag, *spatial)
        Ax_lag = self.A_lag(x)                            # (B, width, lag, *spatial)
        if self.use_spatial:
            Ax_spat = self.A_spat(x)                      # (B, width, lag, *spatial)
            y = self.act(Bx + Ax_lag + Ax_spat)
        else:
            y = self.act(Bx + Ax_lag)
        return y


class LEMOND(nn.Module):
    """Lag-Equivariant Memory Operator over (G=C_n, X=R^{C×*spatial}).

    For spatial_shape=() this reduces to LEMOv2 with the same numerical
    behavior (G2 regression gate).

    Input:  (batch, length, *spatial_shape, in_channels)
    Output: (batch, length, *spatial_shape, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        spatial_shape: Sequence[int] = (),
        spatial_modes: Optional[Sequence[int]] = None,
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
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.width = width
        self.n_layers = n_layers
        self.sigma = sigma
        # Default spatial modes: half the size on each axis, capped at 16.
        if spatial_modes is None:
            spatial_modes = tuple(min(s // 2, 16) for s in self.spatial_shape)
        self.spatial_modes = tuple(spatial_modes)

        self.lift = PointwiseChannelMix(in_channels, width, sigma=sigma)

        self.blocks = nn.ModuleList([
            LEMONDBlock(
                width=width,
                length=length,
                spatial_shape=self.spatial_shape,
                spatial_modes=self.spatial_modes,
                kernel_hidden=kernel_hidden,
                sigma=sigma,
                activation=activation,
                use_spatial=(self.n_spatial > 0),
            )
            for _ in range(n_layers)
        ])

        self.proj1 = PointwiseChannelMix(width, width, sigma=sigma)
        self.proj2 = PointwiseChannelMix(width, out_channels, sigma=sigma)
        if activation == "relu":
            self.proj_act = nn.ReLU()
        else:
            if sigma is not None:
                raise ValueError("LEMO_sigma requires activation='relu'.")
            self.proj_act = nn.GELU()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, *spatial, in_channels).
        Returns (batch, length, *spatial, out_channels).
        """
        # Lift on channel axis (last).
        x = self.lift(x)                                  # (B, lag, *spatial, width)
        # Permute to (B, width, lag, *spatial).
        n_spatial = self.n_spatial
        if n_spatial == 0:
            x = x.permute(0, 2, 1)                         # (B, width, lag)
        else:
            perm = [0, x.dim() - 1, 1] + list(range(2, x.dim() - 1))
            x = x.permute(*perm)
        # Body.
        for blk in self.blocks:
            x = blk(x)
            x = self.dropout(x)
        # Permute back to channel-last for projection.
        if n_spatial == 0:
            x = x.permute(0, 2, 1)                         # (B, lag, width)
        else:
            inv = [0, 2] + list(range(3, 3 + n_spatial)) + [1]
            x = x.permute(*inv)                            # (B, lag, *spatial, width)
        # Two-layer projection head.
        x = self.proj1(x)
        x = self.proj_act(x)
        x = self.proj2(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo_nd(
    in_channels: int,
    out_channels: int,
    config: dict,
) -> nn.Module:
    """Factory; mirrors create_lemo_v2.

    Reads `model.spatial_shape` and `model.spatial_modes` for ND.
    Defaults: empty tuple recovers 1D LEMOv2 behavior.
    """
    model_cfg = config.get("model", config)
    length = model_cfg.get("length", config.get("length", 64))
    sigma = model_cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    spatial_shape = model_cfg.get("spatial_shape", ())
    spatial_modes = model_cfg.get("spatial_modes", None)
    return LEMOND(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        spatial_shape=tuple(spatial_shape),
        spatial_modes=tuple(spatial_modes) if spatial_modes is not None else None,
        width=model_cfg.get("width", 48),
        n_layers=model_cfg.get("n_layers", 3),
        kernel_hidden=model_cfg.get("kernel_hidden", 64),
        sigma=sigma,
        activation=model_cfg.get("activation", "relu"),
        dropout=model_cfg.get("dropout", 0.0),
    )
