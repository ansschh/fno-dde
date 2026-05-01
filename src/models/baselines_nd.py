"""
ND-adapted baselines for Phase 4 sweep.

Wrappers around standard architectures so they accept the same
`(B, T, *spatial, C+aux)` input layout as LEMO_ND / FNO_ND.

Models exposed:
  - markov_fno_nd      : 1 frame in -> 1 frame out, autoregressive rollout
  - windowed_fno_nd    : K frames in -> 1 frame out, autoregressive rollout
  - unet_nd            : 2D/3D U-Net per frame, stacked channels for time
  - memno_nd           : MemNO over time per spatial position
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno_nd import SpectralConvND


# ---------------------------------------------------------------------
# Markov / Windowed FNO ND
# ---------------------------------------------------------------------

class MarkovFNOND(nn.Module):
    """Per-frame FNO predictor: predict u(t+1) from u(t-K+1..t).

    K=1 => Markov (current frame only). K>1 => Windowed-K.

    Spatial: ND spectral conv block stack.
    Time: implicit, via autoregressive rollout at eval and teacher-forced
    training.

    Input layout:  (B, T, *spatial, C+aux)
    Output:        (B, T, *spatial, C)

    During training we teacher-force: the model sees the true previous K
    frames (from the input history if t<n_hist, or from y_target if
    t>=n_hist).  This is the standard one-step-prediction protocol used
    by PDEBench / APEBench.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_total: int,
        n_hist: int,
        spatial_shape: Sequence[int],
        modes: Optional[Sequence[int]] = None,
        width: int = 32,
        n_layers: int = 4,
        window: int = 1,
        teacher_force: bool = True,
        params_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels       # C + aux (mask, t, params)
        self.out_channels = out_channels     # C (just data channels)
        self.n_total = n_total
        self.n_hist = n_hist
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(spatial_shape)
        self.window = window
        self.teacher_force = teacher_force
        self.width = width

        if modes is None:
            modes = tuple(min(s // 2, 16) for s in spatial_shape)
        self.modes = tuple(modes)

        # Per-frame net: input is (window) frames stacked as channels:
        #   in_dim = window * (C + aux)
        # Output is one frame's data channels:
        #   out_dim = C
        in_dim = window * in_channels
        self.lift = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([
            _FNOBlock(width, self.spatial_shape, self.modes)
            for _ in range(n_layers)
        ])
        self.proj1 = nn.Linear(width, width * 2)
        self.proj2 = nn.Linear(width * 2, out_channels)

    def _per_frame(self, x_window: torch.Tensor) -> torch.Tensor:
        """x_window: (B, window, *spatial, in_channels). Returns (B, *spatial, out_channels)."""
        B = x_window.shape[0]
        # Stack time into channels: (B, *spatial, window * in_channels)
        # Move time axis to be adjacent to channel.
        x = x_window.movedim(1, -2).reshape(B, *self.spatial_shape, -1)
        x = self.lift(x)                                  # (B, *spatial, width)
        # Channels-second.
        if self.n_spatial == 0:
            x = x.unsqueeze(1)
        else:
            perm = [0, x.dim() - 1] + list(range(1, x.dim() - 1))
            x = x.permute(*perm)                           # (B, width, *spatial)
        for blk in self.blocks:
            x = blk(x)
        if self.n_spatial == 0:
            x = x.squeeze(1)
        else:
            inv = [0] + list(range(2, 2 + self.n_spatial)) + [1]
            x = x.permute(*inv)                            # (B, *spatial, width)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)                                # (B, *spatial, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, *spatial, C+aux). Returns (B, T, *spatial, out_channels).

        During training: teacher-forced (uses ground-truth previous frames).
        During eval: autoregressive rollout (uses predicted previous frames).
        """
        B, T = x.shape[0], x.shape[1]
        out_channels = self.out_channels
        in_channels = self.in_channels
        spatial = self.spatial_shape
        device = x.device
        # Output buffer.
        y = torch.zeros((B, T) + spatial + (out_channels,),
                        device=device, dtype=x.dtype)
        # First n_hist frames: just copy the data channels from input.
        y[:, :self.n_hist] = x[:, :self.n_hist, ..., :out_channels]
        # Predict t = n_hist..T-1 autoregressively (or teacher-forced).
        # Build the running history: same shape as x but possibly with
        # predicted frames substituted.
        running = x.clone()
        for t in range(self.n_hist, T):
            # Build window: frames [t-window, t).
            t0 = max(0, t - self.window)
            window_data = running[:, t0:t]                   # (B, w, *spatial, in_channels)
            if window_data.shape[1] < self.window:
                # Pad with zeros at the front if window extends before t=0.
                pad_n = self.window - window_data.shape[1]
                pad = torch.zeros((B, pad_n) + spatial + (in_channels,),
                                  device=device, dtype=x.dtype)
                window_data = torch.cat([pad, window_data], dim=1)
            pred = self._per_frame(window_data)              # (B, *spatial, out_channels)
            y[:, t] = pred
            if not self.teacher_force or not self.training:
                # Update running history with the prediction's data channels.
                # Aux channels (mask, t, params) stay as-is.
                running[:, t, ..., :out_channels] = pred
        return y


class _FNOBlock(nn.Module):
    def __init__(self, width, spatial_shape, modes):
        super().__init__()
        self.A = SpectralConvND(width, width, spatial_shape, modes)
        self.W = nn.Linear(width, width)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, width, *spatial)
        Ax = self.A(x)
        x_chan_last = x.movedim(1, -1)
        Wx = self.W(x_chan_last).movedim(-1, 1)
        return self.act(Ax + Wx)


# ---------------------------------------------------------------------
# U-Net ND  (2D/3D, time stacked as channels for joint prediction)
# ---------------------------------------------------------------------

class UNetND(nn.Module):
    """N-D U-Net (2D or 3D) with time stacked into channels for full-window
    prediction.  This matches FNO_ND's input/output style.

    Architecture: a small standard U-Net (encoder + decoder + skip
    connections) built from ND convolutions.  Designed to be parameter-
    comparable to FNO_ND.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_total: int,
        n_hist: int,
        spatial_shape: Sequence[int],
        width: int = 32,
        depth: int = 3,
        params_dim: int = 0,
        **_,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_total = n_total
        self.n_hist = n_hist
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(spatial_shape)

        in_total = n_total * in_channels
        out_total = n_total * out_channels

        ConvNd = nn.Conv2d if self.n_spatial == 2 else (
                 nn.Conv3d if self.n_spatial == 3 else nn.Conv1d)
        TConvNd = nn.ConvTranspose2d if self.n_spatial == 2 else (
                 nn.ConvTranspose3d if self.n_spatial == 3 else nn.ConvTranspose1d)
        BNNd = nn.BatchNorm2d if self.n_spatial == 2 else (
                 nn.BatchNorm3d if self.n_spatial == 3 else nn.BatchNorm1d)
        Pool = nn.AvgPool2d if self.n_spatial == 2 else (
                 nn.AvgPool3d if self.n_spatial == 3 else nn.AvgPool1d)

        # Encoder.
        self.enc = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_total
        cur = width
        for d in range(depth):
            self.enc.append(nn.Sequential(
                ConvNd(c, cur, 3, padding=1),
                nn.GELU(),
                ConvNd(cur, cur, 3, padding=1),
                nn.GELU(),
            ))
            self.pools.append(Pool(2))
            c = cur
            cur = cur * 2
        self.bottleneck = nn.Sequential(
            ConvNd(c, cur, 3, padding=1), nn.GELU(),
            ConvNd(cur, cur, 3, padding=1), nn.GELU(),
        )
        # Decoder.
        self.upconv = nn.ModuleList()
        self.dec = nn.ModuleList()
        for d in range(depth):
            self.upconv.append(TConvNd(cur, cur // 2, 2, stride=2))
            cur = cur // 2
            self.dec.append(nn.Sequential(
                ConvNd(cur * 2, cur, 3, padding=1),
                nn.GELU(),
                ConvNd(cur, cur, 3, padding=1),
                nn.GELU(),
            ))
        self.out_conv = ConvNd(cur, out_total, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, *spatial, in_channels). Returns (B, T, *spatial, out_channels)."""
        B, T = x.shape[0], x.shape[1]
        spatial = self.spatial_shape
        # Stack time × channels along channel axis.
        x_stacked = x.movedim(1, -2).reshape(B, *spatial, -1)  # (B, *spatial, T*in)
        # Channels first for ConvNd.
        perm = [0, x_stacked.dim() - 1] + list(range(1, x_stacked.dim() - 1))
        h = x_stacked.permute(*perm)                       # (B, T*in, *spatial)
        # Pad spatial to multiple of 2^depth so pooling works.
        pad_amounts = []
        depth = len(self.enc)
        for s in spatial:
            target = ((s + 2**depth - 1) // 2**depth) * 2**depth
            pad_amounts.append(target - s)
        # Apply padding (symmetric, last axis first per F.pad convention).
        if any(p > 0 for p in pad_amounts):
            pad_arg = []
            for p in reversed(pad_amounts):
                pad_arg.extend([0, p])
            h = F.pad(h, pad_arg, mode="constant", value=0.0)

        # Encoder.
        skips = []
        for enc_blk, pool in zip(self.enc, self.pools):
            h = enc_blk(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        # Decoder.
        for upc, dec_blk, skip in zip(self.upconv, self.dec, reversed(skips)):
            h = upc(h)
            # Match shape if needed.
            h = torch.cat([h, skip], dim=1)
            h = dec_blk(h)
        h = self.out_conv(h)                                # (B, T*out, *padded_spatial)
        # Unpad.
        if any(p > 0 for p in pad_amounts):
            slc = [slice(None), slice(None)]
            for i, s in enumerate(spatial):
                slc.append(slice(0, s))
            h = h[tuple(slc)]
        # Channels-last: (B, *spatial, T*out)
        inv = [0] + list(range(2, 2 + self.n_spatial)) + [1]
        h = h.permute(*inv)
        # Reshape (T*out) -> (T, out).
        h = h.reshape(B, *spatial, T, self.out_channels)
        # Permute T to position 1.
        n_sp = self.n_spatial
        perm2 = [0, 1 + n_sp] + list(range(1, 1 + n_sp)) + [2 + n_sp]
        return h.permute(*perm2)                            # (B, T, *spatial, out)


# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------

def create_markov_fno_nd(in_channels: int, out_channels: int, config: dict,
                         length: Optional[int] = None) -> nn.Module:
    cfg = config.get("model", config)
    n_total = cfg.get("length", length or 64)
    n_hist = cfg.get("n_hist", n_total // 2)
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    modes = cfg.get("spatial_modes", None)
    if modes is not None and not isinstance(modes, (list, tuple)):
        modes = (modes,)
    return MarkovFNOND(
        in_channels=in_channels, out_channels=out_channels,
        n_total=n_total, n_hist=n_hist,
        spatial_shape=spatial_shape,
        modes=tuple(modes) if modes else None,
        width=cfg.get("width", 32),
        n_layers=cfg.get("n_layers", 4),
        window=cfg.get("window", 1),
        teacher_force=cfg.get("teacher_force", True),
    )


def create_windowed_fno_nd(in_channels: int, out_channels: int, config: dict,
                           length: Optional[int] = None) -> nn.Module:
    cfg = dict(config.get("model", config))
    cfg["window"] = cfg.get("window", 10)
    return create_markov_fno_nd(in_channels, out_channels, {"model": cfg}, length)


def create_unet_nd(in_channels: int, out_channels: int, config: dict,
                   length: Optional[int] = None) -> nn.Module:
    cfg = config.get("model", config)
    n_total = cfg.get("length", length or 64)
    n_hist = cfg.get("n_hist", n_total // 2)
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    return UNetND(
        in_channels=in_channels, out_channels=out_channels,
        n_total=n_total, n_hist=n_hist,
        spatial_shape=spatial_shape,
        width=cfg.get("width", 32),
        depth=cfg.get("depth", 3),
    )


# ---------------------------------------------------------------------
# MemNO ND adapter (Buitrago et al. 2024) — published delay-aware operator
# ---------------------------------------------------------------------

class MemNOND(nn.Module):
    """ND adapter for the 1D MemNO from `memno_paper.MemNO`.

    Applies MemNO independently at each spatial position over the time axis.
    Layout:  input  (B, T, *spatial, in_channels) → output (B, T, *spatial, out_channels)

    MemNO's per-position model is unchanged — it sees a (T, in_channels) sequence
    and emits (T, out_channels).  Spatial positions are processed in a flat
    batch dim (no spatial mixing — that's MarkovFNO's job).  This matches the
    paper's intended use of the architecture (memory in time, locality in space).

    Channel/lag inductive bias: same as the 1D MemNO — pointwise channel
    mixers + a single causal S4D state-space layer over the time axis.

    Cost: O(B * prod(spatial) * T * width^2) per markovian layer + O(B * prod(spatial) * T * state) for S4D.
    """

    def __init__(self, in_channels: int, out_channels: int, length: int,
                 spatial_shape: Sequence[int],
                 hidden: int = 64, n_markovian: int = 4,
                 memory_position: int = 2, s4_state: int = 64) -> None:
        super().__init__()
        from .memno_paper import MemNO as _MemNO
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = _MemNO(
            in_channels=in_channels,
            out_channels=out_channels,
            length=length,
            hidden=hidden,
            n_markovian=n_markovian,
            memory_position=memory_position,
            s4_state=s4_state,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, *spatial, in_channels)."""
        B = x.shape[0]
        T = x.shape[1]
        # Flatten spatial into batch: (B, T, prod_S, C) -> (B*prod_S, T, C)
        spatial_size = 1
        for s in self.spatial_shape:
            spatial_size *= s
        x_flat = x.reshape(B, T, spatial_size, self.in_channels)
        # Move spatial to batch dim: (B*S, T, C)
        x_seq = x_flat.permute(0, 2, 1, 3).reshape(B * spatial_size, T, self.in_channels)
        y_seq = self.net(x_seq)  # (B*S, T, out_channels)
        # Reshape back to (B, T, *spatial, out_channels)
        y = y_seq.reshape(B, spatial_size, T, self.out_channels)
        y = y.permute(0, 2, 1, 3).reshape(B, T, *self.spatial_shape, self.out_channels)
        return y


def create_memno_nd(in_channels: int, out_channels: int, config: dict,
                    length: Optional[int] = None) -> nn.Module:
    cfg = config.get("model", config)
    n_total = cfg.get("length", length or 64)
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    return MemNOND(
        in_channels=in_channels, out_channels=out_channels,
        length=n_total,
        spatial_shape=spatial_shape,
        hidden=cfg.get("width", 64),  # alias width→hidden for sweep consistency
        n_markovian=cfg.get("n_markovian", 4),
        memory_position=cfg.get("memory_position", 2),
        s4_state=cfg.get("s4_state", 64),
    )


# ---------------------------------------------------------------------
# Factorized FNO ND (F-FNO, Tran et al. ICLR 2023)
# ---------------------------------------------------------------------

class FFactorizedSpectralConvND(nn.Module):
    """Per-axis 1D spectral conv applied AXIS-BY-AXIS (Tran et al. F-FNO).

    Replaces the dense ND spectral conv with a sum of 1D spectral convs,
    one per axis, sharing only width-many channels.  Asymptotically far
    fewer params than vanilla FNO (O(modes_per_axis × C^2) per axis vs
    O(prod(modes_per_axis) × C^2) for full ND).

    Empirically a strong FNO improvement on 2D Navier-Stokes; the standard
    "improved FNO" reviewers ask for.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_per_axis: Sequence[int], n_spatial_axes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_spatial_axes = n_spatial_axes
        self.modes_per_axis = tuple(modes_per_axis)
        scale = 1.0 / (in_channels * out_channels)
        # One complex weight tensor per axis: (in, out, modes_axis)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.rand(in_channels, out_channels,
                                              modes_per_axis[a],
                                              dtype=torch.cfloat))
            for a in range(n_spatial_axes)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, *spatial). Returns (B, out_channels, *spatial)."""
        # Sum of axis-wise 1D spectral convs.
        out = None
        for axis in range(self.n_spatial_axes):
            spatial_axis = axis + 2  # skip B, C
            x_hat = torch.fft.rfft(x, dim=spatial_axis)
            modes = self.modes_per_axis[axis]
            modes = min(modes, x_hat.shape[spatial_axis])
            # Slice along spatial_axis to keep first `modes` modes
            slc = [slice(None)] * x_hat.dim()
            slc[spatial_axis] = slice(0, modes)
            x_block = x_hat[tuple(slc)]
            # einsum: (in, out, modes) × (B, in, *other_spatial, modes) → (B, out, *other_spatial, modes)
            # Move axis to last position for einsum simplicity
            x_block_perm = x_block.movedim(spatial_axis, -1)
            # Now (B, in, *other, modes_a)
            y_block = torch.einsum("iom,bi...m->bo...m", self.weights[axis], x_block_perm)
            y_block = y_block.movedim(-1, spatial_axis)
            # Pad back to full freq length and irfft
            y_full = torch.zeros(x.shape[0], self.out_channels, *x_hat.shape[2:],
                                  dtype=torch.cfloat, device=x.device)
            slc[1] = slice(None)
            y_full[tuple(slc)] = y_block
            y_axis = torch.fft.irfft(y_full, n=x.shape[spatial_axis], dim=spatial_axis)
            out = y_axis if out is None else out + y_axis
        return out


class FactorizedFNOND(nn.Module):
    """F-FNO ND (Tran et al. ICLR 2023): per-axis factorized spectral convolution
    with depth ≥ FNO baseline. Time treated identically to MarkovFNO (per-frame
    + autoregressive teacher-forced rollout)."""

    def __init__(self, in_channels: int, out_channels: int,
                 n_total: int, n_hist: int,
                 spatial_shape: Sequence[int],
                 modes: Optional[Sequence[int]] = None,
                 width: int = 32, n_layers: int = 4,
                 window: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_total = n_total
        self.n_hist = n_hist
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(spatial_shape)
        self.window = window
        if modes is None:
            modes = tuple(min(s // 2, 16) for s in spatial_shape)
        self.modes = tuple(modes)

        in_dim = window * in_channels
        self.lift = nn.Linear(in_dim, width)
        self.spectral = nn.ModuleList([
            FFactorizedSpectralConvND(width, width, self.modes, self.n_spatial)
            for _ in range(n_layers)
        ])
        self.local = nn.ModuleList([
            nn.Conv2d(width, width, 1) if self.n_spatial == 2
            else nn.Conv3d(width, width, 1) if self.n_spatial == 3
            else nn.Conv1d(width, width, 1)
            for _ in range(n_layers)
        ])
        self.proj1 = nn.Linear(width, width * 2)
        self.proj2 = nn.Linear(width * 2, out_channels)

    def _per_frame(self, x_window):
        B = x_window.shape[0]
        x = x_window.movedim(1, -2).reshape(B, *self.spatial_shape, -1)
        x = self.lift(x)  # (B, *spatial, width)
        perm = [0, x.dim() - 1] + list(range(1, x.dim() - 1))
        x = x.permute(*perm)  # (B, width, *spatial)
        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))
        inv = [0] + list(range(2, 2 + self.n_spatial)) + [1]
        x = x.permute(*inv)  # (B, *spatial, width)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        spatial = self.spatial_shape
        device = x.device
        y = torch.zeros((B, T) + spatial + (self.out_channels,),
                          device=device, dtype=x.dtype)
        y[:, :self.n_hist] = x[:, :self.n_hist, ..., :self.out_channels]
        running = x.clone()
        for t in range(self.n_hist, T):
            t0 = max(0, t - self.window)
            window_data = running[:, t0:t]
            if window_data.shape[1] < self.window:
                pad_n = self.window - window_data.shape[1]
                pad = torch.zeros((B, pad_n) + spatial + (self.in_channels,),
                                    device=device, dtype=x.dtype)
                window_data = torch.cat([pad, window_data], dim=1)
            pred = self._per_frame(window_data)
            y[:, t] = pred
            if not self.training:
                running[:, t, ..., :self.out_channels] = pred
        return y


def create_ffno_nd(in_channels: int, out_channels: int, config: dict,
                   length: Optional[int] = None) -> nn.Module:
    cfg = config.get("model", config)
    n_total = cfg.get("length", length or 64)
    n_hist = cfg.get("n_hist", n_total // 2)
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    modes = cfg.get("spatial_modes", None)
    if modes is not None and not isinstance(modes, (list, tuple)):
        modes = (modes,)
    return FactorizedFNOND(
        in_channels=in_channels, out_channels=out_channels,
        n_total=n_total, n_hist=n_hist,
        spatial_shape=spatial_shape,
        modes=tuple(modes) if modes else None,
        width=cfg.get("width", 32),
        n_layers=cfg.get("n_layers", 4),
        window=cfg.get("window", 1),
    )
