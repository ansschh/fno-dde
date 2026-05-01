"""
FiLM-vs-equivariance ablation suite — model factories.

Round 3 fix for REVIEW_PUNCH_LIST.md top risk #2:
"No ablation that cleanly isolates lag-equivariance from FiLM and the
pointwise channel-mixing path; need FNO+FiLM and a non-equivariant FiLM-
conditioned lag mixer, and a corrected FiLM-free variant that includes
the B_ell term; ensure all models see the same history window."

Three NEW variants (A4 = full LEMO-PC ND lives in lemo_pc_nd.py):
    A1 — FNO + FiLM                     (create_fno_film_nd)
    A2 — non-equivariant FiLM mixer    (create_noneq_film_nd)
    A3 — LEMO + B_ell corrected, FiLM-free (create_lemo_bcorrect_nd)

All variants share lift / projection heads and consume the same
(lag, *spatial, in_channels) input layout as LEMOPCND so that param
counts sit in a narrow band; the only architectural difference is
in the lag path.

Build interface (matches build_model contract in src/train/build_model.py):
    create_<variant>(in_channels: int,
                     out_channels: int,
                     config: dict,
                     length: int | None = None) -> nn.Module
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the well-tested building blocks from the existing codebase.
# All variants share lift/proj/spatial path so the only difference is the
# *lag* path, exactly the variable being isolated.
from models.lemo_pc_nd import FiLMLagSpectralND, SpatialFNOND, LEMOPCND  # noqa: F401
from models.lemo_nd import LagConvND, SpatialSpectralConvND, PointwiseChannelMix  # noqa: F401
from models.fno_nd import FNOND, SpectralConvND  # noqa: F401


# =====================================================================
# Shared FiLM head used by A1 (FNO+FiLM) and A2 (non-eq lag mixer + FiLM).
# Mirrors the (gamma, beta) head in FiLMLagSpectralND so that the
# conditioning channel has identical capacity / init across variants.
# =====================================================================

class FiLMHead(nn.Module):
    """Per-(out_channel, lag_mode) FiLM head from per-sample params.

    Same architecture as the conditioning network inside
    `FiLMLagSpectralND`: 2-layer MLP from params -> 2 * out_channels *
    lag_modes, last layer initialised to (gamma=1, beta=0).
    """

    def __init__(self, params_dim: int, out_channels: int, lag_modes: int,
                 hidden: int = 64) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.lag_modes = lag_modes
        self.net = nn.Sequential(
            nn.Linear(params_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * out_channels * lag_modes),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            b = torch.zeros(2 * out_channels * lag_modes)
            b[: out_channels * lag_modes] = 1.0
            self.net[-1].bias.copy_(b)

    def forward(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """params: (B, P).  Returns (gamma, beta) of shape (B, OC, M)."""
        B = params.shape[0]
        film = self.net(params)
        OC, M = self.out_channels, self.lag_modes
        gamma = film[:, : OC * M].view(B, OC, M)
        beta = film[:, OC * M:].view(B, OC, M)
        return gamma, beta


def _split_x_params(x: torch.Tensor, params_dim: int, n_spatial: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the broadcast-across-(time, space) parameter slot from x.

    Mirrors `LEMOPCND._split_x_params`.  Used by every variant so that all
    four see *exactly* the same input and the same params channel.
    """
    if params_dim == 0:
        B = x.shape[0]
        return x, torch.zeros(B, 0, device=x.device, dtype=x.dtype)
    idx = [slice(None), 0]
    for _ in range(n_spatial):
        idx.append(0)
    idx.append(slice(-params_dim, None))
    params = x[tuple(idx)]
    return x, params


# =====================================================================
# A1 — FNO + FiLM
# Vanilla FNO over (lag, *spatial) with the same per-(out_channel,
# lag_mode) FiLM head appended to each block's spectral output.  Tests
# whether FiLM conditioning ALONE (without lag-equivariance as a
# parameter-sharing inductive bias) explains the headline gap.
# =====================================================================

class FNOFiLMBlock(nn.Module):
    """FNO block + per-(out, lag_mode) FiLM modulation in the LAG-axis
    spectral domain (so the FiLM head is the same shape as in LEMO-PC).

    The block first runs the standard FNO spectral conv over the FULL
    physical shape (lag, *spatial), then takes a 1D rfft along the lag
    axis ONLY of the post-spectral activation, applies (gamma, beta) on
    the first `lag_modes` modes, and irffts back.  This is the mildest
    drop-in: it adds FiLM with the same parameter budget as LEMO-PC's
    FiLM, on top of an otherwise unchanged FNO.
    """

    def __init__(self, width: int, lag_len: int, lag_modes: int,
                 spatial_shape: Sequence[int], spatial_modes: Sequence[int],
                 params_dim: int, film_hidden: int = 64) -> None:
        super().__init__()
        self.width = width
        self.lag_len = lag_len
        self.lag_modes = lag_modes
        self.physical_shape = (lag_len,) + tuple(spatial_shape)
        self.modes = (lag_modes,) + tuple(spatial_modes)
        self.A = SpectralConvND(width, width, self.physical_shape, self.modes)
        self.W = nn.Linear(width, width)
        self.film = FiLMHead(params_dim, width, lag_modes, hidden=film_hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, width, lag, *spatial). Returns same shape."""
        Ax = self.A(x)                                              # full FNO spectral conv
        x_chan_last = x.movedim(1, -1)
        Wx = self.W(x_chan_last).movedim(-1, 1)
        h = Ax + Wx                                                  # (B, w, lag, *spatial)
        # Lag-axis rfft -> per-mode FiLM -> irfft, matching LEMO-PC's
        # FiLM placement (so the conditioning capacity is identical).
        n_spatial = h.dim() - 3
        perm = [0, 1] + list(range(3, 3 + n_spatial)) + [2]
        h_moved = h.permute(*perm)                                   # (B, w, *sp, lag)
        h_hat = torch.fft.rfft(h_moved, dim=-1)                       # (B, w, *sp, lag//2+1)
        gamma, beta = self.film(params)                              # (B, w, M)
        M = min(self.lag_modes, h_hat.shape[-1])
        view_shape = [h.shape[0], self.width] + [1] * n_spatial + [M]
        gamma_b = gamma[:, :, :M].view(*view_shape).to(torch.cfloat)
        beta_b = beta[:, :, :M].view(*view_shape).to(torch.cfloat)
        # Out-of-place: build modulated low-modes block, then concatenate
        # with the unmodulated tail to avoid an in-place write into the
        # rfft output (which breaks autograd on the complex view).
        modulated = h_hat[..., :M] * gamma_b + beta_b
        if h_hat.shape[-1] > M:
            h_hat_new = torch.cat([modulated, h_hat[..., M:]], dim=-1)
        else:
            h_hat_new = modulated
        h2 = torch.fft.irfft(h_hat_new, n=self.lag_len, dim=-1)
        inv = [0, 1, h2.dim() - 1] + list(range(2, h2.dim() - 1))
        return self.act(h2.permute(*inv))


class FNOFiLMND(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 spatial_shape: Sequence[int], spatial_modes: Sequence[int],
                 lag_modes: int, params_dim: int, width: int = 64,
                 n_layers: int = 3, film_hidden: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.params_dim = params_dim
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            FNOFiLMBlock(width, length, lag_modes,
                          self.spatial_shape, spatial_modes,
                          params_dim, film_hidden=film_hidden)
            for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_full, params = _split_x_params(x, self.params_dim, self.n_spatial)
        x_lift = self.lift(x_full)                                   # (B, lag, *sp, w)
        if self.n_spatial == 0:
            x_chan = x_lift.permute(0, 2, 1)
        else:
            perm = [0, x_lift.dim() - 1, 1] + list(range(2, x_lift.dim() - 1))
            x_chan = x_lift.permute(*perm)
        for blk in self.blocks:
            x_chan = blk(x_chan, params)
        if self.n_spatial == 0:
            x_seq = x_chan.permute(0, 2, 1)
        else:
            inv = [0, 2] + list(range(3, 3 + self.n_spatial)) + [1]
            x_seq = x_chan.permute(*inv)
        h = self.head_act(self.head1(x_seq))
        return self.head2(h)


def create_fno_film_nd(in_channels: int, out_channels: int, config: dict,
                       length: Optional[int] = None) -> nn.Module:
    """A1 — FNO + FiLM."""
    cfg = config.get("model", config)
    if length is None:
        length = cfg.get("length", config.get("length", 64))
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    spatial_modes = cfg.get("spatial_modes", None)
    if spatial_modes is None:
        spatial_modes = tuple(min(s // 2, 12) for s in spatial_shape)
    spatial_modes = tuple(spatial_modes)
    default_params_dim = max(in_channels - 4, 1)
    return FNOFiLMND(
        in_channels=in_channels, out_channels=out_channels, length=length,
        spatial_shape=spatial_shape, spatial_modes=spatial_modes,
        lag_modes=cfg.get("lag_modes", 24),
        params_dim=cfg.get("params_dim", default_params_dim),
        width=cfg.get("width", 64),
        n_layers=cfg.get("n_layers", 3),
        film_hidden=cfg.get("film_hidden", 64),
    )


# =====================================================================
# A2 — Non-equivariant FiLM-conditioned lag mixer
# Replaces the cyclic-FFT lag conv with a per-lag-position MLP whose
# weights are NOT shared across lag positions (W_t for t = 0..L-1).
# This keeps lag mixing in the network and keeps FiLM, but breaks the
# parameter-sharing that gives lag-equivariance.  Tests whether ANY
# lag mixer + FiLM (regardless of equivariance) closes the gap.
# =====================================================================

class NonEqLagMixerFiLM(nn.Module):
    """Per-lag-position MLP: y[:, :, t] = sigma( W_t x[:, :, t] + b_t ),
    followed by per-(out, lag_mode) FiLM in the lag-axis spectral domain
    (matching LEMO-PC's FiLM placement and capacity).

    Implementation: `weights_per_lag` is shape (L, out, in) — one full
    in->out matrix per lag position.  Asymptotic capacity matches a dense
    learned absolute-position embedding; equivariance under cyclic shift
    is broken because W_t differs from W_{t+k}.
    """

    def __init__(self, in_channels: int, out_channels: int, length: int,
                 lag_modes: int, params_dim: int, film_hidden: int = 64
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.lag_modes = lag_modes
        scale = 1.0 / math.sqrt(in_channels)
        self.weights_per_lag = nn.Parameter(
            scale * torch.randn(length, out_channels, in_channels))
        self.bias_per_lag = nn.Parameter(torch.zeros(length, out_channels))
        self.film = FiLMHead(params_dim, out_channels, lag_modes, hidden=film_hidden)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, in, lag, *spatial).  Returns (B, out, lag, *spatial)."""
        n_spatial = x.dim() - 3
        sp_letters = "pqrstuv"[:n_spatial]
        eq = f"loi,bil{sp_letters}->bol{sp_letters}"
        y = torch.einsum(eq, self.weights_per_lag, x)
        # Add per-lag bias broadcast across spatial.
        view_shape = [1, self.out_channels, self.length] + [1] * n_spatial
        y = y + self.bias_per_lag.permute(1, 0).reshape(*view_shape)
        # Lag-axis rfft -> FiLM -> irfft (same placement as LEMO-PC).
        perm = [0, 1] + list(range(3, 3 + n_spatial)) + [2]
        y_moved = y.permute(*perm)
        y_hat = torch.fft.rfft(y_moved, dim=-1)
        gamma, beta = self.film(params)
        M = min(self.lag_modes, y_hat.shape[-1])
        OC = self.out_channels
        view_shape2 = [x.shape[0], OC] + [1] * n_spatial + [M]
        gamma_b = gamma[:, :, :M].view(*view_shape2).to(torch.cfloat)
        beta_b = beta[:, :, :M].view(*view_shape2).to(torch.cfloat)
        # Out-of-place: build modulated low-modes block, then concatenate
        # with the unmodulated tail to avoid an in-place write into the
        # rfft output (which breaks autograd on the complex view).
        modulated = y_hat[..., :M] * gamma_b + beta_b
        if y_hat.shape[-1] > M:
            y_hat_new = torch.cat([modulated, y_hat[..., M:]], dim=-1)
        else:
            y_hat_new = modulated
        y_back = torch.fft.irfft(y_hat_new, n=self.length, dim=-1)
        inv = [0, 1, y_back.dim() - 1] + list(range(2, y_back.dim() - 1))
        return y_back.permute(*inv)


class NonEqFiLMNDBlock(nn.Module):
    """Block: B (1x1 channel mix) + non-eq lag mixer + (optional) spatial
    FNO conv + GELU + LayerNorm.  Same skeleton as LEMOPCNDBlock, only
    A_lag is replaced.
    """

    def __init__(self, width: int, length: int, lag_modes: int,
                 spatial_shape: Sequence[int], spatial_modes: Sequence[int],
                 params_dim: int, film_hidden: int = 64) -> None:
        super().__init__()
        self.width = width
        self.use_spatial = len(spatial_shape) > 0
        self.B = nn.Conv1d(width, width, 1)
        self.A_lag = NonEqLagMixerFiLM(width, width, length, lag_modes,
                                          params_dim, film_hidden=film_hidden)
        if self.use_spatial:
            self.A_spat = SpatialFNOND(width, width,
                                          spatial_shape=spatial_shape,
                                          modes=spatial_modes)
        else:
            self.A_spat = None
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x_chan_last = x.movedim(1, -1)
        Bx_flat = F.linear(x_chan_last, self.B.weight.squeeze(-1), self.B.bias)
        Bx = Bx_flat.movedim(-1, 1)
        Ax_lag = self.A_lag(x, params)
        if self.use_spatial:
            Ax_spat = self.A_spat(x)
            y = self.act(Bx + Ax_lag + Ax_spat)
        else:
            y = self.act(Bx + Ax_lag)
        y_chan_last = y.movedim(1, -1)
        y_chan_last = self.norm(y_chan_last)
        return y_chan_last.movedim(-1, 1)


class NonEqFiLMND(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 spatial_shape: Sequence[int], spatial_modes: Sequence[int],
                 lag_modes: int, params_dim: int, width: int = 64,
                 n_layers: int = 3, film_hidden: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.params_dim = params_dim
        self.length = length
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            NonEqFiLMNDBlock(width, length, lag_modes,
                              self.spatial_shape, spatial_modes,
                              params_dim, film_hidden=film_hidden)
            for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_full, params = _split_x_params(x, self.params_dim, self.n_spatial)
        x_lift = self.lift(x_full)
        if self.n_spatial == 0:
            x_chan = x_lift.permute(0, 2, 1)
        else:
            perm = [0, x_lift.dim() - 1, 1] + list(range(2, x_lift.dim() - 1))
            x_chan = x_lift.permute(*perm)
        for blk in self.blocks:
            x_chan = blk(x_chan, params)
        if self.n_spatial == 0:
            x_seq = x_chan.permute(0, 2, 1)
        else:
            inv = [0, 2] + list(range(3, 3 + self.n_spatial)) + [1]
            x_seq = x_chan.permute(*inv)
        h = self.head_act(self.head1(x_seq))
        return self.head2(h)


def create_noneq_film_nd(in_channels: int, out_channels: int, config: dict,
                         length: Optional[int] = None) -> nn.Module:
    """A2 — non-equivariant FiLM-conditioned lag mixer."""
    cfg = config.get("model", config)
    if length is None:
        length = cfg.get("length", config.get("length", 64))
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    spatial_modes = cfg.get("spatial_modes", None)
    if spatial_modes is None:
        spatial_modes = tuple(min(s // 2, 12) for s in spatial_shape)
    spatial_modes = tuple(spatial_modes)
    default_params_dim = max(in_channels - 4, 1)
    return NonEqFiLMND(
        in_channels=in_channels, out_channels=out_channels, length=length,
        spatial_shape=spatial_shape, spatial_modes=spatial_modes,
        lag_modes=cfg.get("lag_modes", 24),
        params_dim=cfg.get("params_dim", default_params_dim),
        width=cfg.get("width", 64),
        n_layers=cfg.get("n_layers", 3),
        film_hidden=cfg.get("film_hidden", 64),
    )


# =====================================================================
# A3 — LEMO + B_ell channel-mix only (FiLM-free, CORRECTED)
#
# The original `lemo_nd.LEMOND` *does* include a `PointwiseChannelMix`
# `B` term inside `LEMONDBlock.forward` (line 298: `Bx = self.B(...)`),
# but in the production headline run the `B` path was unintentionally
# disabled because (a) `PointwiseChannelMix` defaults to `sigma=None`
# returning a plain Linear which initialised to (1/sqrt(C))-scale weights
# and (b) the post-block ReLU activation in `LEMONDBlock.act` (used when
# `sigma=None` is overridden) clipped the B contribution before it could
# matter — the upshot was a near-zero contribution from B in the no-FiLM
# variant in some configs.
#
# The corrected variant in this suite:
#   - uses GELU (matches LEMO-PC and FNO baselines, not ReLU)
#   - keeps B + A_lag + A_spat with all three contributing
#   - matches LEMOPCND's lift / head / norm / params_dim / lag_modes /
#     spatial_modes / width / n_layers exactly
#   - removes FiLM entirely (no per-sample conditioning)
# This is the FAIR FiLM-free variant the reviewers asked for.
# =====================================================================

class LEMOFreeFiLMBlock(nn.Module):
    """LEMO block (B + A_lag + A_spat + GELU + LayerNorm), FiLM-free.

    A_lag here is the cyclic-FFT lag conv from `lemo_nd.LagConvND` (NOT
    the FiLM-modulated `FiLMLagSpectralND` from lemo_pc_nd).  The
    parameterisation is identical to the headline LEMO-PC except that
    the per-sample FiLM head is removed and the lag kernel is shared
    across all samples (true T1-equivariant kernel).
    """

    def __init__(self, width: int, length: int,
                 spatial_shape: Sequence[int], spatial_modes: Sequence[int],
                 kernel_hidden: int = 64) -> None:
        super().__init__()
        self.width = width
        self.use_spatial = len(spatial_shape) > 0
        self.B = nn.Conv1d(width, width, 1)
        self.A_lag = LagConvND(width, width, length,
                                kernel_hidden=kernel_hidden, sigma=None)
        if self.use_spatial:
            self.A_spat = SpatialFNOND(width, width,
                                          spatial_shape=spatial_shape,
                                          modes=spatial_modes)
        else:
            self.A_spat = None
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_chan_last = x.movedim(1, -1)
        Bx_flat = F.linear(x_chan_last, self.B.weight.squeeze(-1), self.B.bias)
        Bx = Bx_flat.movedim(-1, 1)
        Ax_lag = self.A_lag(x)
        if self.use_spatial:
            Ax_spat = self.A_spat(x)
            y = self.act(Bx + Ax_lag + Ax_spat)
        else:
            y = self.act(Bx + Ax_lag)
        y_chan_last = y.movedim(1, -1)
        y_chan_last = self.norm(y_chan_last)
        return y_chan_last.movedim(-1, 1)


class LEMOFreeFiLMND(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 spatial_shape: Sequence[int], spatial_modes: Sequence[int],
                 width: int = 64, n_layers: int = 3,
                 kernel_hidden: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            LEMOFreeFiLMBlock(width, length,
                                self.spatial_shape, spatial_modes,
                                kernel_hidden=kernel_hidden)
            for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No params extraction — variant is FiLM-free — but we still
        # consume the params slot so the input layout matches all other
        # variants exactly (same history window, same channel layout).
        x_lift = self.lift(x)
        if self.n_spatial == 0:
            x_chan = x_lift.permute(0, 2, 1)
        else:
            perm = [0, x_lift.dim() - 1, 1] + list(range(2, x_lift.dim() - 1))
            x_chan = x_lift.permute(*perm)
        for blk in self.blocks:
            x_chan = blk(x_chan)
        if self.n_spatial == 0:
            x_seq = x_chan.permute(0, 2, 1)
        else:
            inv = [0, 2] + list(range(3, 3 + self.n_spatial)) + [1]
            x_seq = x_chan.permute(*inv)
        h = self.head_act(self.head1(x_seq))
        return self.head2(h)


def create_lemo_bcorrect_nd(in_channels: int, out_channels: int, config: dict,
                            length: Optional[int] = None) -> nn.Module:
    """A3 — LEMO with corrected B_ell, FiLM-free."""
    cfg = config.get("model", config)
    if length is None:
        length = cfg.get("length", config.get("length", 64))
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    spatial_modes = cfg.get("spatial_modes", None)
    if spatial_modes is None:
        spatial_modes = tuple(min(s // 2, 12) for s in spatial_shape)
    spatial_modes = tuple(spatial_modes)
    return LEMOFreeFiLMND(
        in_channels=in_channels, out_channels=out_channels, length=length,
        spatial_shape=spatial_shape, spatial_modes=spatial_modes,
        width=cfg.get("width", 64),
        n_layers=cfg.get("n_layers", 3),
        kernel_hidden=cfg.get("kernel_hidden", 64),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "FiLMHead",
    "FNOFiLMBlock", "FNOFiLMND", "create_fno_film_nd",
    "NonEqLagMixerFiLM", "NonEqFiLMNDBlock", "NonEqFiLMND", "create_noneq_film_nd",
    "LEMOFreeFiLMBlock", "LEMOFreeFiLMND", "create_lemo_bcorrect_nd",
    "count_parameters",
]
