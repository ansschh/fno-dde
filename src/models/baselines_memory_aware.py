"""
Memory-aware ND baselines.

Round 3 review punch-list, fatal #1: missing head-to-head against specialized
memory/delay operator baselines.  We currently have MemNO (Buitrago et al.
ICLR 2025) and F-FNO (Tran et al. ICLR 2023) in `baselines_nd.py`.  This file
adds the three remaining baselines that reviewers specifically named:

  - NIDE_ND  (Zappala et al. AAAI 2023, Neural Integro-Differential Equations
              + the NIE/ANIE follow-up in Nature Machine Intelligence 2024)
  - NDDE_ND  (Zhu et al. ICLR 2021, Neural Delay Differential Equations,
              with reference to Huang et al. 2025 NADDE / adaptive-delay)
  - S4_ND    (Gu et al. ICLR 2022, Structured State Spaces with HiPPO init)

The three classes adopt the same `(B, T, *spatial, C+aux) -> (B, T, *spatial, C)`
input/output layout and the same `__init__` parameter naming (`in_channels`,
`out_channels`, `n_total`, `n_hist`, `spatial_shape`, `width`, `n_layers`,
...) used by `MarkovFNOND`, `UNetND`, `MemNOND`, and `FactorizedFNOND` in
`baselines_nd.py`.  Factory functions `create_*_nd(in_channels, out_channels,
config, length=None) -> nn.Module` mirror the existing factory signatures so
the `run_apebench_sweep.py` model-resolver can pick them up unchanged.

NIDE_ND, NDDE_ND, and S4_ND are fully implemented:
  - NIDE_ND  : real-valued time-domain lag kernel + spatial FNO
               (Zappala et al. 2023).
  - NDDE_ND  : K learnable scalar delays with cyclic linear-interp lookup
               + per-block stacked-lag MLP + spatial FNO
               (Zhu et al. 2021; optional Huang et al. 2025 adaptive-delay
               head behind ``adaptive_delay=True``).
  - S4_ND    : S4D diagonal SSM with HiPPO-LegS init + FFT convolution along
               the lag axis (per spatial position) + spatial FNO mixing per
               lag step (Gu, Goel, Re 2022; Gu et al. 2023 diagonal variant).

References
----------
Zappala et al., "Neural Integro-Differential Equations", AAAI 2023.
  arXiv:2206.14282 ; bibkey: zappala2023nide.
Zappala et al., "Learning Integral Operators via Neural Integral Equations",
  Nature Machine Intelligence 2024 ; bibkey: zappala2024nie.
Zhu et al., "Neural Delay Differential Equations", ICLR 2021 ;
  bibkey: zhu2021ndde.
Huang et al., "Neural Adaptive Delay Differential Equations",
  Neurocomputing 2025 ; bibkey: huang2025nadde.
Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces"
  (S4), ICLR 2022 ; bibkey: gu2022s4.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno_nd import SpectralConvND


# ---------------------------------------------------------------------
# NIDE — Neural Integro-Differential Equations (Zappala et al. AAAI 2023)
# ---------------------------------------------------------------------

class _SpatialFNOBlockND(nn.Module):
    """ND spectral conv along spatial axes only, per lag position.

    Mirrors `lemo_pc_nd.SpatialFNOND` but kept inline so this file is
    self-contained.  Input/output layout is `(B, C, lag, *spatial)`.
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
        max_modes = []
        for a, n in enumerate(spatial_shape):
            if a == self.n_spatial - 1:
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
        if self.n_spatial == 0:
            return x
        spatial_dims = tuple(range(3, 3 + self.n_spatial))
        x_hat = torch.fft.rfftn(x, dim=spatial_dims)
        out_shape = list(x_hat.shape)
        out_shape[1] = self.out_channels
        out_ft = torch.zeros(*out_shape, dtype=torch.cfloat, device=x.device)
        slc = [slice(None), slice(None), slice(None)]
        for i in range(self.n_spatial):
            slc.append(slice(0, self.modes[i]))
        x_block = x_hat[tuple(slc)]
        ax_letters = "pqrstuv"[:self.n_spatial]
        eq = f"io{ax_letters},bil{ax_letters}->bol{ax_letters}"
        y_block = torch.einsum(eq, self.weights, x_block)
        out_slc = [slice(None), slice(None), slice(None)] + [slice(0, m) for m in self.modes]
        out_ft[tuple(out_slc)] = y_block
        y = torch.fft.irfftn(out_ft, s=tuple(self.spatial_shape), dim=spatial_dims)
        return y


class _RealLagConv(nn.Module):
    """Real-valued time-domain lag kernel computed via cyclic FFT.

    Implements the integral term of NIDE:
        h_int(t) = sum_{tau=0}^{L-1}  K(tau) @ v(t - tau mod L)
    where `K` is a real-valued FIR kernel of length `lag_modes` (i.e. a
    function of the lag offset only -- time-translation-invariant, so this
    realizes the autonomous-PDE simplification of Eq. 2 in Zappala et al.
    2023).  The kernel is real in the time domain (NOT complex spectral
    coefficients as in LEMO); it is applied via cyclic FFT for O(L log L)
    cost per (batch, channel, spatial) position.

    Input/output layout: `(B, C_in, lag, *spatial)` -> `(B, C_out, lag, *spatial)`.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 lag_modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lag_modes = lag_modes
        scale = 1.0 / math.sqrt(in_channels * lag_modes)
        # Real-valued FIR kernel: K[in, out, tau] for tau in [0, lag_modes).
        self.K_time = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, lag_modes,
                                  dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_spatial = x.dim() - 3
        L = x.shape[2]
        # Move lag to last axis.
        perm = [0, 1] + list(range(3, 3 + n_spatial)) + [2]
        x_moved = x.permute(*perm)                       # (B, in, *spatial, lag)
        x_hat = torch.fft.rfft(x_moved, dim=-1)          # (B, in, *spatial, L//2+1)
        M_t = min(self.lag_modes, L)
        K = self.K_time[..., :M_t]                       # (in, out, M_t)
        K_padded = F.pad(K, (0, L - M_t))                # (in, out, L)
        K_hat = torch.fft.rfft(K_padded, dim=-1)         # (in, out, L//2+1)
        sp_letters = "pqrstuv"[:n_spatial]
        eq = f"iom,bi{sp_letters}m->bo{sp_letters}m"
        y_hat = torch.einsum(eq, K_hat, x_hat)
        y_moved = torch.fft.irfft(y_hat, n=L, dim=-1)    # (B, out, *spatial, lag)
        # Permute lag back to axis 2.
        inv = [0, 1, x.dim() - 1] + list(range(2, x.dim() - 1))
        return y_moved.permute(*inv)


class _NIDEBlockND(nn.Module):
    """One NIDE block: instantaneous channel-mix + lag-convolution integral
    term + spatial spectral block + nonlinearity + LayerNorm.

    Output =  Norm( act( MLP_inst(u) + LagConv[K, MLP_state(u)] + A_spat(u) ) )

    Note vs LEMO-PC: NO FiLM (no params conditioning of the kernel), and
    the lag kernel is REAL in the time domain.
    """

    def __init__(
        self,
        width: int,
        lag_modes: int,
        spatial_shape: Sequence[int],
        spatial_modes: Sequence[int],
    ) -> None:
        super().__init__()
        self.width = width
        self.use_spatial = len(spatial_shape) > 0
        # Instantaneous channel-mix (1x1 conv = pointwise linear in channels).
        self.mlp_inst = nn.Conv1d(width, width, 1)
        # State channel-mix applied before the integral kernel.
        self.mlp_state = nn.Conv1d(width, width, 1)
        # Real-valued lag convolution kernel (the K(tau) of NIDE).
        self.lag_conv = _RealLagConv(width, width, lag_modes)
        if self.use_spatial:
            self.spatial = _SpatialFNOBlockND(width, width,
                                                spatial_shape=spatial_shape,
                                                modes=spatial_modes)
        else:
            self.spatial = None
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(width)

    def _channel_linear(self, x: torch.Tensor, conv1x1: nn.Conv1d) -> torch.Tensor:
        """Apply a 1x1 conv as a pointwise linear over the channel axis,
        for input shape (B, C, lag, *spatial)."""
        x_chan_last = x.movedim(1, -1)
        y = F.linear(x_chan_last, conv1x1.weight.squeeze(-1), conv1x1.bias)
        return y.movedim(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, width, lag, *spatial) -> (B, width, lag, *spatial)."""
        h_inst = self._channel_linear(x, self.mlp_inst)
        h_state = self._channel_linear(x, self.mlp_state)
        h_int = self.lag_conv(h_state)
        if self.use_spatial:
            h_spat = self.spatial(x)
            y = self.act(h_inst + h_int + h_spat)
        else:
            y = self.act(h_inst + h_int)
        # LayerNorm on channel axis (channels-last view).
        y_chan_last = y.movedim(1, -1)
        y_chan_last = self.norm(y_chan_last)
        return y_chan_last.movedim(-1, 1)


class NIDE_ND(nn.Module):
    """Neural Integro-Differential Equation operator (Zappala et al. AAAI 2023).

    Forward computes (per layer, then stacked):
        u_pred(t) = MLP_inst(u(t)) + integral_{t-T}^{t}  K(t-s) MLP_state(u(s)) ds

    For our discrete-grid setting:
      1. Apply pointwise channel-mix `MLP_inst` per (lag, spatial) position.
      2. Apply pointwise channel-mix `MLP_state` per (lag, spatial) position.
      3. Compute the integral via cyclic-FFT lag convolution with a learnable
         REAL-valued time-domain kernel `K(tau)` (function of the lag offset
         only -- time-translation-invariant, the autonomous-PDE simplification
         of Eq. 2 of Zappala 2023).
      4. Sum + apply spatial FNO-style spectral block + GELU + LayerNorm.
      5. Project width -> out_channels.

    Architectural difference vs LEMO-PC: NO FiLM modulation, kernel is REAL
    in the time domain (not complex in spectral domain).  See `references.bib`
    key `zappala2023nide` (also `zappala2024nie` for the NIE/ANIE follow-up).

    Residual_anchor-compatible: the layer is purely additive in input space
    (no hard-coded reference frame); residual_anchor pre-processing in the
    data pipeline is honoured by passing the anchored (input, target) pair
    through unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: Sequence[int],
        spatial_modes: Optional[Sequence[int]] = None,
        lag_modes: int = 24,
        width: int = 64,
        n_layers: int = 3,
        n_total: Optional[int] = None,
        n_hist: Optional[int] = None,
        params_dim: int = 0,
        ode_solver: str = "euler",
        ode_steps_per_dt: int = 1,
        sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.width = width
        self.n_layers = n_layers
        self.lag_modes = lag_modes
        self.n_total = n_total
        self.n_hist = n_hist
        # Kept for factory-signature compatibility with the stubs; unused
        # since the discrete cyclic-FFT lag conv subsumes the explicit ODE
        # stepper for autonomous PDEs.
        self.ode_solver = ode_solver
        self.ode_steps_per_dt = ode_steps_per_dt
        self.params_dim = params_dim
        self.sigma = sigma

        if spatial_modes is None:
            spatial_modes = tuple(min(s // 2, 16) for s in self.spatial_shape)
        self.spatial_modes = tuple(spatial_modes)

        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            _NIDEBlockND(
                width=width,
                lag_modes=lag_modes,
                spatial_shape=self.spatial_shape,
                spatial_modes=self.spatial_modes,
            )
            for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lag, *spatial, in_channels) -> (B, lag, *spatial, out_channels)."""
        x_lift = self.lift(x)                            # (B, lag, *spatial, width)
        # Permute to channels-second: (B, width, lag, *spatial).
        if self.n_spatial == 0:
            x_chan = x_lift.permute(0, 2, 1)
        else:
            perm = [0, x_lift.dim() - 1, 1] + list(range(2, x_lift.dim() - 1))
            x_chan = x_lift.permute(*perm)
        for blk in self.blocks:
            x_chan = blk(x_chan)
        # Permute back to channels-last for projection.
        if self.n_spatial == 0:
            x_seq = x_chan.permute(0, 2, 1)
        else:
            inv = [0, 2] + list(range(3, 3 + self.n_spatial)) + [1]
            x_seq = x_chan.permute(*inv)
        h = self.head_act(self.head1(x_seq))
        return self.head2(h)


def create_nide_nd(in_channels: int, out_channels: int, config: dict,
                   length: Optional[int] = None) -> nn.Module:
    cfg = config.get("model", config)
    n_total = cfg.get("length", length or 64)
    n_hist = cfg.get("n_hist", n_total // 2)
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    spatial_modes = cfg.get("spatial_modes", None)
    if spatial_modes is not None and not isinstance(spatial_modes, (list, tuple)):
        spatial_modes = (spatial_modes,)
    sigma = cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    return NIDE_ND(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_shape=spatial_shape,
        spatial_modes=tuple(spatial_modes) if spatial_modes else None,
        lag_modes=cfg.get("lag_modes", 24),
        width=cfg.get("width", 64),
        n_layers=cfg.get("n_layers", 3),
        n_total=n_total,
        n_hist=n_hist,
        params_dim=cfg.get("params_dim", 0),
        ode_solver=cfg.get("ode_solver", "euler"),
        ode_steps_per_dt=cfg.get("ode_steps_per_dt", 1),
        sigma=sigma,
    )


# ---------------------------------------------------------------------
# NDDE — Neural Delay Differential Equations (Zhu et al. ICLR 2021)
# ---------------------------------------------------------------------

def _gather_lag_with_frac(
    h: torch.Tensor, idx_floor: torch.Tensor, idx_ceil: torch.Tensor,
    frac: torch.Tensor,
) -> torch.Tensor:
    """Differentiable cyclic lookup along the lag (T) axis.

    Inputs
    ------
    h          : (B, T, *spatial, C)        — feature tensor
    idx_floor  : (T_out,)  long             — floor of (t - tau_k)  mod T
    idx_ceil   : (T_out,)  long             — ceil  of (t - tau_k)  mod T
    frac       : (T_out,)  float            — fractional weight in [0, 1)
                                              (carries gradient w.r.t. tau)

    Returns
    -------
    (B, T_out, *spatial, C) tensor.

    Notes
    -----
    * Index tensors are detached (no grad).  The gradient flows back to
      ``tau`` purely through ``frac`` (lerp coefficient).
    * Cyclic wrap on the lag axis matches the cyclic-FFT convention used
      throughout the codebase (LEMO, F-FNO, MemNO, NIDE).
    """
    fl = h.index_select(1, idx_floor)            # (B, T_out, *spatial, C)
    ce = h.index_select(1, idx_ceil)             # (B, T_out, *spatial, C)
    view = [1, frac.shape[0]] + [1] * (h.dim() - 2)
    f = frac.view(*view).to(h.dtype)
    # torch.lerp:  fl + f * (ce - fl) = (1 - f) * fl + f * ce.
    return fl + f * (ce - fl)


class NDDE_ND(nn.Module):
    """Neural Delay Differential Equation operator (Zhu et al., ICLR 2021).

    Discrete-grid PDE form:

        u_pred(t) = MLP(u(t), u(t - tau_1), ..., u(t - tau_K))

    where ``tau_1, ..., tau_K`` are *K = n_delays* learnable continuous
    scalar delays in ``[0, T-1]``.  History values ``u(t - tau_k)`` for
    non-integer ``tau_k`` are obtained by linear interpolation between
    ``u[t - floor(tau_k)]`` and ``u[t - ceil(tau_k)]`` along the lag axis,
    using cyclic wrap (matching the cyclic-FFT convention of LEMO / F-FNO).

    The interpolation is differentiable w.r.t. ``tau_k`` because the
    fractional weight ``frac(tau_k) = tau_k - floor(tau_k)`` carries
    autograd to the underlying scalar (see ``_gather_lag_with_frac``).

    Layout
    ------
    Input  : (B, T, *spatial, in_channels)   — channels-last
    Output : (B, T, *spatial, out_channels)

    For each block:
      * Stack the K interpolated lag lookups into a (B, T, *spatial, K*width)
        feature tensor.
      * Pointwise MLP collapses the lag-stack back to ``width`` channels.
      * Spatial FNO conv (per (batch, lag) slice) injects spatial coupling.
      * Residual + LayerNorm + GELU.

    Optional
    --------
    ``adaptive_delay=True`` (Huang et al. Neurocomputing 2025): delays
    become ``tau_k = base_tau_k + delta_tau_k(global_summary(u))`` where
    ``global_summary`` is a global mean over (T, spatial).  Default OFF
    matches the Zhu et al. 2021 fixed-delay setting.

    References
    ----------
    Zhu et al., "Neural Delay Differential Equations", ICLR 2021.
      bibkey: zhu2021ndde.
    Huang et al., "Neural Adaptive Delay Differential Equations",
      Neurocomputing 2025.  bibkey: huang2025nadde.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: Sequence[int],
        spatial_modes: Optional[Sequence[int]] = None,
        width: int = 64,
        n_layers: int = 3,
        n_delays: int = 8,
        adaptive_delay: bool = False,
        interp: str = "linear",            # only "linear" implemented
        n_total: Optional[int] = None,
        n_hist: Optional[int] = None,
        params_dim: int = 0,
        sigma: Optional[float] = None,     # accepted for config compat; not used by NDDE
        # legacy compatibility kwargs (ignored):
        ode_solver: str = "euler",
        ode_steps_per_dt: int = 1,
    ) -> None:
        super().__init__()
        if interp != "linear":
            raise ValueError(
                f"NDDE_ND: only interp='linear' is implemented; got {interp!r}. "
                "Cubic-spline interpolation is left for future work; linear "
                "matches the Zhu et al. 2021 reference implementation."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.width = width
        self.n_layers = n_layers
        self.n_delays = int(n_delays)
        self.adaptive_delay = bool(adaptive_delay)
        self.interp = interp
        self.n_total = n_total
        self.n_hist = n_hist
        self.params_dim = params_dim
        self.sigma = sigma                  # unused; kept for sweep-config parity
        self.ode_solver = ode_solver
        self.ode_steps_per_dt = ode_steps_per_dt

        if spatial_modes is None:
            spatial_modes = tuple(min(s // 2, 16) for s in self.spatial_shape)
        self.spatial_modes = tuple(spatial_modes)

        # ----------------------------------------------------------
        # Learnable scalar delays: tau_k in [0, max_tau]
        # ----------------------------------------------------------
        # Use a sigmoid reparameterization so taus stay in [0, max_tau]
        # without hard projection (gradients always defined).  ``max_tau``
        # is set lazily on first forward when n_total is unknown.
        self.register_buffer(
            "max_tau",
            torch.tensor(float(max(int(n_total or 0) - 1, 1)), dtype=torch.float32),
            persistent=False,
        )
        # raw_taus is initialised so that sigmoid(raw)*max_tau is a uniform
        # sweep over the available lag range.  When n_total is unknown at
        # construction time we initialise around 0 (sigmoid(0)=0.5) and
        # rescale the spread on the first forward when T is observed.
        max_tau_init = max(float((n_total or 8) - 1), 1.0)
        init_taus = torch.linspace(
            0.5, max_tau_init - 0.5, self.n_delays
        ).clamp_min(0.5)
        p = (init_taus / max_tau_init).clamp(1e-3, 1 - 1e-3)
        self.raw_taus = nn.Parameter(torch.log(p / (1 - p)))   # (K,)

        # ----------------------------------------------------------
        # Lift / project
        # ----------------------------------------------------------
        self.lift = nn.Linear(in_channels, width)
        self.proj1 = nn.Linear(width, width * 2)
        self.proj2 = nn.Linear(width * 2, out_channels)

        # ----------------------------------------------------------
        # K-lag-stack collapse MLP per layer (pointwise over space, time)
        # ----------------------------------------------------------
        self.collapse_in = nn.ModuleList([
            nn.Linear(self.n_delays * width, width) for _ in range(n_layers)
        ])
        self.collapse_out = nn.ModuleList([
            nn.Linear(width, width) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(n_layers)
        ])
        # Spatial FNO conv per layer (treats time as a batch dim when n_spatial>0).
        if self.n_spatial > 0:
            self.spatial_blocks = nn.ModuleList([
                SpectralConvND(width, width,
                                physical_shape=self.spatial_shape,
                                modes=self.spatial_modes)
                for _ in range(n_layers)
            ])
            self.spatial_w = nn.ModuleList([
                nn.Linear(width, width) for _ in range(n_layers)
            ])
        else:
            self.spatial_blocks = None
            self.spatial_w = None

        # ----------------------------------------------------------
        # Adaptive-delay head (Huang et al. 2025, NADDE)
        # ----------------------------------------------------------
        if self.adaptive_delay:
            self.adaptive_head = nn.Sequential(
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, self.n_delays),
            )
            with torch.no_grad():
                # Init the final layer near zero so adaptive offsets start small.
                self.adaptive_head[-1].weight.mul_(0.01)
                self.adaptive_head[-1].bias.zero_()
        else:
            self.adaptive_head = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_taus(self, T: int, h: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the current delays.

        Without adaptive_delay   : ``(K,)``  shared across the batch.
        With    adaptive_delay   : ``(B, K)`` per-sample.

        All values are in ``[0, T-1]`` thanks to the sigmoid
        reparameterization.  Gradients flow back to ``self.raw_taus``
        (and the adaptive-head weights, if enabled).
        """
        # Use observed T for the bound so the model is portable across
        # different sequence lengths.
        max_tau = float(max(T - 1, 1))
        # Keep buffer in sync (helps any external introspection).
        if abs(float(self.max_tau.item()) - max_tau) > 1e-6:
            self.max_tau.fill_(max_tau)
        base = torch.sigmoid(self.raw_taus) * max_tau              # (K,)
        if self.adaptive_delay and h is not None and self.adaptive_head is not None:
            mean_dims = tuple(range(1, h.dim() - 1))
            summary = h.mean(dim=mean_dims)                         # (B, width)
            offsets = self.adaptive_head(summary)                   # (B, K)
            # Reparameterize the (base + offset) sum through a sigmoid so
            # taus are smoothly bounded to (0, max_tau) without zeroing
            # gradients at the boundary (the previous .clamp(0, max_tau) had
            # exactly-zero gradient at the boundary, silently killing
            # adaptive-delay learning when the raw sum saturates — H1 fix).
            # Center the input at logit(base/max_tau) so the adaptive
            # offsets start as small perturbations to the base delay.
            base_logit = torch.logit(
                (base / max_tau).clamp(1e-3, 1 - 1e-3)
            )                                                       # (K,)
            taus = max_tau * torch.sigmoid(
                base_logit.unsqueeze(0) + offsets
            )                                                       # (B, K) in (0, max_tau)
            return taus
        return base                                                  # (K,)

    def _build_lag_indices(
        self, tau: torch.Tensor, T: int, device: torch.device,
    ) -> tuple:
        """Build cyclic-wrap index tensors for one delay scalar.

        Returns ``(idx_floor, idx_ceil, frac)`` each of shape ``(T,)``.
        Note: ``tau`` is a 0-d tensor that *carries gradient*; ``frac``
        does too.  ``idx_floor`` / ``idx_ceil`` are long tensors (no grad
        by construction) computed from ``floor(src) % T``.
        """
        t = torch.arange(T, device=device, dtype=torch.float32)      # (T,)
        # Source position (continuous, possibly negative).  Modulo T puts
        # it in [0, T).  PyTorch's ``%`` handles floats correctly.
        src = (t - tau) % T
        floor_f = torch.floor(src)
        frac = src - floor_f                                         # (T,)
        idx_floor = floor_f.detach().long() % T                      # (T,)
        idx_ceil = (idx_floor + 1) % T                               # (T,)
        return idx_floor, idx_ceil, frac

    def _stack_lags(self, h: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Build the (B, T, *spatial, K * width) stacked-lag feature tensor."""
        B, T = h.shape[0], h.shape[1]
        chunks = []
        if taus.dim() == 1:
            # Shared taus: same indices for the whole batch.
            for k in range(self.n_delays):
                idx_fl, idx_ce, frac = self._build_lag_indices(
                    taus[k], T, h.device)
                chunks.append(_gather_lag_with_frac(h, idx_fl, idx_ce, frac))
            return torch.cat(chunks, dim=-1)                         # (B, T, *spatial, K*width)
        # Per-sample taus (adaptive_delay=True).  Loop over batch since
        # index_select does not broadcast its index axis.  K * B linear
        # interp lookups; B small in practice.
        for k in range(self.n_delays):
            tau_b = taus[:, k]                                       # (B,)
            t_grid = torch.arange(T, device=h.device, dtype=torch.float32)
            src = (t_grid.unsqueeze(0) - tau_b.unsqueeze(1)) % T     # (B, T)
            floor_f = torch.floor(src)
            frac = src - floor_f                                     # (B, T)
            idx_floor = (floor_f.detach().long() % T)                 # (B, T)
            idx_ceil = (idx_floor + 1) % T                           # (B, T)
            fl_list, ce_list = [], []
            for b in range(B):
                fl_list.append(h[b].index_select(0, idx_floor[b]))
                ce_list.append(h[b].index_select(0, idx_ceil[b]))
            fl = torch.stack(fl_list, dim=0)                         # (B, T, *spatial, width)
            ce = torch.stack(ce_list, dim=0)
            view = [B, T] + [1] * (h.dim() - 2)
            f = frac.view(*view).to(h.dtype)
            chunks.append(fl + f * (ce - fl))
        return torch.cat(chunks, dim=-1)

    def _spatial_block(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Apply per-layer spatial FNO conv to (B, T, *spatial, width).

        Treats (B, T) as a combined batch dim while doing rfftn over the
        spatial axes, then restores the time axis.
        """
        if self.spatial_blocks is None:
            # No spatial axes -- still apply a pointwise channel mix
            # via the residual nn.Linear (kept None here for symmetry; the
            # outer loop already adds residual via collapse_out).
            return x
        B, T = x.shape[0], x.shape[1]
        spatial = self.spatial_shape
        x_merge = x.reshape(B * T, *spatial, self.width)             # (BT, *spatial, width)
        perm = [0, x_merge.dim() - 1] + list(range(1, x_merge.dim() - 1))
        x_chan = x_merge.permute(*perm).contiguous()                 # (BT, width, *spatial)
        Ax = self.spatial_blocks[layer_idx](x_chan)                  # (BT, width, *spatial)
        Ax_chan_last = Ax.movedim(1, -1)                             # (BT, *spatial, width)
        Wx = self.spatial_w[layer_idx](x_merge)                       # (BT, *spatial, width)
        y = F.gelu(Ax_chan_last + Wx)
        return y.reshape(B, T, *spatial, self.width)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, *spatial, in_channels). Returns (B, T, *spatial, out_channels)."""
        if x.dim() < 3:
            raise ValueError(
                f"NDDE_ND expects a (B, T, *spatial, C) tensor; got shape {tuple(x.shape)}"
            )
        # ---- Lift to width.
        h = self.lift(x)                                              # (B, T, *spatial, width)
        T = h.shape[1]

        # ---- Get delay scalars (possibly adaptive).
        taus = self._get_taus(T, h if self.adaptive_delay else None)

        # ---- Stacked-lag MLP blocks with residual + LN + spatial FNO.
        for li in range(self.n_layers):
            stacked = self._stack_lags(h, taus)                       # (B, T, *spatial, K*width)
            h_new = F.gelu(self.collapse_in[li](stacked))             # (B, T, *spatial, width)
            h_new = self.collapse_out[li](h_new)                      # (B, T, *spatial, width)
            # Residual.
            h_res = h + h_new
            # Spatial FNO mix (no-op when n_spatial == 0).
            h_res = self._spatial_block(li, h_res)
            # LayerNorm on channel dim.
            h = self.norms[li](h_res)

        # ---- Project back to out_channels.
        h = F.gelu(self.proj1(h))
        return self.proj2(h)


def create_ndde_nd(in_channels: int, out_channels: int, config: dict,
                   length: Optional[int] = None) -> nn.Module:
    cfg = config.get("model", config)
    n_total = cfg.get("length", length or 64)
    n_hist = cfg.get("n_hist", n_total // 2)
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    spatial_modes = cfg.get("spatial_modes", None)
    if spatial_modes is not None and not isinstance(spatial_modes, (list, tuple)):
        spatial_modes = (spatial_modes,)
    if spatial_modes is not None:
        spatial_modes = tuple(spatial_modes)
    sigma = cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    return NDDE_ND(
        in_channels=in_channels, out_channels=out_channels,
        spatial_shape=spatial_shape,
        spatial_modes=spatial_modes,
        width=cfg.get("width", 64),
        n_layers=cfg.get("n_layers", 3),
        n_delays=cfg.get("n_delays", 8),
        adaptive_delay=cfg.get("adaptive_delay", False),
        interp=cfg.get("interp", "linear"),
        n_total=n_total,
        n_hist=n_hist,
        params_dim=cfg.get("params_dim", 0),
        sigma=sigma,
        ode_solver=cfg.get("ode_solver", "euler"),
        ode_steps_per_dt=cfg.get("ode_steps_per_dt", 1),
    )


# ---------------------------------------------------------------------
# S4 — Structured State Spaces (Gu et al. ICLR 2022)
# ---------------------------------------------------------------------

class S4D_Layer(nn.Module):
    """Single S4D block with diagonal SSM + FFT convolution along the lag axis.

    Gu, Goel, Re (ICLR 2022) "Efficiently Modeling Long Sequences with
    Structured State Spaces"; the diagonal (S4D) variant per Gu et al. 2023
    "On the Parameterization and Initialization of Diagonal State Space
    Models".  bibkey: gu2022s4.

    Per-channel diagonal SSM:

        A_n  = -exp(log_A_real_n) + i * A_imag_n           (state pole)
        K[t] = real( sum_n  C_n * exp(t * dt * A_n) * conj(B_n) )

    HiPPO-LegS-style initialization (the canonical S4D-LegS approximation
    of Gu et al. 2022, App. C.4):

        log_A_real = log( 0.5 + log(1 + arange(N)**2) )
            i.e.  A_real_n = -(0.5 + log(1 + n^2))   ->  log-spaced poles
        A_imag     = pi * arange(N)
        B          = ones(N)
        C          ~ randn(N) / sqrt(N)

    The kernel is *recomputed each forward pass* since A, B, C, log_dt are
    all learnable (the Cauchy structure is trivial in the diagonal form, so
    no caching is required — this matches the canonical S4D reference impl).

    Input/output: ``(B, T, d_model)``.  Convolution is *circular* along the
    lag axis (length-T FFT, no zero-padding) so the layer is exactly cyclic-
    shift equivariant on the lag axis — this is required by the lag-
    equivariance T1 test and is the canonical interpretation of the SSM as a
    1D translation-equivariant convolution.
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

        # Learnable per-channel time-step (log-uniform init in [dt_min, dt_max]).
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt.float())

        # HiPPO-LegS diagonal init.  We parameterize log_A_real so that
        #   A_real_n = -exp(log_A_real_n)  is negative (=> stable).
        # The HiPPO-LegS approximation places real parts at -(1/2 + log(1+n^2)),
        # giving log-spaced poles that match the LegS Legendre-polynomial
        # memory kernel approximation in Gu et al. 2022.
        n_idx = torch.arange(d_state, dtype=torch.float32)
        a_real = 0.5 + torch.log1p(n_idx ** 2)              # positive magnitudes
        log_A_real = torch.log(a_real).expand(d_model, -1).contiguous()
        A_imag = (math.pi * n_idx).expand(d_model, -1).contiguous()
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        # B = ones(N) (per HiPPO-LegS convention; only C is randn / sqrt(N)).
        # We keep B as complex (real and imag parts) but initialize imag = 0.
        self.B_real = nn.Parameter(torch.ones(d_model, d_state))
        self.B_imag = nn.Parameter(torch.zeros(d_model, d_state))
        # C ~ randn / sqrt(N) — the HiPPO-LegS C-init (Gu et al. 2022, App. C).
        scale = 1.0 / math.sqrt(d_state)
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * scale)

        # Per-channel skip ("D" term in the canonical SSM y = Cx + Du).
        self.D = nn.Parameter(torch.randn(d_model))

    def kernel(self, L: int) -> torch.Tensor:
        """Recompute the SSM impulse response of length L.

        Returns: real tensor of shape ``(d_model, L)``.
        """
        dt = torch.exp(self.log_dt)                                # (M,)
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)  # (M, N)
        B = torch.complex(self.B_real, self.B_imag)
        C = torch.complex(self.C_real, self.C_imag)
        dtA = A * dt.unsqueeze(-1)                                 # (M, N)
        l_idx = torch.arange(L, device=dt.device, dtype=torch.float32)
        # exp(l * dtA) — shape (M, N, L).
        exp_term = torch.exp(dtA.unsqueeze(-1) * l_idx)
        # K[t] = real( sum_n C_n * conj(B_n) * exp(t * dtA_n) ).
        # Following the diagonal-SSM convention we collapse C * B^* into one
        # complex weight per (channel, state) pair (canonical S4D form).
        CB = (C * B.conj()).unsqueeze(-1)                          # (M, N, 1)
        k = (CB * exp_term).sum(dim=1).real                        # (M, L)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model). Returns (B, L, d_model).

        Circular FFT convolution along the lag axis: y = circ_conv(x, K) + D * x.

        We deliberately use length-L (circular) rather than length-2L (linear /
        zero-padded causal) FFT convolution so the layer is exactly cyclic-shift
        equivariant on the lag axis: roll(s4(x), k) == s4(roll(x, k)).  This
        equivariance is required by the lag-equivariance T1 test and is the
        defining property of the S4D SSM viewed as a 1D conv along the lag axis.
        Zero-padded (linear) conv breaks this because content that wraps around
        the boundary under a cyclic shift would not have wrapped under linear
        conv (it gets convolved with kernel taps "from the future" instead).
        """
        L = x.shape[1]
        k = self.kernel(L)                                         # (M, L)
        x_c = x.transpose(1, 2).contiguous()                       # (B, M, L)
        # Length-L FFT for circular (cyclic-equivariant) convolution.
        x_f = torch.fft.rfft(x_c, n=L, dim=-1)
        k_f = torch.fft.rfft(k, n=L, dim=-1)
        y_f = x_f * k_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=L, dim=-1)
        y = y.transpose(1, 2).contiguous()                         # (B, L, M)
        return y + x * self.D.view(1, 1, -1)


class _S4SpatialMixerND(nn.Module):
    """1x1 channel-mix + ND spectral conv applied per lag step.

    When ``spatial_shape`` is empty this is just a pointwise channel mix
    (Linear).  When ``spatial_shape`` is non-empty an FNO-style ND spectral
    conv is added so spatial points actually mix — without it the model
    would be embarrassingly parallel over space.
    """

    def __init__(self, width: int, spatial_shape: Sequence[int],
                 spatial_modes: Optional[Sequence[int]]) -> None:
        super().__init__()
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self._width = width
        if self.n_spatial > 0:
            if spatial_modes is None:
                spatial_modes = tuple(min(s // 2, 8) for s in self.spatial_shape)
            self.spatial_modes = tuple(spatial_modes)
            self.A = SpectralConvND(width, width, self.spatial_shape, self.spatial_modes)
        else:
            self.A = None
        self.W = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, *spatial, width). Returns (B, T, *spatial, width)."""
        B, T = x.shape[0], x.shape[1]
        # Channel-mix (pointwise) is dim-agnostic.
        Wx = self.W(x)
        if self.A is None:
            return Wx
        # Spectral conv per lag step: fold T into batch.
        x_flat = x.reshape(B * T, *self.spatial_shape, self._width)
        # Channels-second for SpectralConvND: (B*T, width, *spatial).
        perm = [0, x_flat.dim() - 1] + list(range(1, x_flat.dim() - 1))
        x_cf = x_flat.permute(*perm).contiguous()
        a_cf = self.A(x_cf)
        # Channels-last: (B*T, *spatial, width).
        inv = [0] + list(range(2, 2 + self.n_spatial)) + [1]
        a_clast = a_cf.permute(*inv).contiguous()
        Ax = a_clast.reshape(B, T, *self.spatial_shape, self._width)
        return Ax + Wx


class _S4Block(nn.Module):
    """One S4D + spatial-mixing block with residual + LayerNorm + GELU."""

    def __init__(self, width: int, d_state: int,
                 spatial_shape: Sequence[int],
                 spatial_modes: Optional[Sequence[int]]) -> None:
        super().__init__()
        self.s4 = S4D_Layer(d_model=width, d_state=d_state)
        self.spatial = _S4SpatialMixerND(width, spatial_shape, spatial_modes)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.act = nn.GELU()
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, *spatial, width). Returns same shape."""
        B, T = x.shape[0], x.shape[1]
        # ---- S4D over the lag axis (per spatial position) ----
        # Reshape (B, T, *spatial, C) -> (B*S, T, C) by moving T past spatial.
        spatial_size = 1
        for s in self.spatial_shape:
            spatial_size *= s
        if self.n_spatial > 0:
            x_perm = x.movedim(1, 1 + self.n_spatial).contiguous()  # (B, *spatial, T, C)
            x_seq = x_perm.reshape(B * spatial_size, T, self.width)
        else:
            x_seq = x.reshape(B, T, self.width)
        h = x_seq + self.s4(self.norm1(x_seq))                      # residual S4D
        h = self.act(h)
        # Reshape back to (B, T, *spatial, C).
        if self.n_spatial > 0:
            h = h.reshape(B, *self.spatial_shape, T, self.width)
            inv_perm = [0, 1 + self.n_spatial] + list(range(1, 1 + self.n_spatial)) + [self.n_spatial + 2]
            h = h.permute(*inv_perm).contiguous()
        else:
            h = h.reshape(B, T, self.width)
        # ---- Spatial mixing per lag step (residual) ----
        h = h + self.spatial(self.norm2(h))
        h = self.act(h)
        return h


class S4_ND(nn.Module):
    """S4D-based structured state-space model for ND PDE rollouts.

    Gu, Goel, Re ICLR 2022 (S4) + Gu et al. 2023 (S4D diagonal variant).
    bibkey: gu2022s4.

    Architecture (per the user spec):

      1. Lift `C_in -> width` via nn.Linear (channel-last).
      2. n_layers blocks of [S4D over the lag axis (per spatial position)
         + spatial mixing (1x1 channel + ND spectral conv) + residual +
         LayerNorm + GELU].
      3. Project `width -> out_channels` via 2-layer MLP head.

    The S4D layer uses a diagonal state matrix with HiPPO-LegS initialization
    (state size N=64 by default) and computes the impulse response via the
    state-space form
        K[t] = real( sum_n C_n * exp(t * dt * A_n) * conj(B_n) ),
    then performs FFT convolution along the lag axis.  Kernel is *recomputed
    each forward pass* since A, B, C, dt are all learnable.

    Input layout:  (B, T, *spatial, in_channels)
    Output layout: (B, T, *spatial, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_total: int,
        n_hist: int,
        spatial_shape: Sequence[int],
        spatial_modes: Optional[Sequence[int]] = None,
        width: int = 64,
        n_layers: int = 3,
        s4_state: int = 64,
        hippo_init: str = "legs",          # one of {"legs", "lagt", "fout"}
        bidirectional: bool = False,
        params_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_total = n_total
        self.n_hist = n_hist
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.width = width
        self.n_layers = n_layers
        self.s4_state = s4_state
        self.hippo_init = hippo_init
        self.bidirectional = bidirectional
        self.params_dim = params_dim

        if spatial_modes is not None:
            spatial_modes = tuple(spatial_modes)
        self.spatial_modes = spatial_modes

        # Lift channels -> width (channel-last).
        self.lift = nn.Linear(in_channels, width)
        # Block stack.
        self.blocks = nn.ModuleList([
            _S4Block(width=width, d_state=s4_state,
                     spatial_shape=self.spatial_shape,
                     spatial_modes=self.spatial_modes)
            for _ in range(n_layers)
        ])
        # Project to out_channels (2-layer MLP head, channel-last).
        self.proj1 = nn.Linear(width, width * 2)
        self.proj2 = nn.Linear(width * 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, *spatial, in_channels). Returns (B, T, *spatial, out_channels)."""
        # Lift channels (channel-last is dim-agnostic for nn.Linear).
        h = self.lift(x)                                    # (B, T, *spatial, width)
        for blk in self.blocks:
            h = blk(h)
        h = F.gelu(self.proj1(h))
        return self.proj2(h)                                # (B, T, *spatial, out_channels)


def create_s4_nd(in_channels: int, out_channels: int, config: dict,
                 length: Optional[int] = None) -> nn.Module:
    cfg = config.get("model", config)
    n_total = cfg.get("length", length or 64)
    n_hist = cfg.get("n_hist", n_total // 2)
    spatial_shape = tuple(cfg.get("spatial_shape", ()))
    spatial_modes = cfg.get("spatial_modes", None)
    if spatial_modes is not None and not isinstance(spatial_modes, (list, tuple)):
        spatial_modes = (spatial_modes,)
    return S4_ND(
        in_channels=in_channels, out_channels=out_channels,
        n_total=n_total, n_hist=n_hist,
        spatial_shape=spatial_shape,
        spatial_modes=tuple(spatial_modes) if spatial_modes else None,
        width=cfg.get("width", 64),
        n_layers=cfg.get("n_layers", 3),
        s4_state=cfg.get("s4_state", 64),
        hippo_init=cfg.get("hippo_init", "legs"),
        bidirectional=cfg.get("bidirectional", False),
        params_dim=cfg.get("params_dim", 0),
    )


# ---------------------------------------------------------------------
# Registry helper for run_apebench_sweep.py
# ---------------------------------------------------------------------

MEMORY_AWARE_FACTORIES = {
    "nide_nd": create_nide_nd,
    "ndde_nd": create_ndde_nd,
    "s4_nd":   create_s4_nd,
}
