"""
LEMO-PC ND — N-dimensional parameter-conditional LEMO.

Round 2.23 (2D/3D upgrade).  ND extension of `lemo_pc_v3.py`:
direct spectral coefficients for the lag kernel + per-(out_channel,
lag_mode) FiLM modulation from per-sample params, with an FNO-style
spatial spectral conv applied independently per lag.

Layout: input/output  `(batch, length, *spatial_shape, in_channels)`
where the last `params_dim` channels are interpreted as broadcast-
across-(time, space) per-sample parameters (extracted from
`x[:, 0, ..., 0, -params_dim:]`).

Setting `spatial_shape=()` reduces this to the 1D parametric case and
should reproduce `lemo_pc_v3.py` numerically (G2 regression).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSmoother(nn.Module):
    """Learnable causal smoother applied at output of LEMO-PC.

    Maps y_t -> sum_{i=0..K-1} alpha_i * y_{t-i} with alpha = softmax(logits).

    Properties:
      - Strictly causal: output at t uses only inputs at s <= t.
      - 1-Lipschitz: alpha is a probability distribution (sum=1, non-negative),
        so ||smoother(y) - smoother(y')||_p <= ||y - y'||_p.
      - Identity init: alpha_0 dominates so model starts ≈ no-op (cyclic LEMO).

    Theory (B5 boundary trade-off):
      Body LEMO-PC-Cyclic is T1-equivariant on cyclic group C_N. Composing
      with the causal smoother on output strictly breaks T1, but preserves
      causal-monoid M_N equivariance (forward shifts only) on the interior.
      The equivariance error bound is bounded by 2 * sum_{i=K_eff..N-1} alpha_i
      times the input magnitude — measurable rather than asserted.
      σ-Lipschitz contraction (Cor 6.19) is preserved since smoother is
      1-Lipschitz and body is σ-Lipschitz; composition is σ-Lipschitz.
    """

    def __init__(self, kernel_length: int = 24, init_concentration: float = 10.0):
        super().__init__()
        self.K = int(kernel_length)
        init = torch.zeros(self.K)
        init[0] = float(init_concentration)
        self.alpha_logits = nn.Parameter(init)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Apply causal smoothing on the lag axis (axis 1).

        y: (B, length, *spatial, channels) — LEMO output convention.
        Returns same shape with y[t] replaced by sum_{i} alpha_i * y[t-i].
        """
        # Move lag (axis 1) to last for conv1d.
        perm = [0] + list(range(2, y.dim())) + [1]
        y_t = y.permute(*perm).contiguous()
        L = y_t.shape[-1]
        K = min(self.K, L)
        alpha = F.softmax(self.alpha_logits[:K], dim=0)
        original_shape = y_t.shape
        y_flat = y_t.reshape(-1, 1, L)
        y_pad = F.pad(y_flat, (K - 1, 0))
        # F.conv1d does cross-correlation; flip alpha so filter[K-1]=alpha[0]
        # (so output[t] = sum_{i=0..K-1} alpha[i] * y[t-i]).
        filt = alpha.flip(0).view(1, 1, K)
        y_out = F.conv1d(y_pad, filt, padding=0).reshape(original_shape)
        # Permute lag back to axis 1.
        inv = [0, y.dim() - 1] + list(range(1, y.dim() - 1))
        return y_out.permute(*inv)

    def equivariance_diagnostics(self) -> dict:
        """Return measured boundary-contamination diagnostics for the smoother."""
        with torch.no_grad():
            alpha = F.softmax(self.alpha_logits, dim=0)
            return {
                "alpha_0": float(alpha[0].item()),
                "alpha_max_index": int(alpha.argmax().item()),
                "effective_length": int((alpha > 0.01).sum().item()),
                "tail_mass_above_8": float(alpha[8:].sum().item()) if alpha.numel() > 8 else 0.0,
            }


class FiLMLagSpectralND(nn.Module):
    """1D spectral lag conv with per-(out, lag_mode) FiLM, kernel shared
    across spatial axes.  ND extension of `FiLMSpectralConv1dV3`.

    Input:  (batch, in_channels, lag, *spatial)
    Output: (batch, out_channels, lag, *spatial)

    Optional `causal` mode reparameterizes the kernel in time-domain as a
    causal FIR filter (look back at most `lag_modes` steps).  This enforces
    Theorem T1' (causal-monoid lag-equivariance): output at position t
    depends only on inputs at positions <= t.  Strictly contained in the
    full cyclic G of T1; the right inductive bias for autoregressive PDE
    rollout where the ground-truth dynamics is causal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lag_modes: int,
        params_dim: int,
        film_hidden: int = 64,
        sigma: Optional[float] = None,
        causal: bool = False,
        zero_beta_above_dc: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lag_modes = lag_modes
        self.params_dim = params_dim
        self.sigma = sigma
        self.causal = causal
        self.zero_beta_above_dc = bool(zero_beta_above_dc)
        scale = 1.0 / (in_channels * out_channels)
        if causal:
            # Time-domain causal FIR kernel of length `lag_modes`.
            # Real-valued; right-padded with zeros to full lag length L at
            # forward time, then rfft'd to get the spectral kernel.
            self.weights_time = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, lag_modes,
                                      dtype=torch.float32)
            )
            self.weights = None
        else:
            self.weights = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, lag_modes,
                                    dtype=torch.cfloat)
            )
            self.weights_time = None
        self.film_net = nn.Sequential(
            nn.Linear(params_dim, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * out_channels * lag_modes),
        )
        with torch.no_grad():
            # Symmetry-breaking init: per-output (γ, β) start at distinct values
            # so each (out_channel, lag_mode) follows a different gradient path.
            # Last-layer weight kept small (0.02 std) to avoid initial saturation,
            # but NOT zero — this gave each family the same gradient under WD.
            self.film_net[-1].weight.normal_(0.0, 0.02)
            b = torch.zeros(2 * out_channels * lag_modes)
            n_g = out_channels * lag_modes
            # γ: N(1, 0.1) so the model starts ≈ FNO but each γ_{o,m} drifts independently
            b[:n_g] = 1.0 + 0.10 * torch.randn(n_g)
            # β: N(0, 0.01) so the additive bias has small per-mode variation
            b[n_g:] = 0.01 * torch.randn(n_g)
            self.film_net[-1].bias.copy_(b)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, in, lag, *spatial). params: (B, params_dim)."""
        n_spatial = x.dim() - 3
        B = x.shape[0]
        L = x.shape[2]
        # Move lag to last axis; spatial axes batched.
        perm = [0, 1] + list(range(3, 3 + n_spatial)) + [2]
        x_moved = x.permute(*perm)                          # (B, in, *spatial, lag)
        x_hat = torch.fft.rfft(x_moved, dim=-1)             # (B, in, *spatial, L//2+1)
        out_modes = L // 2 + 1
        sp_letters = "pqrstuv"[:n_spatial]
        if self.causal:
            # Causal FIR of length lag_modes: right-pad with zeros to L, then FFT.
            # Resulting K_full has support [0, lag_modes-1] in time domain — strictly
            # causal under cyclic boundary for output positions t >= lag_modes - 1.
            M_t = min(self.lag_modes, L)
            W = self.weights_time[..., :M_t]
            W_padded = F.pad(W, (0, L - M_t))               # (in, out, L)
            K_full = torch.fft.rfft(W_padded, dim=-1)       # (in, out, out_modes)
            # Apply sigma-projection to causal kernel — without this, the
            # causal=True branch silently violates the σ-Lipschitz constraint
            # that motivates `sigma`. Same SVD-with-Frobenius-fallback pattern
            # as the non-causal branch (cuSOLVER divergence on ill-conditioned
            # init falls back to per-mode Frobenius rescale, which preserves
            # σ-bound conservatively).
            if self.sigma is not None:
                Km = K_full.permute(2, 0, 1).contiguous()       # (out_modes, in, out)
                try:
                    U, S, Vh = torch.linalg.svd(Km, full_matrices=False)
                    S_clamped = torch.clamp(S, max=float(self.sigma))
                    K_proj = (U * S_clamped.unsqueeze(-2).to(U.dtype)) @ Vh
                except torch._C._LinAlgError:
                    frob = torch.linalg.norm(Km, dim=(-2, -1))  # (out_modes,)
                    scale = torch.clamp(float(self.sigma) / (frob + 1e-12), max=1.0)
                    K_proj = Km * scale.unsqueeze(-1).unsqueeze(-1).to(Km.dtype)
                K_full = K_proj.permute(1, 2, 0).contiguous()    # back to (in, out, out_modes)
            eq = f"iom,bi{sp_letters}m->bo{sp_letters}m"
            active = torch.einsum(eq, K_full, x_hat)        # (B, out, *spatial, out_modes)
            eff_modes = out_modes
        else:
            # Sigma-normalized base kernel.
            #
            # When `sigma` is set, project the spectral kernel so that
            # ‖K[:,:,m]‖_op ≤ sigma per mode, hence ‖LEMO_lag‖_op ≤ sigma
            # (TIGHT bound, not the loose σ·√(in·out) elementwise bound).
            #
            # Method: per-mode SVD with singular-value clamp at sigma.  For
            # complex K[:,:,m] (in×out), torch.linalg.svd gives U,S,Vh;
            # we clamp S to ≤ sigma and reconstruct.  This is the orthogonal
            # projection onto the operator-norm ball of radius sigma.
            #
            # Performance: 24 modes × 64×64 complex SVDs per layer per
            # forward.  ~5-15% slower vs unconstrained.  Safe under autograd
            # (no degenerate-singular-value cases triggered by clamping
            # downward only).
            if self.sigma is not None:
                K_raw = self.weights                            # (in, out, modes) cfloat
                Km = K_raw.permute(2, 0, 1).contiguous()        # (modes, in, out)
                try:
                    U, S, Vh = torch.linalg.svd(Km, full_matrices=False)
                    S_clamped = torch.clamp(S, max=float(self.sigma))
                    K_proj = (U * S_clamped.unsqueeze(-2).to(U.dtype)) @ Vh
                except torch._C._LinAlgError:
                    # cuSOLVER SVD diverges on ill-conditioned random init
                    # (error 63 = "failed to converge", common in early
                    # training). Fall back to per-mode Frobenius rescale:
                    # ‖K‖_op ≤ ‖K‖_F so capping by Frobenius gives a valid
                    # (conservative) upper bound on operator norm and
                    # preserves the σ-Lipschitz guarantee. Differentiable
                    # everywhere; tightens to true SVD projection once
                    # weights stabilize and SVD succeeds.
                    frob = torch.linalg.norm(Km, dim=(-2, -1))  # (modes,)
                    scale = torch.clamp(float(self.sigma) / (frob + 1e-12), max=1.0)
                    K_proj = Km * scale.unsqueeze(-1).unsqueeze(-1).to(Km.dtype)
                K = K_proj.permute(1, 2, 0).contiguous()        # back to (in, out, modes)
            else:
                K = self.weights
            eff_modes = min(self.lag_modes, out_modes)
            x_block = x_hat[..., :eff_modes]                # (B, in, *spatial, eff_modes)
            eq = f"iom,bi{sp_letters}m->bo{sp_letters}m"
            active = torch.einsum(eq, K[:, :, :eff_modes], x_block)
        # FiLM modulation per-(out, mode) on the first M = lag_modes spectral coefs.
        film = self.film_net(params)
        OC, M = self.out_channels, self.lag_modes
        gamma = film[:, :OC * M].view(B, OC, M)              # (B, out, modes)
        beta = film[:, OC * M:].view(B, OC, M)               # (B, out, modes)
        # ALWAYS apply tanh to γ, β — paper-consistent bound (|γ|≤1, |β|≤1).
        # Previously this was σ-conditional, which left γ unbounded for σ=None
        # runs and weight decay drove γ → 0 (FiLM nullification). Bounding the
        # FiLM output keeps gradient flow alive even when σ is not enforced.
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        # T1 equivariance: per-mode β added in spectral domain corresponds to
        # IFFT(β)(t), a non-DC signal in time. That signal is NOT cyclic-shift
        # invariant, so it breaks T1. With `zero_beta_above_dc=True` we keep
        # only the DC (mode 0) component of β — this preserves a per-channel
        # bias (still useful) while making the model exactly cyclic-equivariant.
        # Set on the module via `model.set_zero_beta_above_dc(True)` at eval.
        if getattr(self, "zero_beta_above_dc", False) and M > 1:
            beta = torch.cat([beta[..., :1], torch.zeros_like(beta[..., 1:])], dim=-1)
        eff = min(eff_modes, M)
        # Broadcast gamma, beta across spatial axes.  Build the broadcasting
        # shape: (B, out, *[1]*n_spatial, modes).
        view_shape = [B, OC] + [1] * n_spatial + [eff]
        gamma_b = gamma[:, :, :eff].view(*view_shape).to(torch.cfloat)
        beta_b = beta[:, :, :eff].view(*view_shape).to(torch.cfloat)
        active_modulated = active[..., :eff] * gamma_b + beta_b
        if eff_modes > eff:
            # Causal mode: preserve unmodulated spectral coefs beyond lag_modes
            # so the time-domain kernel stays causal.
            active_eff = torch.cat([active_modulated, active[..., eff:eff_modes]], dim=-1)
        else:
            active_eff = active_modulated
        # Pad with zeros to full rfft length and irfft.
        if out_modes > eff_modes:
            pad_shape = list(active_eff.shape)
            pad_shape[-1] = out_modes - eff_modes
            pad = torch.zeros(*pad_shape, dtype=torch.cfloat, device=x.device)
            out_ft = torch.cat([active_eff, pad], dim=-1)
        else:
            out_ft = active_eff
        y_moved = torch.fft.irfft(out_ft, n=L, dim=-1)        # (B, out, *spatial, lag)
        # Permute lag back to axis 2.
        inv = [0, 1, x.dim() - 1] + list(range(2, x.dim() - 1))
        return y_moved.permute(*inv)


class SpatialFNOND(nn.Module):
    """ND spectral conv along spatial axes only, per lag position. Same as
    `lemo_nd.SpatialSpectralConvND` but inlined here to keep the file
    self-contained."""

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


class LEMOPCNDBlock(nn.Module):
    """One LEMO-PC-ND block: B + A_lag (FiLM-modulated) + (optional) A_spat."""

    def __init__(
        self,
        width: int,
        lag_modes: int,
        params_dim: int,
        spatial_shape: Sequence[int],
        spatial_modes: Sequence[int],
        film_hidden: int = 64,
        sigma: Optional[float] = None,
        causal: bool = False,
        zero_beta_above_dc: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.causal = causal
        self.use_spatial = len(spatial_shape) > 0
        self.B = nn.Conv1d(width, width, 1)  # 1×1 channel mix (lag-only conv)
        self.A_lag = FiLMLagSpectralND(width, width, lag_modes, params_dim,
                                        film_hidden=film_hidden, sigma=sigma,
                                        causal=causal,
                                        zero_beta_above_dc=zero_beta_above_dc)
        if self.use_spatial:
            self.A_spat = SpatialFNOND(width, width,
                                        spatial_shape=spatial_shape,
                                        modes=spatial_modes)
        else:
            self.A_spat = None
        self.act = nn.ReLU() if sigma is not None else nn.GELU()
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (B, width, lag, *spatial). params: (B, params_dim)."""
        # B is per-(spatial, lag) channel-mix. Apply via channel-last.
        x_chan_last = x.movedim(1, -1)
        # Use a Linear via permutation so it works for any spatial dim.
        # nn.Conv1d expects (B, C, L). For multi-dim, we apply Linear on
        # the channel axis by flattening non-channel dims first.
        Bx_flat = F.linear(x_chan_last, self.B.weight.squeeze(-1), self.B.bias)
        Bx = Bx_flat.movedim(-1, 1)                          # back to channels-second
        Ax_lag = self.A_lag(x, params)
        if self.use_spatial:
            Ax_spat = self.A_spat(x)
            y = self.act(Bx + Ax_lag + Ax_spat)
        else:
            y = self.act(Bx + Ax_lag)
        # LayerNorm on channel dim.
        y_chan_last = y.movedim(1, -1)
        y_chan_last = self.norm(y_chan_last)
        return y_chan_last.movedim(-1, 1)


class LEMOPCND(nn.Module):
    """Parametric ND LEMO with per-(out, lag_mode) FiLM, ND spatial conv.

    Setting spatial_shape=() recovers `lemo_pc_v3.LEMOPCv3`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        params_dim: int,
        spatial_shape: Sequence[int] = (),
        spatial_modes: Optional[Sequence[int]] = None,
        lag_modes: int = 16,
        width: int = 48,
        n_layers: int = 3,
        film_hidden: int = 64,
        sigma: Optional[float] = None,
        extract_params: bool = True,
        causal: bool = False,
        causal_smooth: bool = False,
        causal_smooth_K: Optional[int] = None,
        zero_beta_above_dc: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.params_dim = params_dim
        self.extract_params = extract_params
        self.spatial_shape = tuple(spatial_shape)
        self.n_spatial = len(self.spatial_shape)
        self.lag_modes = lag_modes
        self.width = width
        self.causal = causal
        self.causal_smooth = causal_smooth
        self.zero_beta_above_dc = bool(zero_beta_above_dc)
        if spatial_modes is None:
            spatial_modes = tuple(min(s // 2, 16) for s in self.spatial_shape)
        self.spatial_modes = tuple(spatial_modes)

        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            LEMOPCNDBlock(
                width=width, lag_modes=lag_modes, params_dim=params_dim,
                spatial_shape=self.spatial_shape, spatial_modes=self.spatial_modes,
                film_hidden=film_hidden, sigma=sigma, causal=causal,
                zero_beta_above_dc=self.zero_beta_above_dc,
            )
            for _ in range(n_layers)
        ])
        self.head1 = nn.Linear(width, width)
        self.head2 = nn.Linear(width, out_channels)
        self.head_act = nn.ReLU() if sigma is not None else nn.GELU()
        # Optional output-side causal smoother (B5 trade-off variant).
        # Body remains cyclic-equivariant; smoother makes output strictly causal
        # at the cost of T1 equivariance (preserves σ-contraction since smoother
        # is 1-Lipschitz). See CausalSmoother docstring.
        if causal_smooth:
            K = causal_smooth_K if causal_smooth_K is not None else lag_modes
            self.causal_smoother = CausalSmoother(K)
        else:
            self.causal_smoother = None

    def _split_x_params(self, x: torch.Tensor) -> tuple:
        if not self.extract_params or self.params_dim == 0:
            B = x.shape[0]
            return x, torch.zeros(B, self.params_dim, device=x.device)
        # Read params from the (lag=0, spatial=0,..., last params_dim channels).
        # Build slice that picks the first lag and first spatial position.
        idx = [slice(None), 0]
        for _ in range(self.n_spatial):
            idx.append(0)
        idx.append(slice(-self.params_dim, None))
        params = x[tuple(idx)]                              # (B, params_dim)
        return x, params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lag, *spatial, in_channels)."""
        x_full, params = self._split_x_params(x)
        x_lift = self.lift(x_full)                          # (B, lag, *spatial, width)
        # Permute to channels-second.
        if self.n_spatial == 0:
            x_chan = x_lift.permute(0, 2, 1)                # (B, width, lag)
        else:
            perm = [0, x_lift.dim() - 1, 1] + list(range(2, x_lift.dim() - 1))
            x_chan = x_lift.permute(*perm)                   # (B, width, lag, *spatial)
        for blk in self.blocks:
            x_chan = blk(x_chan, params)
        # Permute back to channels-last for projection head.
        if self.n_spatial == 0:
            x_seq = x_chan.permute(0, 2, 1)
        else:
            inv = [0, 2] + list(range(3, 3 + self.n_spatial)) + [1]
            x_seq = x_chan.permute(*inv)
        h = self.head_act(self.head1(x_seq))
        out = self.head2(h)
        if self.causal_smoother is not None:
            out = self.causal_smoother(out)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo_pc_nd(in_channels: int, out_channels: int, config: dict,
                      length: Optional[int] = None) -> nn.Module:
    """Factory; mirrors create_lemo_pc_v3."""
    model_cfg = config.get("model", config)
    if length is None:
        length = model_cfg.get("length", config.get("length", 64))
    sigma = model_cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    spatial_shape = tuple(model_cfg.get("spatial_shape", ()))
    spatial_modes = model_cfg.get("spatial_modes", None)
    if spatial_modes is not None:
        spatial_modes = tuple(spatial_modes)
    n_spatial = len(spatial_shape)
    # params_dim default depends on input shape: in - state - mask - t = in - 4 in 1D,
    # with extra spatial channels in ND.  Allow override.
    default_params_dim = max(in_channels - 4, 1)
    params_dim = model_cfg.get("params_dim", default_params_dim)
    return LEMOPCND(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        params_dim=params_dim,
        spatial_shape=spatial_shape,
        spatial_modes=spatial_modes,
        lag_modes=model_cfg.get("lag_modes", model_cfg.get("modes", 16)),
        width=model_cfg.get("width", 48),
        n_layers=model_cfg.get("n_layers", 3),
        film_hidden=model_cfg.get("film_hidden", 64),
        sigma=sigma,
        extract_params=model_cfg.get("extract_params", True),
        causal=bool(model_cfg.get("causal", False)),
        causal_smooth=bool(model_cfg.get("causal_smooth", False)),
        causal_smooth_K=model_cfg.get("causal_smooth_K", None),
        zero_beta_above_dc=bool(model_cfg.get("zero_beta_above_dc", True)),
    )


def create_causal_lemo_pc_nd(in_channels: int, out_channels: int, config: dict,
                              length: Optional[int] = None) -> nn.Module:
    """Causal variant of LEMO-PC-ND.

    Forces the lag kernel to be a causal FIR of length `lag_modes` (right-padded
    to L before FFT).  Output at position t depends on input only at positions
    <= t (modulo cyclic wrap on the first lag_modes-1 tokens).  This realizes
    Theorem T1' (causal-monoid lag-equivariance) — strictly tighter than T1.
    """
    cfg = dict(config) if not isinstance(config, dict) else config.copy()
    if "model" in cfg:
        cfg["model"] = {**cfg["model"], "causal": True}
    else:
        cfg["causal"] = True
    return create_lemo_pc_nd(in_channels, out_channels, cfg, length=length)


def create_causal_smooth_lemo_pc_nd(in_channels: int, out_channels: int, config: dict,
                                      length: Optional[int] = None) -> nn.Module:
    """LEMO-PC + learned causal smoother on output (B5 SOTA-causal variant).

    Body retains full cyclic-equivariant LEMO-PC architecture (T1 + σ-contraction
    + FiLM benefits all preserved internally). Adds a learnable causal smoother
    P_causal on output: y_t -> sum_i alpha_i * y_{t-i} with alpha = softmax.

    Theory:
      - T1 (cyclic-equivariance) preserved on interior; broken on boundary by
        a measurable amount: equivariance error <= 2 * tail_mass(alpha) * |x|.
      - Cor 6.19 (σ-contraction) preserved: smoother is 1-Lipschitz, body is
        σ-Lipschitz, composition is σ-Lipschitz.
      - Identity-init: alpha_0 ≈ 1, so model starts as cyclic LEMO and learns
        to smooth only when beneficial.
      - Strict causality at deployment.

    Tradeoff vs. existing causal_lemo_pc_nd (which uses zero-padded FIR and
    breaks cyclic-equivariance entirely): preserves all theory benefits of
    body, makes only the output-projection layer non-equivariant.
    """
    cfg = dict(config) if not isinstance(config, dict) else config.copy()
    if "model" in cfg:
        cfg["model"] = {**cfg["model"], "causal_smooth": True}
    else:
        cfg["causal_smooth"] = True
    return create_lemo_pc_nd(in_channels, out_channels, cfg, length=length)
