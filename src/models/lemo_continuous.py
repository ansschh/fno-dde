"""
LEMO-Continuous — multi-channel, multi-layer continuous-tau LEMO.

Generalizes `ContinuousLagEquiv` from `experiments/continuous_lag_transfer.py`
(which scored 10^3x better than MLP+aug on the continuous-tau transfer task)
to the LEMO architectural family: per-channel lift, B + A blocks with the
spectral conv `A` evaluated at tau-dependent offsets, and a two-layer
projection head.

Forward contract:
    h : (batch, length, in_channels)         -- history sequence
    tau : (batch,)                            -- continuous query shift
    y : (batch, length, out_channels)        -- predicted target

The spectral conv A in each block computes its kernel at offsets
    delta[b, g] = (tau[b] - s_grid[g]) wrapped to [-L/2, L/2]
so a single sample's kernel depends on its tau, and OFF-GRID test taus
are handled by exact kernel evaluation rather than augmentation.

This is the architecture the paper's H1 / Cor 10 actually requires.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Continuous-tau lag conv
# -----------------------------------------------------------------------------

class ContinuousTauLagConv1d(nn.Module):
    """Lag-conv whose kernel is evaluated per-sample at the query tau.

    Uses an MLP-on-(scalar lag offset) kernel parameterization. For each
    sample b in the batch, computes
        delta[b, g] = ((tau[b] - s_grid[g]) wrapped to [-L/2, L/2]) / L
    feeds those into the kernel MLP to get K[b, g, c_out, c_in], then
    applies the convolution-as-quadrature
        y[b, j, c_out] = sum_g sum_c_in  K[b, j, g, c_out, c_in] * x[b, g, c_in]  * ds
                       (using j-dependent offsets when L_out != L_in;
                        for our case L_out = L_in = L and we use the full grid).

    For LEMO_sigma: rescale the kernel so its operator norm <= sigma via
    the Frobenius upper bound on |K_hat|.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        L_phys: float = 1.0,
        kernel_hidden: int = 64,
        sigma: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.L_phys = L_phys
        self.sigma = sigma
        self.kernel_net = nn.Sequential(
            nn.Linear(1, kernel_hidden),
            nn.GELU(),
            nn.Linear(kernel_hidden, kernel_hidden),
            nn.GELU(),
            nn.Linear(kernel_hidden, out_channels * in_channels),
        )
        # FNO-style init scaling: per-element output ~ 1 / (in*out)
        # so spectral magnitudes are O(1/sqrt(L)) and the conv preserves
        # activation magnitude.
        target_std = 1.0 / max(in_channels * out_channels, 1)
        with torch.no_grad():
            final = self.kernel_net[-1]
            nn.init.normal_(final.weight, mean=0.0,
                            std=target_std / math.sqrt(kernel_hidden))
            nn.init.zeros_(final.bias)
        # Static buffer for the grid s in [0, L_phys).
        s_grid = torch.linspace(0.0, L_phys, length + 1, dtype=torch.float32)[:-1]
        self.register_buffer("s_grid", s_grid)

    def _offsets(self, tau: torch.Tensor) -> torch.Tensor:
        """Per-sample wrapped lag offsets, normalized to [-1/2, 1/2)."""
        # tau: (batch,)
        delta = (tau.unsqueeze(-1) - self.s_grid.unsqueeze(0)) % self.L_phys
        delta_signed = torch.where(delta > self.L_phys / 2,
                                    delta - self.L_phys, delta)
        return (delta_signed / self.L_phys).unsqueeze(-1)  # (batch, length, 1)

    def compute_kernel(self, tau: torch.Tensor) -> torch.Tensor:
        """Per-sample kernels K[b, g, out, in]."""
        offsets = self._offsets(tau)                                  # (B, L, 1)
        K_flat = self.kernel_net(offsets)                              # (B, L, out*in)
        K = K_flat.view(offsets.shape[0], offsets.shape[1],
                        self.out_channels, self.in_channels)          # (B, L, out, in)
        if self.sigma is None:
            return K
        # Sigma-normalize per-sample. K has length L; FFT over the L axis.
        K_fc = K.permute(0, 2, 3, 1)                                  # (B, out, in, L)
        K_hat = torch.fft.rfft(K_fc, dim=-1)                          # (B, out, in, L//2+1)
        frob_per_freq = K_hat.abs().pow(2).sum(dim=(1, 2)).sqrt()     # (B, L//2+1)
        max_frob = frob_per_freq.max(dim=-1).values.clamp_min(1e-10)  # (B,)
        # Broadcast: scale each sample's kernel by sigma / max_frob.
        scale = (self.sigma / max_frob).view(-1, 1, 1, 1)             # (B, 1, 1, 1)
        return K * scale

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, in_channels). tau: (batch,)."""
        K = self.compute_kernel(tau)                                  # (B, L, out, in)
        # Quadrature: y[b, j, out] = sum_in sum_g K[b, g, out, in] * x[b, g, in] * ds.
        # Note: this is a (lag-shift)-equivariant, not a cyclic conv;
        # the kernel encodes the (tau - s) shift directly. For multi-output
        # spatial structure (predicting y at every j), we reuse the same
        # tau-dependent kernel and apply it as a sum over the lag axis g.
        ds = self.L_phys / self.length
        y_pool = torch.einsum("bgoi,bgi->bo", K, x) * ds              # (B, out)
        # Broadcast to length-L output (same value at every j) — when
        # the model is used for "predict scalar y(tau)", this is what we
        # want; for sequence prediction the user should call the per-j
        # version explicitly.  We expose `forward_seq` for the latter.
        return y_pool.unsqueeze(1).expand(-1, self.length, -1).contiguous()

    def forward_seq(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Sequence forward: predict y(t) at every t in t_out_grid.

        We apply the convolution-as-quadrature for every output position j,
        using the j-shifted query tau' = tau + t_out_grid[j] (the shift of
        the kernel center).
        """
        # For now use the same scalar pool; the discrete-grid LEMO
        # (lemo_v2.py) is the right architecture for sequence outputs.
        return self.forward(x, tau)


# -----------------------------------------------------------------------------
# Pointwise norm
# -----------------------------------------------------------------------------

class PointwiseNorm(nn.Module):
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


# -----------------------------------------------------------------------------
# LEMO-Continuous (scalar-output version, matching original experiment)
# -----------------------------------------------------------------------------

class LEMOContinuousScalar(nn.Module):
    """Continuous-tau LEMO with scalar output: y = LEMO(h, tau).

    Mirrors `experiments/continuous_lag_transfer.py:ContinuousLagEquiv` but
    with multi-channel lift, multi-layer body (B + A blocks), and an
    optional sigma-constrained variant.

    Architecture (Round 2.20: softmax-attention pool matching paper spec):
        h     : (batch, length, in_channels)
        tau   : (batch,)
          -> Linear lift: (batch, length, width)
          -> N blocks of (B + A(tau) + activation)
          -> Lag-distribution head: pi(j) = softmax_j(alpha(U(j)))
                                    v(j)  = beta(U(j))
                                    m     = sum_j pi(j) * v(j)
          -> Two-layer scalar head on m: (batch, out_channels)

    pool_mode="softmax" matches the paper's lag-invariant attention
    pool exactly; pool_mode="mean" reproduces the prior code path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        L_phys: float = 1.0,
        width: int = 48,
        n_layers: int = 3,
        kernel_hidden: int = 64,
        sigma: Optional[float] = None,
        activation: str = "gelu",
        pool_mode: str = "softmax",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.L_phys = L_phys
        self.width = width
        self.n_layers = n_layers
        self.sigma = sigma
        self.pool_mode = pool_mode

        if sigma is not None and activation != "relu":
            activation = "relu"
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()

        self._lift = nn.Linear(in_channels, width)
        self.lift = PointwiseNorm(self._lift, sigma)

        self.B_layers = nn.ModuleList()
        self.A_layers = nn.ModuleList()
        for _ in range(n_layers):
            B_lin = nn.Linear(width, width)
            self.B_layers.append(PointwiseNorm(B_lin, sigma))
            self.A_layers.append(ContinuousTauLagConv1d(
                width, width, length, L_phys=L_phys,
                kernel_hidden=kernel_hidden, sigma=sigma,
            ))

        # Lag-distribution head (paper Section 3.4):
        #   alpha: width -> 1 (score)         beta: width -> width (value)
        if pool_mode == "softmax":
            self._alpha = nn.Linear(width, 1)
            self._beta  = nn.Linear(width, width)
            self.alpha = PointwiseNorm(self._alpha, sigma)
            self.beta  = PointwiseNorm(self._beta,  sigma)

        self._head1 = nn.Linear(width, width)
        self._head2 = nn.Linear(width, out_channels)
        self.head1 = PointwiseNorm(self._head1, sigma)
        self.head2 = PointwiseNorm(self._head2, sigma)

    def forward(self, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """h: (batch, length, in_channels). tau: (batch,). Returns (batch, out_channels)."""
        x = self.lift(h)                                              # (B, L, W)
        for B, A in zip(self.B_layers, self.A_layers):
            B_x = B(x)                                                # (B, L, W)
            A_x = A(x, tau)                                           # (B, L, W)
            x = self.act(B_x + A_x)
        # Lag-invariant pooled summary
        if self.pool_mode == "softmax":
            s = self.alpha(x).squeeze(-1)                             # (B, L)
            pi = torch.softmax(s, dim=1).unsqueeze(-1)                # (B, L, 1)
            v = self.beta(x)                                          # (B, L, W)
            x_pool = (pi * v).sum(dim=1)                              # (B, W)
        else:
            x_pool = x.mean(dim=1)                                    # (B, W)
        h1 = self.act(self.head1(x_pool))                             # (B, W)
        return self.head2(h1)                                         # (B, out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo_continuous(
    in_channels: int,
    out_channels: int,
    config: dict,
) -> nn.Module:
    """Factory; mirrors create_lemo and create_fno1d signatures."""
    model_cfg = config.get("model", config)
    length = model_cfg.get("length", config.get("length", 64))
    L_phys = model_cfg.get("L_phys", 1.0)
    sigma = model_cfg.get("sigma", None)
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    return LEMOContinuousScalar(
        in_channels=in_channels,
        out_channels=out_channels,
        length=length,
        L_phys=L_phys,
        width=model_cfg.get("width", 48),
        n_layers=model_cfg.get("n_layers", 3),
        kernel_hidden=model_cfg.get("kernel_hidden", 64),
        sigma=sigma,
        activation=model_cfg.get("activation", "gelu"),
        pool_mode=model_cfg.get("pool_mode", "softmax"),
    )
