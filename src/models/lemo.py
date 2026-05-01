"""
Lag-Equivariant Memory Operator (LEMO) for 1D DDE data.

Implements a lag-equivariant cyclic-convolution architecture and its
spectral-norm-constrained subclass LEMO_sigma with certified rollout
stability (see Cor 16 in sections/main_theoretical_results.tex).

Key implementation detail: the lag-convolution kernels are parameterized
directly in the physical domain (or via a small MLP for continuous-tau
evaluation) and normalized via FFT-based operator-norm rescaling. This
is NOT the same as torch.nn.utils.spectral_norm applied to nn.Conv1d,
which bounds the SVD of the weight tensor reshaped as a matrix rather
than the convolution's operator norm. The FFT-based normalization is
what Cor 16's proof requires; the local validation experiment showed a
~36 orders-of-magnitude gap between the two normalizations on an
adversarially expanding target.

Architecture, forward contract matches FNO1d:
    input  (batch, length, in_channels)
    output (batch, length, out_channels)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Lag-convolution primitives
# -----------------------------------------------------------------------------

class ContinuousLagConv1d(nn.Module):
    """Cyclic lag convolution with a kernel parameterized as an MLP of the
    (signed, normalized) lag offset. This supports:

    - Grid training: evaluate the MLP at the L training-grid offsets and
      apply cyclic convolution via FFT.
    - Off-grid inference: evaluate the MLP at arbitrary lag offsets.

    If `sigma` is provided, the kernel is FFT-normalized so the
    convolution's operator norm (in Euclidean L2) satisfies
    ||K*||_op <= sigma, via the Frobenius upper bound
        max_omega ||K_hat(omega)||_F  >=  max_omega ||K_hat(omega)||_2
                                       =  ||K*||_op.
    Using Frobenius is a conservative but valid normalization; the proof
    of Cor 16 only requires an upper bound on the operator norm.
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

    def _offsets(self, length: int, device: torch.device) -> torch.Tensor:
        """Signed, normalized lag offsets in [-0.5, 0.5)."""
        idx = torch.arange(length, device=device, dtype=torch.float32)
        signed = torch.where(idx > length / 2, idx - length, idx)
        return (signed / length).unsqueeze(-1)  # (length, 1)

    def compute_kernel(self, length: Optional[int] = None) -> torch.Tensor:
        """Return the normalized kernel of shape (out_channels, in_channels, length)."""
        length = length or self.length
        device = next(self.parameters()).device
        offsets = self._offsets(length, device)
        K_flat = self.kernel_net(offsets)              # (length, out*in)
        K = K_flat.view(length, self.out_channels, self.in_channels).permute(1, 2, 0)
        if self.sigma is None:
            return K
        K_hat = torch.fft.rfft(K, dim=-1)               # (out, in, L//2+1) complex
        frob = K_hat.abs().pow(2).sum(dim=(0, 1)).sqrt()  # (L//2+1,)
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
    """Enforce operator norm <= sigma on a pointwise linear layer.
    The operator norm of a pointwise linear map (Linear along channels,
    shared across length) equals the spectral norm of its weight matrix.
    We use the Frobenius upper bound, same as above.
    """
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
# LEMO model
# -----------------------------------------------------------------------------

class LEMO(nn.Module):
    """Lag-Equivariant Memory Operator.

    Architecture:
        x : (batch, length, in_channels)
          -> Linear lift to `width` channels
          -> N blocks of (ContinuousLagConv1d + ReLU)
          -> Linear projection to `out_channels`
        y : (batch, length, out_channels)

    If `sigma` is not None, the architecture is the LEMO_sigma subclass
    (Cor 16 in main_theoretical_results.tex): every lag-conv kernel is
    FFT-normalized so its operator norm is bounded by sigma, and the
    lift/project layers have weight norm bounded by sigma. With
    sigma < 1 and ReLU nonlinearities (exactly 1-Lipschitz), the
    history-update map is contractive by construction with rate
    rho = sigma^((L+1)/n) < 1.

    Activation is fixed to ReLU (exactly 1-Lipschitz) by default;
    GELU's maximum slope (~1.13) violates the 1-Lipschitz requirement
    of Cor 16's proof.
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

        self.conv_blocks = nn.ModuleList([
            ContinuousLagConv1d(width, width, length, kernel_hidden=kernel_hidden, sigma=sigma)
            for _ in range(n_layers)
        ])

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu" and sigma is None:
            self.act = nn.GELU()
        else:
            # GELU is not exactly 1-Lipschitz; if sigma is set, refuse.
            if sigma is not None and activation != "relu":
                raise ValueError(
                    f"LEMO_sigma requires activation='relu' for exact 1-Lipschitz guarantee; "
                    f"got activation={activation!r}"
                )
            self.act = nn.ReLU()

        self._proj_linear = nn.Linear(width, out_channels)
        self.proj = PointwiseNorm(self._proj_linear, sigma)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, in_channels). Returns (batch, length, out_channels)."""
        x = self.lift(x)                      # (batch, length, width)
        x = x.permute(0, 2, 1)                # (batch, width, length)
        for conv in self.conv_blocks:
            x = conv(x)
            x = self.act(x)
            x = self.dropout(x)
        x = x.permute(0, 2, 1)                # (batch, length, width)
        x = self.proj(x)                      # (batch, length, out_channels)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lemo(
    in_channels: int,
    out_channels: int,
    config: dict,
) -> nn.Module:
    """Factory. Reads sigma from config['model']['sigma'] (None => unconstrained LEMO).
    Mirrors the signature of create_fno1d for drop-in use in train_fno.py.
    """
    model_cfg = config.get("model", config)
    length = model_cfg.get("length", config.get("length", 64))
    sigma = model_cfg.get("sigma", None)
    # Convert string 'null' / 'none' to None for YAML convenience
    if isinstance(sigma, str) and sigma.lower() in ("null", "none", ""):
        sigma = None
    return LEMO(
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
