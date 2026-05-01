"""
Faithful PyTorch implementation of the MFNO architecture described in
Lee, Kim, and Park, "Fourier Neural Operators for Non-Markovian Processes".

The paper reports the scalar SDE MFNO as:
  input: paired path (xi, B), so input_channels = 2
  output: solution path X, so output_channels = 1
  mirror padding to double the temporal domain
  lifting to 32 latent channels
  5 Fourier layers
  Fourier cutoff / modes = 64
  projection back to output dimension

This file also contains ZFNO and vanilla FNO variants using the same FNO core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ActivationName = Literal["gelu", "relu", "tanh", "silu", "identity"]
PaddingName = Literal["mirror", "zero", "none"]


def get_activation(name: ActivationName) -> Callable[[Tensor], Tensor]:
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name == "silu":
        return F.silu
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unknown activation: {name}")


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class SpectralConv1d(nn.Module):
    """1D Fourier integral layer with learned complex weights on low modes.

    Input shape:  (batch, in_channels, n_grid)
    Output shape: (batch, out_channels, n_grid)

    We store real and imaginary parts separately so PyTorch's parameter count
    matches the real-valued parameter count usually reported in papers.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            raise ValueError("modes must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (batch, channels, grid), got {x.shape}")
        batch_size, _, n_grid = x.shape

        x_ft = torch.fft.rfft(x, dim=-1)
        n_freq = x_ft.size(-1)
        used_modes = min(self.modes, n_freq)

        out_ft = x_ft.new_zeros(batch_size, self.out_channels, n_freq)
        weight = torch.complex(
            self.weight_real[..., :used_modes],
            self.weight_imag[..., :used_modes],
        )
        out_ft[..., :used_modes] = torch.einsum(
            "bim,iom->bom", x_ft[..., :used_modes], weight
        )
        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)


class FourierLayer1d(nn.Module):
    """One layer L_l(v)=sigma(W_l v + F^{-1}(P_l F(v)))."""

    def __init__(
        self,
        channels: int,
        modes: int,
        activation: ActivationName = "gelu",
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor, *, apply_activation: bool = True) -> Tensor:
        x = self.spectral(x) + self.pointwise(x)
        if apply_activation:
            x = self.activation(x)
        return x


class ProjectionHead(nn.Module):
    """Pointwise projection from latent channels to output channels.

    The first projection bias is disabled so the scalar SDE model has exactly
    664,961 trainable real parameters, matching Table 3 of the paper:

      5 * [2 * 32 * 32 * 64 + (32 * 32 + 32)]
      + (2 * 32 + 32)
      + (32 * 128 + 128 * 1 + 1)
      = 664,961.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        activation: ActivationName = "gelu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=True)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, channels, grid) -> (batch, grid, channels)
        x = x.transpose(1, 2)
        x = self.activation(self.fc1(x))
        return self.fc2(x)


class FNO1dCore(nn.Module):
    """Discretized 1D FNO core: Q o L_L o ... o L_1 o R.

    Input shape:  (batch, n_grid, input_channels)
    Output shape: (batch, n_grid, output_channels)
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        latent_channels: int = 32,
        modes: int = 64,
        num_layers: int = 5,
        projection_channels: int = 128,
        activation: ActivationName = "gelu",
        activate_last: bool = True,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        if latent_channels <= 0 or num_layers <= 0:
            raise ValueError("latent_channels and num_layers must be positive")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_channels = latent_channels
        self.modes = modes
        self.num_layers = num_layers
        self.activate_last = activate_last
        self.activation = get_activation(activation)

        self.lifting = nn.Linear(input_channels, latent_channels, bias=True)
        self.layers = nn.ModuleList(
            [FourierLayer1d(latent_channels, modes, activation=activation) for _ in range(num_layers)]
        )
        self.projection = ProjectionHead(
            latent_channels,
            output_channels,
            hidden_channels=projection_channels,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (batch, grid, channels), got {x.shape}")
        if x.size(-1) != self.input_channels:
            raise ValueError(
                f"Expected last dim {self.input_channels}, got {x.size(-1)}. "
                "For scalar SDEs use channels [xi_constant_path, brownian_path]."
            )

        x = self.activation(self.lifting(x))
        x = x.transpose(1, 2)  # (batch, channels, grid)
        for layer_id, layer in enumerate(self.layers):
            apply_activation = self.activate_last or (layer_id != len(self.layers) - 1)
            x = layer(x, apply_activation=apply_activation)
        return self.projection(x)


class PaddedFNO1d(nn.Module):
    """FNO with optional mirror or zero padding, followed by truncation.

    padding='mirror' implements MFNO:
        x_pad = concat(x, reverse(x))
        y = FNO(x_pad)
        return y restricted to the original grid

    padding='zero' implements ZFNO with the same doubled domain size.
    padding='none' implements vanilla FNO.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        latent_channels: int = 32,
        modes: int = 64,
        num_layers: int = 5,
        projection_channels: int = 128,
        activation: ActivationName = "gelu",
        padding: PaddingName = "mirror",
        activate_last: bool = True,
    ) -> None:
        super().__init__()
        if padding not in {"mirror", "zero", "none"}:
            raise ValueError("padding must be one of: 'mirror', 'zero', 'none'")
        self.padding = padding
        self.core = FNO1dCore(
            input_channels=input_channels,
            output_channels=output_channels,
            latent_channels=latent_channels,
            modes=modes,
            num_layers=num_layers,
            projection_channels=projection_channels,
            activation=activation,
            activate_last=activate_last,
        )

    def pad(self, x: Tensor) -> Tensor:
        if self.padding == "none":
            return x
        if self.padding == "mirror":
            return torch.cat([x, torch.flip(x, dims=[1])], dim=1)
        zeros = torch.zeros_like(x)
        return torch.cat([x, zeros], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        original_n = x.size(1)
        x_pad = self.pad(x)
        y_pad = self.core(x_pad)
        if self.padding == "none":
            return y_pad
        return y_pad[:, :original_n, :]


class MFNO1d(PaddedFNO1d):
    def __init__(self, input_channels: int, output_channels: int, **kwargs) -> None:
        super().__init__(input_channels, output_channels, padding="mirror", **kwargs)


class ZFNO1d(PaddedFNO1d):
    def __init__(self, input_channels: int, output_channels: int, **kwargs) -> None:
        super().__init__(input_channels, output_channels, padding="zero", **kwargs)


class VanillaFNO1d(PaddedFNO1d):
    def __init__(self, input_channels: int, output_channels: int, **kwargs) -> None:
        super().__init__(input_channels, output_channels, padding="none", **kwargs)


@dataclass(frozen=True)
class PaperMFNOConfig:
    input_channels: int = 2        # scalar initial condition as constant path + scalar Brownian path
    output_channels: int = 1       # scalar solution path
    latent_channels: int = 32      # 32-channel latent space
    modes: int = 64                # Fourier cutoff / width W
    num_layers: int = 5            # five Fourier layers
    projection_channels: int = 128 # projection MLP hidden width, needed to match reported params
    activation: ActivationName = "gelu"
    activate_last: bool = True


def build_paper_mfno(
    input_channels: int = 2,
    output_channels: int = 1,
    activation: ActivationName = "gelu",
    activate_last: bool = True,
) -> MFNO1d:
    return MFNO1d(
        input_channels=input_channels,
        output_channels=output_channels,
        latent_channels=32,
        modes=64,
        num_layers=5,
        projection_channels=128,
        activation=activation,
        activate_last=activate_last,
    )


def make_sde_input(xi: Tensor, brownian_path: Tensor) -> Tensor:
    """Create the paper's input pair ((xi, B), X) for scalar SDE tasks.

    xi:            (batch,) or (batch, 1)
    brownian_path: (batch, n_grid) or (batch, n_grid, 1)

    Returns x with shape (batch, n_grid, 2):
      channel 0 = xi repeated as a constant path
      channel 1 = Brownian path values
    """
    if brownian_path.ndim == 3:
        if brownian_path.size(-1) != 1:
            raise ValueError("brownian_path with 3 dims must have final dim 1")
        brownian_path = brownian_path[..., 0]
    if brownian_path.ndim != 2:
        raise ValueError("brownian_path must have shape (batch, n_grid) or (batch, n_grid, 1)")

    xi = xi.reshape(-1, 1).to(dtype=brownian_path.dtype, device=brownian_path.device)
    xi_path = xi.expand(-1, brownian_path.size(1))
    return torch.stack([xi_path, brownian_path], dim=-1)


def relative_l2_error(pred: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    """Mean relative l2 error over batch for path tensors."""
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    return (torch.linalg.norm(pred - target, dim=1) / (torch.linalg.norm(target, dim=1) + eps)).mean()


def relative_linf_error(pred: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    """Mean relative l-infinity error over batch for path tensors."""
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    return ((pred - target).abs().amax(dim=1) / (target.abs().amax(dim=1) + eps)).mean()


if __name__ == "__main__":
    torch.manual_seed(0)

    model = build_paper_mfno(input_channels=2, output_channels=1)
    print("MFNO parameter count:", count_parameters(model))

    batch, n_grid = 4, 128
    xi = torch.rand(batch) * 20.0
    brownian = torch.randn(batch, n_grid).cumsum(dim=1) * (0.1 ** 0.5)
    x = make_sde_input(xi, brownian)
    y = model(x)
    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
