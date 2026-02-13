"""
Fourier Neural Operator for 2D spatial data.

Implements FNO2D for learning PDE solution operators on 2D grids.
Architecture based on Li et al. (2020) extended to 2D spectral convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer using rfft2."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            (batch, out_channels, height, width)
        """
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Positive frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Negative frequencies (conjugate symmetric part)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock2d(nn.Module):
    """Single FNO2d block: spectral conv + pointwise conv + activation."""

    def __init__(self, width: int, modes1: int, modes2: int,
                 activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(width, width, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        return self.dropout(self.activation(x1 + x2))


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.

    Input:  (batch, height, width, in_channels)
    Output: (batch, height, width, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
        padding: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding

        self.lift = nn.Linear(in_channels, width)

        self.blocks = nn.ModuleList([
            FNOBlock2d(width, modes1, modes2, activation, dropout)
            for _ in range(n_layers)
        ])

        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, out_channels)
        self.final_activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, height, width, in_channels)
        x = self.lift(x)  # (batch, H, W, width)

        # -> (batch, width, H, W) for conv
        x = x.permute(0, 3, 1, 2)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for block in self.blocks:
            x = block(x)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        # -> (batch, H, W, width)
        x = x.permute(0, 2, 3, 1)

        x = self.proj1(x)
        x = self.final_activation(x)
        x = self.proj2(x)

        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_fno2d(
    in_channels: int,
    out_channels: int,
    config: dict,
) -> FNO2d:
    """Factory function for FNO2d from config dict."""
    return FNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes1=config.get('modes1', config.get('modes', 12)),
        modes2=config.get('modes2', config.get('modes', 12)),
        width=config.get('width', 32),
        n_layers=config.get('n_layers', 4),
        activation=config.get('activation', 'gelu'),
        padding=config.get('padding', 0),
        dropout=config.get('dropout', 0.0),
    )
