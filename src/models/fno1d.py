"""
Fourier Neural Operator for 1D (time-series) data.

Implements FNO1D for learning DDE solution operators.
Architecture based on Li et al. (2020) "Fourier Neural Operator for Parametric PDEs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer.
    
    Performs convolution in Fourier space by multiplying with learnable weights
    for a fixed number of Fourier modes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to keep
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for batched tensors."""
        # input: (batch, in_channels, modes)
        # weights: (in_channels, out_channels, modes)
        # output: (batch, out_channels, modes)
        return torch.einsum("bim,iom->bom", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Output tensor of shape (batch, out_channels, length)
        """
        batch_size = x.shape[0]
        
        # Compute FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBlock1d(nn.Module):
    """
    Single FNO block consisting of:
    - Spectral convolution (global, in Fourier space)
    - Pointwise convolution (local, 1x1 conv)
    - Nonlinearity
    - Optional dropout for regularization
    """
    
    def __init__(
        self,
        width: int,
        modes: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.modes = modes
        
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.pointwise_conv = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, width, length)
            
        Returns:
            Output tensor of shape (batch, width, length)
        """
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        x = x1 + x2
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator.
    
    Architecture:
    1. Lifting layer: project input channels to hidden dimension
    2. N FNO blocks with spectral convolutions
    3. Projection layer: project back to output channels
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        activation: str = "gelu",
        padding: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to keep
            width: Hidden channel dimension
            n_layers: Number of FNO blocks
            activation: Activation function ('gelu', 'relu', 'tanh')
            padding: Padding to add for non-periodic boundaries
            layer_norm: Whether to use layer normalization after each block
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        self.dropout_rate = dropout
        self.use_layer_norm = layer_norm
        
        # Lifting layer
        self.lift = nn.Linear(in_channels, width)
        
        # FNO blocks with dropout
        self.blocks = nn.ModuleList([
            FNOBlock1d(width, modes, activation, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Optional layer normalization after each block
        if layer_norm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(width) for _ in range(n_layers)
            ])
        
        # Projection layers
        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, out_channels)
        
        if activation == "gelu":
            self.final_activation = nn.GELU()
        elif activation == "relu":
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, length, in_channels)
            
        Returns:
            Output tensor of shape (batch, length, out_channels)
        """
        # x: (batch, length, in_channels)
        
        # Lifting
        x = self.lift(x)  # (batch, length, width)
        
        # Transpose for conv: (batch, width, length)
        x = x.permute(0, 2, 1)
        
        # Pad if needed (for non-periodic boundaries)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
        
        # FNO blocks with optional layer norm
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.use_layer_norm:
                # LayerNorm expects (batch, length, channels), so transpose
                x = x.permute(0, 2, 1)
                x = self.norms[i](x)
                x = x.permute(0, 2, 1)
        
        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding]
        
        # Transpose back: (batch, length, width)
        x = x.permute(0, 2, 1)
        
        # Projection
        x = self.proj1(x)
        x = self.final_activation(x)
        x = self.proj2(x)  # (batch, length, out_channels)
        
        return x


class FNO1dResidual(nn.Module):
    """
    FNO1d with residual connections between blocks.
    
    Can improve training stability for deeper networks.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        activation: str = "gelu",
        padding: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        self.dropout_rate = dropout
        
        # Lifting layer
        self.lift = nn.Linear(in_channels, width)
        
        # FNO blocks with separate norm layers and dropout
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(FNOBlock1d(width, modes, activation, dropout=dropout))
            self.norms.append(nn.LayerNorm(width))
        
        # Projection layers
        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, out_channels)
        
        if activation == "gelu":
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, length, in_channels)
            
        Returns:
            Output tensor of shape (batch, length, out_channels)
        """
        # Lifting
        x = self.lift(x)  # (batch, length, width)
        
        # FNO blocks with residual connections
        for block, norm in zip(self.blocks, self.norms):
            # Transpose for conv: (batch, width, length)
            x_conv = x.permute(0, 2, 1)
            
            if self.padding > 0:
                x_conv = F.pad(x_conv, [0, self.padding])
            
            x_conv = block(x_conv)
            
            if self.padding > 0:
                x_conv = x_conv[..., :-self.padding]
            
            # Transpose back and add residual
            x_conv = x_conv.permute(0, 2, 1)
            x = norm(x + x_conv)
        
        # Projection
        x = self.proj1(x)
        x = self.final_activation(x)
        x = self.proj2(x)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_fno1d(
    in_channels: int,
    out_channels: int,
    config: dict,
    use_residual: bool = False,
) -> nn.Module:
    """
    Factory function to create FNO1d model from config.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        config: Dictionary with model hyperparameters
        use_residual: Whether to use residual connections
        
    Returns:
        FNO1d or FNO1dResidual model
    """
    model_class = FNO1dResidual if use_residual else FNO1d
    
    return model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=config.get('modes', 16),
        width=config.get('width', 64),
        n_layers=config.get('n_layers', 4),
        activation=config.get('activation', 'gelu'),
        padding=config.get('padding', 0),
        dropout=config.get('dropout', 0.0),
    )
