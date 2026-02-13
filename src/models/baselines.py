"""
Baseline Models for DDE Operator Learning

Provides baselines to compare against FNO:
1. Naive baseline (constant continuation)
2. TCN (Temporal Convolutional Network)
3. Simple MLP baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class NaiveBaseline(nn.Module):
    """
    Baseline 0: Predict constant continuation from history.
    
    Output = φ(0) for all t > 0.
    
    This sets a floor and catches trivial datasets.
    """
    
    def __init__(self, state_dim: int = 1):
        super().__init__()
        self.state_dim = state_dim
    
    def forward(self, x: torch.Tensor, n_hist: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, length, in_channels)
            n_hist: Number of history points
            
        Returns:
            Predicted trajectory (batch, length, state_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract final history value (state dimensions only)
        phi_0 = x[:, n_hist - 1, :self.state_dim]  # (batch, state_dim)
        
        # Broadcast to full sequence
        output = phi_0.unsqueeze(1).expand(-1, seq_len, -1)
        
        return output


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connection.
    
    Uses dilated causal convolutions for sequence-to-sequence modeling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, channels, length)
            
        Returns:
            Output (batch, out_channels, length)
        """
        # Causal: remove future padding
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Trim to original length
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.norm2(out)
        
        # Residual
        res = self.residual(x)
        out = self.activation(out + res)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network for sequence-to-sequence modeling.
    
    Baseline 1: Strong non-operator baseline that often competes well.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # TCN blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(
                hidden_channels, hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, length, in_channels)
            
        Returns:
            Output (batch, length, out_channels)
        """
        # Input projection
        x = self.input_proj(x)  # (batch, length, hidden)
        
        # Transpose for conv: (batch, hidden, length)
        x = x.permute(0, 2, 1)
        
        # TCN blocks
        for block in self.blocks:
            x = block(x)
        
        # Transpose back: (batch, length, hidden)
        x = x.permute(0, 2, 1)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline.
    
    Flattens input and uses fully connected layers.
    Only works for fixed sequence lengths.
    """
    
    def __init__(
        self,
        seq_length: int,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.out_channels = out_channels
        
        input_dim = seq_length * in_channels
        output_dim = seq_length * out_channels
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, length, in_channels)
            
        Returns:
            Output (batch, length, out_channels)
        """
        batch_size = x.size(0)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # MLP
        x = self.net(x)
        
        # Reshape
        x = x.view(batch_size, self.seq_length, self.out_channels)
        
        return x


class LinearODEBaseline(nn.Module):
    """
    ODE baseline: Ignore delay terms.
    
    For families where dropping delays gives meaningful comparison.
    Trains a neural ODE-style model.
    """
    
    def __init__(
        self,
        state_dim: int,
        n_params: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # RHS network: f(x, params) -> dx/dt
        self.rhs_net = nn.Sequential(
            nn.Linear(state_dim + n_params, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        n_hist: int,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """
        Euler integration ignoring delays.
        
        Args:
            x: Input (batch, length, in_channels)
            n_hist: Number of history points
            dt: Time step
            
        Returns:
            Output (batch, length, state_dim)
        """
        batch_size, seq_len, in_channels = x.shape
        
        # Extract initial state and params
        x0 = x[:, n_hist - 1, :self.state_dim]  # (batch, state_dim)
        
        # Assume params are last channels, broadcast
        n_params = in_channels - self.state_dim - 2  # state + mask + time
        if n_params > 0:
            params = x[:, 0, -n_params:]  # (batch, n_params)
        else:
            params = torch.zeros(batch_size, 1, device=x.device)
        
        # Integrate
        outputs = []
        state = x0
        
        for t_idx in range(seq_len):
            outputs.append(state)
            
            if t_idx >= n_hist - 1:  # Only integrate in future
                # RHS input
                rhs_input = torch.cat([state, params], dim=-1)
                dx = self.rhs_net(rhs_input)
                state = state + dt * dx
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, state_dim)
        
        return output


def create_baseline(
    baseline_type: str,
    in_channels: int,
    out_channels: int,
    seq_length: int = 256,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        baseline_type: One of "naive", "tcn", "mlp", "ode"
        in_channels: Number of input channels
        out_channels: Number of output channels
        seq_length: Sequence length (for MLP)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Baseline model
    """
    if baseline_type == "naive":
        return NaiveBaseline(state_dim=out_channels)
    elif baseline_type == "tcn":
        return TCN(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=kwargs.get("hidden_channels", 64),
            n_layers=kwargs.get("n_layers", 6),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif baseline_type == "mlp":
        return MLPBaseline(
            seq_length=seq_length,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=kwargs.get("hidden_dim", 256),
            n_layers=kwargs.get("n_layers", 4),
        )
    elif baseline_type == "ode":
        return LinearODEBaseline(
            state_dim=out_channels,
            n_params=in_channels - out_channels - 2,
            hidden_dim=kwargs.get("hidden_dim", 64),
        )
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
