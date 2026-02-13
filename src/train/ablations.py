"""
Ablation Study Utilities

Implements ablation-specific functionality:
- Input encoding variants
- Loss weighting schemes
- Capacity sweep runner
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
import yaml


# =============================================================================
# Input Encoding Ablations
# =============================================================================

class InputEncoder:
    """
    Configurable input encoder for ablation studies.
    
    Supports different channel combinations:
    - history: φ(t) for t≤0, 0 for t>0
    - mask: 1 for t≤0, 0 for t>0
    - time: normalized time coordinate
    - params: parameters broadcast across time
    """
    
    def __init__(self, channels: List[str], state_dim: int, n_params: int):
        """
        Args:
            channels: List of channel types to include
            state_dim: Dimension of state
            n_params: Number of parameters
        """
        self.channels = channels
        self.state_dim = state_dim
        self.n_params = n_params
        
        # Compute total channels
        self.n_channels = 0
        for ch in channels:
            if ch == "history":
                self.n_channels += state_dim
            elif ch == "mask":
                self.n_channels += 1
            elif ch == "time":
                self.n_channels += 1
            elif ch == "params":
                self.n_channels += n_params
    
    def encode(
        self,
        phi: torch.Tensor,      # (batch, n_hist, state_dim)
        t: torch.Tensor,        # (batch, n_total)
        params: torch.Tensor,   # (batch, n_params)
        n_hist: int,
    ) -> torch.Tensor:
        """
        Encode inputs according to channel configuration.
        
        Returns:
            Encoded input (batch, n_total, n_channels)
        """
        batch_size = phi.shape[0]
        n_total = t.shape[1]
        
        channels_list = []
        
        for ch in self.channels:
            if ch == "history":
                # History signal: φ for t≤0, 0 for t>0
                hist_signal = torch.zeros(batch_size, n_total, self.state_dim, device=phi.device)
                hist_signal[:, :n_hist] = phi
                channels_list.append(hist_signal)
                
            elif ch == "mask":
                # Mask: 1 for t≤0, 0 for t>0
                mask = torch.zeros(batch_size, n_total, 1, device=phi.device)
                mask[:, :n_hist] = 1.0
                channels_list.append(mask)
                
            elif ch == "time":
                # Normalized time
                t_norm = (t - t.min(dim=1, keepdim=True)[0]) / (
                    t.max(dim=1, keepdim=True)[0] - t.min(dim=1, keepdim=True)[0] + 1e-8
                )
                channels_list.append(t_norm.unsqueeze(-1))
                
            elif ch == "params":
                # Parameters broadcast
                params_broadcast = params.unsqueeze(1).expand(-1, n_total, -1)
                channels_list.append(params_broadcast)
        
        return torch.cat(channels_list, dim=-1)


def load_encoding_config(config_path: Path, variant: str) -> Dict:
    """Load encoding configuration for a specific variant."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if variant not in config:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(config.keys())}")
    
    return config[variant]


# =============================================================================
# Loss Weighting Ablations
# =============================================================================

class WeightedMSELoss(nn.Module):
    """
    MSE loss with configurable temporal weighting.
    """
    
    def __init__(
        self,
        weight_type: str = "uniform",
        start_weight: float = 1.0,
        end_weight: float = 1.0,
        n_segments: int = 1,
        segment_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.weight_type = weight_type
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.n_segments = n_segments
        self.segment_weights = segment_weights or [1.0]
    
    def forward(
        self,
        pred: torch.Tensor,    # (batch, length, channels)
        target: torch.Tensor,  # (batch, length, channels)
        mask: torch.Tensor,    # (batch, length)
    ) -> torch.Tensor:
        """Compute weighted MSE loss."""
        # Expand mask
        mask = mask.unsqueeze(-1)
        
        # Get temporal positions (normalized)
        n_future = int(mask.sum(dim=1).max().item())
        
        # Compute weights based on type
        if self.weight_type == "uniform":
            weights = torch.ones_like(mask)
            
        elif self.weight_type == "linear":
            # Linear from start_weight to end_weight
            t_rel = torch.cumsum(mask, dim=1) / (n_future + 1e-8)
            weights = self.start_weight + (self.end_weight - self.start_weight) * t_rel
            
        elif self.weight_type == "exponential":
            t_rel = torch.cumsum(mask, dim=1) / (n_future + 1e-8)
            weights = self.start_weight * torch.exp(t_rel)
            
        elif self.weight_type == "segment":
            # Segment-based weights
            weights = torch.ones_like(mask)
            indices = torch.cumsum(mask, dim=1).long()
            segment_size = n_future // self.n_segments
            
            for i, w in enumerate(self.segment_weights):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < self.n_segments - 1 else n_future
                seg_mask = (indices >= start_idx) & (indices < end_idx)
                weights = torch.where(seg_mask, torch.tensor(w, device=weights.device), weights)
        else:
            weights = torch.ones_like(mask)
        
        # Apply mask
        weights = weights * mask
        
        # Compute weighted MSE
        sq_error = (pred - target) ** 2
        weighted_error = sq_error * weights
        
        loss = weighted_error.sum() / (weights.sum() * pred.shape[-1] + 1e-8)
        
        return loss


class TwoStageLoss:
    """
    Two-stage loss: uniform first, then late-focus.
    """
    
    def __init__(
        self,
        stage1_epochs: int = 50,
        late_focus_start: float = 0.5,
    ):
        self.stage1_epochs = stage1_epochs
        self.late_focus_start = late_focus_start
        self.current_epoch = 0
        
        self.stage1_loss = WeightedMSELoss(weight_type="uniform")
        self.stage2_loss = WeightedMSELoss(
            weight_type="linear",
            start_weight=0.5,
            end_weight=2.0,
        )
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.current_epoch < self.stage1_epochs:
            return self.stage1_loss(pred, target, mask)
        else:
            return self.stage2_loss(pred, target, mask)


def create_loss_function(config: Dict) -> nn.Module:
    """Create loss function from config."""
    loss_type = config.get("type", "uniform")
    
    if loss_type == "uniform":
        return WeightedMSELoss(weight_type="uniform")
    elif loss_type == "linear":
        return WeightedMSELoss(
            weight_type="linear",
            start_weight=config.get("start_weight", 0.5),
            end_weight=config.get("end_weight", 1.5),
        )
    elif loss_type == "exponential":
        return WeightedMSELoss(
            weight_type="exponential",
            start_weight=config.get("base_weight", 0.5),
        )
    elif loss_type == "segment":
        return WeightedMSELoss(
            weight_type="segment",
            n_segments=config.get("n_segments", 4),
            segment_weights=config.get("weights", [0.5, 0.75, 1.0, 1.5]),
        )
    elif loss_type == "two_stage":
        return TwoStageLoss(
            stage1_epochs=config.get("stage1_epochs", 50),
            late_focus_start=config.get("late_focus_start", 0.5),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =============================================================================
# Capacity Sweep Runner
# =============================================================================

def load_capacity_configs(config_path: Path) -> Dict:
    """Load capacity ablation configurations."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    base = config.get("base", {})
    variants = config.get("variants", {})
    
    # Merge base into each variant
    full_configs = {}
    for name, variant in variants.items():
        full_config = {**base, **variant}
        full_configs[name] = full_config
    
    return full_configs


def estimate_parameters(modes: int, width: int, n_layers: int, in_channels: int, out_channels: int) -> int:
    """Estimate number of parameters for FNO configuration."""
    # Lifting: in_channels -> width
    lift_params = in_channels * width
    
    # Spectral conv per layer: width * width * modes (complex)
    spectral_params = n_layers * width * width * modes * 2
    
    # Pointwise conv per layer: width * width
    pointwise_params = n_layers * width * width
    
    # Projection: width -> width -> out_channels
    proj_params = width * width + width * out_channels
    
    return lift_params + spectral_params + pointwise_params + proj_params


def create_sweep_table(config_path: Path, in_channels: int = 6, out_channels: int = 1) -> str:
    """Create a table of sweep configurations with estimated parameters."""
    configs = load_capacity_configs(config_path)
    
    lines = [
        "| Name | Modes | Width | Layers | Est. Params |",
        "|------|-------|-------|--------|-------------|",
    ]
    
    for name, cfg in configs.items():
        params = estimate_parameters(
            cfg["modes"], cfg["width"], cfg["n_layers"],
            in_channels, out_channels
        )
        lines.append(
            f"| {name} | {cfg['modes']} | {cfg['width']} | {cfg['n_layers']} | {params:,} |"
        )
    
    return "\n".join(lines)
