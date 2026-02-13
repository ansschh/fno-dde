"""
PyTorch Dataset for DDE Operator Learning

Provides the DDEDataset class that loads HDF5 data and prepares
inputs/outputs in the combined-grid format suitable for FNO.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Dict


class DDEDataset(Dataset):
    """
    Dataset for DDE operator learning.
    
    Input encoding (combined grid on [-tau_max, T]):
    - Channel 0: history signal (phi(t) for t <= 0, else 0)
    - Channel 1: mask (1 for t <= 0, else 0)
    - Channel 2: normalized time coordinate
    - Channels 3+: parameters (broadcast)
    
    Output:
    - Full solution on [-tau_max, T] (history + future)
    - Loss mask indicating which points to include in loss (typically t >= 0)
    """
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        return_loss_mask: bool = True,
    ):
        """
        Args:
            data_path: Path to HDF5 file
            normalize: Whether to normalize inputs/outputs
            return_loss_mask: Whether to return mask for loss computation
        """
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.return_loss_mask = return_loss_mask
        
        # Load data
        with h5py.File(self.data_path, 'r') as f:
            self.history = f['history'][:]  # (N, n_hist, state_dim)
            self.params = f['params'][:]  # (N, n_params)
            self.solution = f['solution'][:]  # (N, n_future, state_dim)
            self.t_hist = f['t_hist'][:]  # (n_hist,)
            self.t_future = f['t_future'][:]  # (n_future,)
            
            self.family = f.attrs['family']
            self.state_dim = f.attrs['state_dim']
            self.tau_max = f.attrs['tau_max']
            self.T = f.attrs['T']
        
        self.n_samples = len(self.history)
        self.n_hist = len(self.t_hist)
        self.n_future = len(self.t_future)
        self.n_total = self.n_hist + self.n_future
        self.n_params = self.params.shape[1]
        
        # Create combined time grid
        self.t_combined = np.concatenate([self.t_hist, self.t_future])
        
        # Compute normalization statistics if needed
        if self.normalize:
            self._compute_stats()
    
    def _compute_stats(self):
        """Compute mean/std for normalization."""
        # Solution statistics (combine history and future)
        all_values = np.concatenate([
            self.history.reshape(-1, self.state_dim),
            self.solution.reshape(-1, self.state_dim)
        ], axis=0)
        
        self.sol_mean = all_values.mean(axis=0)
        self.sol_std = all_values.std(axis=0) + 1e-8
        
        # Parameter statistics
        self.param_mean = self.params.mean(axis=0)
        self.param_std = self.params.std(axis=0) + 1e-8
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
            - 'input': (n_total, n_channels) input tensor
            - 'target': (n_total, state_dim) target solution
            - 'loss_mask': (n_total,) mask for loss (1 for t >= 0)
            - 'params': (n_params,) raw parameters
        """
        history = self.history[idx]  # (n_hist, state_dim)
        params = self.params[idx]  # (n_params,)
        solution = self.solution[idx]  # (n_future, state_dim)
        
        # Normalize if needed
        if self.normalize:
            history = (history - self.sol_mean) / self.sol_std
            solution = (solution - self.sol_mean) / self.sol_std
            params_norm = (params - self.param_mean) / self.param_std
        else:
            params_norm = params
        
        # Build combined grid representation
        # Channel 0: history signal (history values, then zeros)
        hist_signal = np.zeros((self.n_total, self.state_dim))
        hist_signal[:self.n_hist] = history
        
        # Channel 1: mask (1 for history region)
        mask = np.zeros((self.n_total, 1))
        mask[:self.n_hist] = 1.0
        
        # Channel 2: normalized time
        t_norm = (self.t_combined - self.t_combined.min()) / (
            self.t_combined.max() - self.t_combined.min()
        )
        t_channel = t_norm.reshape(-1, 1)
        
        # Channels 3+: parameters (broadcast)
        param_channel = np.tile(params_norm, (self.n_total, 1))
        
        # Combine input channels
        input_tensor = np.concatenate([
            hist_signal, mask, t_channel, param_channel
        ], axis=1)
        
        # Build target (full solution on combined grid)
        target = np.zeros((self.n_total, self.state_dim))
        target[:self.n_hist] = history  # Include history in target
        target[self.n_hist:] = solution
        
        # Loss mask (1 for t >= 0, which is the future region)
        loss_mask = np.zeros(self.n_total)
        loss_mask[self.n_hist:] = 1.0
        
        result = {
            'input': torch.from_numpy(input_tensor).float(),
            'target': torch.from_numpy(target).float(),
            'loss_mask': torch.from_numpy(loss_mask).float(),
            'params': torch.from_numpy(params).float(),
            't': torch.from_numpy(self.t_combined).float(),
        }
        
        return result
    
    def get_input_channels(self) -> int:
        """Return number of input channels."""
        return self.state_dim + 1 + 1 + self.n_params  # history + mask + time + params
    
    def get_output_channels(self) -> int:
        """Return number of output channels."""
        return self.state_dim
    
    def denormalize_solution(self, solution: torch.Tensor) -> torch.Tensor:
        """Denormalize a solution tensor."""
        if not self.normalize:
            return solution
        
        mean = torch.from_numpy(self.sol_mean).to(solution.device)
        std = torch.from_numpy(self.sol_std).to(solution.device)
        
        return solution * std + mean


def create_dataloaders(
    data_dir: str,
    family: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/val/test dataloaders for a DDE family.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    train_dataset = DDEDataset(data_dir / f"{family}_train.h5")
    val_dataset = DDEDataset(data_dir / f"{family}_val.h5")
    test_dataset = DDEDataset(data_dir / f"{family}_test.h5")
    
    # Share normalization stats from training set
    val_dataset.sol_mean = train_dataset.sol_mean
    val_dataset.sol_std = train_dataset.sol_std
    val_dataset.param_mean = train_dataset.param_mean
    val_dataset.param_std = train_dataset.param_std
    
    test_dataset.sol_mean = train_dataset.sol_mean
    test_dataset.sol_std = train_dataset.sol_std
    test_dataset.param_mean = train_dataset.param_mean
    test_dataset.param_std = train_dataset.param_std
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
