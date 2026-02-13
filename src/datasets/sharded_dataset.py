"""
PyTorch Dataset for Sharded DDE Data

Loads sharded .npz files and provides combined-grid encoding for FNO training.
Supports streaming from multiple shards without loading everything into memory.
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Iterator
import warnings


class ShardedDDEDataset(Dataset):
    """
    Map-style dataset that loads all shards into memory.
    Use for small-to-medium datasets that fit in RAM.
    """
    
    def __init__(
        self,
        data_dir: str,
        family: str,
        split: str = "train",
        normalize: bool = True,
        include_history_in_target: bool = True,
    ):
        """
        Args:
            data_dir: Root data directory
            family: DDE family name
            split: 'train', 'val', or 'test'
            normalize: Whether to normalize inputs/outputs
            include_history_in_target: Include history region in target
        """
        self.data_dir = Path(data_dir)
        self.family = family
        self.split = split
        self.normalize = normalize
        self.include_history_in_target = include_history_in_target
        
        # Load manifest
        manifest_path = self.data_dir / family / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        self.state_dim = self.manifest["state_dim"]
        self.param_names = self.manifest["param_names"]
        self.n_params = len(self.param_names)
        
        # Load all shards
        split_dir = self.data_dir / family / split
        shard_files = sorted(split_dir.glob("shard_*.npz"))
        
        if len(shard_files) == 0:
            raise FileNotFoundError(f"No shards found in {split_dir}")
        
        # Load first shard to get dimensions
        first_shard = np.load(shard_files[0])
        self.t_hist = first_shard["t_hist"]
        self.t_out = first_shard["t_out"]
        self.n_hist = len(self.t_hist)
        self.n_out = len(self.t_out)
        self.n_total = self.n_hist + self.n_out
        
        # Build combined time grid
        self.t_combined = np.concatenate([self.t_hist, self.t_out])
        
        # Load all data
        phi_list = []
        y_list = []
        params_list = []
        lags_list = []
        
        for shard_path in shard_files:
            data = np.load(shard_path, allow_pickle=True)
            phi_list.append(data["phi"])
            y_list.append(data["y"])
            params_list.append(data["params"])
            lags_list.append(data["lags"])
        
        self.phi = np.concatenate(phi_list, axis=0)      # (N, N_hist, d_hist)
        self.y = np.concatenate(y_list, axis=0)          # (N, N_out, d_state)
        self.params = np.concatenate(params_list, axis=0)  # (N, P)
        self.lags = np.concatenate(lags_list, axis=0)    # (N, L)
        
        self.n_samples = len(self.phi)
        self.hist_dim = self.phi.shape[2]
        
        # Compute normalization statistics
        if self.normalize:
            self._compute_stats()
    
    def _compute_stats(self):
        """Compute mean/std for normalization."""
        # For history, use phi values
        self.phi_mean = self.phi.mean(axis=(0, 1))
        self.phi_std = self.phi.std(axis=(0, 1)) + 1e-8
        
        # For solution
        self.y_mean = self.y.mean(axis=(0, 1))
        self.y_std = self.y.std(axis=(0, 1)) + 1e-8
        
        # For params
        self.param_mean = self.params.mean(axis=0)
        self.param_std = self.params.std(axis=0) + 1e-8
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns combined-grid encoding suitable for FNO.
        
        Input channels:
        - history signal (phi for t <= 0, zeros for t > 0)
        - mask (1 for t <= 0, 0 for t > 0)
        - normalized time coordinate
        - parameters (broadcast)
        
        Output:
        - full solution on combined grid
        - loss mask (1 for t > 0)
        """
        phi = self.phi[idx]      # (N_hist, d_hist)
        y = self.y[idx]          # (N_out, d_state)
        params = self.params[idx]  # (P,)
        
        # Normalize
        if self.normalize:
            phi_norm = (phi - self.phi_mean) / self.phi_std
            y_norm = (y - self.y_mean) / self.y_std
            params_norm = (params - self.param_mean) / self.param_std
        else:
            phi_norm = phi
            y_norm = y
            params_norm = params
        
        # Build input channels
        # Channel: history signal (pad to state_dim if hist_dim differs)
        hist_signal = np.zeros((self.n_total, self.state_dim))
        if self.hist_dim == self.state_dim:
            hist_signal[:self.n_hist] = phi_norm
        else:
            # Only use first hist_dim channels
            hist_signal[:self.n_hist, :self.hist_dim] = phi_norm
        
        # Channel: mask
        mask = np.zeros((self.n_total, 1))
        mask[:self.n_hist] = 1.0
        
        # Channel: normalized time
        t_norm = (self.t_combined - self.t_combined.min()) / (
            self.t_combined.max() - self.t_combined.min()
        )
        t_channel = t_norm.reshape(-1, 1)
        
        # Channel: parameters (broadcast)
        param_channel = np.tile(params_norm, (self.n_total, 1))
        
        # Combine input channels
        input_tensor = np.concatenate([
            hist_signal, mask, t_channel, param_channel
        ], axis=1)
        
        # Build target (full solution on combined grid)
        target = np.zeros((self.n_total, self.state_dim))
        if self.include_history_in_target and self.hist_dim == self.state_dim:
            target[:self.n_hist] = phi_norm
        elif self.include_history_in_target:
            target[:self.n_hist, :self.hist_dim] = phi_norm
        target[self.n_hist:] = y_norm
        
        # Loss mask (1 for future region t > 0)
        loss_mask = np.zeros(self.n_total)
        loss_mask[self.n_hist:] = 1.0
        
        return {
            "input": torch.from_numpy(input_tensor).float(),
            "target": torch.from_numpy(target).float(),
            "loss_mask": torch.from_numpy(loss_mask).float(),
            "params": torch.from_numpy(params).float(),
            "t": torch.from_numpy(self.t_combined).float(),
            # Include normalization stats for proper denormalization in evaluation
            "target_mean": torch.from_numpy(self.y_mean).float().unsqueeze(0),
            "target_std": torch.from_numpy(self.y_std).float().unsqueeze(0),
        }
    
    def get_input_channels(self) -> int:
        """Number of input channels for FNO."""
        return self.state_dim + 1 + 1 + self.n_params
    
    def get_output_channels(self) -> int:
        """Number of output channels for FNO."""
        return self.state_dim
    
    def denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize solution tensor."""
        if not self.normalize:
            return y_norm
        mean = torch.from_numpy(self.y_mean).to(y_norm.device).float()
        std = torch.from_numpy(self.y_std).to(y_norm.device).float()
        return y_norm * std + mean


class StreamingShardedDataset(IterableDataset):
    """
    Iterable dataset that streams shards without loading all into memory.
    Use for large datasets that don't fit in RAM.
    """
    
    def __init__(
        self,
        data_dir: str,
        family: str,
        split: str = "train",
        normalize: bool = True,
        shuffle_shards: bool = True,
        normalization_stats: Optional[Dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.family = family
        self.split = split
        self.normalize = normalize
        self.shuffle_shards = shuffle_shards
        
        # Load manifest
        manifest_path = self.data_dir / family / "manifest.json"
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        self.state_dim = self.manifest["state_dim"]
        self.n_params = len(self.manifest["param_names"])
        
        # Get shard files
        split_dir = self.data_dir / family / split
        self.shard_files = sorted(split_dir.glob("shard_*.npz"))
        
        # Load first shard for dimensions
        first_shard = np.load(self.shard_files[0])
        self.t_hist = first_shard["t_hist"]
        self.t_out = first_shard["t_out"]
        self.n_hist = len(self.t_hist)
        self.n_out = len(self.t_out)
        self.n_total = self.n_hist + self.n_out
        self.t_combined = np.concatenate([self.t_hist, self.t_out])
        self.hist_dim = first_shard["phi"].shape[2]
        
        # Normalization stats (compute from training set or use provided)
        if normalization_stats:
            self.phi_mean = normalization_stats["phi_mean"]
            self.phi_std = normalization_stats["phi_std"]
            self.y_mean = normalization_stats["y_mean"]
            self.y_std = normalization_stats["y_std"]
            self.param_mean = normalization_stats["param_mean"]
            self.param_std = normalization_stats["param_std"]
        elif normalize:
            self._compute_stats_streaming()
        else:
            self.phi_mean = self.phi_std = None
            self.y_mean = self.y_std = None
            self.param_mean = self.param_std = None
    
    def _compute_stats_streaming(self):
        """Compute stats by streaming through shards."""
        phi_sum = phi_sq_sum = 0
        y_sum = y_sq_sum = 0
        param_sum = param_sq_sum = 0
        n_phi = n_y = n_param = 0
        
        for shard_path in self.shard_files:
            data = np.load(shard_path)
            phi = data["phi"]
            y = data["y"]
            params = data["params"]
            
            phi_sum += phi.sum(axis=(0, 1))
            phi_sq_sum += (phi ** 2).sum(axis=(0, 1))
            n_phi += phi.shape[0] * phi.shape[1]
            
            y_sum += y.sum(axis=(0, 1))
            y_sq_sum += (y ** 2).sum(axis=(0, 1))
            n_y += y.shape[0] * y.shape[1]
            
            param_sum += params.sum(axis=0)
            param_sq_sum += (params ** 2).sum(axis=0)
            n_param += params.shape[0]
        
        self.phi_mean = phi_sum / n_phi
        self.phi_std = np.sqrt(phi_sq_sum / n_phi - self.phi_mean ** 2) + 1e-8
        
        self.y_mean = y_sum / n_y
        self.y_std = np.sqrt(y_sq_sum / n_y - self.y_mean ** 2) + 1e-8
        
        self.param_mean = param_sum / n_param
        self.param_std = np.sqrt(param_sq_sum / n_param - self.param_mean ** 2) + 1e-8
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        shard_order = list(range(len(self.shard_files)))
        if self.shuffle_shards:
            import random
            random.shuffle(shard_order)
        
        for shard_idx in shard_order:
            data = np.load(self.shard_files[shard_idx])
            phi = data["phi"]
            y = data["y"]
            params = data["params"]
            
            for i in range(phi.shape[0]):
                yield self._encode_sample(phi[i], y[i], params[i])
    
    def _encode_sample(self, phi, y, params) -> Dict[str, torch.Tensor]:
        """Encode a single sample."""
        if self.normalize:
            phi_norm = (phi - self.phi_mean) / self.phi_std
            y_norm = (y - self.y_mean) / self.y_std
            params_norm = (params - self.param_mean) / self.param_std
        else:
            phi_norm, y_norm, params_norm = phi, y, params
        
        hist_signal = np.zeros((self.n_total, self.state_dim))
        hist_signal[:self.n_hist, :self.hist_dim] = phi_norm
        
        mask = np.zeros((self.n_total, 1))
        mask[:self.n_hist] = 1.0
        
        t_norm = (self.t_combined - self.t_combined.min()) / (
            self.t_combined.max() - self.t_combined.min()
        )
        t_channel = t_norm.reshape(-1, 1)
        
        param_channel = np.tile(params_norm, (self.n_total, 1))
        
        input_tensor = np.concatenate([hist_signal, mask, t_channel, param_channel], axis=1)
        
        target = np.zeros((self.n_total, self.state_dim))
        target[:self.n_hist, :self.hist_dim] = phi_norm
        target[self.n_hist:] = y_norm
        
        loss_mask = np.zeros(self.n_total)
        loss_mask[self.n_hist:] = 1.0
        
        return {
            "input": torch.from_numpy(input_tensor).float(),
            "target": torch.from_numpy(target).float(),
            "loss_mask": torch.from_numpy(loss_mask).float(),
            "params": torch.from_numpy(params).float(),
            "t": torch.from_numpy(self.t_combined).float(),
        }
    
    def get_normalization_stats(self) -> Dict:
        """Get normalization stats to share with val/test sets."""
        return {
            "phi_mean": self.phi_mean,
            "phi_std": self.phi_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "param_mean": self.param_mean,
            "param_std": self.param_std,
        }


def create_sharded_dataloaders(
    data_dir: str,
    family: str,
    batch_size: int = 32,
    num_workers: int = 4,
    streaming: bool = False,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for sharded dataset.

    Args:
        data_dir: Root data directory
        family: DDE family name
        batch_size: Batch size
        num_workers: Number of data loading workers
        streaming: Use streaming dataset (for large data)
        distributed: Use DistributedSampler for DDP training

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if streaming:
        train_ds = StreamingShardedDataset(data_dir, family, "train")
        stats = train_ds.get_normalization_stats()
        val_ds = StreamingShardedDataset(data_dir, family, "val",
                                         shuffle_shards=False, normalization_stats=stats)
        test_ds = StreamingShardedDataset(data_dir, family, "test",
                                          shuffle_shards=False, normalization_stats=stats)

        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    else:
        # Auto-detect PDE vs DDE family by checking manifest for input_type
        is_pde = False
        manifest_path = Path(data_dir) / family / "manifest.json"
        if manifest_path.exists():
            import json as _json
            with open(manifest_path) as _f:
                _manifest = _json.load(_f)
            is_pde = "input_type" in _manifest or _manifest.get("generator") == "python_pilot" and "domain" in _manifest.get("config", {})
            # More reliable: check if shards have "input_func" key (PDE) vs "phi" key (DDE)
            split_dir = Path(data_dir) / family / "train"
            shard_files = sorted(split_dir.glob("shard_*.npz"))
            if shard_files:
                import numpy as _np
                _shard = _np.load(shard_files[0], allow_pickle=True)
                is_pde = "input_func" in _shard.files

        if is_pde:
            from datasets.pde_dataset import ShardedPDEDataset
            train_ds = ShardedPDEDataset(data_dir, family, "train")
            val_ds = ShardedPDEDataset(data_dir, family, "val")
            test_ds = ShardedPDEDataset(data_dir, family, "test")

            # Share normalization from training set
            for ds in [val_ds, test_ds]:
                ds.input_mean = train_ds.input_mean
                ds.input_std = train_ds.input_std
                ds.y_mean = train_ds.y_mean
                ds.y_std = train_ds.y_std
                ds.param_mean = train_ds.param_mean
                ds.param_std = train_ds.param_std
        else:
            train_ds = ShardedDDEDataset(data_dir, family, "train")
            val_ds = ShardedDDEDataset(data_dir, family, "val")
            test_ds = ShardedDDEDataset(data_dir, family, "test")

            # Share normalization from training set
            for ds in [val_ds, test_ds]:
                ds.phi_mean = train_ds.phi_mean
                ds.phi_std = train_ds.phi_std
                ds.y_mean = train_ds.y_mean
                ds.y_std = train_ds.y_std
                ds.param_mean = train_ds.param_mean
                ds.param_std = train_ds.param_std

        # Distributed training uses DistributedSampler instead of shuffle
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_ds, shuffle=True)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True
            )

        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader, test_loader
