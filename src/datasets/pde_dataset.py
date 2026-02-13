"""
PyTorch Dataset for Sharded PDE Data

Loads sharded .npz files and provides combined-grid encoding for FNO training.
Follows the same pattern as ShardedDDEDataset but adapted for PDE data:
- Input: [input_func(x) | x_coord_normalized | params_broadcast]
- Output: solution(x)
- Loss mask: 1 everywhere (no history/future split for PDEs)
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class ShardedPDEDataset(Dataset):
    """
    Map-style dataset for PDE data stored in NPZ shards.

    Shard format:
        x_grid: (N_x,) spatial grid
        input_func: (B, N_x, d_in) input function(s)
        solution: (B, N_x, d_out) solution field(s)
        params: (B, P) PDE parameters
        meta_json: str JSON metadata

    Encoding for FNO:
        Input channels: [input_func | x_normalized | params_broadcast]
        Target: solution
        Loss mask: all ones (every spatial point contributes to loss)
    """

    def __init__(
        self,
        data_dir: str,
        family: str,
        split: str = "train",
        normalize: bool = True,
    ):
        """
        Args:
            data_dir: Root data directory.
            family: PDE family name (e.g., "burgers", "ks", "helmholtz", "wave").
            split: "train", "val", or "test".
            normalize: Whether to normalize inputs/outputs.
        """
        self.data_dir = Path(data_dir)
        self.family = family
        self.split = split
        self.normalize = normalize

        # Load manifest
        manifest_path = self.data_dir / family / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.state_dim = self.manifest["state_dim"]
        self.param_names = self.manifest["param_names"]
        self.n_params = len(self.param_names)
        self.input_type = self.manifest.get("input_type", "initial_condition")

        # Load all shards for this split
        split_dir = self.data_dir / family / split
        shard_files = sorted(split_dir.glob("shard_*.npz"))

        if len(shard_files) == 0:
            raise FileNotFoundError(f"No shards found in {split_dir}")

        # Load first shard to get dimensions
        first_shard = np.load(shard_files[0], allow_pickle=True)
        self.x_grid = first_shard["x_grid"]
        self.n_spatial = len(self.x_grid)

        # Determine input dimension from first shard
        sample_input = first_shard["input_func"]
        if sample_input.ndim == 2:
            # (B, N_x) -> single channel
            self.input_dim = 1
        else:
            # (B, N_x, d_in)
            self.input_dim = sample_input.shape[2]

        # Load all data
        input_list = []
        solution_list = []
        params_list = []

        for shard_path in shard_files:
            data = np.load(shard_path, allow_pickle=True)
            inp = data["input_func"]
            sol = data["solution"]
            par = data["params"]

            # Ensure 3D: (B, N_x, d_in)
            if inp.ndim == 2:
                inp = inp[:, :, np.newaxis]
            if sol.ndim == 2:
                sol = sol[:, :, np.newaxis]

            input_list.append(inp)
            solution_list.append(sol)
            params_list.append(par)

        self.inputs = np.concatenate(input_list, axis=0)      # (N, N_x, d_in)
        self.solutions = np.concatenate(solution_list, axis=0)  # (N, N_x, d_out)
        self.params = np.concatenate(params_list, axis=0)      # (N, P)

        self.n_samples = len(self.inputs)

        # Compute normalization statistics
        if self.normalize:
            self._compute_stats()

    def _compute_stats(self):
        """Compute per-channel mean/std for normalization."""
        # Input stats: mean/std over (samples, spatial)
        self.input_mean = self.inputs.mean(axis=(0, 1))   # (d_in,)
        self.input_std = self.inputs.std(axis=(0, 1)) + 1e-8

        # Solution stats — use y_mean/y_std to match DDE dataset interface
        self.y_mean = self.solutions.mean(axis=(0, 1))   # (d_out,)
        self.y_std = self.solutions.std(axis=(0, 1)) + 1e-8

        # Parameter stats
        self.param_mean = self.params.mean(axis=0)   # (P,)
        self.param_std = self.params.std(axis=0) + 1e-8

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns FNO-compatible encoding.

        Input channels (concatenated along last axis):
          - input_func: (N_x, d_in) normalized input function
          - x_coord: (N_x, 1) normalized spatial coordinate in [0, 1]
          - params: (N_x, P) parameters broadcast to every spatial point

        Target: (N_x, d_out) solution

        Loss mask: (N_x,) all ones
        """
        inp = self.inputs[idx]       # (N_x, d_in)
        sol = self.solutions[idx]    # (N_x, d_out)
        params = self.params[idx]    # (P,)

        # Normalize
        if self.normalize:
            inp_norm = (inp - self.input_mean) / self.input_std
            sol_norm = (sol - self.y_mean) / self.y_std
            params_norm = (params - self.param_mean) / self.param_std
        else:
            inp_norm = inp
            sol_norm = sol
            params_norm = params

        # Normalized spatial coordinate in [0, 1]
        x_min, x_max = self.x_grid.min(), self.x_grid.max()
        x_norm = (self.x_grid - x_min) / (x_max - x_min + 1e-10)
        x_channel = x_norm.reshape(-1, 1)  # (N_x, 1)

        # Broadcast parameters to every spatial point
        param_channel = np.tile(params_norm, (self.n_spatial, 1))  # (N_x, P)

        # Concatenate input channels
        input_tensor = np.concatenate([
            inp_norm,       # (N_x, d_in)
            x_channel,      # (N_x, 1)
            param_channel,  # (N_x, P)
        ], axis=1)

        # Loss mask: 1 everywhere for PDEs
        loss_mask = np.ones(self.n_spatial, dtype=np.float32)

        return {
            "input": torch.from_numpy(input_tensor).float(),
            "target": torch.from_numpy(sol_norm).float(),
            "loss_mask": torch.from_numpy(loss_mask).float(),
            "params": torch.from_numpy(params).float(),
            "x": torch.from_numpy(self.x_grid).float(),
            # Normalization stats for denormalization during evaluation
            "target_mean": torch.from_numpy(self.y_mean).float().unsqueeze(0),
            "target_std": torch.from_numpy(self.y_std).float().unsqueeze(0),
        }

    def get_input_channels(self) -> int:
        """Number of input channels for FNO."""
        return self.input_dim + 1 + self.n_params

    def get_output_channels(self) -> int:
        """Number of output channels for FNO."""
        return self.state_dim

    def denormalize_solution(self, sol_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize solution tensor."""
        if not self.normalize:
            return sol_norm
        mean = torch.from_numpy(self.y_mean).to(sol_norm.device).float()
        std = torch.from_numpy(self.y_std).to(sol_norm.device).float()
        return sol_norm * std + mean


def create_pde_dataloaders(
    data_dir: str,
    family: str,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for PDE dataset.

    Args:
        data_dir: Root data directory.
        family: PDE family name.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        distributed: Use DistributedSampler for DDP training.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = ShardedPDEDataset(data_dir, family, "train")
    val_ds = ShardedPDEDataset(data_dir, family, "val")
    test_ds = ShardedPDEDataset(data_dir, family, "test")

    # Share normalization statistics from training set
    for ds in [val_ds, test_ds]:
        ds.input_mean = train_ds.input_mean
        ds.input_std = train_ds.input_std
        ds.y_mean = train_ds.y_mean
        ds.y_std = train_ds.y_std
        ds.param_mean = train_ds.param_mean
        ds.param_std = train_ds.param_std

    # Build dataloaders
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
