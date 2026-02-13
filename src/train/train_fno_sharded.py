"""
FNO Training Script for Sharded DDE Datasets

Updated training script that works with the new sharded dataset format.
Includes proper loss masking, learning rate scheduling, and comprehensive logging.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.sharded_dataset import ShardedDDEDataset, create_sharded_dataloaders
from models import FNO1d, FNO1dResidual, create_fno1d, count_parameters
from utils.config import load_config


def setup_distributed():
    """Initialize DDP from SLURM/torchrun environment variables.

    Returns:
        (rank, world_size, is_distributed)
    """
    if 'WORLD_SIZE' not in os.environ or int(os.environ['WORLD_SIZE']) <= 1:
        return 0, 1, False

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    return rank, world_size, True


def cleanup_distributed():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss only on masked (future) region.
    
    Args:
        pred: Predicted values (batch, length, channels)
        target: Target values (batch, length, channels)
        mask: Loss mask (batch, length), 1 where loss should be computed
        
    Returns:
        Scalar loss value
    """
    mask = mask.unsqueeze(-1)
    sq_error = (pred - target) ** 2
    masked_error = sq_error * mask
    loss = masked_error.sum() / (mask.sum() * pred.shape[-1] + 1e-8)
    return loss


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute relative L2 error on masked region.
    """
    mask = mask.unsqueeze(-1)
    diff = (pred - target) * mask
    target_masked = target * mask
    
    diff_norm = torch.sqrt((diff ** 2).sum(dim=(1, 2)) + 1e-8)
    target_norm = torch.sqrt((target_masked ** 2).sum(dim=(1, 2)) + 1e-8)
    
    rel_error = (diff_norm / target_norm).mean()
    return rel_error


class ShardedTrainer:
    """Trainer class for FNO models with sharded datasets.

    Supports single-GPU and multi-GPU (DDP) training. When distributed,
    logging/checkpointing is gated to rank 0.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        output_dir: Path,
        rank: int = 0,
        world_size: int = 1,
        is_distributed: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.is_main = (rank == 0)

        self.model = model.to(device)
        if is_distributed:
            self.model = DDP(self.model, device_ids=[rank])

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir

        # Optimizer (use unwrapped model params)
        model_params = self.model.module.parameters() if is_distributed else self.model.parameters()
        self.optimizer = optim.AdamW(
            model_params,
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 100),
                eta_min=config.get('lr_min', 1e-6),
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('lr_step', 30),
                gamma=config.get('lr_gamma', 0.5),
            )
        else:
            self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

        # Tensorboard (rank 0 only)
        self.writer = SummaryWriter(output_dir / 'logs') if self.is_main else None

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rel_l2': [],
            'lr': [],
        }

        # Early stopping
        self.patience = config.get('patience', 20)
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Set epoch for distributed sampler
        if self.is_distributed and hasattr(self.train_loader, 'sampler'):
            sampler = self.train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False,
                    disable=not self.is_main)
        for batch in pbar:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            mask = batch['loss_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = masked_mse_loss(outputs, targets, mask)
            
            loss.backward()
            
            if self.config.get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_rel_l2 = 0.0
        n_batches = 0
        
        for batch in self.val_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            mask = batch['loss_mask'].to(self.device)
            
            outputs = self.model(inputs)
            
            loss = masked_mse_loss(outputs, targets, mask)
            rel_l2 = relative_l2_error(outputs, targets, mask)
            
            total_loss += loss.item()
            total_rel_l2 += rel_l2.item()
            n_batches += 1
        
        return total_loss / n_batches, total_rel_l2 / n_batches
    
    def train(self, epochs: int) -> bool:
        """
        Full training loop.
        
        Returns:
            True if training completed, False if early stopped
        """
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_rel_l2 = self.validate()
            
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Logging (all ranks track history, only rank 0 writes tensorboard)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rel_l2'].append(val_rel_l2)
            self.history['lr'].append(current_lr)

            if self.is_main:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Metrics/rel_l2', val_rel_l2, epoch)
                self.writer.add_scalar('LR', current_lr, epoch)

                print(f"Epoch {epoch:3d}: train={train_loss:.6f} val={val_loss:.6f} "
                      f"rel_L2={val_rel_l2:.4f} lr={current_lr:.2e}")
            
            # Best model (only rank 0 saves)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                if self.is_main:
                    self.save_checkpoint('best_model.pt')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                if self.is_main:
                    print(f"\nEarly stopping at epoch {epoch} (patience={self.patience})")
                break

            # Periodic checkpoint (rank 0 only)
            if self.is_main and epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Final save (rank 0 only)
        if self.is_main:
            self.save_checkpoint('final_model.pt')
            self.save_history()
            print(f"\nTraining complete. Best val_loss={self.best_val_loss:.6f} at epoch {self.best_epoch}")

        return self.patience_counter < self.patience
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint (unwraps DDP if needed)."""
        model_to_save = self.model.module if self.is_distributed else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_history(self):
        """Save training history."""
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray = None,
    y_std: np.ndarray = None,
) -> dict:
    """
    Comprehensive evaluation of model on a data loader.
    
    Reports metrics in both normalized and original space.
    
    Args:
        model: Trained model
        loader: Data loader
        device: Torch device
        y_mean: Mean for denormalization (required for original space)
        y_std: Std for denormalization (required for original space)
    
    Returns:
        Dictionary with various metrics in both spaces
    """
    model.eval()
    
    total_loss = 0.0
    all_rel_l2_norm = []  # normalized space
    all_rel_l2_orig = []  # original space
    all_rmse_norm = []
    all_rmse_orig = []
    n_batches = 0
    
    # Convert stats to tensors if provided
    if y_mean is not None and y_std is not None:
        y_mean_t = torch.from_numpy(y_mean).float().to(device)
        y_std_t = torch.from_numpy(y_std).float().to(device)
        has_denorm = True
    else:
        has_denorm = False
    
    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        mask = batch['loss_mask'].to(device)
        
        outputs = model(inputs)
        
        loss = masked_mse_loss(outputs, targets, mask)
        
        # Per-sample metrics in normalized space
        mask_exp = mask.unsqueeze(-1)
        diff_norm = (outputs - targets) * mask_exp
        target_masked_norm = targets * mask_exp
        
        diff_l2 = torch.sqrt((diff_norm ** 2).sum(dim=(1, 2)) + 1e-8)
        target_l2 = torch.sqrt((target_masked_norm ** 2).sum(dim=(1, 2)) + 1e-8)
        rel_l2_norm = (diff_l2 / target_l2).cpu().numpy()
        all_rel_l2_norm.extend(rel_l2_norm)
        
        # RMSE in normalized space
        n_masked = mask_exp.sum(dim=(1, 2))
        rmse_norm = torch.sqrt((diff_norm ** 2).sum(dim=(1, 2)) / (n_masked + 1e-8)).cpu().numpy()
        all_rmse_norm.extend(rmse_norm)
        
        # Original space metrics
        if has_denorm:
            outputs_orig = outputs * y_std_t + y_mean_t
            targets_orig = targets * y_std_t + y_mean_t
            
            diff_orig = (outputs_orig - targets_orig) * mask_exp
            target_masked_orig = targets_orig * mask_exp
            
            diff_l2_orig = torch.sqrt((diff_orig ** 2).sum(dim=(1, 2)) + 1e-8)
            target_l2_orig = torch.sqrt((target_masked_orig ** 2).sum(dim=(1, 2)) + 1e-8)
            rel_l2_orig = (diff_l2_orig / target_l2_orig).cpu().numpy()
            all_rel_l2_orig.extend(rel_l2_orig)
            
            rmse_orig = torch.sqrt((diff_orig ** 2).sum(dim=(1, 2)) / (n_masked + 1e-8)).cpu().numpy()
            all_rmse_orig.extend(rmse_orig)
        
        total_loss += loss.item()
        n_batches += 1
    
    all_rel_l2_norm = np.array(all_rel_l2_norm)
    all_rmse_norm = np.array(all_rmse_norm)
    
    metrics = {
        'mse_normalized': total_loss / n_batches,
        'rel_l2_normalized_mean': all_rel_l2_norm.mean(),
        'rel_l2_normalized_std': all_rel_l2_norm.std(),
        'rel_l2_normalized_median': np.median(all_rel_l2_norm),
        'rel_l2_normalized_p95': np.percentile(all_rel_l2_norm, 95),
        'rmse_normalized_mean': all_rmse_norm.mean(),
        'rmse_normalized_median': np.median(all_rmse_norm),
    }
    
    if has_denorm:
        all_rel_l2_orig = np.array(all_rel_l2_orig)
        all_rmse_orig = np.array(all_rmse_orig)
        metrics.update({
            'rel_l2_original_mean': all_rel_l2_orig.mean(),
            'rel_l2_original_std': all_rel_l2_orig.std(),
            'rel_l2_original_median': np.median(all_rel_l2_orig),
            'rel_l2_original_p95': np.percentile(all_rel_l2_orig, 95),
            'rmse_original_mean': all_rmse_orig.mean(),
            'rmse_original_median': np.median(all_rmse_orig),
        })
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train FNO on sharded DDE dataset')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing sharded data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--overfit_n', type=int, default=0,
                        help='If > 0, overfit on N samples (for debugging)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Setup distributed training (no-op if not launched with torchrun)
    rank, world_size, is_distributed = setup_distributed()
    is_main = (rank == 0)

    # Set seeds for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    # Load config
    config = load_config(args.config)

    # Setup output directory (rank 0 only)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{config['family']}_seed{args.seed}_{timestamp}"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    # Barrier to ensure directory exists before other ranks proceed
    if is_distributed:
        dist.barrier()

    # Device
    if is_distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if is_main:
        print(f"Using device: {device}" +
              (f" (DDP: {world_size} GPUs)" if is_distributed else ""))

    # Create dataloaders
    if is_main:
        print(f"\nLoading data for family: {config['family']}")

    if args.overfit_n > 0:
        from datasets.sharded_dataset import ShardedDDEDataset
        from torch.utils.data import DataLoader as DL, Subset

        train_ds = ShardedDDEDataset(args.data_dir, config['family'], 'train')
        val_ds = ShardedDDEDataset(args.data_dir, config['family'], 'val')
        test_ds = ShardedDDEDataset(args.data_dir, config['family'], 'test')

        for ds in [val_ds, test_ds]:
            ds.y_mean, ds.y_std = train_ds.y_mean, train_ds.y_std
            ds.phi_mean, ds.phi_std = train_ds.phi_mean, train_ds.phi_std
            ds.param_mean, ds.param_std = train_ds.param_mean, train_ds.param_std

        subset_indices = list(range(min(args.overfit_n, len(train_ds))))
        train_subset = Subset(train_ds, subset_indices)

        train_loader = DL(train_subset, batch_size=min(args.overfit_n, 32), shuffle=False)
        val_loader = DL(train_subset, batch_size=min(args.overfit_n, 32), shuffle=False)
        test_loader = DL(train_subset, batch_size=min(args.overfit_n, 32), shuffle=False)

        y_mean, y_std = train_ds.y_mean, train_ds.y_std
        if is_main:
            print(f"\n*** OVERFIT TEST: training on {len(subset_indices)} samples ***")
    else:
        train_loader, val_loader, test_loader = create_sharded_dataloaders(
            data_dir=args.data_dir,
            family=config['family'],
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4),
            streaming=config.get('streaming', False),
            distributed=is_distributed,
        )
        y_mean = train_loader.dataset.y_mean
        y_std = train_loader.dataset.y_std

    # Get dimensions from dataset
    sample = next(iter(train_loader))
    in_channels = sample['input'].shape[-1]
    out_channels = sample['target'].shape[-1]
    seq_length = sample['input'].shape[1]

    if is_main:
        print(f"Input: ({seq_length}, {in_channels}), Output: ({seq_length}, {out_channels})")
        print(f"Train samples: ~{len(train_loader) * config.get('batch_size', 32)}")

    # Create model
    model = create_fno1d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=config.get('model', {}),
        use_residual=config.get('use_residual', False),
    )

    if is_main:
        print(f"Model parameters: {count_parameters(model):,}")

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if is_main:
            print(f"Resumed from {args.resume}")

    # Create trainer
    trainer = ShardedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        is_distributed=is_distributed,
    )

    # Train
    if is_main:
        print(f"\nStarting training for {config.get('epochs', 100)} epochs...")
    trainer.train(epochs=config.get('epochs', 100))

    # Final evaluation on test set (rank 0 only)
    if is_main:
        print("\nEvaluating on test set...")

        # Load best model (unwrapped)
        eval_model = create_fno1d(
            in_channels=in_channels,
            out_channels=out_channels,
            config=config.get('model', {}),
            use_residual=config.get('use_residual', False),
        )
        best_checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
        eval_model.load_state_dict(best_checkpoint['model_state_dict'])
        eval_model.to(device)

        test_metrics = evaluate_model(eval_model, test_loader, device,
                                       y_mean=y_mean, y_std=y_std)

        print("\nTest Results:")
        print("  [Normalized space]")
        print(f"    MSE: {test_metrics['mse_normalized']:.6f}")
        print(f"    Rel L2 (median): {test_metrics['rel_l2_normalized_median']:.4f}")
        print(f"    Rel L2 (p95): {test_metrics['rel_l2_normalized_p95']:.4f}")
        print(f"    RMSE (median): {test_metrics['rmse_normalized_median']:.4f}")

        if 'rel_l2_original_median' in test_metrics:
            print("  [Original space] (PRIMARY)")
            print(f"    Rel L2 (mean): {test_metrics['rel_l2_original_mean']:.4f} "
                  f"\u00b1 {test_metrics['rel_l2_original_std']:.4f}")
            print(f"    Rel L2 (median): {test_metrics['rel_l2_original_median']:.4f}")
            print(f"    Rel L2 (p95): {test_metrics['rel_l2_original_p95']:.4f}")
            print(f"    RMSE (median): {test_metrics['rmse_original_median']:.4f}")

        test_metrics_json = {k: float(v) for k, v in test_metrics.items()}
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics_json, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    # Clean up
    cleanup_distributed()


if __name__ == '__main__':
    main()
