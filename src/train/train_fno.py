"""
FNO Training Script for DDE Operator Learning

Trains FNO1d models on DDE datasets with proper loss masking,
learning rate scheduling, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

from datasets import DDEDataset, create_dataloaders
from models import FNO1d, FNO1dResidual, create_fno1d, count_parameters
from utils.config import load_config
from utils.logging import setup_logger


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
    # Expand mask to match channel dimension
    mask = mask.unsqueeze(-1)  # (batch, length, 1)
    
    # Compute squared error
    sq_error = (pred - target) ** 2
    
    # Apply mask and compute mean
    masked_error = sq_error * mask
    loss = masked_error.sum() / (mask.sum() * pred.shape[-1] + 1e-8)
    
    return loss


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute relative L2 error on masked region.
    
    Args:
        pred: Predicted values (batch, length, channels)
        target: Target values (batch, length, channels)
        mask: Loss mask (batch, length)
        
    Returns:
        Relative L2 error (scalar)
    """
    mask = mask.unsqueeze(-1)
    
    diff = (pred - target) * mask
    target_masked = target * mask
    
    # L2 norms
    diff_norm = torch.sqrt((diff ** 2).sum(dim=(1, 2)) + 1e-8)
    target_norm = torch.sqrt((target_masked ** 2).sum(dim=(1, 2)) + 1e-8)
    
    rel_error = (diff_norm / target_norm).mean()
    return rel_error


class Trainer:
    """Trainer class for FNO models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('lr_min', 1e-6),
        )
        
        # Tensorboard
        self.writer = SummaryWriter(output_dir / 'logs')
        
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
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            mask = batch['loss_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = masked_mse_loss(outputs, targets, mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
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
    
    def train(self, epochs: int):
        """Full training loop."""
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_rel_l2 = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Logging
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rel_l2'].append(val_rel_l2)
            self.history['lr'].append(current_lr)
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/rel_l2', val_rel_l2, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, rel_L2={val_rel_l2:.4f}, "
                  f"lr={current_lr:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pt')
            
            # Periodic checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        print(f"\nTraining complete. Best val_loss={self.best_val_loss:.6f} at epoch {self.best_epoch}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_history(self):
        """Save training history."""
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train FNO for DDE')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{config['family']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        family=config['family'],
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
    )
    
    # Get input/output dimensions from dataset
    sample = train_loader.dataset[0]
    in_channels = sample['input'].shape[-1]
    out_channels = sample['target'].shape[-1]
    
    print(f"Input channels: {in_channels}, Output channels: {out_channels}")
    
    # Create model
    model = create_fno1d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=config.get('model', {}),
        use_residual=config.get('use_residual', False),
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    
    # Train
    trainer.train(epochs=config.get('epochs', 100))
    
    # Evaluate on test set
    test_loss, test_rel_l2 = evaluate_model(
        model, test_loader, device
    )
    print(f"\nTest results: loss={test_loss:.6f}, rel_L2={test_rel_l2:.4f}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_rel_l2': test_rel_l2,
        }, f, indent=2)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    total_rel_l2 = 0.0
    n_batches = 0
    
    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        mask = batch['loss_mask'].to(device)
        
        outputs = model(inputs)
        
        loss = masked_mse_loss(outputs, targets, mask)
        rel_l2 = relative_l2_error(outputs, targets, mask)
        
        total_loss += loss.item()
        total_rel_l2 += rel_l2.item()
        n_batches += 1
    
    return total_loss / n_batches, total_rel_l2 / n_batches


if __name__ == '__main__':
    main()
