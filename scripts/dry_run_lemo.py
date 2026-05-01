"""
Dry-run script for LEMO: one config per spectral-norm σ level, short training,
verifies end-to-end plumbing against the existing sharded DDE dataset.

Reads configs from configs/dry_run/*.yaml. For each, trains the LEMO model for
the configured number of epochs and reports test MSE plus (for LEMO_σ) peak
activation magnitude on the held-out test set as a rollout-proxy diagnostic.

The goal of a dry-run is to exercise the pipeline, not to produce publishable
numbers. Epoch count and training samples are deliberately small.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from datasets.sharded_dataset import create_sharded_dataloaders
from models import create_lemo, count_parameters


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    m = mask.unsqueeze(-1).expand_as(diff2)
    return (diff2 * m).sum() / m.sum().clamp_min(1.0)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch['input'].to(device)
        y = batch['target'].to(device)
        mask = batch.get('mask', torch.ones_like(y[..., 0])).to(device)
        pred = model(x)
        loss = masked_mse(pred, y, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    peak = 0.0
    for batch in loader:
        x = batch['input'].to(device)
        y = batch['target'].to(device)
        mask = batch.get('mask', torch.ones_like(y[..., 0])).to(device)
        pred = model(x)
        loss = masked_mse(pred, y, mask)
        total += loss.item() * x.shape[0]
        count += x.shape[0]
        peak = max(peak, float(pred.abs().max().item()))
    return total / max(count, 1), peak


def run_one(config_path: Path, data_dir: str, device: torch.device) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    family = config['family']
    epochs = config.get('epochs', 2)
    lr = config.get('lr', 1e-3)
    batch = config.get('batch_size', 32)

    t0 = time.time()
    train_loader, val_loader, test_loader = create_sharded_dataloaders(
        data_dir=data_dir,
        family=family,
        batch_size=batch,
        num_workers=0,
    )

    sample = train_loader.dataset[0]
    in_channels = sample['input'].shape[-1]
    out_channels = sample['target'].shape[-1]
    length_in = sample['input'].shape[0]

    config.setdefault('model', {})
    config['model']['length'] = length_in
    model = create_lemo(in_channels, out_channels, config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for ep in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

    val_mse, val_peak = evaluate(model, val_loader, device)
    test_mse, test_peak = evaluate(model, test_loader, device)

    return {
        "config": config_path.name,
        "sigma": config['model'].get('sigma', None),
        "params": count_parameters(model),
        "train_losses": train_losses,
        "val_mse": val_mse,
        "val_peak": val_peak,
        "test_mse": test_mse,
        "test_peak": test_peak,
        "wall_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config_dir', type=str, default='configs/dry_run')
    p.add_argument('--data_dir', type=str, default='data_baseline_v2')
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config_dir = REPO / args.config_dir
    configs = sorted(config_dir.glob('*.yaml'))
    if not configs:
        print(f"No configs found in {config_dir}")
        return 1

    results = []
    for cp in configs:
        print(f"\n=== {cp.name} ===", flush=True)
        try:
            r = run_one(cp, args.data_dir, device)
            results.append(r)
            print(f"  params={r['params']:,}  "
                  f"train[last]={r['train_losses'][-1]:.4e}  "
                  f"val_mse={r['val_mse']:.4e}  "
                  f"test_mse={r['test_mse']:.4e}  "
                  f"test_peak={r['test_peak']:.3e}  "
                  f"{r['wall_s']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"config": cp.name, "error": str(e)})

    print("\n\n=== Summary ===")
    print(f"{'config':<28} {'sigma':>8} {'params':>9} {'test_mse':>12} {'test_peak':>12} {'s':>6}")
    print("-" * 86)
    for r in results:
        if 'error' in r:
            print(f"{r['config']:<28} {'--':>8} {'--':>9} {'ERR':>12} {'--':>12} {'--':>6}   {r['error'][:40]}")
            continue
        sigma = r['sigma'] if r['sigma'] is not None else 'none'
        print(f"{r['config']:<28} {str(sigma):>8} {r['params']:>9,} "
              f"{r['test_mse']:>12.4e} {r['test_peak']:>12.3e} {r['wall_s']:>6.1f}")

    return 0 if all('error' not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
