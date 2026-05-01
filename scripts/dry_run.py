"""
Unified dry-run driver for LEMO, LEMO_sigma, and non-equivariant baselines
(PlainMLP with optional shift augmentation).

Reads any yaml config in --config_dir; each config specifies `model_class`:
  - 'lemo'      -> dde-fno/src/models/lemo.py:create_lemo
  - 'mlp'       -> dde-fno/src/models/baselines.py:MLPBaseline
  - 'fno1d'     -> dde-fno/src/models/fno1d.py:create_fno1d
Optionally, `shift_aug_p: float` enables training-time cyclic shift
augmentation (applied to both input and target in lock-step).

The goal of a dry-run is to exercise the pipeline, not to produce
publishable numbers. Use short epoch counts and compare relative
behaviors across models, not absolute accuracy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from datasets.sharded_dataset import create_sharded_dataloaders
from models import (
    create_lemo, create_fno1d, MLPBaseline, apply_cyclic_shift, count_parameters,
    create_research_baseline,
)

RESEARCH_BASELINES = {"deeponet", "memno", "anie", "nie", "localno", "lno", "mfno", "zfno", "vanillafno"}


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    m = mask.unsqueeze(-1).expand_as(diff2)
    return (diff2 * m).sum() / m.sum().clamp_min(1.0)


def build_model(config: dict, in_channels: int, out_channels: int, length: int) -> torch.nn.Module:
    model_class = config.get('model_class', 'fno1d').lower()
    if model_class == 'lemo':
        c = dict(config)
        c.setdefault('model', {})
        c['model'] = {**c['model'], 'length': length}
        return create_lemo(in_channels, out_channels, c)
    elif model_class == 'mlp':
        mcfg = config.get('model', {})
        return MLPBaseline(
            seq_length=length,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=mcfg.get('hidden_dim', 256),
            n_layers=mcfg.get('n_layers', 4),
        )
    elif model_class == 'fno1d':
        return create_fno1d(in_channels, out_channels, config.get('model', {}),
                            use_residual=config.get('use_residual', False))
    elif model_class in RESEARCH_BASELINES:
        return create_research_baseline(model_class, in_channels, out_channels,
                                         length, config)
    else:
        raise ValueError(f"unknown model_class: {model_class!r}")


def train_one_epoch(model, loader, optimizer, device, shift_aug_p: float = 0.0,
                    grad_clip: float = 0.0) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        # Move to device
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if shift_aug_p > 0:
            batch_gpu = apply_cyclic_shift(batch_gpu, shift_probability=shift_aug_p)
        x = batch_gpu['input']
        y = batch_gpu['target']
        mask = batch_gpu.get('mask', torch.ones_like(y[..., 0]))
        pred = model(x)
        loss = masked_mse(pred, y, mask)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
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
    shift_aug_p = float(config.get('shift_aug_p', 0.0))
    grad_clip = float(config.get('grad_clip', 0.0))

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
    length = sample['input'].shape[0]

    model = build_model(config, in_channels, out_channels, length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for ep in range(epochs):
        train_losses.append(
            train_one_epoch(model, train_loader, optimizer, device,
                            shift_aug_p=shift_aug_p, grad_clip=grad_clip)
        )

    val_mse, val_peak = evaluate(model, val_loader, device)
    test_mse, test_peak = evaluate(model, test_loader, device)

    return {
        "config": config_path.name,
        "model_class": config.get('model_class', 'fno1d'),
        "sigma": config.get('model', {}).get('sigma', None),
        "shift_aug_p": shift_aug_p,
        "params": count_parameters(model),
        "train_last": train_losses[-1],
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
            print(f"  class={r['model_class']:<8} sigma={str(r['sigma']):>6} "
                  f"aug_p={r['shift_aug_p']:.2f}  "
                  f"params={r['params']:,}  train[last]={r['train_last']:.3e}  "
                  f"val={r['val_mse']:.3e}  test={r['test_mse']:.3e}  "
                  f"peak={r['test_peak']:.2e}  {r['wall_s']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"config": cp.name, "error": str(e)})

    print("\n\n=== Summary ===")
    header = f"{'config':<30} {'class':>6} {'sigma':>6} {'aug':>4} {'params':>10} {'test_mse':>12} {'peak':>10} {'s':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        if 'error' in r:
            print(f"{r['config']:<30} ERROR  {r['error'][:50]}")
            continue
        sigma = r['sigma'] if r['sigma'] is not None else 'none'
        print(f"{r['config']:<30} {r['model_class']:>6} {str(sigma):>6} "
              f"{r['shift_aug_p']:>4.1f} {r['params']:>10,} "
              f"{r['test_mse']:>12.3e} {r['test_peak']:>10.2e} {r['wall_s']:>6.1f}")

    return 0 if all('error' not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
