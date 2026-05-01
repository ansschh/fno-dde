#!/usr/bin/env python3
"""
LEMO architectural diagnostic.

For a single T1 batch, profile activation magnitudes at every layer of:
  (a) LEMO with default init (broken, blows up to 1e7)
  (b) LEMO with the 1/length init patch (collapses to mean)
  (c) FNO1d (the working baseline)

Also profile gradient magnitudes after one backward pass.

Output: a per-layer table with mean(|x|), max(|x|), std(|x|), and the
same for gradients.  This pins down WHERE the cascade goes wrong and
informs the principled init fix.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.lemo import LEMO, ContinuousLagConv1d
from models.fno1d import FNO1d, SpectralConv1d


def stats(t: torch.Tensor, name: str) -> str:
    if t.numel() == 0:
        return f"{name:<40} empty"
    a = t.detach().abs()
    return (f"{name:<40} mean={a.mean().item():.3e}  "
            f"max={a.max().item():.3e}  "
            f"std={t.detach().std().item():.3e}  "
            f"shape={tuple(t.shape)}")


def hook_fwd(name: str, store: list):
    def _hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            store.append((name, output.detach().clone()))
    return _hook


def profile_lemo(model: LEMO, x: torch.Tensor, label: str) -> None:
    print(f"\n{'='*80}\n{label}\n{'='*80}")
    print(f"input: {stats(x, 'input')}")

    handles = []
    fwd_log: list = []

    handles.append(model.lift.register_forward_hook(hook_fwd("lift", fwd_log)))
    for i, conv in enumerate(model.conv_blocks):
        handles.append(conv.register_forward_hook(hook_fwd(f"conv{i}", fwd_log)))
    handles.append(model.proj.register_forward_hook(hook_fwd("proj", fwd_log)))

    out = model(x)
    for h in handles:
        h.remove()

    for name, t in fwd_log:
        print(f"  {stats(t, name)}")
    print(f"  {stats(out, 'OUTPUT')}")

    # Kernel magnitudes
    print(f"\nKernels:")
    for i, conv in enumerate(model.conv_blocks):
        K = conv.compute_kernel()
        K_hat = torch.fft.rfft(K, dim=-1)
        print(f"  conv{i} K (out, in, L):  {stats(K, f'K_{i}')}")
        print(f"  conv{i} |K_hat|:         "
              f"mean={K_hat.abs().mean().item():.3e}  "
              f"max={K_hat.abs().max().item():.3e}")

    # Backward: simple MSE loss against zeros
    target = torch.zeros_like(out)
    loss = F.mse_loss(out, target)
    print(f"\nLoss (vs zeros): {loss.item():.3e}")
    loss.backward()

    print("\nGradients (parameter-group):")
    groups = {
        "lift.weight":  model._lift_linear.weight,
        "lift.bias":    model._lift_linear.bias,
        "proj.weight":  model._proj_linear.weight,
        "proj.bias":    model._proj_linear.bias,
    }
    for i, conv in enumerate(model.conv_blocks):
        for j, layer in enumerate(conv.kernel_net):
            if isinstance(layer, nn.Linear):
                groups[f"conv{i}.kernel_net[{j}].weight"] = layer.weight
                groups[f"conv{i}.kernel_net[{j}].bias"] = layer.bias

    for name, p in groups.items():
        g = p.grad
        if g is None:
            print(f"  {name:<40} grad=None")
        else:
            a = g.abs()
            print(f"  {name:<40} mean={a.mean().item():.3e}  "
                  f"max={a.max().item():.3e}")


def profile_fno(model: FNO1d, x: torch.Tensor, label: str) -> None:
    print(f"\n{'='*80}\n{label}\n{'='*80}")
    print(f"input: {stats(x, 'input')}")

    handles = []
    fwd_log: list = []

    handles.append(model.lift.register_forward_hook(hook_fwd("lift", fwd_log)))
    for i, blk in enumerate(model.blocks):
        handles.append(blk.spectral_conv.register_forward_hook(
            hook_fwd(f"block{i}.spectral", fwd_log)))
        handles.append(blk.pointwise_conv.register_forward_hook(
            hook_fwd(f"block{i}.pointwise", fwd_log)))
        handles.append(blk.register_forward_hook(hook_fwd(f"block{i}.out", fwd_log)))
    handles.append(model.proj1.register_forward_hook(hook_fwd("proj1", fwd_log)))
    handles.append(model.proj2.register_forward_hook(hook_fwd("proj2", fwd_log)))

    out = model(x)
    for h in handles:
        h.remove()

    for name, t in fwd_log:
        print(f"  {stats(t, name)}")
    print(f"  {stats(out, 'OUTPUT')}")

    # Spectral weights
    print(f"\nSpectral weights:")
    for i, blk in enumerate(model.blocks):
        w = blk.spectral_conv.weights
        print(f"  block{i}.spectral.weights:  "
              f"|w| mean={w.abs().mean().item():.3e}  "
              f"max={w.abs().max().item():.3e}  "
              f"shape={tuple(w.shape)}")

    target = torch.zeros_like(out)
    loss = F.mse_loss(out, target)
    print(f"\nLoss (vs zeros): {loss.item():.3e}")
    loss.backward()

    print("\nGradients:")
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"  {name:<40} grad=None")
        else:
            a = p.grad.abs()
            print(f"  {name:<40} mean={a.mean().item():.3e}  "
                  f"max={a.max().item():.3e}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_shard",
                    default="data_phase_a/t1_continuous_lag/train/shard_000.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load a small batch from the shard.
    d = np.load(args.data_shard)
    print("Shard keys + shapes:")
    for k in d.keys():
        v = d[k]
        if hasattr(v, "shape"):
            print(f"  {k:<10} {v.shape} {v.dtype}")
    # Build a (batch, length, in_channels) input matching what the dataset
    # loader produces. For T1 (expose_tau=False), in_channels = phi (1) +
    # auxiliary lift (e.g., positional encoding). Inspect what the loader
    # actually does in src/datasets/sharded_dataset.py if needed; for now
    # use phi alone as input.
    phi = torch.tensor(d["phi"][:8], dtype=torch.float32)   # (8, 256, 1)
    print(f"\nUsing input: phi[:8] of shape {phi.shape}")

    torch.manual_seed(args.seed)
    in_c = phi.shape[-1]
    out_c = 1
    L = phi.shape[1]

    # (a) LEMO with current source code (small-init patch already applied)
    lemo = LEMO(in_channels=in_c, out_channels=out_c, length=L,
                width=48, n_layers=3, kernel_hidden=64, sigma=None)
    profile_lemo(lemo, phi.clone(), "LEMO (unconstrained, post-patch)")

    # (b) LEMO with sigma=0.99 (known-stable variant)
    torch.manual_seed(args.seed)
    lemo_sigma = LEMO(in_channels=in_c, out_channels=out_c, length=L,
                     width=48, n_layers=3, kernel_hidden=64, sigma=0.99)
    profile_lemo(lemo_sigma, phi.clone(), "LEMO_sigma=0.99")

    # (c) FNO1d (the working baseline)
    torch.manual_seed(args.seed)
    fno = FNO1d(in_channels=in_c, out_channels=out_c,
                modes=16, width=48, n_layers=3, activation="gelu")
    profile_fno(fno, phi.clone(), "FNO1d (reference)")


if __name__ == "__main__":
    main()
