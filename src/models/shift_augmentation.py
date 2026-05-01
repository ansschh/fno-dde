"""
Shift-augmentation utility for non-equivariant models.

Applies a cyclic roll to both input and target tensors along the length
(time/lag) dimension, simulating the augmentation baseline that a
non-equivariant model uses to approximate lag-equivariance.

This is a training-time transform, NOT a model wrapper. It does not
change the model architecture; it changes the training data
distribution. See Remark `rem:augmentation` and Corollary
`cor:augmentation-lower-bound` in the paper.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def apply_cyclic_shift(batch: Dict[str, torch.Tensor], shift_probability: float = 1.0,
                       m: Optional[int] = None,
                       rng: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """Shift every sample in a batch by an independent random k along the
    length axis (dim=1).

    Inputs are dictionaries matching the DDEDataset convention:
      'input': (batch, length, in_channels)
      'target': (batch, length, out_channels)
      'mask': (batch, length)   [optional; shifted in lock-step]

    Args:
        batch: dict with tensors to shift.
        shift_probability: probability that shift is applied per-sample;
            the same random k is drawn regardless, but with probability
            (1 - p) the sample is passed through unshifted. Use 1.0 for
            full augmentation (every sample shifted); use < 1.0 for partial.
        m: number of discrete shift points sampled from `{0, length/m,
            2·length/m, ..., (m-1)·length/m}`. `None` or `m >= length`
            means sample from the full cyclic group `{0, ..., length-1}`.
            This is the knob that distinguishes MLP+Aug(m) baselines in
            Phase A.
        rng: optional torch random generator for reproducibility.
    """
    x = batch['input']
    batch_size, length = x.shape[0], x.shape[1]

    if m is None or m >= length:
        if rng is None:
            ks = torch.randint(0, length, (batch_size,), device=x.device)
        else:
            ks = torch.randint(0, length, (batch_size,), device=x.device, generator=rng)
    else:
        # Draw from the m-point discrete subgroup. Candidates are length·i/m
        # floored to integer indices.
        candidates = torch.tensor(
            [int(length * i / m) for i in range(m)],
            device=x.device, dtype=torch.long,
        )
        if rng is None:
            idx = torch.randint(0, m, (batch_size,), device=x.device)
        else:
            idx = torch.randint(0, m, (batch_size,), device=x.device, generator=rng)
        ks = candidates[idx]

    if shift_probability < 1.0:
        mask = torch.rand(batch_size, device=x.device) < shift_probability
        ks = torch.where(mask, ks, torch.zeros_like(ks))

    # Build per-sample index tensor: (batch, length) where row i is
    # (arange(length) - ks[i]) mod length.
    arange = torch.arange(length, device=x.device)
    idx = (arange[None, :] - ks[:, None]) % length  # (batch, length)

    def shift_tensor(t: torch.Tensor) -> torch.Tensor:
        # t: (batch, length, ...) -> gather along length with per-row idx
        idx_ = idx
        while idx_.dim() < t.dim():
            idx_ = idx_.unsqueeze(-1)
        idx_ = idx_.expand_as(t)
        return torch.gather(t, 1, idx_)

    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[0] == batch_size \
                and v.shape[1] == length:
            out[k] = shift_tensor(v)
        else:
            out[k] = v
    return out
