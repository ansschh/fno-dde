"""
Continuous-τ off-grid OOD data generator.

For each target family (DistExp, DistUniform, Hutchinson), generates a
held-out test set in which the delay parameter τ is sampled freshly
from the family's default uniform range.  These samples test
sample-level generalization across the continuous τ axis — the
theoretical story (plan §4 / theorem T14) predicts that architectural
lag-equivariant models transfer exactly to unseen continuous τ, while
finite-augmentation baselines pay a covering-radius-sized error.

Shard layout mirrors the existing seed-agnostic OOD layout:

    data_ood_continuous_tau/<family>/test_ood/shard_*.npz
    data_ood_continuous_tau/<family>/manifest.json

so `sweep_phase_b.py` can pick it up with no configuration changes once
the split_data_dir entry points here.

Usage:

    python3 src/datasets/generate_continuous_tau_ood.py \\
        --families dist_exp dist_uniform hutch \\
        --n_test 2000 \\
        --output_dir data_ood_continuous_tau \\
        --seed 9999

The separate seed (default 9999) ensures the continuous-τ test samples
are disjoint from the frozen baseline's train/val/test RNG stream.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from dde.families import get_family
from datasets.generate_python import (
    ShardConfig, generate_sample_python, run_qc, write_shard_npz,
)


# Families for which continuous-τ OOD is applicable per plan §4.
CONTINUOUS_TAU_FAMILIES = ("dist_exp", "dist_uniform", "hutch")


def generate_test_ood_split(
    family_name: str,
    n_samples: int,
    output_dir: Path,
    seed: int,
    tau_max: float = 2.0,
    T: float = 20.0,
    dt_out: float = 0.05,
    n_hist: int = 256,
    shard_size: int = 128,
    y_clip: float = 100.0,
    max_retries: int = 10,
) -> int:
    """Generate a single `test_ood` split for one family.

    Returns the number of samples actually written (may be slightly
    less than `n_samples` if QC rejects many candidates).
    """
    family = get_family(family_name)
    fam_dir = output_dir / family_name
    split_dir = fam_dir / "test_ood"
    split_dir.mkdir(parents=True, exist_ok=True)

    n_shards = (n_samples + shard_size - 1) // shard_size
    total = 0
    fail_counts: dict[str, int] = {}

    for shard_id in range(n_shards):
        shard_path = split_dir / f"shard_{shard_id:03d}.npz"
        if shard_path.exists():
            total += shard_size
            continue

        remaining = n_samples - shard_id * shard_size
        B = min(shard_size, remaining)
        rng = np.random.default_rng(seed + shard_id * 1000)
        samples: list[dict] = []
        retries = 0
        pbar = tqdm(total=B, desc=f"  {family_name} shard {shard_id}", leave=False)

        while len(samples) < B and retries < B * max_retries:
            s = generate_sample_python(family, rng, tau_max, T, dt_out, n_hist)
            if s is None:
                fail_counts["solver_failed"] = fail_counts.get("solver_failed", 0) + 1
                retries += 1
                continue
            passed, reasons = run_qc(s["y"], s["phi"], family, y_clip)
            if not passed:
                for r in reasons:
                    fail_counts[r] = fail_counts.get(r, 0) + 1
                retries += 1
                continue
            samples.append(s)
            pbar.update(1)
            retries = 0

        pbar.close()
        if samples:
            meta = {
                "family": family_name,
                "split": "test_ood",
                "shard_id": shard_id,
                "n_samples": len(samples),
                "config": {"tau_max": tau_max, "T": T, "dt_out": dt_out, "n_hist": n_hist},
                "seed": seed + shard_id * 1000,
                "generator": "continuous_tau_ood",
            }
            write_shard_npz(str(shard_path), samples, meta)
            total += len(samples)

    if fail_counts:
        print(f"  {family_name} failures: {fail_counts}")

    # Per-family manifest.
    manifest = {
        "family": family_name,
        "split": "test_ood",
        "ood_type": "continuous_tau",
        "description": (
            "Fresh independent sample of (phi, params, y) with the family's "
            "default continuous tau range, different random seed from the "
            "frozen baseline. Tests sample-level continuous-tau generalization "
            "per plan Theorem T14 (continuous-lag equivariance) and T15 "
            "(continuous OOD transfer)."
        ),
        "config": {
            "tau_max": tau_max, "T": T, "dt_out": dt_out, "n_hist": n_hist,
        },
        "param_names": family.config.param_names,
        "param_ranges": {k: list(v) for k, v in family.config.param_ranges.items()},
        "state_dim": family.config.state_dim,
        "n_samples": total,
        "n_shards": n_shards,
        "seed": seed,
        "generator": "continuous_tau_ood",
    }
    with open(fam_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", nargs="+", default=list(CONTINUOUS_TAU_FAMILIES),
                   help="Which families to generate. Default: dist_exp, dist_uniform, hutch.")
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--output_dir", type=str, default="data_ood_continuous_tau")
    p.add_argument("--seed", type=int, default=9999,
                   help="Base seed for the test-OOD stream. Default: 9999 "
                        "(disjoint from the frozen baseline's 0/1/2-prefixed seeds).")
    p.add_argument("--tau_max", type=float, default=2.0)
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--dt_out", type=float, default=0.05)
    p.add_argument("--n_hist", type=int, default=256)
    p.add_argument("--shard_size", type=int, default=128)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating continuous-τ OOD data under {output_dir}/")
    for family in args.families:
        print(f"\n=== {family} ===")
        n = generate_test_ood_split(
            family_name=family,
            n_samples=args.n_test,
            output_dir=output_dir,
            seed=args.seed,
            tau_max=args.tau_max,
            T=args.T,
            dt_out=args.dt_out,
            n_hist=args.n_hist,
            shard_size=args.shard_size,
        )
        print(f"  → wrote {n} samples to {output_dir / family / 'test_ood'}")


if __name__ == "__main__":
    main()
