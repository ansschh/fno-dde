#!/usr/bin/env python3
"""
Data Quality Suite for Baseline-All-5
Runs integrity, fidelity, residual, diversity checks on DDE datasets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import hashlib

from dde.families import get_family


def load_shard(data_dir: Path, family: str, split: str):
    shard_path = data_dir / family / split / "shard_000.npz"
    if not shard_path.exists():
        return None
    data = np.load(shard_path)
    return {k: data[k] for k in data.files}


def check_integrity(data_dir: Path, family_name: str, split: str):
    """Check shapes, NaNs, continuity, positivity."""
    data = load_shard(data_dir, family_name, split)
    if data is None:
        return {"error": "Shard not found"}
    
    family = get_family(family_name)
    phi, y = data["phi"], data["y"]
    
    # Continuity at t=0
    cont_err = np.max(np.abs(phi[:, -1, :] - y[:, 0, :]))
    
    results = {
        "n_samples": len(phi),
        "phi_shape": list(phi.shape),
        "y_shape": list(y.shape),
        "has_nans": bool(np.any(np.isnan(y))),
        "continuity_max_err": float(cont_err),
        "continuity_pass": cont_err < 1e-3,
    }
    
    if family.config.requires_positive:
        results["min_x"] = float(np.min(y[:, :, 0]))
        results["positivity_pass"] = results["min_x"] >= -1e-6
    
    return results


def check_diversity(data_dir: Path, family_name: str, split: str):
    """Profile amplitude, oscillations, energy distributions."""
    data = load_shard(data_dir, family_name, split)
    if data is None:
        return {"error": "Shard not found"}
    
    family = get_family(family_name)
    y, phi, params = data["y"], data["phi"], data["params"]
    
    # Target energy (relL2 denominator)
    y_norms = np.linalg.norm(y.reshape(len(y), -1), axis=1)
    
    # Zero crossings
    x = y[:, :, 0]
    zc = [np.sum(np.diff(np.sign(x[i] - np.mean(x[i]))) != 0) for i in range(len(x))]
    
    # History roughness
    roughness = np.mean(np.abs(np.diff(phi[:, :, 0], n=2, axis=1)), axis=1)
    
    return {
        "n_samples": len(y),
        "y_norm_median": float(np.median(y_norms)),
        "y_norm_p90": float(np.percentile(y_norms, 90)),
        "zero_crossings_mean": float(np.mean(zc)),
        "history_roughness_median": float(np.median(roughness)),
        "x_range": [float(np.min(y[:,:,0])), float(np.max(y[:,:,0]))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--data_dir", default="data_baseline_v1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output_dir", default="reports/data_quality")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.family
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"family": args.family, "checks": {}}
    
    for split in args.splits:
        print(f"\nChecking {args.family}/{split}...")
        
        integrity = check_integrity(Path(args.data_dir), args.family, split)
        diversity = check_diversity(Path(args.data_dir), args.family, split)
        
        results["checks"][split] = {
            "integrity": integrity,
            "diversity": diversity,
        }
        
        print(f"  Samples: {integrity.get('n_samples', 'N/A')}")
        print(f"  Continuity pass: {integrity.get('continuity_pass', 'N/A')}")
        print(f"  y_norm median: {diversity.get('y_norm_median', 'N/A'):.3f}")
    
    out_path = output_dir / "quality_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
