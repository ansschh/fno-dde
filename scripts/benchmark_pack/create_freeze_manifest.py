#!/usr/bin/env python3
"""
Create freeze manifest for the baseline benchmark pack.
This captures all dataset paths, model checkpoints, configs, and commit hash.
"""
import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent

FAMILIES = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]

MODEL_PATHS = {
    "dist_exp": "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403",
    "hutch": "outputs/baseline_v1/hutch_seed42_20251228_131919",
    "linear2": "outputs/baseline_v1/linear2_seed42_20251228_142839",
    "vdp": "outputs/baseline_v1/vdp_seed42_20251229_020516",
    "dist_uniform": "outputs/baseline_v1/dist_uniform_seed42_20251229_030851",
}

DATA_PATHS = {
    "dist_exp": "data_baseline_v2/dist_exp",
    "hutch": "data_baseline_v1/hutch",
    "linear2": "data_baseline_v1/linear2",
    "vdp": "data_baseline_v1/vdp",
    "dist_uniform": "data_baseline_v1/dist_uniform",
}


def get_git_info():
    """Get current git commit and status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0
        
        # Try to get tag
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags", "--always"], cwd=ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            tag = commit[:8]
        
        return {"commit": commit, "tag": tag, "dirty": dirty}
    except:
        return {"commit": "unknown", "tag": "unknown", "dirty": True}


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def hash_directory(path: Path, extensions: list = None) -> str:
    """Compute combined hash of all files in directory."""
    if not path.exists():
        return "MISSING"
    
    h = hashlib.sha256()
    files = sorted(path.rglob("*"))
    for f in files:
        if f.is_file():
            if extensions and f.suffix not in extensions:
                continue
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


def load_config(model_dir: Path) -> dict:
    """Load model config from checkpoint."""
    import torch
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        return {}
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("config", {})
    except:
        return {}


def main():
    manifest = {
        "created": datetime.now().isoformat(),
        "git": get_git_info(),
        "families": {},
    }
    
    print("Creating freeze manifest...")
    
    for family in FAMILIES:
        print(f"  Processing {family}...")
        
        data_path = ROOT / DATA_PATHS[family]
        model_path = ROOT / MODEL_PATHS[family]
        
        family_entry = {
            "data_path": DATA_PATHS[family],
            "model_path": MODEL_PATHS[family],
            "data_hash": hash_directory(data_path, extensions=[".npy", ".npz", ".json"]),
            "model_hash": hash_file(model_path / "best_model.pt"),
            "config": load_config(model_path).get("model", {}),
        }
        
        # Load manifest from data directory
        manifest_path = data_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                family_entry["data_manifest"] = json.load(f)
        
        # Load training history
        history_path = model_path / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
                family_entry["training_summary"] = {
                    "epochs": len(history.get("train_loss", [])),
                    "best_val_loss": min(history.get("val_loss", [999])),
                    "best_val_rel_l2": min(history.get("val_rel_l2", [999])),
                }
        
        manifest["families"][family] = family_entry
    
    # Save manifest
    output_path = ROOT / "freeze_manifest_all5_v2.json"
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"\n✓ Saved freeze manifest to: {output_path}")
    print(f"  Git: {manifest['git']['tag']} ({'dirty' if manifest['git']['dirty'] else 'clean'})")


if __name__ == "__main__":
    main()
