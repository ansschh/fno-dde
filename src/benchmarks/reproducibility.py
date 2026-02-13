"""
Reproducibility Verification

Ensures dataset generation is fully reproducible:
- Shard-level verification (same seed → same data)
- Hash-based regression tests
- Config tracking
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess


def compute_array_hash(arr: np.ndarray) -> str:
    """Compute SHA256 hash of numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def compute_shard_hash(shard_path: Path) -> Dict[str, str]:
    """Compute hashes for all arrays in a shard."""
    data = np.load(shard_path, allow_pickle=True)
    
    hashes = {}
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            hashes[key] = compute_array_hash(arr)
    
    return hashes


def verify_shard_reproducibility(
    shard_path: Path,
    expected_hashes: Dict[str, str],
) -> Tuple[bool, Dict[str, str]]:
    """
    Verify a shard matches expected hashes.
    
    Returns:
        (passed, actual_hashes)
    """
    actual_hashes = compute_shard_hash(shard_path)
    
    passed = True
    for key, expected in expected_hashes.items():
        if key not in actual_hashes:
            passed = False
        elif actual_hashes[key] != expected:
            passed = False
    
    return passed, actual_hashes


def create_reproducibility_manifest(
    data_dir: Path,
    family: str,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Create a manifest with hashes for reproducibility verification.
    """
    family_dir = data_dir / family
    
    manifest = {
        "family": family,
        "splits": {},
    }
    
    for split in ["train", "val", "test"]:
        split_dir = family_dir / split
        if not split_dir.exists():
            continue
        
        manifest["splits"][split] = {}
        
        # Hash first shard only (for quick regression test)
        shard_files = sorted(split_dir.glob("shard_*.npz"))
        if shard_files:
            first_shard = shard_files[0]
            hashes = compute_shard_hash(first_shard)
            manifest["splits"][split]["first_shard"] = {
                "path": first_shard.name,
                "hashes": hashes,
            }
            manifest["splits"][split]["n_shards"] = len(shard_files)
    
    # Add git info if available
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=data_dir,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        manifest["git_commit"] = git_hash
    except:
        pass
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    return manifest


def run_reproducibility_test(
    data_dir: Path,
    family: str,
    manifest_path: Path,
) -> Tuple[bool, str]:
    """
    Run reproducibility regression test.
    
    Returns:
        (passed, message)
    """
    if not manifest_path.exists():
        return False, f"Manifest not found: {manifest_path}"
    
    with open(manifest_path, "r") as f:
        expected = json.load(f)
    
    if expected.get("family") != family:
        return False, f"Family mismatch: expected {expected.get('family')}, got {family}"
    
    all_passed = True
    messages = []
    
    for split, split_data in expected.get("splits", {}).items():
        first_shard_info = split_data.get("first_shard", {})
        shard_name = first_shard_info.get("path")
        expected_hashes = first_shard_info.get("hashes", {})
        
        if not shard_name:
            continue
        
        shard_path = data_dir / family / split / shard_name
        
        if not shard_path.exists():
            all_passed = False
            messages.append(f"Shard not found: {shard_path}")
            continue
        
        passed, actual_hashes = verify_shard_reproducibility(shard_path, expected_hashes)
        
        if not passed:
            all_passed = False
            messages.append(f"Hash mismatch in {split}/{shard_name}")
            
            # Show differences
            for key in expected_hashes:
                if key in actual_hashes and actual_hashes[key] != expected_hashes[key]:
                    messages.append(f"  {key}: expected {expected_hashes[key]}, got {actual_hashes[key]}")
    
    if all_passed:
        return True, "Reproducibility test PASSED"
    else:
        return False, "Reproducibility test FAILED:\n" + "\n".join(messages)


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    versions = {}
    
    packages = ["numpy", "scipy", "torch", "h5py"]
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    
    return versions


def create_generation_record(
    config: Dict,
    output_path: Path,
):
    """
    Create a complete record of generation settings for reproducibility.
    """
    import sys
    import platform
    
    record = {
        "config": config,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": get_package_versions(),
        },
    }
    
    # Git info
    try:
        record["git"] = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip(),
            "branch": subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip(),
            "dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()),
        }
    except:
        record["git"] = None
    
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    
    return record
