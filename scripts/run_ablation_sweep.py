"""
Ablation Sweep Runner

Runs systematic ablation studies:
1. Capacity sweep (modes, width, depth)
2. Input encoding ablation
3. Loss weighting ablation

Usage:
    python scripts/run_ablation_sweep.py --family hutch --sweep capacity
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import yaml
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_sweep_config(sweep_type: str) -> dict:
    """Load sweep configuration."""
    config_dir = Path(__file__).parent.parent / "configs" / "ablations"
    
    if sweep_type == "capacity":
        config_path = config_dir / "fno_capacity.yaml"
    elif sweep_type == "encoding":
        config_path = config_dir / "input_encoding.yaml"
    elif sweep_type == "loss":
        config_path = config_dir / "loss_weighting.yaml"
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_variant_config(
    base_config: dict,
    variant_config: dict,
    family: str,
    output_path: Path,
) -> Path:
    """Create a training config for a specific variant."""
    config = {
        "family": family,
        **base_config,
        "model": {
            "modes": variant_config.get("modes", 16),
            "width": variant_config.get("width", 64),
            "n_layers": variant_config.get("n_layers", 4),
            "activation": base_config.get("activation", "gelu"),
            "padding": base_config.get("padding", 8),
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    
    return output_path


def run_capacity_sweep(
    family: str,
    data_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    variants: list = None,
):
    """Run capacity ablation sweep."""
    config = load_sweep_config("capacity")
    base = config.get("base", {})
    all_variants = config.get("variants", {})
    
    if variants is None:
        variants = config.get("sweep_order", list(all_variants.keys()))
    
    results = {
        "family": family,
        "sweep_type": "capacity",
        "timestamp": datetime.now().isoformat(),
        "variants": {},
    }
    
    for variant_name in variants:
        if variant_name not in all_variants:
            print(f"Unknown variant: {variant_name}")
            continue
        
        variant = all_variants[variant_name]
        print(f"\n{'='*60}")
        print(f"Running variant: {variant_name}")
        print(f"  {variant.get('description', '')}")
        print('='*60)
        
        # Create config
        variant_dir = output_dir / "capacity" / variant_name
        config_path = variant_dir / "config.yaml"
        create_variant_config(base, variant, family, config_path)
        
        # Run training
        train_script = Path(__file__).parent.parent / "src" / "train" / "train_fno_sharded.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            f"--config={config_path}",
            f"--data_dir={data_dir}",
            f"--output_dir={variant_dir}",
            f"--device={device}",
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False)
        
        # Load results if available
        results_path = variant_dir / "test_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                variant_results = json.load(f)
            results["variants"][variant_name] = {
                "config": variant,
                "test_results": variant_results,
            }
        else:
            results["variants"][variant_name] = {
                "config": variant,
                "status": "failed" if result.returncode != 0 else "no_results",
            }
    
    # Save sweep results
    sweep_results_path = output_dir / "capacity" / "sweep_results.json"
    with open(sweep_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSweep results saved to: {sweep_results_path}")
    
    return results


def run_encoding_ablation(
    family: str,
    data_dir: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """Run input encoding ablation."""
    config = load_sweep_config("encoding")
    
    results = {
        "family": family,
        "sweep_type": "encoding",
        "timestamp": datetime.now().isoformat(),
        "variants": {},
    }
    
    # This would require modifying the dataset to use different encodings
    # For now, just create the configs
    
    for variant_name, variant_config in config.items():
        if variant_name in ["description"]:
            continue
        
        print(f"\nEncoding variant: {variant_name}")
        print(f"  Channels: {variant_config.get('channels', [])}")
        
        results["variants"][variant_name] = {
            "config": variant_config,
            "status": "config_created",
        }
    
    # Save
    output_path = output_dir / "encoding" / "ablation_configs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEncoding configs saved to: {output_path}")
    print("Note: Actual training requires dataset encoder modification")
    
    return results


def run_loss_ablation(
    family: str,
    data_dir: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """Run loss weighting ablation."""
    config = load_sweep_config("loss")
    
    results = {
        "family": family,
        "sweep_type": "loss",
        "timestamp": datetime.now().isoformat(),
        "variants": {},
    }
    
    for variant_name, variant_config in config.items():
        if variant_name in ["description"]:
            continue
        
        print(f"\nLoss variant: {variant_name}")
        print(f"  Type: {variant_config.get('type', 'unknown')}")
        
        results["variants"][variant_name] = {
            "config": variant_config,
            "status": "config_created",
        }
    
    # Save
    output_path = output_dir / "loss" / "ablation_configs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nLoss configs saved to: {output_path}")
    print("Note: Actual training requires trainer modification to use weighted loss")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation sweeps")
    parser.add_argument("--family", type=str, required=True,
                        help="DDE family name")
    parser.add_argument("--sweep", type=str, required=True,
                        choices=["capacity", "encoding", "loss", "all"],
                        help="Type of sweep to run")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="ablation_results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        help="Specific variants to run (for capacity sweep)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.family
    
    if args.sweep == "capacity" or args.sweep == "all":
        run_capacity_sweep(
            args.family, data_dir, output_dir, args.device, args.variants
        )
    
    if args.sweep == "encoding" or args.sweep == "all":
        run_encoding_ablation(
            args.family, data_dir, output_dir, args.device
        )
    
    if args.sweep == "loss" or args.sweep == "all":
        run_loss_ablation(
            args.family, data_dir, output_dir, args.device
        )
    
    print("\nAblation sweep complete!")


if __name__ == "__main__":
    main()
