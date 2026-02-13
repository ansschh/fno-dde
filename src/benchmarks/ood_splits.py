"""
Out-of-Distribution (OOD) Test Split Generation

Creates specialized test splits for generalization benchmarks:
- OOD-delay: Test on delays outside training range
- OOD-params: Test on parameters outside training range
- OOD-history: Test on different history function distributions
- Horizon: Test on longer time horizons
- Resolution: Test on different time discretizations
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import yaml
import json


@dataclass
class OODSplitConfig:
    """Configuration for an OOD split."""
    name: str
    description: str
    param_overrides: Dict[str, Tuple[float, float]]  # Parameter range overrides
    tau_max: Optional[float] = None
    T: Optional[float] = None
    dt_out: Optional[float] = None
    history_type: str = "fourier"  # "fourier", "spline", "step"
    n_samples: int = 100
    seed_offset: int = 0


class OODSplitGenerator:
    """
    Generates OOD test splits based on configuration.
    """
    
    def __init__(self, base_config_path: Path):
        """
        Args:
            base_config_path: Path to dataset_v1.yaml
        """
        with open(base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)
    
    def get_ood_delay_config(self, family: str) -> OODSplitConfig:
        """Get OOD-delay split configuration."""
        family_config = self.base_config["families"].get(family, {})
        ood_delay = family_config.get("ood_delay", {})
        
        # Extract test ranges
        param_overrides = {}
        for key, range_dict in ood_delay.get("test", {}).items():
            if isinstance(range_dict, dict):
                param_overrides[key] = (range_dict.get("min", 0.1), range_dict.get("max", 2.0))
            elif isinstance(range_dict, list):
                param_overrides[key] = tuple(range_dict)
        
        return OODSplitConfig(
            name="ood_delay",
            description="Out-of-distribution delay test",
            param_overrides=param_overrides,
            seed_offset=300000,
        )
    
    def get_ood_params_config(self, family: str) -> OODSplitConfig:
        """Get OOD-params split configuration."""
        family_config = self.base_config["families"].get(family, {})
        ood_params = family_config.get("ood_params", {})
        
        param_overrides = {}
        for key, range_dict in ood_params.get("test", {}).items():
            if isinstance(range_dict, dict):
                param_overrides[key] = (range_dict.get("min", 0.5), range_dict.get("max", 2.0))
            elif isinstance(range_dict, list):
                param_overrides[key] = tuple(range_dict)
        
        return OODSplitConfig(
            name="ood_params",
            description="Out-of-distribution parameter test",
            param_overrides=param_overrides,
            seed_offset=400000,
        )
    
    def get_ood_history_config(self, history_type: str = "spline") -> OODSplitConfig:
        """Get OOD-history split configuration."""
        return OODSplitConfig(
            name="ood_history",
            description=f"Out-of-distribution history test ({history_type})",
            param_overrides={},
            history_type=history_type,
            seed_offset=600000,
        )
    
    def get_horizon_config(self, T_factor: float = 2.0) -> OODSplitConfig:
        """Get extended horizon test configuration."""
        base_T = self.base_config["global"]["T"]
        
        return OODSplitConfig(
            name="horizon",
            description=f"Extended horizon test (T={base_T * T_factor})",
            param_overrides={},
            T=base_T * T_factor,
            seed_offset=500000,
        )
    
    def get_resolution_config(self, dt_factor: float = 0.5) -> OODSplitConfig:
        """Get resolution generalization test configuration."""
        base_dt = self.base_config["global"]["dt_out"]
        
        return OODSplitConfig(
            name="resolution",
            description=f"Resolution test (dt={base_dt * dt_factor})",
            param_overrides={},
            dt_out=base_dt * dt_factor,
            seed_offset=700000,
        )
    
    def get_all_ood_configs(self, family: str) -> List[OODSplitConfig]:
        """Get all OOD configurations for a family."""
        configs = [
            self.get_ood_delay_config(family),
            self.get_ood_params_config(family),
            self.get_ood_history_config("spline"),
            self.get_horizon_config(2.0),
            self.get_resolution_config(0.5),
        ]
        return configs


def sample_params_ood(
    base_ranges: Dict[str, Tuple[float, float]],
    overrides: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Sample parameters with OOD overrides.
    
    Args:
        base_ranges: Base parameter ranges
        overrides: OOD range overrides
        rng: Random number generator
        
    Returns:
        Sampled parameters
    """
    params = {}
    
    for name, (lo, hi) in base_ranges.items():
        if name in overrides:
            lo, hi = overrides[name]
        
        if lo == hi:
            params[name] = lo
        else:
            params[name] = rng.uniform(lo, hi)
    
    return params


def sample_spline_history(
    rng: np.random.Generator,
    t_hist: np.ndarray,
    n_knots: int = 5,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Sample history using cubic spline interpolation.
    
    Different from Fourier series - tests function space robustness.
    """
    from scipy.interpolate import CubicSpline
    
    # Random knot positions
    knot_t = np.linspace(t_hist[0], t_hist[-1], n_knots)
    knot_y = rng.uniform(-amplitude, amplitude, n_knots)
    
    # Interpolate
    spline = CubicSpline(knot_t, knot_y)
    return spline(t_hist)


def sample_step_history(
    rng: np.random.Generator,
    t_hist: np.ndarray,
    n_steps: int = 3,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Sample piecewise constant history.
    
    Tests robustness to non-smooth histories.
    """
    # Random step positions
    step_t = np.sort(rng.uniform(t_hist[0], t_hist[-1], n_steps - 1))
    step_t = np.concatenate([[t_hist[0]], step_t, [t_hist[-1]]])
    step_y = rng.uniform(-amplitude, amplitude, n_steps)
    
    # Build piecewise constant
    history = np.zeros_like(t_hist)
    for i in range(n_steps):
        mask = (t_hist >= step_t[i]) & (t_hist < step_t[i + 1])
        history[mask] = step_y[i]
    
    # Handle last point
    history[-1] = step_y[-1]
    
    return history


def generate_ood_samples(
    family,
    ood_config: OODSplitConfig,
    n_samples: int,
    base_seed: int,
    output_dir: Path,
):
    """
    Generate OOD samples for a family.
    
    Note: This is a template - actual implementation depends on solver choice.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from dde.families import get_family
    
    family_obj = get_family(family) if isinstance(family, str) else family
    config = family_obj.config
    
    rng = np.random.default_rng(base_seed + ood_config.seed_offset)
    
    # Override settings
    T = ood_config.T or config.T
    tau_max = ood_config.tau_max or config.tau_max
    
    samples = []
    
    for i in range(n_samples):
        # Sample params with overrides
        base_ranges = config.param_ranges
        params = sample_params_ood(base_ranges, ood_config.param_overrides, rng)
        
        # Sample history based on type
        t_hist = np.linspace(-tau_max, 0, 256)
        
        if ood_config.history_type == "spline":
            phi = sample_spline_history(rng, t_hist)
        elif ood_config.history_type == "step":
            phi = sample_step_history(rng, t_hist)
        else:
            phi = family_obj.sample_history(rng, t_hist)[:, 0]
        
        # For positive families, ensure positive
        if config.requires_positive:
            phi = np.abs(phi) + 0.1
        
        samples.append({
            "params": params,
            "history": phi,
            "t_hist": t_hist,
        })
    
    return samples


def create_ood_dataset_config(
    base_config_path: Path,
    family: str,
    ood_type: str,
    output_path: Path,
):
    """
    Create a configuration file for generating OOD datasets.
    
    Args:
        base_config_path: Path to dataset_v1.yaml
        family: Family name
        ood_type: One of "delay", "params", "history", "horizon", "resolution"
        output_path: Where to save the config
    """
    generator = OODSplitGenerator(base_config_path)
    
    if ood_type == "delay":
        config = generator.get_ood_delay_config(family)
    elif ood_type == "params":
        config = generator.get_ood_params_config(family)
    elif ood_type == "history":
        config = generator.get_ood_history_config()
    elif ood_type == "horizon":
        config = generator.get_horizon_config()
    elif ood_type == "resolution":
        config = generator.get_resolution_config()
    else:
        raise ValueError(f"Unknown OOD type: {ood_type}")
    
    # Convert to dict and save
    config_dict = {
        "name": config.name,
        "description": config.description,
        "family": family,
        "param_overrides": config.param_overrides,
        "tau_max": config.tau_max,
        "T": config.T,
        "dt_out": config.dt_out,
        "history_type": config.history_type,
        "n_samples": config.n_samples,
        "seed_offset": config.seed_offset,
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f)
