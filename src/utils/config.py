"""
Configuration Utilities

Handles loading, merging, and validating configuration files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import copy


DEFAULT_CONFIG = {
    # Data settings
    'family': 'linear',
    'data_dir': 'data/processed',
    'batch_size': 32,
    'num_workers': 4,
    
    # Model settings
    'model': {
        'modes': 16,
        'width': 64,
        'n_layers': 4,
        'activation': 'gelu',
        'padding': 8,
    },
    'use_residual': False,
    
    # Training settings
    'epochs': 100,
    'lr': 1e-3,
    'lr_min': 1e-6,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'save_every': 10,
    
    # Dataset generation settings
    'n_samples': 1000,
    'n_hist_points': 64,
    'n_future_points': 192,
    'seed': 42,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and merge with defaults.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
    
    # Deep merge with defaults
    config = deep_merge(copy.deepcopy(DEFAULT_CONFIG), user_config)
    
    return config


def deep_merge(base: Dict, update: Dict) -> Dict:
    """
    Recursively merge update dict into base dict.
    
    Args:
        base: Base dictionary
        update: Dictionary with updates
        
    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError if invalid
    """
    required_fields = ['family']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Dynamic validation: accept any registered DDE or PDE family
    try:
        from dde.families import DDE_FAMILIES
        valid_families = set(DDE_FAMILIES.keys())
    except ImportError:
        valid_families = set()

    try:
        from pde.families import PDE_FAMILIES
        valid_families |= set(PDE_FAMILIES.keys())
    except ImportError:
        pass

    # Only validate if we have a registry loaded
    if valid_families and config['family'] not in valid_families:
        raise ValueError(f"Invalid family: {config['family']}. Must be one of {sorted(valid_families)}")
    
    if config.get('epochs', 0) <= 0:
        raise ValueError("epochs must be positive")
    
    if config.get('batch_size', 0) <= 0:
        raise ValueError("batch_size must be positive")
    
    return True
