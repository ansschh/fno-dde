"""
PDE Family Definitions

Defines each PDE family with parameter sampling for operator learning.
Each family specifies:
  - The PDE equation and domain
  - Parameter ranges
  - Input function generation (initial conditions, coefficient fields, etc.)
  - Solver dispatch
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod


@dataclass
class PDEConfig:
    """Configuration for a PDE family."""
    name: str
    spatial_dim: int              # Number of spatial dimensions (1 for 1D PDEs)
    n_spatial: int                # Number of spatial grid points
    n_time: int                   # Number of time steps saved (0 for steady-state)
    domain: Tuple[float, float]   # Spatial domain [a, b]
    T: float                      # Final time (0 for steady-state)
    state_dim: int                # Dimension of the solution (scalar = 1)
    param_names: List[str]        # Names of parameters
    param_ranges: Dict[str, Tuple[float, float]]  # Parameter sampling ranges
    input_type: str = "initial_condition"  # "initial_condition", "coefficient_field", "multi_channel"


class PDEFamily(ABC):
    """Abstract base class for PDE families."""

    def __init__(self, config: PDEConfig):
        self.config = config

    @abstractmethod
    def solve(
        self,
        input_func: np.ndarray,
        x_grid: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        Solve the PDE for a given input function and parameters.

        Args:
            input_func: Input function evaluated on x_grid.
                Shape depends on input_type:
                  - "initial_condition": (N_x,) or (N_x, d_in)
                  - "coefficient_field": (N_x, d_in)
                  - "multi_channel": (N_x, d_in)
            x_grid: Spatial grid, shape (N_x,).
            params: Dictionary of PDE parameters.

        Returns:
            Solution array. For time-dependent PDEs: (N_x,) at final time
            or (N_x, state_dim). For steady-state: (N_x,) or (N_x, state_dim).
        """
        pass

    @abstractmethod
    def sample_input_function(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random input function evaluated on x_grid.

        Args:
            rng: NumPy random number generator.
            x_grid: Spatial grid, shape (N_x,).

        Returns:
            Input function values on x_grid.
            Shape: (N_x,) for single-channel, (N_x, d_in) for multi-channel.
        """
        pass

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample parameters uniformly from ranges."""
        params = {}
        for name, (low, high) in self.config.param_ranges.items():
            params[name] = rng.uniform(low, high)
        return params

    def get_spatial_grid(self) -> np.ndarray:
        """Return the default spatial grid for this PDE family."""
        a, b = self.config.domain
        # Periodic domain: exclude right endpoint
        return np.linspace(a, b, self.config.n_spatial, endpoint=False)


# Registry of all PDE families (populated by individual modules)
PDE_FAMILIES: Dict[str, type] = {}


def get_pde_family(name: str) -> PDEFamily:
    """Get a PDE family instance by name."""
    if name not in PDE_FAMILIES:
        # Trigger imports of family modules to populate the registry
        _import_all_families()

    if name not in PDE_FAMILIES:
        raise ValueError(
            f"Unknown PDE family: {name}. Available: {list(PDE_FAMILIES.keys())}"
        )
    return PDE_FAMILIES[name]()


def _import_all_families():
    """Import all PDE family modules to populate the registry."""
    # These imports trigger registration via module-level code
    from . import burgers
    from . import kuramoto_sivashinsky
    from . import helmholtz
    from . import wave
