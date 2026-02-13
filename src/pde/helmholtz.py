"""
Helmholtz Equation PDE Family

Equation: u_xx + k^2 * n(x)^2 * u = f(x)  on periodic [0, 2*pi]

This is a steady-state (time-independent) PDE. The operator learning task
maps the pair (n(x), f(x)) -> u(x) for varying wavenumber k.

n(x) is the refractive index field (smooth, centered at 1.0).
f(x) is the source term (localized Gaussian).

The Helmholtz equation arises in acoustics, electromagnetics, and quantum
mechanics. For large k, the solution is highly oscillatory and the problem
becomes increasingly ill-conditioned near resonances.
"""

import numpy as np
from typing import Dict

from .families import PDEFamily, PDEConfig, PDE_FAMILIES
from .solvers.finite_difference import helmholtz_solve_1d


class HelmholtzPDE(PDEFamily):
    """
    1D Helmholtz equation on periodic domain [0, 2*pi].

    u_xx + k^2 * n(x)^2 * u = f(x)

    Input: 2-channel field (n(x), f(x))
    Output: u(x)
    Parameter: k (wavenumber)

    Uses n_spatial=512 to ensure >= 10 points per wavelength at k=50
    (wavelength ~ 2*pi/50 ~ 0.126, dx = 2*pi/512 ~ 0.012, so ~10 pts).
    """

    def __init__(self):
        config = PDEConfig(
            name="helmholtz",
            spatial_dim=1,
            n_spatial=512,
            n_time=0,              # Steady-state
            domain=(0.0, 2.0 * np.pi),
            T=0.0,                 # No time evolution
            state_dim=1,
            param_names=["k"],
            param_ranges={"k": (5.0, 50.0)},
            input_type="coefficient_field",
        )
        super().__init__(config)

    def solve(
        self,
        input_func: np.ndarray,
        x_grid: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        Solve the Helmholtz equation using finite differences.

        Args:
            input_func: 2-channel input (n(x), f(x)), shape (N, 2).
            x_grid: Spatial grid, shape (N,).
            params: {"k": wavenumber}.

        Returns:
            Solution u(x), shape (N,).
        """
        k = params["k"]
        n_field = input_func[:, 0]
        f_source = input_func[:, 1]

        u = helmholtz_solve_1d(
            k=k,
            n_field=n_field,
            f_source=f_source,
            x_grid=x_grid,
            bc_type="periodic",
        )

        return u

    def sample_input_function(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random (n(x), f(x)) pair.

        n(x): Smooth random Fourier field with 3-5 modes, centered at 1.0.
            n(x) = 1 + epsilon * sum_k (a_k cos + b_k sin) / k
            where epsilon ~ 0.1-0.3 to keep n(x) > 0.

        f(x): Localized Gaussian source.
            f(x) = A * exp(-(x - x0)^2 / (2 * sigma^2))
            with random center x0, width sigma, and amplitude A.

        Args:
            rng: NumPy random generator.
            x_grid: Spatial grid, shape (N,).

        Returns:
            (n(x), f(x)) stacked as shape (N, 2).
        """
        N = len(x_grid)
        L = x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])

        # --- Refractive index n(x) ---
        n_modes = rng.integers(3, 6)  # 3 to 5 modes
        epsilon = rng.uniform(0.1, 0.3)

        n_field = np.ones(N)  # Centered at 1.0
        for mode in range(1, n_modes + 1):
            ak = rng.standard_normal() / mode
            bk = rng.standard_normal() / mode
            n_field += epsilon * (
                ak * np.cos(2 * np.pi * mode * x_grid / L)
                + bk * np.sin(2 * np.pi * mode * x_grid / L)
            )

        # Ensure n(x) > 0 (should be, but clip for safety)
        n_field = np.maximum(n_field, 0.1)

        # --- Source term f(x) ---
        # Localized Gaussian source
        x0 = rng.uniform(x_grid[0] + 0.5, x_grid[-1] - 0.5)
        sigma = rng.uniform(0.1, 0.5)
        amplitude = rng.uniform(0.5, 5.0)

        # For periodic domain, use periodic Gaussian
        # (approximate by wrapping contributions)
        f_source = np.zeros(N)
        for shift in [-L, 0.0, L]:
            f_source += amplitude * np.exp(
                -0.5 * ((x_grid - x0 - shift) / sigma) ** 2
            )

        # Stack as 2-channel input
        return np.stack([n_field, f_source], axis=-1)

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample wavenumber k, biased toward moderate values for stability."""
        # Log-uniform sampling gives better coverage across scales
        log_k_min = np.log10(5.0)
        log_k_max = np.log10(50.0)
        log_k = rng.uniform(log_k_min, log_k_max)
        return {"k": 10.0 ** log_k}


# Register in the global PDE family registry
PDE_FAMILIES["helmholtz"] = HelmholtzPDE
