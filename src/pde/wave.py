"""
Heterogeneous Wave Equation PDE Family

Equation: u_tt = c(x)^2 * u_xx  on periodic [0, 2*pi]

The operator learning task maps the 3-channel input (u0(x), v0(x), c(x))
to u(x, T), where u0 is the initial displacement, v0 is the initial
velocity, and c(x) is the spatially varying wave speed.

This tests the network's ability to handle:
- Wave propagation and interference
- Spatially varying media (heterogeneous c(x))
- Multiple input channels with different physical meanings
"""

import numpy as np
from typing import Dict

from .families import PDEFamily, PDEConfig, PDE_FAMILIES
from .solvers.finite_difference import leapfrog_wave_1d


class HeterogeneousWavePDE(PDEFamily):
    """
    Heterogeneous 1D wave equation on periodic domain [0, 2*pi].

    u_tt = c(x)^2 * u_xx

    Input: 3-channel field (u0(x), v0(x), c(x))
    Output: u(x, T)
    Parameters: T (final time), c_contrast (controls wave speed variation)
    """

    def __init__(self):
        config = PDEConfig(
            name="wave",
            spatial_dim=1,
            n_spatial=256,
            n_time=0,
            domain=(0.0, 2.0 * np.pi),
            T=1.0,  # Placeholder; actual T sampled per instance
            state_dim=1,
            param_names=["T", "c_contrast"],
            param_ranges={
                "T": (1.0, 5.0),
                "c_contrast": (1.0, 3.0),
            },
            input_type="multi_channel",
        )
        super().__init__(config)

    def solve(
        self,
        input_func: np.ndarray,
        x_grid: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        Solve the wave equation using leapfrog (Stormer-Verlet) scheme.

        Args:
            input_func: 3-channel input (u0, v0, c), shape (N, 3).
            x_grid: Spatial grid, shape (N,).
            params: {"T": final_time, "c_contrast": speed_contrast}.

        Returns:
            Solution u(x, T), shape (N,).
        """
        T = params["T"]
        u0 = input_func[:, 0]
        v0 = input_func[:, 1]
        c_field = input_func[:, 2]

        N = len(x_grid)
        dx = x_grid[1] - x_grid[0]

        # CFL condition: dt < dx / max(c)
        c_max = np.max(np.abs(c_field))
        dt = 0.8 * dx / c_max  # Safety factor 0.8

        n_steps = int(np.ceil(T / dt))
        # Adjust dt to hit T exactly
        dt = T / n_steps

        u_final = leapfrog_wave_1d(
            u0=u0,
            v0=v0,
            c_field=c_field,
            x_grid=x_grid,
            dt=dt,
            n_steps=n_steps,
        )

        return u_final

    def sample_input_function(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
        c_contrast: float = None,
    ) -> np.ndarray:
        """
        Sample random (u0, v0, c) as 3-channel input.

        u0, v0: Random Fourier series with 5-10 modes.
        c(x): 1 + contrast * smooth_variation(x), where smooth_variation
            is a piecewise-smooth field built from a few Fourier modes.

        Args:
            rng: NumPy random generator.
            x_grid: Spatial grid, shape (N,).
            c_contrast: Wave speed contrast. If None, uses a default of 1.5.

        Returns:
            (u0, v0, c) stacked as shape (N, 3).
        """
        N = len(x_grid)
        L = x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])

        if c_contrast is None:
            c_contrast = 1.5

        # --- Initial displacement u0(x) ---
        u0 = self._random_fourier(rng, x_grid, L, n_mode_range=(5, 11))

        # --- Initial velocity v0(x) ---
        v0 = self._random_fourier(rng, x_grid, L, n_mode_range=(5, 11))
        v0 *= 0.5  # Slightly smaller amplitude for velocity

        # --- Wave speed c(x) ---
        # c(x) = 1 + c_contrast * smooth_variation(x)
        # smooth_variation is normalized to [-0.5, 0.5] so c > 0 always
        n_c_modes = rng.integers(2, 6)
        variation = np.zeros(N)
        for mode in range(1, n_c_modes + 1):
            ak = rng.standard_normal() / mode
            bk = rng.standard_normal() / mode
            variation += (
                ak * np.cos(2 * np.pi * mode * x_grid / L)
                + bk * np.sin(2 * np.pi * mode * x_grid / L)
            )

        # Normalize variation to [-0.5, 0.5]
        v_range = np.max(variation) - np.min(variation)
        if v_range > 1e-10:
            variation = (variation - np.min(variation)) / v_range - 0.5

        # c(x) = 1 + contrast * variation, so c in [1 - 0.5*contrast, 1 + 0.5*contrast]
        # With contrast in [1, 3], c is in [0.5, 2.5] at worst -- always positive
        c_field = 1.0 + c_contrast * variation

        # Safety: ensure c > 0
        c_field = np.maximum(c_field, 0.1)

        return np.stack([u0, v0, c_field], axis=-1)

    def _random_fourier(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
        L: float,
        n_mode_range: tuple = (5, 11),
    ) -> np.ndarray:
        """Generate a random Fourier series on x_grid."""
        n_modes = rng.integers(*n_mode_range)
        u = np.zeros_like(x_grid)

        for mode in range(1, n_modes + 1):
            amplitude = 1.0 / mode
            ak = rng.uniform(-amplitude, amplitude)
            bk = rng.uniform(-amplitude, amplitude)
            u += ak * np.cos(2 * np.pi * mode * x_grid / L)
            u += bk * np.sin(2 * np.pi * mode * x_grid / L)

        # Add small constant
        u += rng.uniform(-0.3, 0.3)

        return u


# Register in the global PDE family registry
PDE_FAMILIES["wave"] = HeterogeneousWavePDE
