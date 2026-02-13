"""
Burgers' Equation PDE Family

Equation: u_t + u * u_x = nu * u_xx  on periodic [0, 2*pi]

This is the viscous Burgers equation, a canonical nonlinear PDE that
develops sharp gradients (shocks in the inviscid limit). The operator
learning task maps initial condition u(x, 0) -> u(x, T) for varying
viscosity nu.

Linear part in Fourier space: -nu * k^2 (diffusion)
Nonlinear part: -0.5 * d/dx(u^2) = -0.5 * ik * FFT(u^2) with dealiasing
"""

import numpy as np
from typing import Dict

from .families import PDEFamily, PDEConfig, PDE_FAMILIES
from .solvers.spectral import SpectralSolver1D


class BurgersPDE(PDEFamily):
    """
    Viscous Burgers equation on periodic domain [0, 2*pi].

    u_t + u * u_x = nu * u_xx

    Operator: u_0(x) -> u(x, T) given viscosity nu.
    """

    def __init__(self):
        config = PDEConfig(
            name="burgers",
            spatial_dim=1,
            n_spatial=256,
            n_time=0,             # We only predict the final-time snapshot
            domain=(0.0, 2.0 * np.pi),
            T=1.0,
            state_dim=1,
            param_names=["nu"],
            param_ranges={"nu": (1e-3, 1e-1)},
            input_type="initial_condition",
        )
        super().__init__(config)

    def solve(
        self,
        input_func: np.ndarray,
        x_grid: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        Solve Burgers equation using ETDRK4 pseudospectral method.

        Args:
            input_func: Initial condition u(x, 0), shape (N,).
            x_grid: Spatial grid, shape (N,).
            params: {"nu": viscosity}.

        Returns:
            Solution u(x, T) at final time, shape (N,).
        """
        nu = params["nu"]
        N = len(x_grid)
        L_domain = x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])  # Full period

        # Choose dt based on viscosity and CFL-like considerations
        # For stiff problems (small nu), we rely on ETDRK4's implicit handling
        # of the linear part. The nonlinear CFL gives dt ~ dx / max(|u|).
        dx = L_domain / N
        u_max = max(np.max(np.abs(input_func)), 1.0)
        dt = min(0.5 * dx / u_max, 0.01)
        dt = min(dt, 1e-2)  # Cap for safety

        solver = SpectralSolver1D(N, L_domain, dt)

        # Linear operator: -nu * k^2
        linear_op = -nu * solver.k**2

        # Nonlinear RHS: -0.5 * ik * FFT(u^2)
        def nonlinear_rhs(u_hat):
            # Dealias before computing nonlinear term
            u_hat_d = solver.dealias(u_hat)
            u = solver.inverse_transform(u_hat_d)
            u_sq = u**2
            u_sq_hat = solver.forward_transform(u_sq)
            # -0.5 * d/dx(u^2) in Fourier: -0.5 * ik * FFT(u^2)
            return solver.dealias(-0.5j * solver.k * u_sq_hat)

        u_final = solver.solve(input_func, self.config.T, linear_op, nonlinear_rhs)

        return u_final

    def sample_input_function(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random initial condition as a Fourier series.

        Uses 3-8 modes with 1/k amplitude decay to produce smooth,
        non-trivial initial conditions that lead to interesting dynamics.

        Args:
            rng: NumPy random generator.
            x_grid: Spatial grid, shape (N,).

        Returns:
            Initial condition u(x, 0), shape (N,).
        """
        L = x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])
        n_modes = rng.integers(3, 9)  # 3 to 8 modes

        u = np.zeros_like(x_grid)

        for mode in range(1, n_modes + 1):
            # 1/k amplitude decay
            amplitude = 1.0 / mode
            ak = rng.uniform(-amplitude, amplitude)
            bk = rng.uniform(-amplitude, amplitude)
            u += ak * np.cos(2 * np.pi * mode * x_grid / L)
            u += bk * np.sin(2 * np.pi * mode * x_grid / L)

        # Add a small constant offset
        u += rng.uniform(-0.5, 0.5)

        return u

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample viscosity nu, log-uniform for better coverage of low nu."""
        log_nu_min = np.log10(1e-3)
        log_nu_max = np.log10(1e-1)
        log_nu = rng.uniform(log_nu_min, log_nu_max)
        return {"nu": 10.0 ** log_nu}


# Register in the global PDE family registry
PDE_FAMILIES["burgers"] = BurgersPDE
