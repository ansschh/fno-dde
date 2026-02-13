"""
Kuramoto-Sivashinsky (KS) Equation PDE Family

Equation: u_t + u * u_x + u_xx + u_xxxx = 0  on periodic [0, L]

Equivalently: u_t = -u * u_x - u_xx - u_xxxx

This is a prototypical equation for spatiotemporal chaos. The u_xx term
is destabilizing (negative diffusion) while the u_xxxx term provides
short-wavelength stabilization. The parameter L controls the number of
unstable modes and hence the complexity of the dynamics.

Linear part in Fourier space:
  The equation u_t = -u*u_x - u_xx - u_xxxx has linear terms -u_xx - u_xxxx.
  In Fourier space: -(-k^2) - (k^4) = k^2 - k^4
  So the linear operator eigenvalues are: L_k = k^2 - k^4

Nonlinear part: -u * u_x = -0.5 * d/dx(u^2) = -0.5 * ik * FFT(u^2)
"""

import numpy as np
from typing import Dict

from .families import PDEFamily, PDEConfig, PDE_FAMILIES
from .solvers.spectral import SpectralSolver1D


class KuramotoSivashinskyPDE(PDEFamily):
    """
    Kuramoto-Sivashinsky equation on periodic domain [0, L].

    u_t + u * u_x + u_xx + u_xxxx = 0

    The parameter L (domain length) controls the level of chaos:
    - L ~ 22: onset of chaos (1-2 unstable modes)
    - L ~ 50-100: well-developed spatiotemporal chaos

    Operator: u_0(x) -> u(x, T) given domain length L.
    """

    def __init__(self):
        config = PDEConfig(
            name="ks",
            spatial_dim=1,
            n_spatial=256,
            n_time=0,
            domain=(0.0, 1.0),  # Placeholder; actual domain depends on L
            T=1.0,              # Placeholder; set per-sample as L/4
            state_dim=1,
            param_names=["L"],
            param_ranges={"L": (22.0, 100.0)},
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
        Solve KS equation using ETDRK4 pseudospectral method.

        Args:
            input_func: Initial condition u(x, 0), shape (N,).
            x_grid: Spatial grid, shape (N,).
            params: {"L": domain_length}.

        Returns:
            Solution u(x, T) at final time, shape (N,).
        """
        L_domain = params["L"]
        N = len(x_grid)
        T = L_domain / 4.0  # Reasonable integration time

        # dt for KS: the k^4 term is handled implicitly by ETDRK4,
        # but the nonlinear CFL gives a constraint.
        # A safe choice for KS with N=256 modes:
        dt = 0.5 * (L_domain / N)  # ~ dx/2
        dt = min(dt, 0.05)  # Cap

        solver = SpectralSolver1D(N, L_domain, dt)

        # Linear operator: L_k = k^2 - k^4
        # From: u_t = -u*u_x - u_xx - u_xxxx
        # In Fourier: hat(u_xx) = -k^2 * hat(u), so -hat(u_xx) = k^2 * hat(u)
        #             hat(u_xxxx) = k^4 * hat(u), so -hat(u_xxxx) = -k^4 * hat(u)
        # Linear part: k^2 - k^4
        k = solver.k
        linear_op = k**2 - k**4

        # Nonlinear RHS: -0.5 * ik * FFT(u^2)
        def nonlinear_rhs(u_hat):
            u_hat_d = solver.dealias(u_hat)
            u = solver.inverse_transform(u_hat_d)
            u_sq = u**2
            u_sq_hat = solver.forward_transform(u_sq)
            return solver.dealias(-0.5j * k * u_sq_hat)

        u_final = solver.solve(input_func, T, linear_op, nonlinear_rhs)

        return u_final

    def sample_input_function(
        self,
        rng: np.random.Generator,
        x_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a small random perturbation as initial condition.

        KS dynamics are largely determined by L and the equation itself;
        the initial condition just needs to seed the instability. We use
        small-amplitude random Fourier modes.

        Args:
            rng: NumPy random generator.
            x_grid: Spatial grid, shape (N,).

        Returns:
            Initial condition u(x, 0), shape (N,).
        """
        L = x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])
        N = len(x_grid)

        # Small amplitude perturbation with a few modes
        u = np.zeros(N)
        n_modes = rng.integers(3, 8)

        for mode in range(1, n_modes + 1):
            amplitude = 0.1 / mode  # Small 1/k decay
            ak = rng.uniform(-amplitude, amplitude)
            bk = rng.uniform(-amplitude, amplitude)
            u += ak * np.cos(2 * np.pi * mode * x_grid / L)
            u += bk * np.sin(2 * np.pi * mode * x_grid / L)

        return u

    def get_spatial_grid(self, L: float = None) -> np.ndarray:
        """Return spatial grid for given domain length L."""
        if L is None:
            L = 50.0  # Default
        return np.linspace(0, L, self.config.n_spatial, endpoint=False)

    def get_T(self, params: Dict[str, float]) -> float:
        """Get integration time for given parameters."""
        return params["L"] / 4.0


# Register in the global PDE family registry
PDE_FAMILIES["ks"] = KuramotoSivashinskyPDE
