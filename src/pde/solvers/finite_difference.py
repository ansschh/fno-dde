"""
Finite Difference Solvers for 1D PDEs

Contains:
- leapfrog_wave_1d: Stormer-Verlet (leapfrog) for the wave equation u_tt = c(x)^2 u_xx
- helmholtz_solve_1d: Sparse direct solve for u_xx + k^2 n(x)^2 u = f(x)
"""

import numpy as np
from typing import Optional
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def leapfrog_wave_1d(
    u0: np.ndarray,
    v0: np.ndarray,
    c_field: np.ndarray,
    x_grid: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """
    Stormer-Verlet (leapfrog) scheme for the 1D wave equation:
        u_tt = c(x)^2 * u_xx

    with periodic boundary conditions.

    The scheme is second-order in both space and time:
        u^{n+1} = 2*u^n - u^{n-1} + dt^2 * c(x)^2 * D_xx u^n

    where D_xx is the standard second-order finite difference Laplacian
    for periodic domains.

    For initialization, we use the Taylor expansion:
        u^1 = u^0 + dt * v0 + 0.5 * dt^2 * c^2 * D_xx(u^0)

    Args:
        u0: Initial displacement, shape (N,).
        v0: Initial velocity, shape (N,).
        c_field: Spatially varying wave speed, shape (N,).
        x_grid: Spatial grid, shape (N,). Assumed uniform and periodic.
        dt: Time step size. Should satisfy CFL: dt < dx / max(c).
        n_steps: Number of time steps to take.

    Returns:
        u_final: Solution at final time, shape (N,).
    """
    N = len(u0)
    dx = x_grid[1] - x_grid[0]

    # CFL check
    cfl = np.max(np.abs(c_field)) * dt / dx
    if cfl > 1.0:
        import warnings
        warnings.warn(
            f"CFL number {cfl:.3f} > 1. Leapfrog may be unstable. "
            f"Consider reducing dt or increasing spatial resolution."
        )

    c2 = c_field ** 2  # c(x)^2

    def laplacian_periodic(u):
        """Second-order centered finite difference Laplacian, periodic BC."""
        # u_{i+1} - 2*u_i + u_{i-1}
        return (np.roll(u, -1) + np.roll(u, 1) - 2.0 * u) / dx**2

    # Initialize with Taylor expansion for first step
    Lxx_u0 = laplacian_periodic(u0)
    u_prev = u0.copy()
    u_curr = u0 + dt * v0 + 0.5 * dt**2 * c2 * Lxx_u0

    # Leapfrog iteration
    for _ in range(1, n_steps):
        Lxx_u = laplacian_periodic(u_curr)
        u_next = 2.0 * u_curr - u_prev + dt**2 * c2 * Lxx_u
        u_prev = u_curr
        u_curr = u_next

    return u_curr


def helmholtz_solve_1d(
    k: float,
    n_field: np.ndarray,
    f_source: np.ndarray,
    x_grid: np.ndarray,
    bc_type: str = "periodic",
) -> np.ndarray:
    """
    Solve the 1D Helmholtz equation:
        u_xx + k^2 * n(x)^2 * u = f(x)

    using second-order finite differences and a sparse direct solver.

    For periodic BC, we use a circulant Laplacian matrix. For Dirichlet,
    we set u(0) = u(L) = 0.

    The discretized system is:
        (D_xx + k^2 * diag(n^2)) * u = f

    where D_xx is the finite difference Laplacian matrix.

    Note: For large k, the system can become ill-conditioned near
    resonances. We add a small imaginary part to k to regularize
    (limiting absorption principle).

    Args:
        k: Wavenumber (positive real).
        n_field: Refractive index field, shape (N,).
        f_source: Source term f(x), shape (N,).
        x_grid: Spatial grid, shape (N,). Assumed uniform.
        bc_type: Boundary condition type. "periodic" or "dirichlet".

    Returns:
        u: Solution field, shape (N,). Real part of the solution.
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]

    # Build Laplacian matrix (sparse)
    # D_xx[i,i] = -2/dx^2, D_xx[i,i+1] = D_xx[i,i-1] = 1/dx^2
    diag_main = np.full(N, -2.0 / dx**2)
    diag_off = np.full(N - 1, 1.0 / dx**2)

    if bc_type == "periodic":
        # Circulant: wrap-around entries
        D_xx = sp.diags(
            [diag_off, diag_main, diag_off],
            [-1, 0, 1],
            shape=(N, N),
            format="lil",
        )
        # Add periodic wrap-around
        D_xx[0, N - 1] = 1.0 / dx**2
        D_xx[N - 1, 0] = 1.0 / dx**2
        D_xx = D_xx.tocsc()
    elif bc_type == "dirichlet":
        D_xx = sp.diags(
            [diag_off, diag_main, diag_off],
            [-1, 0, 1],
            shape=(N, N),
            format="csc",
        )
    else:
        raise ValueError(f"Unknown BC type: {bc_type}. Use 'periodic' or 'dirichlet'.")

    # Helmholtz operator: D_xx + k^2 * diag(n^2)
    # Add small imaginary part for regularization (limiting absorption)
    k_reg = k + 1e-4j
    n2_diag = sp.diags(n_field**2, 0, shape=(N, N), format="csc")
    A = D_xx + k_reg**2 * n2_diag

    # Solve A * u = f
    # Use complex arithmetic, return real part
    f_complex = f_source.astype(complex)
    u_complex = spla.spsolve(A, f_complex)

    return np.real(u_complex)
