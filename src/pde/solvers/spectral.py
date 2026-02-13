"""
ETDRK4 Pseudospectral Solver for 1D Periodic PDEs

Implements the Exponential Time Differencing Runge-Kutta 4th order scheme
from Kassam & Trefethen, SIAM J. Sci. Comput. 26(4), 2005.

The method solves equations of the form:
    u_hat_t = L * u_hat + N_hat(u, t)

where L is the diagonal linear operator in Fourier space and N_hat is the
nonlinear term (computed in physical space, transformed to Fourier space).

The contour integral formula is used to compute the ETDRK4 coefficients
for numerical stability near zero eigenvalues.
"""

import numpy as np
from typing import Callable, Tuple, Optional


def etdrk4_coefficients(
    L: np.ndarray,
    dt: float,
    M: int = 64,
) -> Tuple[np.ndarray, ...]:
    """
    Precompute ETDRK4 coefficient arrays using contour integral evaluation
    for numerical stability.

    Reference: Kassam & Trefethen (2005), Eq. (26)-(29).

    The key insight: the expressions (e^z - 1)/z, etc. are entire functions
    but direct evaluation is numerically unstable near z=0. The contour
    integral approach evaluates them stably for all z.

    Args:
        L: Linear operator (diagonal in Fourier space), shape (N,).
           Complex array of eigenvalues.
        dt: Time step size.
        M: Number of quadrature points on the contour (default 64).

    Returns:
        Tuple of (E, E2, Q, f1, f2, f3) where:
        - E  = exp(L*dt)          : propagator for full step
        - E2 = exp(L*dt/2)        : propagator for half step
        - Q  = dt * phi_1(L*dt/2) : coefficient for RK substeps
        - f1 = dt * phi_1(L*dt) - 3*phi_2(L*dt) + 4*phi_3(L*dt)
        - f2 = dt * 2*phi_2(L*dt) - 4*phi_3(L*dt)
        - f3 = dt * -phi_2(L*dt) + 4*phi_3(L*dt)

        where phi_1(z) = (e^z - 1)/z
              phi_2(z) = (e^z - 1 - z)/z^2
              phi_3(z) = (e^z - 1 - z - z^2/2)/z^3
    """
    N = len(L)

    # Contour: circle of radius 1 around each L*dt value
    theta = np.linspace(0, 2 * np.pi, M + 1)[:-1]  # M points on unit circle
    z_circle = np.exp(1j * theta)  # shape (M,)

    # L*dt values, shape (N,)
    Ldt = L * dt
    Ldt_half = L * dt / 2.0

    # For each eigenvalue, evaluate on a circle around it
    # LR shape: (N, M) - contour points for full step
    LR = Ldt[:, None] + z_circle[None, :]

    # LR2 shape: (N, M) - contour points for half step
    LR2 = Ldt_half[:, None] + z_circle[None, :]

    # Exact propagators (no contour needed for these)
    E = np.exp(Ldt)
    E2 = np.exp(Ldt_half)

    # Q = dt * phi_1(L*dt/2)  =  dt * mean_contour[ (e^z - 1) / z ] at z = L*dt/2
    # But since phi_1(z) = (e^z - 1)/z, we evaluate on the contour:
    Q = dt * np.real(np.mean((np.exp(LR2) - 1.0) / LR2, axis=1))

    # For f1, f2, f3 we need phi_1, phi_2, phi_3 at z = L*dt
    exp_LR = np.exp(LR)

    # phi_1(z) = (e^z - 1) / z
    # phi_2(z) = (e^z - 1 - z) / z^2
    # phi_3(z) = (e^z - 1 - z - z^2/2) / z^3

    # f1 = dt * [ phi_1 - 3*phi_2 + 4*phi_3 ]
    #    = dt * mean[ (e^z - 1)/z - 3*(e^z - 1 - z)/z^2 + 4*(e^z - 1 - z - z^2/2)/z^3 ]
    # Simplify: combine into single expression evaluated on contour
    # f1 = mean[ (-4 - z + e^z * (4 - 3z + z^2)) / z^3 ] * dt
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + exp_LR * (4.0 - 3.0 * LR + LR**2)) / LR**3,
        axis=1,
    ))

    # f2 = dt * [ 2*phi_2 - 4*phi_3 ]
    #    = mean[ (2 + z + e^z * (-2 + z)) ... ] but let's use direct form:
    # f2 = dt * mean[ 2*(e^z - 1 - z)/z^2 - 4*(e^z - 1 - z - z^2/2)/z^3 ]
    #    = dt * mean[ (2 + z + e^z*(-2 + z)) / z^3 ]  -- after algebra
    #    Verify: 2*(e^z-1-z)/z^2 = (2e^z - 2 - 2z)/z^2
    #            4*(e^z-1-z-z^2/2)/z^3 = (4e^z - 4 - 4z - 2z^2)/z^3
    #    Combined: (2e^z-2-2z)/z^2 - (4e^z-4-4z-2z^2)/z^3
    #            = [(2e^z-2-2z)*z - 4e^z+4+4z+2z^2] / z^3
    #            = [2ze^z - 2z - 2z^2 - 4e^z + 4 + 4z + 2z^2] / z^3
    #            = [2ze^z + 2z - 4e^z + 4] / z^3
    #            = [2(z-2)e^z + 2z + 4] / z^3
    #            = 2*[(z-2)e^z + z + 2] / z^3
    f2 = dt * np.real(np.mean(
        2.0 * ((LR - 2.0) * exp_LR + LR + 2.0) / LR**3,
        axis=1,
    ))

    # f3 = dt * [ -phi_2 + 4*phi_3 ]
    #    = dt * mean[ -(e^z-1-z)/z^2 + 4*(e^z-1-z-z^2/2)/z^3 ]
    #    Combine: [-(e^z-1-z)*z + 4(e^z-1-z-z^2/2)] / z^3
    #           = [-ze^z+z+z^2 + 4e^z-4-4z-2z^2] / z^3
    #           = [(-z+4)e^z - z^2 - 3z - 4] / z^3
    #           = [(4-z)e^z - z^2 - 3z - 4] / z^3
    #    Double check with alternate:
    #           = [-e^z*z + z + z^2 + 4e^z - 4 - 4z - 2z^2] / z^3
    #           = [(4-z)e^z - z^2 - 3z - 4] / z^3
    f3 = dt * np.real(np.mean(
        ((4.0 - LR) * exp_LR - LR**2 - 3.0 * LR - 4.0) / LR**3,
        axis=1,
    ))

    return E, E2, Q, f1, f2, f3


class SpectralSolver1D:
    """
    Pseudospectral solver for 1D periodic PDEs using ETDRK4 time stepping.

    Handles PDEs of the form:
        u_t = L[u] + N(u)

    where L is a linear differential operator (handled exactly in Fourier
    space via the exponential integrator) and N(u) is the nonlinear part
    (evaluated in physical space, dealiased via the 3/2 rule).
    """

    def __init__(self, N: int, L: float, dt: float):
        """
        Initialize the spectral solver.

        Args:
            N: Number of spatial grid points (should be even, ideally power of 2).
            L: Domain length [0, L) with periodic BCs.
            dt: Time step size.
        """
        self.N = N
        self.L = L
        self.dt = dt

        # Spatial grid (exclude right endpoint for periodicity)
        self.x = np.linspace(0, L, N, endpoint=False)
        self.dx = L / N

        # Wavenumbers for rfft (real FFT)
        # For a domain of length L, wavenumber k_j = 2*pi*j/L
        # rfft returns N//2+1 complex coefficients
        self.k = np.fft.rfftfreq(N, d=L / (2 * np.pi * N))
        # Correct: rfftfreq(N, d=dx) gives frequencies in cycles/sample.
        # We want angular wavenumbers: k = 2*pi*freq/L * N... let's be precise.
        # Actually: np.fft.rfftfreq(N, d=1/N) gives [0, 1, 2, ..., N/2]
        # and we want k = 2*pi*j/L for j = 0, 1, ..., N/2.
        self.k = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)

        self.n_modes = len(self.k)  # N//2 + 1

        # Dealiasing mask (2/3 rule): zero out top 1/3 of modes
        self.dealias_mask = np.ones(self.n_modes, dtype=float)
        k_max = N // 3  # 2/3 * N/2 = N/3
        self.dealias_mask[k_max + 1:] = 0.0

    def forward_transform(self, u: np.ndarray) -> np.ndarray:
        """Transform physical-space field to Fourier space (rfft)."""
        return np.fft.rfft(u)

    def inverse_transform(self, u_hat: np.ndarray) -> np.ndarray:
        """Transform Fourier-space field to physical space (irfft)."""
        return np.fft.irfft(u_hat, n=self.N)

    def dealias(self, u_hat: np.ndarray) -> np.ndarray:
        """Apply 2/3-rule dealiasing by zeroing high-frequency modes."""
        return u_hat * self.dealias_mask

    def etdrk4_step(
        self,
        u_hat: np.ndarray,
        E: np.ndarray,
        E2: np.ndarray,
        Q: np.ndarray,
        f1: np.ndarray,
        f2: np.ndarray,
        f3: np.ndarray,
        nonlinear_func: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Perform one ETDRK4 time step.

        Args:
            u_hat: Current solution in Fourier space, shape (n_modes,).
            E, E2, Q, f1, f2, f3: Precomputed ETDRK4 coefficients.
            nonlinear_func: Function that takes u_hat and returns N_hat(u).
                Must handle dealiasing internally or accept dealiased input.

        Returns:
            Updated u_hat after one time step.
        """
        Nu_a = nonlinear_func(u_hat)

        a = E2 * u_hat + Q * Nu_a
        Na = nonlinear_func(a)

        b = E2 * u_hat + Q * Na
        Nb = nonlinear_func(b)

        c = E2 * a + Q * (2.0 * Nb - Nu_a)
        Nc = nonlinear_func(c)

        u_hat_new = E * u_hat + f1 * Nu_a + 2.0 * f2 * (Na + Nb) + f3 * Nc

        return u_hat_new

    def solve(
        self,
        u0: np.ndarray,
        T: float,
        linear_op: np.ndarray,
        nonlinear_rhs: Callable[[np.ndarray], np.ndarray],
        n_snapshots: int = 0,
        dealias_output: bool = True,
    ) -> np.ndarray:
        """
        Integrate from u0 to time T using ETDRK4.

        Args:
            u0: Initial condition in physical space, shape (N,).
            T: Final time.
            linear_op: Linear operator eigenvalues in Fourier space, shape (n_modes,).
                For example, -nu * k^2 for diffusion.
            nonlinear_rhs: Function mapping u_hat -> N_hat(u).
                Should compute the nonlinear RHS in Fourier space.
                The function receives u_hat and should:
                1. Transform to physical space
                2. Compute nonlinear terms
                3. Transform back to Fourier space
                4. Apply dealiasing
            n_snapshots: If > 0, return n_snapshots equally spaced snapshots
                including the final time. If 0, return only final solution.
            dealias_output: Whether to dealias the final output.

        Returns:
            If n_snapshots == 0: final solution in physical space, shape (N,).
            If n_snapshots > 0: solution snapshots, shape (n_snapshots, N).
        """
        dt = self.dt
        n_steps = int(np.ceil(T / dt))
        # Adjust dt slightly so we hit T exactly
        dt_actual = T / n_steps

        # Precompute ETDRK4 coefficients with actual dt
        E, E2, Q, f1, f2, f3 = etdrk4_coefficients(linear_op, dt_actual)

        # Initial transform
        u_hat = self.forward_transform(u0)

        # Snapshot storage
        if n_snapshots > 0:
            snapshot_interval = max(1, n_steps // n_snapshots)
            snapshots = []

        # Time stepping
        for step in range(n_steps):
            u_hat = self.etdrk4_step(u_hat, E, E2, Q, f1, f2, f3, nonlinear_rhs)

            if n_snapshots > 0 and (step + 1) % snapshot_interval == 0:
                u_phys = self.inverse_transform(u_hat)
                snapshots.append(u_phys.copy())

        # Final solution
        if dealias_output:
            u_hat = self.dealias(u_hat)

        u_final = self.inverse_transform(u_hat)

        if n_snapshots > 0:
            # Ensure final snapshot is included
            if len(snapshots) < n_snapshots:
                snapshots.append(u_final.copy())
            return np.array(snapshots[-n_snapshots:])

        return u_final
