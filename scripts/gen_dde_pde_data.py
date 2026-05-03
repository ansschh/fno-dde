"""
DDE-PDE benchmark generator.

Generates numpy-format DDE-PDE training data with explicit delay structure
in the dynamics, suitable for evaluating lag-equivariant operators.

Currently implemented:
  B1: Mackey-Glass + diffusion 2D
        ∂u/∂t = D ∇²u + β u(x, t-τ) / (1 + u(x, t-τ)^n) - γ u(x, t)

Future (Section 1 of FINAL_RESEARCH_PLAN.md):
  B2: Wright + diffusion 2D
  B3: Hutchinson + diffusion 2D
  B4: Distributed-delay reaction-diffusion (2D + 3D)
  B5: Delayed-feedback Burgers 2D
  B6: Ring-coupled Kuramoto 2D

Solver design:
  - Method of steps for the delay term (history buffer indexed by t/dt steps)
  - RK4 time stepping with cubic interpolation of the delayed argument
  - Spectral spatial discretization (FFT-based Laplacian) on torus [0, 2π]^d
  - Periodic boundary conditions (matches FFT requirement)

Output format matches APEBench convention:
  shard_000.npz with keys:
    phi    (N, n_hist, *spatial, C)
    y      (N, n_out, *spatial, C)
    params (N, params_dim)
    t_hist (n_hist,)
    t_out  (n_out,)

Usage:
    python3 scripts/gen_dde_pde_data.py --family mackey_glass_2d \
        --num_train 256 --num_val 64 --num_test 64 \
        --num_points 64 --T 128 --n_hist 64 --n_out 64 \
        --out_dir data_dde_pde
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------
# Spectral spatial helpers (2D, torus)
# ---------------------------------------------------------------------

@dataclass
class SpectralGrid2D:
    """Pre-computed spectral grid for FFT-based ∇² on a 2D torus."""
    n: int                            # grid points per axis
    L: float                          # domain size (default 2π)
    X: np.ndarray                     # (n, n) physical x grid
    Y: np.ndarray                     # (n, n) physical y grid
    KX: np.ndarray                    # (n, n) spectral kx grid
    KY: np.ndarray                    # (n, n) spectral ky grid
    K2: np.ndarray                    # (n, n) kx²+ky² (Laplacian symbol)

    @classmethod
    def make(cls, n: int = 64, L: float = 2 * np.pi) -> "SpectralGrid2D":
        x = np.linspace(0.0, L, n, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        kx_1d = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
        KX, KY = np.meshgrid(kx_1d, kx_1d, indexing="ij")
        K2 = KX * KX + KY * KY
        return cls(n=n, L=L, X=X, Y=Y, KX=KX, KY=KY, K2=K2)


def laplacian2d(u: np.ndarray, grid: SpectralGrid2D) -> np.ndarray:
    """Spectral 2D Laplacian on torus.  u: (n, n)."""
    return np.real(np.fft.ifft2(-(grid.K2) * np.fft.fft2(u)))


# ---------------------------------------------------------------------
# Initial-condition (smooth random fields)
# ---------------------------------------------------------------------

def smooth_random_field_2d(rng: np.random.Generator, grid: SpectralGrid2D,
                            amplitude: float = 1.0, k_max: int = 4,
                            base: float = 1.0) -> np.ndarray:
    """Generate a smooth periodic function u: torus -> R, low-frequency.

    Construction: random Fourier coefficients with |k| <= k_max, IFFT to
    obtain u(x, y) as a real, smooth field.  Add `base` for positivity
    when used as a Mackey-Glass IC (which expects u > 0 typically).
    """
    n = grid.n
    coeffs = np.zeros((n, n), dtype=complex)
    # Build a low-pass random field; for real output, enforce conjugate symmetry.
    for i in range(-k_max, k_max + 1):
        for j in range(-k_max, k_max + 1):
            if i == 0 and j == 0:
                continue
            re = rng.standard_normal()
            im = rng.standard_normal()
            ii = i % n
            jj = j % n
            coeffs[ii, jj] = re + 1j * im
    # Conjugate symmetry: coeffs[-k] = conj(coeffs[k]) to ensure real ifft.
    for i in range(n):
        for j in range(n):
            if (i, j) <= ((n - i) % n, (n - j) % n):
                continue
            coeffs[i, j] = np.conj(coeffs[(n - i) % n, (n - j) % n])
    u = np.real(np.fft.ifft2(coeffs))
    # Normalize to amplitude.
    u = u / (np.abs(u).max() + 1e-12) * amplitude
    return base + u


# ---------------------------------------------------------------------
# B1: Mackey-Glass + diffusion 2D
# ---------------------------------------------------------------------

@dataclass
class MackeyGlassParams:
    """All parameters for a Mackey-Glass + diffusion 2D simulation."""
    beta: float = 2.0          # delayed-source strength
    gamma: float = 1.0         # decay rate
    n: float = 10.0            # Hill exponent (controls nonlinearity)
    tau: float = 2.0           # delay
    D: float = 0.1             # spatial diffusion
    T_total: float = 16.0      # simulation horizon (in same units as tau)
    dt: float = 0.01           # time step; must divide tau evenly; <0.014 for stability at D=0.1, n=64
    n_grid: int = 64
    L: float = 2 * np.pi


def mackey_glass_full_rhs(u: np.ndarray, u_delayed: np.ndarray,
                           p: MackeyGlassParams,
                           grid: SpectralGrid2D) -> np.ndarray:
    """Full RHS: D*laplacian(u) + beta*u_delayed/(1+u_delayed^n) - gamma*u."""
    diffusion = p.D * laplacian2d(u, grid)
    source = p.beta * u_delayed / (1.0 + np.abs(u_delayed) ** p.n)
    decay = p.gamma * u
    return diffusion + source - decay


def simulate_mackey_glass(p: MackeyGlassParams, rng: np.random.Generator,
                           grid: SpectralGrid2D) -> np.ndarray:
    """Simulate one Mackey-Glass + diffusion 2D trajectory.

    Integrator: classical RK4 with cubic Hermite interpolation for the
    delayed argument at intermediate times.

    Stability: explicit RK4 requires `D * k_max^2 * dt < ~2.78`.  For
    `D=0.1, n_grid=64` (k_max^2 = 2*32^2 = 2048), need `dt < 0.0136`.
    We default to `dt = 0.01` which is safely inside this bound.

    Convergence note: an external audit measured EFFECTIVE 1st-order
    convergence in dt for the delayed term (not 4th-order).  Cause: RK4
    stages reuse a frozen history (same hist_arr for k1..k4), and the
    delayed-argument lookup uses linear interpolation (kernel-integral
    families further use trapezoidal quadrature, 2nd-order in s).  The
    benchmark remains internally consistent (all baselines train/test on
    the SAME generated ground-truth), but absolute-accuracy claims must
    describe the integrator as "frozen-history RK4 + trapezoidal kernel
    quadrature, effective 1st-order in dt for delayed term."  Achieving
    true 4th-order requires cubic-Hermite history interpolation + Simpson's
    rule on the kernel integral.
    """
    n_steps = int(round(p.T_total / p.dt))
    history_steps = int(round(p.tau / p.dt))
    if abs(history_steps * p.dt - p.tau) > 1e-9:
        raise ValueError(f"tau={p.tau} not divisible by dt={p.dt}")
    # Stability sanity check (axis-aligned Nyquist mode).
    k_max_sq = 2 * (np.pi * p.n_grid / p.L) ** 2  # 2*kx_max^2 for 2D
    stability_factor = p.D * k_max_sq * p.dt
    if stability_factor > 2.78:
        raise ValueError(
            f"explicit RK4 unstable: D*k^2*dt = {stability_factor:.2f} > 2.78."
            f"  Reduce dt to <= {2.78 / (p.D * k_max_sq):.4f}.")

    # Initial history: smooth random field held constant on [-tau, 0].
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.4, k_max=4, base=0.6)
    history = np.tile(u0[None, ...], (history_steps + 1, 1, 1))

    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0

    def get_delayed(step_real: float) -> np.ndarray:
        """Linear interpolation of u(t - tau) at fractional step.

        step_real: floating-point step index where we want u(step_real * dt).
        """
        if step_real <= 0:
            # Piecewise-constant initial history (handles step_real in [-h, 0]).
            return history[0]
        elif step_real >= step:
            # Should never need this; fallback is the latest computed traj.
            return traj[step]
        else:
            i_low = int(np.floor(step_real))
            frac = step_real - i_low
            return (1.0 - frac) * traj[i_low] + frac * traj[i_low + 1]

    for step in range(n_steps):
        u = traj[step]
        # u(t-tau), u(t+dt/2-tau), u(t+dt-tau)
        ud_n = get_delayed(step - history_steps)
        ud_mid = get_delayed(step + 0.5 - history_steps)
        ud_n1 = get_delayed(step + 1 - history_steps)
        # Classical RK4
        k1 = mackey_glass_full_rhs(u, ud_n, p, grid)
        k2 = mackey_glass_full_rhs(u + 0.5 * p.dt * k1, ud_mid, p, grid)
        k3 = mackey_glass_full_rhs(u + 0.5 * p.dt * k2, ud_mid, p, grid)
        k4 = mackey_glass_full_rhs(u + p.dt * k3, ud_n1, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"NaN/inf at step {step + 1}/{n_steps}")
    return traj


def sample_param_set(rng: np.random.Generator) -> MackeyGlassParams:
    """Sample a parameter regime that lies in the dynamically interesting band.

    For 0D Mackey-Glass at n=10, beta*tau > pi (and gamma small) puts the
    system near the Hopf bifurcation; (beta=2, n=10, gamma=1, tau=2) is a
    documented chaotic regime.  We vary beta and tau across the bifurcation
    boundary; D is held at a single value compatible with explicit RK4
    stability at the reference dt=0.01 (D*k^2*dt < 2.78 requires D < 0.135
    for n_grid=64).
    """
    return MackeyGlassParams(
        beta=float(rng.uniform(1.5, 2.5)),
        gamma=1.0,
        n=10.0,
        tau=float(rng.choice([1.5, 2.0, 2.5])),
        D=float(rng.choice([0.025, 0.05, 0.1])),
    )


# ---------------------------------------------------------------------
# B2: Wright equation + diffusion 2D
#     d/dt u(x, t) = D laplacian(u) - alpha * u(x, t-tau) * (1 + u(x, t))
#
# Wright's classical 0D equation is famous for the Wright conjecture:
# the trivial fixed point u=0 is asymptotically stable iff alpha*tau <=
# 3*pi/2 (proved 2017 by van den Berg-Jaquette).  The Hopf bifurcation
# is at alpha*tau = pi/2.
#
# Coupling to spatial diffusion creates a reaction-diffusion-with-delay
# system whose homogeneous mode obeys the 0D dynamics.
# ---------------------------------------------------------------------

@dataclass
class WrightParams:
    """All parameters for a Wright + diffusion 2D simulation."""
    alpha: float = 1.5         # delayed-source strength
    tau: float = 1.0           # delay
    D: float = 0.05            # spatial diffusion (smaller than B1 to keep stability headroom)
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def wright_full_rhs(u: np.ndarray, u_delayed: np.ndarray,
                     p: WrightParams, grid: SpectralGrid2D) -> np.ndarray:
    """RHS: D laplacian(u) - alpha * u(t-tau) * (1 + u(t))."""
    diffusion = p.D * laplacian2d(u, grid)
    feedback = -p.alpha * u_delayed * (1.0 + u)
    return diffusion + feedback


def simulate_wright(p: WrightParams, rng: np.random.Generator,
                     grid: SpectralGrid2D) -> np.ndarray:
    """Simulate Wright + diffusion 2D using classical RK4 + linear delay interp."""
    n_steps = int(round(p.T_total / p.dt))
    history_steps = int(round(p.tau / p.dt))
    if abs(history_steps * p.dt - p.tau) > 1e-9:
        raise ValueError(f"tau={p.tau} not divisible by dt={p.dt}")
    k_max_sq = 2 * (np.pi * p.n_grid / p.L) ** 2
    sf = p.D * k_max_sq * p.dt
    if sf > 2.78:
        raise ValueError(
            f"explicit RK4 unstable for Wright: D*k^2*dt = {sf:.2f} > 2.78."
            f"  Reduce dt to <= {2.78 / (p.D * k_max_sq):.4f}.")

    # Initial history: small smooth perturbation around 0 (Wright's fixed point).
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.2, k_max=4, base=0.0)
    history = np.tile(u0[None, ...], (history_steps + 1, 1, 1))

    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0

    def get_delayed(step_real: float) -> np.ndarray:
        if step_real <= 0:
            return history[0]
        elif step_real >= step:
            return traj[step]
        else:
            i_low = int(np.floor(step_real))
            frac = step_real - i_low
            return (1.0 - frac) * traj[i_low] + frac * traj[i_low + 1]

    for step in range(n_steps):
        u = traj[step]
        ud_n = get_delayed(step - history_steps)
        ud_mid = get_delayed(step + 0.5 - history_steps)
        ud_n1 = get_delayed(step + 1 - history_steps)
        k1 = wright_full_rhs(u, ud_n, p, grid)
        k2 = wright_full_rhs(u + 0.5 * p.dt * k1, ud_mid, p, grid)
        k3 = wright_full_rhs(u + 0.5 * p.dt * k2, ud_mid, p, grid)
        k4 = wright_full_rhs(u + p.dt * k3, ud_n1, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"Wright NaN/inf at step {step + 1}/{n_steps}")
    return traj


def sample_wright_param_set(rng: np.random.Generator) -> WrightParams:
    """Sample alpha and tau across the alpha*tau ~ pi/2 (Hopf) and 3*pi/2 (Wright)
    bifurcation boundaries -- WITH a safety margin below 3*pi/2 ~ 4.71 to avoid
    blow-up in the strongly-unstable regime (verified empirically: alpha*tau=4
    NaN'd at step 1122/2000).  Range covers stable + oscillatory regimes.
    """
    # alpha in [0.5, 1.75], tau in {1.0, 1.5, 2.0} -> alpha*tau in [0.5, 3.5]
    # which spans the Hopf boundary at pi/2~1.57 with margin below 3*pi/2.
    return WrightParams(
        alpha=float(rng.uniform(0.5, 1.75)),
        tau=float(rng.choice([1.0, 1.5, 2.0])),
        D=float(rng.choice([0.025, 0.05])),
    )


def audit_wright(out_dir: Path, alpha=1.5, tau=1.0, D=0.05,
                  T_total=20.0, dt=0.01, n_grid=64, seed=42):
    """B2 audit: single trajectory + Hopf onset sweep."""
    p = WrightParams(alpha=alpha, tau=tau, D=D, T_total=T_total, dt=dt, n_grid=n_grid)
    grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
    rng = np.random.default_rng(seed)
    print(f"\n=== AUDIT Wright trajectory: alpha={alpha}, tau={tau}, D={D},"
          f" T={T_total}, dt={dt}, grid={n_grid} ===")
    t0 = time.time()
    traj = simulate_wright(p, rng, grid)
    print(f"  wall: {time.time()-t0:.1f}s,  shape: {traj.shape}")
    print(f"  finite: {np.isfinite(traj).all()}")
    print(f"  range: [{traj.min():.4f}, {traj.max():.4f}], mean={traj.mean():.4f}, std={traj.std():.4f}")
    u_bar = traj.mean(axis=(1, 2))
    centered = u_bar - u_bar.mean()
    zc = np.sum(np.diff(np.sign(centered)) != 0)
    print(f"  u_bar(t) range: [{u_bar.min():.4f}, {u_bar.max():.4f}]  zero-crossings={zc}")
    at = alpha * tau
    print(f"  alpha*tau = {at:.3f}  (Hopf at pi/2={np.pi/2:.3f}, Wright bound 3pi/2={3*np.pi/2:.3f})")
    print(f"  predicts: stable for alpha*tau<=pi/2, oscillatory for pi/2<alpha*tau<=3pi/2")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"audit_traj_a{alpha}_t{tau}.npy", traj)
    return traj


def hopf_onset_sweep_wright(out_dir: Path,
                             alphas=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
                             tau=1.0, T_total=20.0, dt=0.01, n_grid=64, seed=42):
    """Vary alpha at fixed tau; observe onset of oscillation past alpha*tau ~ pi/2."""
    print(f"\n=== HOPF ONSET SWEEP Wright (vary alpha, fixed tau={tau}) ===")
    print(f"   {'alpha':>5}  {'a*tau':>7}  {'>pi/2?':>6}  {'osc.amp':>9}  {'mean':>7}")
    for a in alphas:
        p = WrightParams(alpha=a, tau=tau, T_total=T_total, dt=dt, n_grid=n_grid)
        grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
        rng = np.random.default_rng(seed)
        traj = simulate_wright(p, rng, grid)
        u_bar = traj.mean(axis=(1, 2))
        ss = u_bar[len(u_bar) // 2:]
        amp = float(ss.max() - ss.min())
        mean = float(ss.mean())
        at = a * tau
        print(f"   {a:>5.2f}  {at:>7.3f}  {('YES' if at > np.pi/2 else 'no'):>6}"
              f"  {amp:>9.4f}  {mean:>7.4f}")


# ---------------------------------------------------------------------
# B3: Hutchinson's logistic + diffusion 2D
#     d/dt u(x,t) = D laplacian(u) + r * u(x,t) * (1 - u(x,t-tau)/K)
#
# Classical 0D Hutchinson equation: famous logistic-with-delay
# population model.  Linearizing v = u - K around the fixed point K
# gives dv/dt = -r * v(t - tau), the Wright-type linear DDE.  Hopf
# bifurcation at r*tau = pi/2; stable for r*tau < pi/2.
# ---------------------------------------------------------------------

@dataclass
class HutchinsonParams:
    """All parameters for a Hutchinson + diffusion 2D simulation."""
    r: float = 1.0             # intrinsic growth rate
    K: float = 1.0             # carrying capacity
    tau: float = 1.0           # maturation delay
    D: float = 0.05            # spatial diffusion (within stability bound)
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def hutchinson_full_rhs(u: np.ndarray, u_delayed: np.ndarray,
                         p: HutchinsonParams,
                         grid: SpectralGrid2D) -> np.ndarray:
    """RHS: D laplacian(u) + r * u(t) * (1 - u(t-tau)/K)."""
    diffusion = p.D * laplacian2d(u, grid)
    growth = p.r * u * (1.0 - u_delayed / p.K)
    return diffusion + growth


def simulate_hutchinson(p: HutchinsonParams, rng: np.random.Generator,
                         grid: SpectralGrid2D) -> np.ndarray:
    """Simulate Hutchinson + diffusion 2D using classical RK4 + linear delay interp."""
    n_steps = int(round(p.T_total / p.dt))
    history_steps = int(round(p.tau / p.dt))
    if abs(history_steps * p.dt - p.tau) > 1e-9:
        raise ValueError(f"tau={p.tau} not divisible by dt={p.dt}")
    k_max_sq = 2 * (np.pi * p.n_grid / p.L) ** 2
    sf = p.D * k_max_sq * p.dt
    if sf > 2.78:
        raise ValueError(
            f"explicit RK4 unstable for Hutchinson: D*k^2*dt = {sf:.2f} > 2.78."
            f"  Reduce dt to <= {2.78 / (p.D * k_max_sq):.4f}.")

    # Initial history: smooth random field around the carrying capacity K.
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.3 * p.K, k_max=4, base=p.K)
    history = np.tile(u0[None, ...], (history_steps + 1, 1, 1))

    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0

    def get_delayed(step_real: float) -> np.ndarray:
        if step_real <= 0:
            return history[0]
        elif step_real >= step:
            return traj[step]
        else:
            i_low = int(np.floor(step_real))
            frac = step_real - i_low
            return (1.0 - frac) * traj[i_low] + frac * traj[i_low + 1]

    for step in range(n_steps):
        u = traj[step]
        ud_n = get_delayed(step - history_steps)
        ud_mid = get_delayed(step + 0.5 - history_steps)
        ud_n1 = get_delayed(step + 1 - history_steps)
        k1 = hutchinson_full_rhs(u, ud_n, p, grid)
        k2 = hutchinson_full_rhs(u + 0.5 * p.dt * k1, ud_mid, p, grid)
        k3 = hutchinson_full_rhs(u + 0.5 * p.dt * k2, ud_mid, p, grid)
        k4 = hutchinson_full_rhs(u + p.dt * k3, ud_n1, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"Hutchinson NaN/inf at step {step + 1}/{n_steps}")
    return traj


def sample_hutchinson_param_set(rng: np.random.Generator) -> HutchinsonParams:
    """Sample r and tau spanning the r*tau ~ pi/2 (Hopf) bifurcation.

    Hutchinson is robust beyond the Hopf onset (unlike Wright), but very large
    r*tau still causes large transients.  We cap r*tau <= 3.0 for safety.
    """
    return HutchinsonParams(
        r=float(rng.uniform(0.5, 1.5)),
        K=1.0,
        tau=float(rng.choice([1.0, 1.5, 2.0])),
        D=float(rng.choice([0.025, 0.05])),
    )


def audit_hutchinson(out_dir: Path, r=1.0, K=1.0, tau=1.5, D=0.05,
                      T_total=20.0, dt=0.01, n_grid=64, seed=42):
    """B3 audit: single trajectory + Hopf onset sweep."""
    p = HutchinsonParams(r=r, K=K, tau=tau, D=D, T_total=T_total, dt=dt, n_grid=n_grid)
    grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
    rng = np.random.default_rng(seed)
    print(f"\n=== AUDIT Hutchinson trajectory: r={r}, K={K}, tau={tau}, D={D},"
          f" T={T_total}, dt={dt}, grid={n_grid} ===")
    t0 = time.time()
    traj = simulate_hutchinson(p, rng, grid)
    print(f"  wall: {time.time()-t0:.1f}s,  shape: {traj.shape}")
    print(f"  finite: {np.isfinite(traj).all()}")
    print(f"  range: [{traj.min():.4f}, {traj.max():.4f}], mean={traj.mean():.4f}, std={traj.std():.4f}")
    u_bar = traj.mean(axis=(1, 2))
    centered = u_bar - u_bar.mean()
    zc = np.sum(np.diff(np.sign(centered)) != 0)
    print(f"  u_bar(t) range: [{u_bar.min():.4f}, {u_bar.max():.4f}]  zero-crossings={zc}")
    rt = r * tau
    print(f"  r*tau = {rt:.3f}  (Hopf at pi/2={np.pi/2:.3f})")
    print(f"  predicts: stable convergence to K={K} for r*tau<pi/2,"
          f" oscillatory for r*tau>pi/2")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"audit_traj_r{r}_t{tau}.npy", traj)
    return traj


def hopf_onset_sweep_hutchinson(out_dir: Path,
                                 rs=(0.3, 0.6, 1.0, 1.5, 2.0, 2.5),
                                 tau=1.0, T_total=20.0, dt=0.01, n_grid=64, seed=42):
    """Vary r at fixed tau; oscillation onset past r*tau ~ pi/2."""
    print(f"\n=== HOPF ONSET SWEEP Hutchinson (vary r, fixed tau={tau}) ===")
    print(f"   {'r':>5}  {'r*tau':>7}  {'>pi/2?':>6}  {'osc.amp':>9}  {'mean':>7}")
    for r in rs:
        p = HutchinsonParams(r=r, tau=tau, T_total=T_total, dt=dt, n_grid=n_grid)
        grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
        rng = np.random.default_rng(seed)
        try:
            traj = simulate_hutchinson(p, rng, grid)
            u_bar = traj.mean(axis=(1, 2))
            ss = u_bar[len(u_bar) // 2:]
            amp = float(ss.max() - ss.min())
            mean = float(ss.mean())
            rt = r * tau
            print(f"   {r:>5.2f}  {rt:>7.3f}  {('YES' if rt > np.pi/2 else 'no'):>6}"
                  f"  {amp:>9.4f}  {mean:>7.4f}")
        except RuntimeError as e:
            print(f"   {r:>5.2f}  {r*tau:>7.3f}  {'YES':>6}  --BLEW UP-- ({e})")


# ---------------------------------------------------------------------
# B4: Distributed-delay reaction-diffusion 2D
#
#   d/dt u(x, t) = D laplacian(u) + integral_0^tau_max K(s) f(u(x, t-s)) ds
#
# K(s) = (1/tau) * exp(-s/tau) is the exponentially-distributed memory
# kernel.  f(u) = u(1-u) is the logistic source.  Theoretically interesting
# because the integral over the entire past horizon is exactly the
# operator that a lag-equivariant emulator should be ideally suited for.
#
# Numerics: discrete quadrature of the integral via trapezoidal rule over
# the history buffer.  For tau_max = 4*tau (covers >98% of exp kernel
# mass), a buffer of 4*tau/dt entries is sufficient.
# ---------------------------------------------------------------------

@dataclass
class DistDelayRDParams:
    """All parameters for distributed-delay reaction-diffusion 2D."""
    A: float = 1.0             # source amplitude (rescales reaction term)
    tau: float = 0.5           # memory timescale (kernel exp(-s/tau)/tau)
    tau_max: float = 2.0       # truncation horizon for the integral (>=4*tau)
    D: float = 0.05
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def dist_delay_rd_full_rhs(u: np.ndarray, history: np.ndarray,
                            kernel_weights: np.ndarray,
                            p: DistDelayRDParams,
                            grid: SpectralGrid2D) -> np.ndarray:
    """RHS: D laplacian(u) + integral over kernel*f(u(t-s)) ds.

    `history`: array shape (n_quad, *spatial) holding u(x, t-s_i) for the
    quadrature nodes s_i.  `kernel_weights[i]` = K(s_i) * w_i (trapezoidal).
    """
    diffusion = p.D * laplacian2d(u, grid)
    # f(u) = A * u * (1 - u)  (logistic).  Apply at each history slot, then sum.
    f_u = p.A * history * (1.0 - history)              # (n_quad, n, n)
    integral = np.einsum("q,qij->ij", kernel_weights, f_u)
    return diffusion + integral


def simulate_dist_delay_rd(p: DistDelayRDParams, rng: np.random.Generator,
                            grid: SpectralGrid2D) -> np.ndarray:
    """Simulate distributed-delay RD 2D with explicit kernel quadrature."""
    n_steps = int(round(p.T_total / p.dt))
    n_quad = int(round(p.tau_max / p.dt))
    if abs(n_quad * p.dt - p.tau_max) > 1e-9:
        raise ValueError(f"tau_max={p.tau_max} not divisible by dt={p.dt}")
    k_max_sq = 2 * (np.pi * p.n_grid / p.L) ** 2
    sf = p.D * k_max_sq * p.dt
    if sf > 2.78:
        raise ValueError(f"DistDelayRD RK4 unstable: D*k^2*dt = {sf:.2f} > 2.78")

    # Pre-compute the trapezoidal kernel weights:
    #   integral_0^tau_max K(s) g(s) ds approx sum_i w_i * K(s_i) * g(s_i)
    # with s_i = i * dt for i=0..n_quad-1 and trapezoidal w_i = dt
    # (boundary halved).
    s = np.arange(n_quad) * p.dt
    K = (1.0 / p.tau) * np.exp(-s / p.tau)
    w = np.full(n_quad, p.dt, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    kernel_weights = K * w                                # (n_quad,)
    # Renormalize so total mass = 1 (compensates for truncation).
    kernel_weights /= max(kernel_weights.sum(), 1e-12)

    # Initial history: smooth random field around u=0.5 (logistic stable point).
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.2, k_max=4, base=0.5)
    history = np.tile(u0[None, ...], (n_quad, 1, 1))

    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0

    def get_history(step: int) -> np.ndarray:
        """Return (n_quad, n, n) array of u(x, t-s_i) at the current step.

        s_i = i*dt, so we want u at positions [step, step-1, ..., step-n_quad+1]
        Negative indices use the initial history (constant u0).

        Vectorized via array slicing — ~5-10x faster than the per-i loop.
        """
        out = np.empty((n_quad, p.n_grid, p.n_grid))
        n_from_traj = min(step + 1, n_quad)
        n_from_init = n_quad - n_from_traj
        if n_from_init > 0:
            out[n_from_traj:] = history[0]
        if n_from_traj > 0:
            # traj[step], traj[step-1], ..., traj[step - n_from_traj + 1]
            # = traj[step+1-n_from_traj : step+1] reversed
            out[:n_from_traj] = traj[step + 1 - n_from_traj : step + 1][::-1]
        return out

    for step in range(n_steps):
        u = traj[step]
        # For RK4 with distributed delay, all the delayed information is built
        # from the past trajectory; for the intermediate stages we keep the
        # same history (frozen-history approximation, lower order than full
        # interpolation but standard for distributed-delay schemes).
        hist_arr = get_history(step)
        k1 = dist_delay_rd_full_rhs(u, hist_arr, kernel_weights, p, grid)
        k2 = dist_delay_rd_full_rhs(u + 0.5 * p.dt * k1, hist_arr, kernel_weights, p, grid)
        k3 = dist_delay_rd_full_rhs(u + 0.5 * p.dt * k2, hist_arr, kernel_weights, p, grid)
        k4 = dist_delay_rd_full_rhs(u + p.dt * k3, hist_arr, kernel_weights, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"DistDelayRD NaN/inf at step {step + 1}/{n_steps}")
    return traj


def sample_dist_delay_rd_param_set(rng: np.random.Generator) -> DistDelayRDParams:
    """Vary A (source) and tau (memory timescale).  tau_max = 4*tau covers >98%
    of the exponential kernel mass."""
    tau = float(rng.choice([0.25, 0.5, 1.0]))
    return DistDelayRDParams(
        A=float(rng.uniform(0.5, 2.0)),
        tau=tau,
        tau_max=4.0 * tau,
        D=float(rng.choice([0.025, 0.05])),
    )


def audit_dist_delay_rd(out_dir: Path, A=1.0, tau=0.5, D=0.05,
                        T_total=20.0, dt=0.01, n_grid=64, seed=42):
    """B4 audit: trajectory + memory-timescale sweep."""
    p = DistDelayRDParams(A=A, tau=tau, tau_max=4.0 * tau, D=D,
                           T_total=T_total, dt=dt, n_grid=n_grid)
    grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
    rng = np.random.default_rng(seed)
    print(f"\n=== AUDIT DistDelayRD: A={A}, tau={tau}, tau_max={p.tau_max}, D={D} ===")
    t0 = time.time()
    traj = simulate_dist_delay_rd(p, rng, grid)
    print(f"  wall: {time.time()-t0:.1f}s,  shape: {traj.shape}")
    print(f"  finite: {np.isfinite(traj).all()}")
    print(f"  range: [{traj.min():.4f}, {traj.max():.4f}], mean={traj.mean():.4f}")
    u_bar = traj.mean(axis=(1, 2))
    print(f"  u_bar range: [{u_bar.min():.4f}, {u_bar.max():.4f}]")
    print(f"  expected: logistic source pushes u toward 1 (or 0); kernel smooths transients")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"audit_traj_A{A}_t{tau}.npy", traj)
    return traj


# ---------------------------------------------------------------------
# B5: Delayed-feedback Burgers 2D
#
#   d/dt u + u (du/dx + du/dy) = nu (uxx + uyy) + alpha (u(t-tau) - u_target)
#
# Scalar 2D Burgers with delayed feedback control toward a target profile.
# Models closed-loop flow control with sensor delay -- a common engineering
# scenario.  Stability constrained by CFL (advection) and diffusion bounds.
# ---------------------------------------------------------------------

@dataclass
class DelayBurgersParams:
    nu: float = 0.05           # viscosity
    alpha: float = 1.0         # feedback strength
    tau: float = 0.5           # sensor/feedback delay
    u_target_amp: float = 0.0  # target profile amplitude (sin(x))
    T_total: float = 8.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def burgers_full_rhs(u: np.ndarray, u_delayed: np.ndarray,
                      u_target: np.ndarray,
                      p: DelayBurgersParams,
                      grid: SpectralGrid2D) -> np.ndarray:
    """RHS for scalar 2D Burgers with delayed-feedback control.

    -u*u_x - u*u_y (advection)
    + nu*lap(u) (diffusion)
    - alpha*(u(t-tau) - u_target) (NEGATIVE feedback toward target;
                                    positive alpha = stronger control)
    """
    u_hat = np.fft.fft2(u)
    u_x = np.real(np.fft.ifft2(1j * grid.KX * u_hat))
    u_y = np.real(np.fft.ifft2(1j * grid.KY * u_hat))
    advection = -u * (u_x + u_y)
    diffusion = p.nu * laplacian2d(u, grid)
    feedback = -p.alpha * (u_delayed - u_target)
    return advection + diffusion + feedback


def simulate_delay_burgers(p: DelayBurgersParams, rng: np.random.Generator,
                             grid: SpectralGrid2D) -> np.ndarray:
    """Simulate delayed-feedback Burgers 2D using RK4."""
    n_steps = int(round(p.T_total / p.dt))
    history_steps = int(round(p.tau / p.dt))
    if abs(history_steps * p.dt - p.tau) > 1e-9:
        raise ValueError(f"tau={p.tau} not divisible by dt={p.dt}")
    # Diffusion stability check.
    k_max_sq = 2 * (np.pi * p.n_grid / p.L) ** 2
    sf = p.nu * k_max_sq * p.dt
    if sf > 2.78:
        raise ValueError(
            f"DelayBurgers RK4 unstable for diffusion: nu*k^2*dt = {sf:.2f} > 2.78."
            f"  Reduce dt to <= {2.78 / (p.nu * k_max_sq):.4f}.")

    # Target profile: u_target(x, y) = u_target_amp * sin(x) * cos(y)
    u_target = p.u_target_amp * np.sin(grid.X) * np.cos(grid.Y)

    # Initial condition: smooth random field with moderate amplitude.
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.3, k_max=4, base=0.0)
    history = np.tile(u0[None, ...], (history_steps + 1, 1, 1))

    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0

    def get_delayed(step_real: float) -> np.ndarray:
        if step_real <= 0:
            return history[0]
        elif step_real >= step:
            return traj[step]
        else:
            i_low = int(np.floor(step_real))
            frac = step_real - i_low
            return (1.0 - frac) * traj[i_low] + frac * traj[i_low + 1]

    for step in range(n_steps):
        u = traj[step]
        ud_n = get_delayed(step - history_steps)
        ud_mid = get_delayed(step + 0.5 - history_steps)
        ud_n1 = get_delayed(step + 1 - history_steps)
        # CFL check (informational).
        if step == 0:
            cfl = np.abs(u).max() * p.dt / (p.L / p.n_grid)
            if cfl > 0.5:
                print(f"   warning: CFL={cfl:.2f} > 0.5, advection may be unstable")
        k1 = burgers_full_rhs(u, ud_n, u_target, p, grid)
        k2 = burgers_full_rhs(u + 0.5 * p.dt * k1, ud_mid, u_target, p, grid)
        k3 = burgers_full_rhs(u + 0.5 * p.dt * k2, ud_mid, u_target, p, grid)
        k4 = burgers_full_rhs(u + p.dt * k3, ud_n1, u_target, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"DelayBurgers NaN/inf at step {step + 1}/{n_steps}")
    return traj


def sample_delay_burgers_param_set(rng: np.random.Generator) -> DelayBurgersParams:
    """Vary nu (viscosity), alpha (feedback strength), tau (delay), u_target_amp."""
    return DelayBurgersParams(
        nu=float(rng.choice([0.025, 0.05])),
        alpha=float(rng.uniform(0.5, 2.0)),
        tau=float(rng.choice([0.25, 0.5, 1.0])),
        u_target_amp=float(rng.uniform(-0.3, 0.3)),
    )


def audit_delay_burgers(out_dir: Path, nu=0.05, alpha=1.0, tau=0.5,
                          u_target_amp=0.0, T_total=8.0, dt=0.01,
                          n_grid=64, seed=42):
    """B5 audit: trajectory + visualization."""
    p = DelayBurgersParams(nu=nu, alpha=alpha, tau=tau,
                            u_target_amp=u_target_amp, T_total=T_total,
                            dt=dt, n_grid=n_grid)
    grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
    rng = np.random.default_rng(seed)
    print(f"\n=== AUDIT DelayBurgers: nu={nu}, alpha={alpha}, tau={tau},"
          f" u_target_amp={u_target_amp}, T={T_total} ===")
    t0 = time.time()
    traj = simulate_delay_burgers(p, rng, grid)
    print(f"  wall: {time.time()-t0:.1f}s,  shape: {traj.shape}")
    print(f"  finite: {np.isfinite(traj).all()}")
    print(f"  range: [{traj.min():.4f}, {traj.max():.4f}], mean={traj.mean():.4f}")
    u_bar = traj.mean(axis=(1, 2))
    print(f"  u_bar range: [{u_bar.min():.4f}, {u_bar.max():.4f}]")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"audit_traj_a{alpha}_t{tau}.npy", traj)
    return traj


# ---------------------------------------------------------------------
# B6: Ring-coupled Kuramoto field 2D
#
#   d/dt theta(x, t) = omega(x) + K * (G * sin(theta(t-tau) - theta(t)))(x)
#
# Phase field on a torus with Gaussian-kernel non-local coupling and
# delay tau in the coupling term.  The phase variable theta lives on
# S^1 (mod 2*pi); we store it as a real number and the model can output
# real values that we interpret periodically at evaluation.
# ---------------------------------------------------------------------

@dataclass
class KuramotoParams:
    K: float = 1.0             # coupling strength
    sigma: float = 0.5         # Gaussian kernel width
    tau: float = 0.5           # propagation delay
    omega_std: float = 0.1     # std of natural-frequency disorder
    T_total: float = 8.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def kuramoto_full_rhs(theta: np.ndarray, theta_delayed: np.ndarray,
                       omega: np.ndarray, G_hat: np.ndarray,
                       p: KuramotoParams) -> np.ndarray:
    """RHS: omega + K * (G * sin(theta_delayed - theta)).

    Convolution computed in Fourier space.  Uses the addition formula:
      sin(theta_d - theta) = sin(theta_d)*cos(theta) - cos(theta_d)*sin(theta)
    so we convolve sin(theta_d) and cos(theta_d) once each, then combine.
    """
    s_d = np.sin(theta_delayed)
    c_d = np.cos(theta_delayed)
    S = np.real(np.fft.ifft2(G_hat * np.fft.fft2(s_d)))
    C = np.real(np.fft.ifft2(G_hat * np.fft.fft2(c_d)))
    coupling = S * np.cos(theta) - C * np.sin(theta)
    return omega + p.K * coupling


def simulate_kuramoto(p: KuramotoParams, rng: np.random.Generator,
                       grid: SpectralGrid2D) -> np.ndarray:
    """Simulate the ring-coupled Kuramoto field 2D using RK4."""
    n_steps = int(round(p.T_total / p.dt))
    history_steps = int(round(p.tau / p.dt))
    if abs(history_steps * p.dt - p.tau) > 1e-9:
        raise ValueError(f"tau={p.tau} not divisible by dt={p.dt}")

    # Pre-compute Gaussian kernel in Fourier space.
    # G(x-y) = (1/(2*pi*sigma^2)) * exp(-|x-y|^2 / (2*sigma^2)).
    # Its Fourier transform is exp(-(kx^2 + ky^2)*sigma^2 / 2), peak 1 at k=0.
    # Note: we do NOT divide by (2*pi*sigma^2) -- we want G_hat[0,0] = 1
    # so that the convolution preserves the magnitude order of theta.
    G_hat = np.exp(-grid.K2 * p.sigma ** 2 / 2.0)

    # Natural frequency: spatially-varying with omega_std disorder.
    omega = p.omega_std * rng.standard_normal((p.n_grid, p.n_grid))

    # Initial history: random phase field, low-spatial-frequency.
    theta0 = smooth_random_field_2d(rng, grid, amplitude=np.pi / 2,
                                      k_max=4, base=0.0)
    history = np.tile(theta0[None, ...], (history_steps + 1, 1, 1))

    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = theta0

    def get_delayed(step_real: float) -> np.ndarray:
        if step_real <= 0:
            return history[0]
        elif step_real >= step:
            return traj[step]
        else:
            i_low = int(np.floor(step_real))
            frac = step_real - i_low
            return (1.0 - frac) * traj[i_low] + frac * traj[i_low + 1]

    for step in range(n_steps):
        theta = traj[step]
        td_n = get_delayed(step - history_steps)
        td_mid = get_delayed(step + 0.5 - history_steps)
        td_n1 = get_delayed(step + 1 - history_steps)
        k1 = kuramoto_full_rhs(theta, td_n, omega, G_hat, p)
        k2 = kuramoto_full_rhs(theta + 0.5 * p.dt * k1, td_mid, omega, G_hat, p)
        k3 = kuramoto_full_rhs(theta + 0.5 * p.dt * k2, td_mid, omega, G_hat, p)
        k4 = kuramoto_full_rhs(theta + p.dt * k3, td_n1, omega, G_hat, p)
        traj[step + 1] = theta + (p.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"Kuramoto NaN/inf at step {step + 1}/{n_steps}")
    return traj


def sample_kuramoto_param_set(rng: np.random.Generator) -> KuramotoParams:
    """Vary K (coupling), sigma (kernel width), tau (delay), omega_std."""
    return KuramotoParams(
        K=float(rng.uniform(0.5, 2.0)),
        sigma=float(rng.choice([0.3, 0.5, 0.8])),
        tau=float(rng.choice([0.25, 0.5, 1.0])),
        omega_std=float(rng.choice([0.05, 0.1, 0.2])),
    )


def audit_kuramoto(out_dir: Path, K=1.0, sigma=0.5, tau=0.5, omega_std=0.1,
                    T_total=8.0, dt=0.01, n_grid=64, seed=42):
    """B6 audit: trajectory + synchronization measure."""
    p = KuramotoParams(K=K, sigma=sigma, tau=tau, omega_std=omega_std,
                        T_total=T_total, dt=dt, n_grid=n_grid)
    grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
    rng = np.random.default_rng(seed)
    print(f"\n=== AUDIT Kuramoto: K={K}, sigma={sigma}, tau={tau},"
          f" omega_std={omega_std}, T={T_total} ===")
    t0 = time.time()
    traj = simulate_kuramoto(p, rng, grid)
    print(f"  wall: {time.time()-t0:.1f}s,  shape: {traj.shape}")
    print(f"  finite: {np.isfinite(traj).all()}")
    print(f"  range: [{traj.min():.4f}, {traj.max():.4f}], std={traj.std():.4f}")
    # Kuramoto order parameter R(t) = |<exp(i*theta)>_x|, in [0, 1].
    R = np.abs(np.exp(1j * traj).mean(axis=(1, 2)))
    print(f"  Kuramoto order R(t) range: [{R.min():.4f}, {R.max():.4f}], final R={R[-1]:.4f}")
    print(f"  expected: R grows for K large enough (synchronization)")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"audit_traj_K{K}_t{tau}.npy", traj)
    return traj


def synchronization_sweep_kuramoto(out_dir: Path,
                                    Ks=(0.1, 0.5, 1.0, 2.0, 4.0),
                                    sigma=0.5, tau=0.5, omega_std=0.1,
                                    T_total=8.0, dt=0.01, n_grid=64, seed=42):
    """Vary K; observe synchronization onset (Kuramoto critical K_c)."""
    print(f"\n=== SYNC SWEEP Kuramoto (vary K, fixed tau={tau}, sigma={sigma}) ===")
    print(f"   {'K':>5}  {'final R':>9}  {'mean R(2nd half)':>17}")
    for K in Ks:
        p = KuramotoParams(K=K, sigma=sigma, tau=tau, omega_std=omega_std,
                            T_total=T_total, dt=dt, n_grid=n_grid)
        grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
        rng = np.random.default_rng(seed)
        try:
            traj = simulate_kuramoto(p, rng, grid)
            R = np.abs(np.exp(1j * traj).mean(axis=(1, 2)))
            mean_R = float(R[len(R) // 2:].mean())
            print(f"   {K:>5.2f}  {R[-1]:>9.4f}  {mean_R:>17.4f}")
        except RuntimeError as e:
            print(f"   {K:>5.2f}  --BLEW UP-- ({e})")


# ---------------------------------------------------------------------
# Generation pipeline (matches apebench shard format)
# ---------------------------------------------------------------------

def generate_split_generic(family: str, num_samples: int, seed: int,
                            n_hist: int, n_out: int, dt: float, n_grid: int,
                            L: float = 2 * np.pi):
    """Generic split generator dispatched by `family`.

    For each sample i:
      1. Sample family-specific params.
      2. Simulate (n_hist + n_out) timesteps.
      3. Append to phi/y arrays.

    Returns: phi (N, n_hist, *spatial, C), y (N, n_out, *spatial, C),
             params (N, params_dim), p_used (template params).
    """
    rng = np.random.default_rng(seed)
    grid = SpectralGrid2D.make(n=n_grid, L=L)
    n_total = n_hist + n_out
    T_total_needed = (n_total - 1) * dt
    phi_list, y_list, params_list = [], [], []
    for i in range(num_samples):
        if family == "mackey_glass_2d":
            p = sample_param_set(rng)
            p = MackeyGlassParams(**{**p.__dict__, "T_total": T_total_needed,
                                      "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_mackey_glass(p, rng, grid)
            params_vec = [p.beta, p.tau, p.D]
        elif family == "wright_2d":
            p = sample_wright_param_set(rng)
            p = WrightParams(**{**p.__dict__, "T_total": T_total_needed,
                                  "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_wright(p, rng, grid)
            params_vec = [p.alpha, p.tau, p.D]
        elif family == "hutchinson_2d":
            p = sample_hutchinson_param_set(rng)
            p = HutchinsonParams(**{**p.__dict__, "T_total": T_total_needed,
                                      "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_hutchinson(p, rng, grid)
            params_vec = [p.r, p.tau, p.D]
        elif family == "dist_delay_rd_2d":
            p = sample_dist_delay_rd_param_set(rng)
            p = DistDelayRDParams(**{**p.__dict__, "T_total": T_total_needed,
                                       "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_dist_delay_rd(p, rng, grid)
            params_vec = [p.A, p.tau, p.D]
        elif family == "delay_burgers_2d":
            p = sample_delay_burgers_param_set(rng)
            p = DelayBurgersParams(**{**p.__dict__, "T_total": T_total_needed,
                                        "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_delay_burgers(p, rng, grid)
            params_vec = [p.alpha, p.tau, p.nu]
        elif family == "kuramoto_2d":
            p = sample_kuramoto_param_set(rng)
            p = KuramotoParams(**{**p.__dict__, "T_total": T_total_needed,
                                    "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_kuramoto(p, rng, grid)
            params_vec = [p.K, p.tau, p.sigma]
        elif family == "multi_delay_mg_2d":
            p = sample_multi_delay_mg_param_set(rng)
            p = MultiDelayMGParams(**{**p.__dict__, "T_total": T_total_needed,
                                        "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_multi_delay_mg(p, rng, grid)
            params_vec = [float(p.taus[0]), float(p.taus[1]), float(p.taus[2])]
        elif family in ("dist_exp_rd_2d", "dist_gaussian_rd_2d",
                          "dist_gamma_rd_2d", "dist_uniform_rd_2d",
                          "dist_powerlaw_rd_2d"):
            kernel_type = family.replace("dist_", "").replace("_rd_2d", "")
            p = sample_dist_kernel_rd_param_set(rng, kernel_type)
            p = DistKernelRDParams(**{**p.__dict__, "T_total": T_total_needed,
                                        "dt": dt, "n_grid": n_grid, "L": L})
            traj = simulate_dist_kernel_rd(p, rng, grid)
            # Pad params_vec to dim=5: [A, tau, D, e1, e2] with kernel-shape
            # extras so FiLM has enough signal to learn family-specific kernel
            # modulation. Padding scheme:
            #   exp / uniform : (e1, e2) = (0, 0)
            #   gaussian      : (e1, e2) = (mu, sigma)
            #   gamma         : (e1, e2) = (k, 0)
            #   powerlaw      : (e1, e2) = (alpha, 0)   (s0 is fixed at 0.05)
            ke = p.kernel_extra or {}
            if kernel_type == "gaussian":
                e1, e2 = float(ke.get("mu", 0.0)), float(ke.get("sigma", 0.0))
            elif kernel_type == "gamma":
                e1, e2 = float(ke.get("k", 0.0)), 0.0
            elif kernel_type == "powerlaw":
                e1, e2 = float(ke.get("alpha", 0.0)), 0.0
            else:  # exp, uniform
                e1, e2 = 0.0, 0.0
            params_vec = [p.A, p.tau, p.D, e1, e2]
        else:
            raise ValueError(f"family {family} not implemented in generate_split_generic")
        traj = traj[:n_total]
        traj = traj[..., None].astype(np.float32)
        phi_list.append(traj[:n_hist])
        y_list.append(traj[n_hist:n_hist + n_out])
        params_list.append(params_vec)
        if (i + 1) % 16 == 0:
            print(f"    [{i+1}/{num_samples}] simulated.")
    phi = np.stack(phi_list, axis=0)
    y = np.stack(y_list, axis=0)
    params = np.array(params_list, dtype=np.float32)
    return phi, y, params, p


def write_shard(out_dir: Path, split: str, phi, y, params, t_hist, t_out):
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "shard_000.npz",
        phi=phi.astype(np.float32),
        y=y.astype(np.float32),
        params=params.astype(np.float32),
        t_hist=t_hist.astype(np.float32),
        t_out=t_out.astype(np.float32),
        lags=np.zeros((phi.shape[0], 1), dtype=np.float32),
    )
    print(f"  wrote {out_dir}/shard_000.npz  phi={phi.shape}  y={y.shape}")


# ---------------------------------------------------------------------
# Audit functions (run before any large-scale generation)
# ---------------------------------------------------------------------

def audit_one_trajectory(out_dir: Path, beta=2.0, tau=2.0, n=10.0, gamma=1.0,
                          D=0.1, T_total=20.0, dt=0.05, n_grid=64, seed=42):
    """Generate ONE Mackey-Glass + diffusion trajectory and run a sanity audit.

    Reports:
      * NaN/inf check
      * Magnitude range
      * Per-time-step variance (proxy for oscillation onset)
      * Mean trajectory u_bar(t) = mean over space (should oscillate at beta*tau > pi)
      * Spatial spectrum at final t (should be smooth, decaying)
    """
    p = MackeyGlassParams(beta=beta, tau=tau, n=n, gamma=gamma, D=D,
                           T_total=T_total, dt=dt, n_grid=n_grid)
    grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
    rng = np.random.default_rng(seed)
    print(f"\n=== AUDIT trajectory: beta={beta}, tau={tau}, n={n}, gamma={gamma}, D={D},"
          f" T={T_total}, dt={dt}, grid={n_grid} ===")
    t0 = time.time()
    traj = simulate_mackey_glass(p, rng, grid)
    print(f"  wall: {time.time()-t0:.1f}s,  shape: {traj.shape}")
    print(f"  finite: {np.isfinite(traj).all()}")
    print(f"  range: [{traj.min():.4f}, {traj.max():.4f}], mean={traj.mean():.4f}, std={traj.std():.4f}")
    # Mean trajectory u_bar(t).
    u_bar = traj.mean(axis=(1, 2))                        # (n_steps + 1,)
    # Detect oscillations: count zero-crossings of u_bar - u_bar.mean().
    centered = u_bar - u_bar.mean()
    zc = np.sum(np.diff(np.sign(centered)) != 0)
    print(f"  u_bar(t) range: [{u_bar.min():.4f}, {u_bar.max():.4f}]")
    print(f"  u_bar zero-crossings: {zc}  (expect >0 in oscillatory regime, beta*tau > pi)")
    print(f"  beta*tau = {beta * tau:.3f}, pi = {np.pi:.3f}  (oscillation predicted: {beta*tau > np.pi})")
    # Spatial spectrum at t = T_total.
    final = traj[-1]
    spec = np.abs(np.fft.fft2(final))
    print(f"  spatial spectrum: |F[0,0]|={spec[0,0]:.3f}, max|F[k>0]|={spec[1:,1:].max():.3f}")
    print(f"  spectral roll-off (|F[k=4]| / |F[k=1]|): {spec[4,0] / (spec[1,0] + 1e-12):.3e}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"audit_traj_b{beta}_t{tau}_seed{seed}.npy", traj)
    print(f"  saved to {out_dir}")
    return traj


def convergence_study_mg(out_dir: Path, beta=2.0, tau=2.0, T_total=10.0,
                          n_grid=64, seed=42, dt_list=(0.1, 0.05, 0.025, 0.0125)):
    """Run identical IC simulation at multiple dt; report L2 difference between
    successive refinements at the final time.

    NOTE: empirical audit shows EFFECTIVE 1st-order convergence in dt
    (~2x error reduction per halving), not 4th order, due to frozen-history
    RK4 + linear delay interpolation.  The integrator is robust and
    deterministic — its order just isn't 4 — and the benchmark is
    internally consistent.  See convergence audit at scripts/
    audit_pde_solver_convergence.py.
    """
    print(f"\n=== CONVERGENCE STUDY (Mackey-Glass + diffusion 2D) ===")
    print(f"   beta={beta}, tau={tau}, T={T_total}, grid={n_grid}, seed={seed}")
    finals = {}
    for dt in dt_list:
        # Adjust T_total so all simulations end at the same time.
        steps = int(round(T_total / dt))
        if abs(steps * dt - T_total) > 1e-9:
            T_use = steps * dt
        else:
            T_use = T_total
        p = MackeyGlassParams(beta=beta, tau=tau, T_total=T_use, dt=dt,
                              n_grid=n_grid)
        grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
        rng = np.random.default_rng(seed)
        traj = simulate_mackey_glass(p, rng, grid)
        finals[dt] = traj[-1]
        print(f"   dt={dt:.4f}: shape={traj.shape}, |u(T)| mean={np.abs(traj[-1]).mean():.4f}")
    # Compare successive refinements.
    dts = sorted(dt_list, reverse=True)
    print(f"\n   relL2(dt vs dt/2) — empirically ~2x per halving (1st-order); ~16x would be 4th-order (not achieved due to frozen-history RK4):")
    prev_err = None
    for i in range(len(dts) - 1):
        dt_coarse, dt_fine = dts[i], dts[i + 1]
        diff = finals[dt_coarse] - finals[dt_fine]
        rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(finals[dt_fine]) + 1e-12)
        ratio = (prev_err / rel_l2) if prev_err else None
        ratio_str = f"  (ratio: {ratio:.1f}x)" if ratio else ""
        print(f"   dt={dt_coarse:.4f} vs dt={dt_fine:.4f}:  relL2 = {rel_l2:.3e}{ratio_str}")
        prev_err = rel_l2
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for dt, f in finals.items():
        np.save(out_dir / f"final_dt{dt:.4f}.npy", f)


def visualize_trajectory(traj_path_or_arr, out_path: Path, n_frames: int = 8,
                          title: str = "trajectory"):
    """Render a strip of `n_frames` snapshots evenly spaced along time.

    Saves a single PNG to `out_path`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if isinstance(traj_path_or_arr, (str, Path)):
        traj = np.load(traj_path_or_arr)
    else:
        traj = traj_path_or_arr
    T = traj.shape[0]
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2.0, 2.4),
                              constrained_layout=True)
    vmin, vmax = traj.min(), traj.max()
    for ax, k in zip(axes, idxs):
        im = ax.imshow(traj[k], vmin=vmin, vmax=vmax, cmap="viridis",
                        origin="lower")
        ax.set_title(f"t-step {k}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title}  range=[{vmin:.3f}, {vmax:.3f}]")
    fig.colorbar(im, ax=axes, fraction=0.02)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"   wrote {out_path}")


# ---------------------------------------------------------------------
# B7: Multi-delay Mackey-Glass + diffusion 2D
#     d/dt u = D laplacian(u) + sum_i beta_i * u(x, t-tau_i) / (1 + u(x, t-tau_i)^n) - gamma * u
#
# 3 discrete delays summed. This is the benchmark that genuinely
# exercises Theorem T1 — the lag axis has multiple cyclic-buffer slots
# coupled into the dynamics, so a lag-equivariant operator should have
# advantage over local-conv UNet.
# ---------------------------------------------------------------------

@dataclass
class MultiDelayMGParams:
    """Multi-delay Mackey-Glass + diffusion 2D."""
    betas: tuple = (1.0, 0.8, 0.6)        # 3 source amplitudes
    taus: tuple = (1.0, 2.0, 3.0)          # 3 discrete delays
    gamma: float = 1.0
    n: float = 10.0
    D: float = 0.1
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def multi_delay_mg_rhs(u: np.ndarray, u_delayed_list: list,
                        p: MultiDelayMGParams,
                        grid: SpectralGrid2D) -> np.ndarray:
    """RHS: D*lap(u) + sum_i beta_i * u_delay_i / (1 + u_delay_i^n) - gamma*u."""
    diffusion = p.D * laplacian2d(u, grid)
    source = np.zeros_like(u)
    for beta, ud in zip(p.betas, u_delayed_list):
        source = source + beta * ud / (1.0 + np.abs(ud) ** p.n)
    decay = p.gamma * u
    return diffusion + source - decay


def simulate_multi_delay_mg(p: MultiDelayMGParams, rng: np.random.Generator,
                              grid: SpectralGrid2D) -> np.ndarray:
    n_steps = int(round(p.T_total / p.dt))
    history_steps_list = [int(round(t / p.dt)) for t in p.taus]
    for h, t in zip(history_steps_list, p.taus):
        if abs(h * p.dt - t) > 1e-9:
            raise ValueError(f"tau={t} not divisible by dt={p.dt}")
    max_history = max(history_steps_list)
    k_max_sq = 2 * (np.pi * p.n_grid / p.L) ** 2
    sf = p.D * k_max_sq * p.dt
    if sf > 2.78:
        raise ValueError(
            f"explicit RK4 unstable: D*k^2*dt = {sf:.2f} > 2.78."
            f"  Reduce dt to <= {2.78 / (p.D * k_max_sq):.4f}.")

    u0 = smooth_random_field_2d(rng, grid, amplitude=0.4, k_max=4, base=0.6)
    history = np.tile(u0[None, ...], (max_history + 1, 1, 1))

    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0

    def get_delayed(step_real: float) -> np.ndarray:
        if step_real <= 0:
            return history[0]
        elif step_real >= step:
            return traj[step]
        else:
            i_low = int(np.floor(step_real))
            frac = step_real - i_low
            return (1.0 - frac) * traj[i_low] + frac * traj[i_low + 1]

    for step in range(n_steps):
        u = traj[step]
        # Build delayed args for each tau (linear-interp at step + offset)
        # k1 stage uses u_delayed at integer step
        ud_n_list = [get_delayed(step - h) for h in history_steps_list]
        ud_mid_list = [get_delayed(step + 0.5 - h) for h in history_steps_list]
        ud_n1_list = [get_delayed(step + 1 - h) for h in history_steps_list]
        k1 = multi_delay_mg_rhs(u, ud_n_list, p, grid)
        k2 = multi_delay_mg_rhs(u + 0.5 * p.dt * k1, ud_mid_list, p, grid)
        k3 = multi_delay_mg_rhs(u + 0.5 * p.dt * k2, ud_mid_list, p, grid)
        k4 = multi_delay_mg_rhs(u + p.dt * k3, ud_n1_list, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"MultiDelayMG NaN/inf at step {step + 1}/{n_steps}")
    return traj


def sample_multi_delay_mg_param_set(rng: np.random.Generator) -> MultiDelayMGParams:
    """Sample 3 discrete delays + 3 betas. Tau values constrained to be divisible
    by dt=0.01 to satisfy the simulator. Variation: pick 3 taus from a discrete
    set, randomize betas around (1, 0.8, 0.6) baseline."""
    candidate_taus = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    taus = tuple(sorted(rng.choice(candidate_taus, size=3, replace=False)))
    betas = tuple(float(rng.uniform(0.4, 1.2)) for _ in range(3))
    return MultiDelayMGParams(
        betas=betas, taus=taus, gamma=1.0, n=10.0,
        D=float(rng.choice([0.025, 0.05, 0.1])))


# ---------------------------------------------------------------------
# B8: Multi-delay Wright + diffusion 2D
#     d/dt u = D laplacian(u) - sum_i alpha_i * u(t-tau_i) * (1 + u)
# ---------------------------------------------------------------------
@dataclass
class MultiDelayWrightParams:
    alphas: tuple = (0.5, 0.4, 0.3)
    taus: tuple = (0.5, 1.0, 1.5)
    D: float = 0.05
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def multi_delay_wright_rhs(u, ud_list, p, grid):
    diffusion = p.D * laplacian2d(u, grid)
    fb = np.zeros_like(u)
    for a, ud in zip(p.alphas, ud_list):
        fb = fb - a * ud * (1.0 + u)
    return diffusion + fb


def simulate_multi_delay_wright(p, rng, grid):
    n_steps = int(round(p.T_total / p.dt))
    h_list = [int(round(t / p.dt)) for t in p.taus]
    max_h = max(h_list)
    sf = p.D * 2 * (np.pi * p.n_grid / p.L) ** 2 * p.dt
    if sf > 2.78:
        raise ValueError(f"unstable: D*k^2*dt = {sf:.2f}")
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.2, k_max=4, base=0.0)
    history = np.tile(u0[None, ...], (max_h + 1, 1, 1))
    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0
    def get_delayed(s):
        if s <= 0: return history[0]
        if s >= step: return traj[step]
        i = int(np.floor(s)); f = s - i
        return (1 - f) * traj[i] + f * traj[i + 1]
    for step in range(n_steps):
        u = traj[step]
        ud_n = [get_delayed(step - h) for h in h_list]
        ud_m = [get_delayed(step + 0.5 - h) for h in h_list]
        ud_n1 = [get_delayed(step + 1 - h) for h in h_list]
        k1 = multi_delay_wright_rhs(u, ud_n, p, grid)
        k2 = multi_delay_wright_rhs(u + 0.5 * p.dt * k1, ud_m, p, grid)
        k3 = multi_delay_wright_rhs(u + 0.5 * p.dt * k2, ud_m, p, grid)
        k4 = multi_delay_wright_rhs(u + p.dt * k3, ud_n1, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"MultiDelayWright NaN at step {step+1}")
    return traj


def sample_multi_delay_wright_param_set(rng):
    cand = [0.5, 1.0, 1.5, 2.0]
    taus = tuple(sorted(rng.choice(cand, size=3, replace=False)))
    # Keep Σαᵢ × max(τ) small enough to avoid Wright blowup.
    alphas = tuple(float(rng.uniform(0.2, 0.5)) for _ in range(3))
    return MultiDelayWrightParams(
        alphas=alphas, taus=taus,
        D=float(rng.choice([0.025, 0.05])))


# ---------------------------------------------------------------------
# B9: Multi-delay Burgers 2D
#     d/dt u + u (du/dx + du/dy) = nu lap(u) - sum_i alpha_i (u(t-tau_i) - u_target)
# ---------------------------------------------------------------------
@dataclass
class MultiDelayBurgersParams:
    nu: float = 0.05
    alphas: tuple = (0.5, 0.4, 0.3)
    taus: tuple = (0.25, 0.5, 1.0)
    u_target_amp: float = 0.0
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def multi_delay_burgers_rhs(u, ud_list, u_target, p, grid):
    u_hat = np.fft.fft2(u)
    u_x = np.real(np.fft.ifft2(1j * grid.KX * u_hat))
    u_y = np.real(np.fft.ifft2(1j * grid.KY * u_hat))
    advection = -u * (u_x + u_y)
    diffusion = p.nu * laplacian2d(u, grid)
    fb = np.zeros_like(u)
    for a, ud in zip(p.alphas, ud_list):
        fb = fb - a * (ud - u_target)
    return advection + diffusion + fb


def simulate_multi_delay_burgers(p, rng, grid):
    n_steps = int(round(p.T_total / p.dt))
    h_list = [int(round(t / p.dt)) for t in p.taus]
    max_h = max(h_list)
    sf = p.nu * 2 * (np.pi * p.n_grid / p.L) ** 2 * p.dt
    if sf > 2.78:
        raise ValueError(f"unstable: nu*k^2*dt = {sf:.2f}")
    u_target = p.u_target_amp * np.sin(grid.X) * np.cos(grid.Y)
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.3, k_max=4, base=0.0)
    history = np.tile(u0[None, ...], (max_h + 1, 1, 1))
    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0
    def get_delayed(s):
        if s <= 0: return history[0]
        if s >= step: return traj[step]
        i = int(np.floor(s)); f = s - i
        return (1 - f) * traj[i] + f * traj[i + 1]
    for step in range(n_steps):
        u = traj[step]
        ud_n = [get_delayed(step - h) for h in h_list]
        ud_m = [get_delayed(step + 0.5 - h) for h in h_list]
        ud_n1 = [get_delayed(step + 1 - h) for h in h_list]
        k1 = multi_delay_burgers_rhs(u, ud_n, u_target, p, grid)
        k2 = multi_delay_burgers_rhs(u + 0.5*p.dt*k1, ud_m, u_target, p, grid)
        k3 = multi_delay_burgers_rhs(u + 0.5*p.dt*k2, ud_m, u_target, p, grid)
        k4 = multi_delay_burgers_rhs(u + p.dt*k3, ud_n1, u_target, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"MultiDelayBurgers NaN at step {step+1}")
    return traj


def sample_multi_delay_burgers_param_set(rng):
    cand = [0.25, 0.5, 0.75, 1.0, 1.5]
    taus = tuple(sorted(rng.choice(cand, size=3, replace=False)))
    alphas = tuple(float(rng.uniform(0.3, 0.7)) for _ in range(3))
    return MultiDelayBurgersParams(
        nu=float(rng.choice([0.025, 0.05])),
        alphas=alphas, taus=taus,
        u_target_amp=float(rng.uniform(-0.2, 0.2)))


# ---------------------------------------------------------------------
# B10: Multi-delay Kuramoto 2D
#     d/dt theta = omega + sum_i K_i (G * sin(theta(t-tau_i) - theta(t)))
# ---------------------------------------------------------------------
@dataclass
class MultiDelayKuramotoParams:
    Ks: tuple = (0.5, 0.4, 0.3)
    sigma: float = 0.5
    taus: tuple = (0.25, 0.5, 1.0)
    omega_std: float = 0.1
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def multi_delay_kuramoto_rhs(theta, theta_d_list, omega, G_hat, p):
    coupling = np.zeros_like(theta)
    for K, td in zip(p.Ks, theta_d_list):
        s_d = np.sin(td); c_d = np.cos(td)
        S = np.real(np.fft.ifft2(G_hat * np.fft.fft2(s_d)))
        C = np.real(np.fft.ifft2(G_hat * np.fft.fft2(c_d)))
        coupling = coupling + K * (S * np.cos(theta) - C * np.sin(theta))
    return omega + coupling


def simulate_multi_delay_kuramoto(p, rng, grid):
    n_steps = int(round(p.T_total / p.dt))
    h_list = [int(round(t / p.dt)) for t in p.taus]
    max_h = max(h_list)
    G_hat = np.exp(-grid.K2 * p.sigma ** 2 / 2.0)
    omega = p.omega_std * rng.standard_normal((p.n_grid, p.n_grid))
    theta0 = smooth_random_field_2d(rng, grid, amplitude=np.pi / 2, k_max=4, base=0.0)
    history = np.tile(theta0[None, ...], (max_h + 1, 1, 1))
    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = theta0
    def get_delayed(s):
        if s <= 0: return history[0]
        if s >= step: return traj[step]
        i = int(np.floor(s)); f = s - i
        return (1 - f) * traj[i] + f * traj[i + 1]
    for step in range(n_steps):
        th = traj[step]
        td_n = [get_delayed(step - h) for h in h_list]
        td_m = [get_delayed(step + 0.5 - h) for h in h_list]
        td_n1 = [get_delayed(step + 1 - h) for h in h_list]
        k1 = multi_delay_kuramoto_rhs(th, td_n, omega, G_hat, p)
        k2 = multi_delay_kuramoto_rhs(th + 0.5*p.dt*k1, td_m, omega, G_hat, p)
        k3 = multi_delay_kuramoto_rhs(th + 0.5*p.dt*k2, td_m, omega, G_hat, p)
        k4 = multi_delay_kuramoto_rhs(th + p.dt*k3, td_n1, omega, G_hat, p)
        traj[step + 1] = th + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"MultiDelayKuramoto NaN at step {step+1}")
    return traj


def sample_multi_delay_kuramoto_param_set(rng):
    cand = [0.25, 0.5, 0.75, 1.0]
    taus = tuple(sorted(rng.choice(cand, size=3, replace=False)))
    Ks = tuple(float(rng.uniform(0.3, 0.7)) for _ in range(3))
    return MultiDelayKuramotoParams(
        Ks=Ks, sigma=float(rng.choice([0.3, 0.5, 0.8])), taus=taus,
        omega_std=float(rng.choice([0.05, 0.1, 0.2])))


# ---------------------------------------------------------------------
# DISTRIBUTED-DELAY VARIANTS (different K(s) kernels)
# All share f(u) = u*(1-u) logistic reaction; differ by kernel shape.
# ---------------------------------------------------------------------

def _build_kernel_weights(kernel_type: str, n_quad: int, dt: float,
                           tau: float, params: dict) -> np.ndarray:
    """Build trapezoidal-quadrature weights for various K(s) kernels.

    Returns w_i = K(s_i) * dt with ends halved, normalized to sum=1.
    """
    s = np.arange(n_quad) * dt
    if kernel_type == "exp":
        K = (1.0 / tau) * np.exp(-s / tau)
    elif kernel_type == "gaussian":
        mu = params.get("mu", 0.5 * tau)
        sigma = params.get("sigma", 0.2 * tau)
        K = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((s - mu) ** 2) / (2 * sigma ** 2))
    elif kernel_type == "gamma":
        import math as _m
        k = params.get("k", 2.0)
        K = (s ** (k - 1) / (tau ** k * _m.gamma(k))) * np.exp(-s / tau)
    elif kernel_type == "uniform":
        K = np.where(s <= tau, 1.0 / tau, 0.0)
    elif kernel_type == "powerlaw":
        s0 = params.get("s0", dt)
        alpha = params.get("alpha", 1.5)
        K = (s + s0) ** (-alpha)
    else:
        raise ValueError(kernel_type)
    w = np.full(n_quad, dt, dtype=np.float64)
    w[0] *= 0.5; w[-1] *= 0.5
    weights = K * w
    s_total = weights.sum()
    return weights / max(s_total, 1e-12)


@dataclass
class DistKernelRDParams:
    """Distributed-delay RD with parameterized kernel."""
    kernel_type: str = "exp"          # exp, gaussian, gamma, uniform, powerlaw
    A: float = 1.0
    tau: float = 0.5                  # primary memory scale
    tau_max: float = 2.0              # quadrature truncation
    kernel_extra: dict = None         # extra params (mu, sigma, k, alpha, etc.)
    D: float = 0.05
    T_total: float = 16.0
    dt: float = 0.01
    n_grid: int = 64
    L: float = 2 * np.pi


def dist_kernel_rd_rhs(u, history, kernel_weights, p, grid):
    diffusion = p.D * laplacian2d(u, grid)
    f_u = p.A * history * (1.0 - history)
    integral = np.einsum("q,qij->ij", kernel_weights, f_u)
    return diffusion + integral


def simulate_dist_kernel_rd(p, rng, grid):
    n_steps = int(round(p.T_total / p.dt))
    n_quad = int(round(p.tau_max / p.dt))
    if abs(n_quad * p.dt - p.tau_max) > 1e-9:
        raise ValueError(f"tau_max={p.tau_max} not divisible by dt={p.dt}")
    sf = p.D * 2 * (np.pi * p.n_grid / p.L) ** 2 * p.dt
    if sf > 2.78:
        raise ValueError(f"unstable D*k^2*dt = {sf:.2f}")
    kernel_weights = _build_kernel_weights(p.kernel_type, n_quad, p.dt, p.tau,
                                              p.kernel_extra or {})
    u0 = smooth_random_field_2d(rng, grid, amplitude=0.2, k_max=4, base=0.5)
    history = np.tile(u0[None, ...], (n_quad, 1, 1))
    traj = np.zeros((n_steps + 1, p.n_grid, p.n_grid), dtype=np.float64)
    traj[0] = u0
    def get_history(step):
        out = np.empty((n_quad, p.n_grid, p.n_grid))
        n_traj = min(step + 1, n_quad)
        n_init = n_quad - n_traj
        if n_init > 0:
            out[n_traj:] = history[0]
        if n_traj > 0:
            out[:n_traj] = traj[step + 1 - n_traj : step + 1][::-1]
        return out
    for step in range(n_steps):
        u = traj[step]
        hist = get_history(step)
        k1 = dist_kernel_rd_rhs(u, hist, kernel_weights, p, grid)
        k2 = dist_kernel_rd_rhs(u + 0.5 * p.dt * k1, hist, kernel_weights, p, grid)
        k3 = dist_kernel_rd_rhs(u + 0.5 * p.dt * k2, hist, kernel_weights, p, grid)
        k4 = dist_kernel_rd_rhs(u + p.dt * k3, hist, kernel_weights, p, grid)
        traj[step + 1] = u + (p.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(traj[step + 1]).all():
            raise RuntimeError(f"DistKernelRD NaN at step {step+1}")
    return traj


def sample_dist_kernel_rd_param_set(rng, kernel_type: str):
    tau = float(rng.choice([0.25, 0.5, 1.0]))
    tau_max = 4.0 * tau if kernel_type != "uniform" else tau
    extra = {}
    if kernel_type == "gaussian":
        extra = {"mu": float(rng.uniform(0.3, 0.8)) * tau,
                 "sigma": float(rng.uniform(0.1, 0.3)) * tau}
        tau_max = max(tau_max, extra["mu"] + 4 * extra["sigma"])
        # round tau_max to dt grid
        tau_max = round(tau_max / 0.01) * 0.01
    elif kernel_type == "gamma":
        extra = {"k": float(rng.choice([1.5, 2.0, 3.0]))}
    elif kernel_type == "powerlaw":
        extra = {"s0": 0.05, "alpha": float(rng.uniform(1.2, 2.0))}
    return DistKernelRDParams(
        kernel_type=kernel_type, A=float(rng.uniform(0.5, 2.0)),
        tau=tau, tau_max=tau_max, kernel_extra=extra,
        D=float(rng.choice([0.025, 0.05])))


def hopf_onset_sweep(out_dir: Path, taus=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                      beta=2.0, T_total=20.0, dt=0.05, n_grid=64, seed=42):
    """Sweep tau across the predicted Hopf-onset region; verify oscillation
    amplitude grows past beta*tau ~ pi (Wright-style criterion).

    For the diffusion-coupled Mackey-Glass, the spatial-mean field u_bar(t)
    behaves approximately like the 0D Mackey-Glass (since the Laplacian
    averages to 0).  We use peak-to-peak amplitude of u_bar(t) over the
    last half of the trajectory as a proxy for oscillation strength.
    """
    print(f"\n=== HOPF ONSET SWEEP (vary tau, fixed beta={beta}) ===")
    print(f"   T={T_total}, dt={dt}, grid={n_grid}")
    print(f"   {'tau':>5}  {'beta*tau':>9}  {'>pi?':>5}  {'osc.amp':>9}  {'mean':>7}")
    for tau in taus:
        p = MackeyGlassParams(beta=beta, tau=tau, T_total=T_total, dt=dt,
                              n_grid=n_grid)
        grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
        rng = np.random.default_rng(seed)
        traj = simulate_mackey_glass(p, rng, grid)
        u_bar = traj.mean(axis=(1, 2))
        # Use second half for steady-state oscillation amplitude.
        ss = u_bar[len(u_bar) // 2:]
        amp = float(ss.max() - ss.min())
        mean = float(ss.mean())
        bt = beta * tau
        print(f"   {tau:>5.2f}  {bt:>9.3f}  {('YES' if bt > np.pi else 'no'):>5}"
              f"  {amp:>9.4f}  {mean:>7.3f}")
    print("\n   expectation: osc.amp should be small (< 0.05) for beta*tau < pi,"
          " and grow noticeably past pi.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=str, required=True,
                    choices=["mackey_glass_2d", "wright_2d", "hutchinson_2d",
                              "dist_delay_rd_2d", "delay_burgers_2d",
                              "kuramoto_2d", "multi_delay_mg_2d",
                              "dist_exp_rd_2d", "dist_gaussian_rd_2d",
                              "dist_gamma_rd_2d", "dist_uniform_rd_2d",
                              "dist_powerlaw_rd_2d"])
    ap.add_argument("--num_train", type=int, default=256)
    ap.add_argument("--num_val", type=int, default=64)
    ap.add_argument("--num_test", type=int, default=64)
    ap.add_argument("--num_points", type=int, default=64)
    ap.add_argument("--T_total", type=float, default=16.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_hist", type=int, default=64)
    ap.add_argument("--n_out", type=int, default=64)
    ap.add_argument("--out_dir", default="data_dde_pde")
    ap.add_argument("--audit_only", action="store_true",
                    help="Run only the single-trajectory audit, skip data gen.")
    args = ap.parse_args()

    audit_dir = Path(args.out_dir) / args.family / "audit"
    if args.family == "mackey_glass_2d":
        print("--- B1 Mackey-Glass + diffusion 2D audit ---")
        traj = audit_one_trajectory(audit_dir,
                              beta=2.0, tau=2.0, n=10.0, gamma=1.0, D=0.1,
                              T_total=args.T_total, dt=args.dt,
                              n_grid=args.num_points, seed=42)
        try:
            visualize_trajectory(traj, audit_dir / "audit_traj_b2.0_t2.0.png",
                                  n_frames=8, title="MG+diff 2D, beta=2.0, tau=2.0")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
        convergence_study_mg(audit_dir,
                              beta=2.0, tau=2.0, T_total=10.0,
                              n_grid=args.num_points, seed=42,
                              dt_list=(0.01, 0.005, 0.0025, 0.00125))
        hopf_onset_sweep(audit_dir, taus=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                          beta=2.0, T_total=args.T_total, dt=args.dt,
                          n_grid=args.num_points, seed=42)
    elif args.family == "wright_2d":
        print("--- B2 Wright + diffusion 2D audit ---")
        traj = audit_wright(audit_dir,
                              alpha=1.5, tau=1.0, D=0.05,
                              T_total=args.T_total, dt=args.dt,
                              n_grid=args.num_points, seed=42)
        try:
            visualize_trajectory(traj, audit_dir / "audit_traj_a1.5_t1.0.png",
                                  n_frames=8, title="Wright+diff 2D, alpha=1.5, tau=1.0")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
        hopf_onset_sweep_wright(audit_dir,
                                 alphas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                                 tau=1.0, T_total=args.T_total, dt=args.dt,
                                 n_grid=args.num_points, seed=42)
    elif args.family == "hutchinson_2d":
        print("--- B3 Hutchinson + diffusion 2D audit ---")
        traj = audit_hutchinson(audit_dir,
                                 r=1.0, K=1.0, tau=1.5, D=0.05,
                                 T_total=args.T_total, dt=args.dt,
                                 n_grid=args.num_points, seed=42)
        try:
            visualize_trajectory(traj, audit_dir / "audit_traj_r1.0_t1.5.png",
                                  n_frames=8, title="Hutchinson+diff 2D, r=1, tau=1.5")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
        hopf_onset_sweep_hutchinson(audit_dir,
                                     rs=(0.3, 0.6, 1.0, 1.5, 2.0, 2.5),
                                     tau=1.0, T_total=args.T_total, dt=args.dt,
                                     n_grid=args.num_points, seed=42)
    elif args.family == "dist_delay_rd_2d":
        print("--- B4 Distributed-delay reaction-diffusion 2D audit ---")
        traj = audit_dist_delay_rd(audit_dir,
                                     A=1.0, tau=0.5, D=0.05,
                                     T_total=args.T_total, dt=args.dt,
                                     n_grid=args.num_points, seed=42)
        try:
            visualize_trajectory(traj, audit_dir / "audit_traj_A1.0_t0.5.png",
                                  n_frames=8, title="DistDelayRD 2D, A=1.0, tau=0.5")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
    elif args.family == "delay_burgers_2d":
        print("--- B5 Delayed-feedback Burgers 2D audit ---")
        traj = audit_delay_burgers(audit_dir,
                                     nu=0.05, alpha=1.0, tau=0.5, u_target_amp=0.0,
                                     T_total=args.T_total, dt=args.dt,
                                     n_grid=args.num_points, seed=42)
        try:
            visualize_trajectory(traj, audit_dir / "audit_traj_a1.0_t0.5.png",
                                  n_frames=8, title="DelayBurgers 2D, alpha=1.0, tau=0.5")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
    elif args.family in ("dist_exp_rd_2d", "dist_gaussian_rd_2d",
                            "dist_gamma_rd_2d", "dist_uniform_rd_2d",
                            "dist_powerlaw_rd_2d"):
        kernel_type = args.family.replace("dist_", "").replace("_rd_2d", "")
        print(f"--- DistKernelRD audit: kernel={kernel_type} ---")
        # Pick representative kernel params per type
        rng_audit = np.random.default_rng(42)
        p = sample_dist_kernel_rd_param_set(rng_audit, kernel_type)
        p = DistKernelRDParams(**{**p.__dict__, "T_total": args.T_total,
                                     "dt": args.dt, "n_grid": args.num_points,
                                     "L": 2 * np.pi})
        grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
        rng = np.random.default_rng(42)
        t0 = time.time()
        traj = simulate_dist_kernel_rd(p, rng, grid)
        print(f"  wall: {time.time()-t0:.1f}s, shape: {traj.shape}")
        print(f"  finite: {np.isfinite(traj).all()}, range: [{traj.min():.4f}, {traj.max():.4f}]")
        print(f"  kernel: {kernel_type}, A={p.A:.3f}, tau={p.tau:.3f}, tau_max={p.tau_max:.3f}, extra={p.kernel_extra}")
        try:
            visualize_trajectory(traj, audit_dir / f"audit_traj_{kernel_type}.png",
                                  n_frames=8, title=f"Dist-{kernel_type} RD 2D")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
    elif args.family == "multi_delay_mg_2d":
        print("--- B7 Multi-delay Mackey-Glass + diffusion 2D audit ---")
        # Quick single-trajectory sanity test.
        p = MultiDelayMGParams(betas=(1.0, 0.8, 0.6), taus=(1.0, 2.0, 3.0),
                                gamma=1.0, n=10.0, D=0.05,
                                T_total=args.T_total, dt=args.dt,
                                n_grid=args.num_points, L=2 * np.pi)
        grid = SpectralGrid2D.make(n=p.n_grid, L=p.L)
        rng = np.random.default_rng(42)
        t0 = time.time()
        traj = simulate_multi_delay_mg(p, rng, grid)
        print(f"  wall: {time.time()-t0:.1f}s, shape: {traj.shape}")
        print(f"  finite: {np.isfinite(traj).all()}")
        print(f"  range: [{traj.min():.4f}, {traj.max():.4f}]")
        u_bar = traj.mean(axis=(1, 2))
        zc = np.sum(np.diff(np.sign(u_bar - u_bar.mean())) != 0)
        print(f"  u_bar zero-crossings: {zc}  (multi-delay can produce complex oscillations)")
        try:
            visualize_trajectory(traj, audit_dir / "audit_traj_multi.png",
                                  n_frames=8, title="Multi-delay MG 2D, taus=(1,2,3)")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
    elif args.family == "kuramoto_2d":
        print("--- B6 Ring-coupled Kuramoto field 2D audit ---")
        traj = audit_kuramoto(audit_dir,
                                K=1.0, sigma=0.5, tau=0.5, omega_std=0.1,
                                T_total=args.T_total, dt=args.dt,
                                n_grid=args.num_points, seed=42)
        try:
            visualize_trajectory(traj, audit_dir / "audit_traj_K1.0_t0.5.png",
                                  n_frames=8, title="Kuramoto 2D, K=1.0, tau=0.5")
        except Exception as e:
            print(f"   (visualization skipped: {e})")
        synchronization_sweep_kuramoto(audit_dir,
                                         Ks=(0.1, 0.5, 1.0, 2.0, 4.0),
                                         sigma=0.5, tau=0.5, omega_std=0.1,
                                         T_total=args.T_total, dt=args.dt,
                                         n_grid=args.num_points, seed=42)
    else:
        raise ValueError(f"family {args.family} not implemented yet")

    if args.audit_only:
        return

    out_root = Path(args.out_dir) / args.family
    out_root.mkdir(parents=True, exist_ok=True)
    for split, num in [("train", args.num_train), ("val", args.num_val),
                        ("test", args.num_test)]:
        seed = {"train": 0, "val": 100_000, "test": 200_000}[split]
        print(f"\n--- Generating {split} split (n={num}) ---")
        phi, y, params, p_used = generate_split_generic(
            family=args.family, num_samples=num, seed=seed,
            n_hist=args.n_hist, n_out=args.n_out,
            dt=args.dt, n_grid=args.num_points)
        t_hist = np.arange(args.n_hist) * args.dt
        t_out = (args.n_hist + np.arange(args.n_out)) * args.dt
        write_shard(out_root, split, phi, y, params, t_hist, t_out)

    manifest = {
        "family": args.family,
        "spatial_dims": 2,
        "num_channels": 1,
        "spatial_shape": [args.num_points, args.num_points],
        "n_hist": args.n_hist, "n_out": args.n_out,
        "n_samples": {"train": args.num_train, "val": args.num_val,
                       "test": args.num_test},
        "params_dim": 3,
        "source": "scripts/gen_dde_pde_data.py",
        "delay_relevance": "explicit u(x, t-tau) term in dynamics",
        "dt": args.dt,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {out_root}/manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
