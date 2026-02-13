"""
DDE Family Definitions

Defines each DDE family with parameter sampling for operator learning.
Each family specifies:
  - The DDE dynamics
  - Parameter ranges
  - History function generation
  - Auxiliary state transformations (for distributed delays)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class DDEConfig:
    """Configuration for a DDE family."""
    name: str
    tau_max: float  # Maximum delay (history window)
    T: float  # Solution horizon
    n_grid: int  # Number of grid points
    state_dim: int  # Dimension of state
    param_names: List[str]  # Names of parameters
    param_ranges: Dict[str, Tuple[float, float]]  # Parameter sampling ranges
    requires_positive: bool = False  # Whether solution must stay positive
    stiff: bool = False  # Whether to use stiff solver (Radau)


class DDEFamily(ABC):
    """Abstract base class for DDE families."""
    
    def __init__(self, config: DDEConfig):
        self.config = config
    
    @abstractmethod
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray], 
            params: Dict[str, float]) -> np.ndarray:
        """Right-hand side of the DDE: dx/dt = rhs(t, x, x_delayed, params)."""
        pass
    
    @abstractmethod
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        """Return list of delays for given parameters."""
        pass
    
    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample parameters uniformly from ranges."""
        params = {}
        for name, (low, high) in self.config.param_ranges.items():
            params[name] = rng.uniform(low, high)
        return params
    
    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """
        Sample a smooth random history function using Fourier series.
        
        Args:
            rng: Random number generator
            t_hist: Time points on [-tau_max, 0]
            n_fourier: Number of Fourier modes
            
        Returns:
            History values at t_hist, shape (len(t_hist), state_dim)
        """
        L = self.config.tau_max
        state_dim = self.config.state_dim
        
        history = np.zeros((len(t_hist), state_dim))
        
        for d in range(state_dim):
            # Random Fourier coefficients
            c0 = rng.uniform(-1, 1)
            ak = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
            bk = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
            
            # Evaluate Fourier series
            phi = np.full_like(t_hist, c0)
            for k in range(1, n_fourier + 1):
                phi += ak[k-1] * np.cos(2 * np.pi * k * t_hist / L)
                phi += bk[k-1] * np.sin(2 * np.pi * k * t_hist / L)
            
            # Scale to reasonable range
            phi = 0.5 + 0.5 * phi  # Map to roughly [0, 1]
            
            if self.config.requires_positive:
                phi = np.abs(phi) + 0.1  # Ensure positive
            
            history[:, d] = phi
        
        return history


class Linear2DDE(DDEFamily):
    """
    Family 1: Scalar linear DDE with TWO discrete delays.
    
    x'(t) = a*x(t) + b1*x(t-τ1) + b2*x(t-τ2)
    
    Parameters: a, b1, b2, tau1, tau2
    
    Stable-biased sampling:
    - a in [-2.0, -0.1] (damping baseline)
    - b1, b2: 70% negative feedback, 30% full range
    """
    
    def __init__(self):
        config = DDEConfig(
            name="linear2",
            tau_max=2.0,
            T=10.0,
            n_grid=256,
            state_dim=1,
            param_names=["a", "b1", "b2", "tau1", "tau2"],
            param_ranges={
                "a": (-2.0, -0.1),  # Always damping
                "b1": (-1.5, 1.5),
                "b2": (-1.5, 1.5),
                "tau1": (0.1, 2.0),
                "tau2": (0.1, 2.0),
            },
            requires_positive=False,
        )
        super().__init__(config)
    
    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Stable-biased parameter sampling to reduce blowups."""
        a = rng.uniform(-2.0, -0.1)
        
        # 70% negative feedback, 30% full range
        if rng.random() < 0.7:
            b1 = -np.abs(rng.uniform(0, 1.5))
        else:
            b1 = rng.uniform(-1.5, 1.5)
        
        if rng.random() < 0.7:
            b2 = -np.abs(rng.uniform(0, 1.5))
        else:
            b2 = rng.uniform(-1.5, 1.5)
        
        tau1 = rng.uniform(0.1, 2.0)
        tau2 = rng.uniform(0.1, 2.0)
        # Ensure delays are not too close
        while np.abs(tau1 - tau2) < 0.2:
            tau2 = rng.uniform(0.1, 2.0)
        
        return {"a": a, "b1": b1, "b2": b2, "tau1": tau1, "tau2": tau2}
    
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        a, b1, b2 = params["a"], params["b1"], params["b2"]
        x_tau1 = x_delayed["tau1"][0]
        x_tau2 = x_delayed["tau2"][0]
        return np.array([a * x[0] + b1 * x_tau1 + b2 * x_tau2])
    
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau1"], params["tau2"]]


class HutchDDE(DDEFamily):
    """
    Family 2: Hutchinson / delayed logistic equation.
    
    x'(t) = r * x(t) * (1 - x(t - tau) / K)
    
    Parameters: r, K, tau
    """
    
    def __init__(self):
        config = DDEConfig(
            name="hutch",
            tau_max=2.0,
            T=15.0,
            n_grid=256,
            state_dim=1,
            param_names=["r", "K", "tau"],
            param_ranges={
                "r": (0.5, 3.0),
                "K": (0.5, 2.0),
                "tau": (0.1, 2.0),
            },
            requires_positive=True,
        )
        super().__init__(config)
    
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        r, K = params["r"], params["K"]
        x_tau = x_delayed["tau"]
        return np.array([r * x[0] * (1 - x_tau[0] / K)])
    
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]


class MackeyGlassDDE(DDEFamily):
    """
    Family 3: Mackey-Glass equation.
    
    x'(t) = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)
    
    Parameters: beta, gamma, tau (n fixed at 10)
    """
    
    def __init__(self, n: float = 10.0):
        config = DDEConfig(
            name="mackey_glass",
            tau_max=2.0,
            T=20.0,
            n_grid=512,
            state_dim=1,
            param_names=["beta", "gamma", "tau"],
            param_ranges={
                "beta": (1.0, 4.0),
                "gamma": (0.5, 3.0),
                "tau": (0.2, 2.0),
            },
            requires_positive=True,
        )
        super().__init__(config)
        self.n = n
    
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        beta, gamma = params["beta"], params["gamma"]
        x_tau = x_delayed["tau"]
        x_tau_val = max(x_tau[0], 1e-10)  # Avoid division issues
        dx = beta * x_tau_val / (1 + x_tau_val ** self.n) - gamma * x[0]
        return np.array([dx])
    
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]


class PredatorPreyDDE(DDEFamily):
    """
    Family 4: 2D predator-prey with two discrete delays.
    
    x'(t) = x(t) * (alpha - beta * y(t - tau1))
    y'(t) = y(t) * (-delta + gamma * x(t - tau2))
    
    Parameters: alpha, beta, gamma, delta, tau1, tau2
    """
    
    def __init__(self):
        config = DDEConfig(
            name="predator_prey",
            tau_max=2.0,
            T=15.0,
            n_grid=256,
            state_dim=2,
            param_names=["alpha", "beta", "gamma", "delta", "tau1", "tau2"],
            param_ranges={
                "alpha": (0.5, 2.0),
                "beta": (0.5, 2.0),
                "gamma": (0.5, 2.0),
                "delta": (0.5, 2.0),
                "tau1": (0.1, 2.0),
                "tau2": (0.1, 2.0),
            },
            requires_positive=True,
        )
        super().__init__(config)
    
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        alpha, beta = params["alpha"], params["beta"]
        gamma, delta = params["gamma"], params["delta"]
        
        y_tau1 = x_delayed["tau1"][1]  # y at t - tau1
        x_tau2 = x_delayed["tau2"][0]  # x at t - tau2
        
        dx = x[0] * (alpha - beta * y_tau1)
        dy = x[1] * (-delta + gamma * x_tau2)
        
        return np.array([dx, dy])
    
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau1"], params["tau2"]]


class VdPDDE(DDEFamily):
    """
    Family 4: Van der Pol oscillator with delayed feedback.
    
    x'(t) = v(t)
    v'(t) = μ*(1 - x²)*v - x + κ*x(t-τ)
    
    Parameters: mu, kappa, tau
    State: [x, v]
    """
    
    def __init__(self):
        config = DDEConfig(
            name="vdp",
            tau_max=2.0,
            T=20.0,
            n_grid=512,
            state_dim=2,
            param_names=["mu", "kappa", "tau"],
            param_ranges={
                "mu": (0.5, 3.0),
                "kappa": (-2.0, 2.0),
                "tau": (0.1, 2.0),
            },
            requires_positive=False,
        )
        super().__init__(config)
    
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        mu, kappa = params["mu"], params["kappa"]
        x_val, v_val = x[0], x[1]
        x_tau = x_delayed["tau"][0]
        
        dx = v_val
        dv = mu * (1.0 - x_val**2) * v_val - x_val + kappa * x_tau
        
        return np.array([dx, dv])
    
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]
    
    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """Sample consistent history for VdP: v = dx/dt."""
        L = self.config.tau_max
        
        # Sample x as Fourier series
        c0 = rng.uniform(-1, 1)
        ak = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
        bk = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
        
        x_hist = np.full_like(t_hist, c0)
        v_hist = np.zeros_like(t_hist)
        
        for k in range(1, n_fourier + 1):
            omega = 2 * np.pi * k / L
            x_hist += ak[k-1] * np.cos(omega * t_hist)
            x_hist += bk[k-1] * np.sin(omega * t_hist)
            # v = dx/dt
            v_hist += -ak[k-1] * omega * np.sin(omega * t_hist)
            v_hist += bk[k-1] * omega * np.cos(omega * t_hist)
        
        return np.stack([x_hist, v_hist], axis=-1)


class DistUniformDDE(DDEFamily):
    """
    Family 5: Distributed delay - uniform kernel (moving average).
    
    x'(t) = r*x(t)*(1 - m(t)/K)
    m(t) = (1/τ) ∫_{t-τ}^t x(s) ds
    
    Auxiliary form:
        m'(t) = (x(t) - x(t-τ))/τ
    
    Parameters: r, K, tau
    State: [x, m]
    
    Diversity-tuned: larger r, larger tau for more oscillations
    """
    
    def __init__(self):
        config = DDEConfig(
            name="dist_uniform",
            tau_max=2.0,
            T=15.0,
            n_grid=256,
            state_dim=2,
            param_names=["r", "K", "tau"],
            param_ranges={
                "r": (1.0, 6.0),      # Increased from (0.5, 3.0)
                "K": (0.5, 2.0),
                "tau": (0.5, 2.0),    # Bias toward larger tau
            },
            requires_positive=True,
        )
        super().__init__(config)
    
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        r, K, tau = params["r"], params["K"], params["tau"]
        x_val, m_val = x[0], x[1]
        x_tau = x_delayed["tau"][0]
        
        dx = r * x_val * (1.0 - m_val / K)
        dm = (x_val - x_tau) / tau
        
        return np.array([dx, dm])
    
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]
    
    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """Sample positive history with off-equilibrium starts for diversity."""
        L = self.config.tau_max
        
        # Off-equilibrium: baseline multiplier far from K
        c_mult = rng.uniform(0.2, 2.0)
        c0 = c_mult * 0.5  # Scale relative to typical K
        
        # Larger amplitude variations
        ak = rng.uniform(-0.3, 0.3, size=n_fourier) / np.arange(1, n_fourier + 1)
        bk = rng.uniform(-0.3, 0.3, size=n_fourier) / np.arange(1, n_fourier + 1)
        
        x_hist = np.full_like(t_hist, c0)
        for k in range(1, n_fourier + 1):
            x_hist += ak[k-1] * np.cos(2 * np.pi * k * t_hist / L)
            x_hist += bk[k-1] * np.sin(2 * np.pi * k * t_hist / L)
        
        x_hist = np.abs(x_hist) + 0.1  # Ensure positive
        m_val = np.mean(x_hist)
        m_hist = np.full_like(t_hist, m_val)
        
        return np.stack([x_hist, m_hist], axis=-1)


class DistExpDDE(DDEFamily):
    """
    Family 6: Distributed delay - FINITE-WINDOW exponential kernel.
    
    This is a TRUE distributed delay DDE (not a pure ODE).
    
    The distributed delay integral:
        z(t) = (1/C) ∫_{t-τ}^t exp(-λ(t-s)) x(s) ds
    
    where C = (1 - exp(-λτ))/λ is the normalization constant.
    
    Auxiliary ODE form with ONE discrete lag:
        x'(t) = r*x(t)*(1 - z(t)/K)
        z'(t) = -λ*z(t) + (x(t) - exp(-λτ)*x(t-τ))/C
    
    Parameters: r, K, lam (λ), tau (τ)
    State: [x, z]
    
    Diversity-tuned: wider lambda range for memory variation
    """
    
    # Theta regime bounds for delay relevance (tightened for stronger sensitivity)
    THETA_MIN = 0.5   # exp(-0.5) ≈ 0.61 - strong delay dependence
    THETA_MAX = 1.8   # exp(-1.8) ≈ 0.17 - still substantial
    
    def __init__(self):
        config = DDEConfig(
            name="dist_exp",
            tau_max=2.0,
            T=15.0,
            n_grid=256,
            state_dim=2,
            param_names=["r", "K", "lam", "tau", "g"],  # Added coupling gain g
            param_ranges={
                "r": (3.0, 10.0),     # Fast dynamics - delay matters more
                "K": (0.5, 2.0),
                "lam": (0.3, 4.0),    # Will be constrained by theta
                "tau": (0.3, 2.0),    # Finite window size
                "g": (1.0, 2.0),      # Coupling gain for delay sensitivity (stronger feedback)
            },
            requires_positive=True,
        )
        super().__init__(config)
    
    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """
        Sample parameters with θ = λτ constrained to [0.5, 1.8].
        
        This ensures the delay term exp(-λτ)·x(t-τ) remains significant,
        making dist_exp a true delay benchmark (not ODE-ish).
        """
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Sample base parameters
            r = rng.uniform(*self.config.param_ranges["r"])
            K = rng.uniform(*self.config.param_ranges["K"])
            g = rng.uniform(*self.config.param_ranges["g"])
            tau = rng.uniform(*self.config.param_ranges["tau"])
            lam = rng.uniform(*self.config.param_ranges["lam"])
            
            # Check theta constraint
            theta = lam * tau
            if self.THETA_MIN <= theta <= self.THETA_MAX:
                return {"r": r, "K": K, "lam": lam, "tau": tau, "g": g}
        
        # Fallback: force theta into valid range by adjusting lambda
        tau = rng.uniform(*self.config.param_ranges["tau"])
        theta = rng.uniform(self.THETA_MIN, self.THETA_MAX)
        lam = theta / tau
        r = rng.uniform(*self.config.param_ranges["r"])
        K = rng.uniform(*self.config.param_ranges["K"])
        g = rng.uniform(*self.config.param_ranges["g"])
        
        return {"r": r, "K": K, "lam": lam, "tau": tau, "g": g}
    
    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        r, K, lam, tau = params["r"], params["K"], params["lam"], params["tau"]
        g = params.get("g", 1.0)  # Coupling gain (backward compatible)
        x_val, z_val = x[0], x[1]
        x_tau = x_delayed["tau"][0]
        
        # Normalization constant for finite window
        C = (1.0 - np.exp(-lam * tau)) / lam
        
        # x' = r*x*(1 - g*z/K) with coupling gain g
        dx = r * x_val * (1.0 - g * z_val / K)
        dz = -lam * z_val + (x_val - np.exp(-lam * tau) * x_tau) / C
        
        return np.array([dx, dz])
    
    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]  # Now has discrete delay!
    
    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """Sample positive history with off-equilibrium initial conditions."""
        L = self.config.tau_max
        
        # Off-equilibrium baseline - wider range for bigger transients
        c_mult = rng.uniform(0.3, 2.5)  # Increased range
        c0 = c_mult * 0.6
        
        # Higher amplitude perturbations for more diverse histories
        ak = rng.uniform(-0.4, 0.4, size=n_fourier) / np.arange(1, n_fourier + 1)
        bk = rng.uniform(-0.4, 0.4, size=n_fourier) / np.arange(1, n_fourier + 1)
        
        x_hist = np.full_like(t_hist, c0)
        for k in range(1, n_fourier + 1):
            x_hist += ak[k-1] * np.cos(2 * np.pi * k * t_hist / L)
            x_hist += bk[k-1] * np.sin(2 * np.pi * k * t_hist / L)
        
        x_hist = np.abs(x_hist) + 0.1  # Ensure positive
        
        # z(0) should be computed from integral, but for simplicity use mean
        # The solver will handle the transient
        z_val = np.mean(x_hist)
        z_hist = np.full_like(t_hist, z_val)
        
        return np.stack([x_hist, z_hist], axis=-1)


class NeutralDDE(DDEFamily):
    """
    Neutral DDE: derivative depends on past derivatives.

    d/dt[x(t) - a*x(t-tau)] = -b*x(t) + c*tanh(x(t-tau))

    Auxiliary form (retarded DDE in z):
        z(t) = x(t) - a*x(t-tau)
        dz/dt = -b*(z + a*x_tau) + c*tanh(x_tau)

    Recover x(t) = z(t) + a*x(t-tau) during post-processing.
    The family stores z as state, and the dataset pipeline sees z.

    Derivative discontinuities propagate at tau multiples without smoothing,
    producing non-smooth solutions that stress spectral methods.

    Parameters: a in [0.05, 0.45] (|a|<1 for well-posedness),
                b in [0.5, 3.0], c in [-2.0, 2.0], tau in [0.2, 2.0]
    """

    def __init__(self):
        config = DDEConfig(
            name="neutral",
            tau_max=2.0,
            T=15.0,
            n_grid=512,
            state_dim=1,
            param_names=["a", "b", "c", "tau"],
            param_ranges={
                "a": (0.05, 0.45),
                "b": (0.5, 3.0),
                "c": (-2.0, 2.0),
                "tau": (0.2, 2.0),
            },
            requires_positive=False,
        )
        super().__init__(config)

    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        a, b, c = params["a"], params["b"], params["c"]
        z_val = x[0]
        x_tau = x_delayed["tau"][0]

        x_current = z_val + a * x_tau
        dz = -b * x_current + c * np.tanh(x_tau)
        return np.array([dz])

    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]

    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """Sample history for z = x - a*x(t-tau).

        We sample x as a smooth Fourier series and compute z from it.
        For simplicity in the history window, we set z(t) = x(t) - a*x(t)
        since x(t-tau) is not always available analytically at arbitrary t.
        This creates a controlled discontinuity at t=0.
        """
        L = self.config.tau_max
        c0 = rng.uniform(-1, 1)
        ak = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
        bk = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)

        phi = np.full_like(t_hist, c0)
        for k in range(1, n_fourier + 1):
            phi += ak[k-1] * np.cos(2 * np.pi * k * t_hist / L)
            phi += bk[k-1] * np.sin(2 * np.pi * k * t_hist / L)

        return phi.reshape(-1, 1)


class ChaoticMackeyGlassDDE(DDEFamily):
    """
    Mackey-Glass equation in the chaotic regime.

    x'(t) = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    Standard MG is chaotic for tau > ~17, n=10, beta/gamma ~ 2/1.
    Long history window and horizon needed to observe chaotic dynamics.

    Parameters: beta in [1.8, 2.2], gamma in [0.9, 1.1], tau in [17, 30]
    """

    def __init__(self, n: float = 10.0):
        config = DDEConfig(
            name="chaotic_mg",
            tau_max=30.0,
            T=100.0,
            n_grid=1024,
            state_dim=1,
            param_names=["beta", "gamma", "tau"],
            param_ranges={
                "beta": (1.8, 2.2),
                "gamma": (0.9, 1.1),
                "tau": (17.0, 30.0),
            },
            requires_positive=True,
        )
        super().__init__(config)
        self.n = n

    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        beta, gamma = params["beta"], params["gamma"]
        x_tau = x_delayed["tau"]
        x_tau_val = max(x_tau[0], 1e-10)
        dx = beta * x_tau_val / (1 + x_tau_val ** self.n) - gamma * x[0]
        return np.array([dx])

    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]

    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """Sample positive history biased around the MG equilibrium (~1.0)."""
        L = self.config.tau_max
        c0 = rng.uniform(0.5, 1.5)
        ak = rng.uniform(-0.2, 0.2, size=n_fourier) / np.arange(1, n_fourier + 1)
        bk = rng.uniform(-0.2, 0.2, size=n_fourier) / np.arange(1, n_fourier + 1)

        phi = np.full_like(t_hist, c0)
        for k in range(1, n_fourier + 1):
            phi += ak[k-1] * np.cos(2 * np.pi * k * t_hist / L)
            phi += bk[k-1] * np.sin(2 * np.pi * k * t_hist / L)

        phi = np.abs(phi) + 0.1
        return phi.reshape(-1, 1)


class ForcedDelayDuffing(DDEFamily):
    """
    Forced Duffing oscillator with delayed feedback.

    x''(t) + delta*x'(t) + alpha*x(t) + beta*x(t)^3
        = gamma_f*cos(omega*t) + kappa*x(t-tau)

    State: [x, v] where v = x'

    The forcing frequency omega is the key difficulty knob:
    high omega creates multi-scale dynamics (fast oscillations + slow envelope)
    that stress spectral truncation in FNO.

    Parameters: delta in [0.1, 0.5], alpha in [-1.0, 1.0], beta in [0.1, 1.0],
                gamma_f in [0.1, 2.0], omega in [1.0, 10.0],
                kappa in [-1.0, 1.0], tau in [0.1, 2.0]
    """

    def __init__(self):
        config = DDEConfig(
            name="forced_duffing",
            tau_max=2.0,
            T=30.0,
            n_grid=512,
            state_dim=2,
            param_names=["delta", "alpha", "beta", "gamma_f", "omega", "kappa", "tau"],
            param_ranges={
                "delta": (0.1, 0.5),
                "alpha": (-1.0, 1.0),
                "beta": (0.1, 1.0),
                "gamma_f": (0.1, 2.0),
                "omega": (1.0, 10.0),
                "kappa": (-1.0, 1.0),
                "tau": (0.1, 2.0),
            },
            requires_positive=False,
        )
        super().__init__(config)

    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        delta = params["delta"]
        alpha = params["alpha"]
        beta = params["beta"]
        gamma_f = params["gamma_f"]
        omega = params["omega"]
        kappa = params["kappa"]
        x_val, v_val = x[0], x[1]
        x_tau = x_delayed["tau"][0]

        dx = v_val
        dv = (-delta * v_val - alpha * x_val - beta * x_val**3
              + gamma_f * np.cos(omega * t) + kappa * x_tau)
        return np.array([dx, dv])

    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]

    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """Sample consistent history: x from Fourier, v = dx/dt analytically."""
        L = self.config.tau_max

        c0 = rng.uniform(-1, 1)
        ak = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
        bk = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)

        x_hist = np.full_like(t_hist, c0)
        v_hist = np.zeros_like(t_hist)

        for k in range(1, n_fourier + 1):
            omega_k = 2 * np.pi * k / L
            x_hist += ak[k-1] * np.cos(omega_k * t_hist)
            x_hist += bk[k-1] * np.sin(omega_k * t_hist)
            v_hist += -ak[k-1] * omega_k * np.sin(omega_k * t_hist)
            v_hist += bk[k-1] * omega_k * np.cos(omega_k * t_hist)

        return np.stack([x_hist, v_hist], axis=-1)


class MultiDelayComb(DDEFamily):
    """
    Linear DDE with many incommensurate delays.

    x'(t) = a*x(t) + sum_i b_i * x(t - tau_i),  i=0..n_delays-1

    Incommensurate delays (golden-ratio spaced) create nasty interference
    patterns that blow up spectral bias in FNO.

    Parameters: a in [-2.0, -0.1] (always damping),
                b_i in [-1.0, 1.0], tau_i in [0.1, 3.0]
    """

    def __init__(self, n_delays: int = 6):
        self.n_delays = n_delays
        param_names = ["a"]
        param_names += [f"b{i}" for i in range(n_delays)]
        param_names += [f"tau{i}" for i in range(n_delays)]

        param_ranges = {"a": (-2.0, -0.1)}
        for i in range(n_delays):
            param_ranges[f"b{i}"] = (-1.0, 1.0)
            param_ranges[f"tau{i}"] = (0.1, 3.0)

        config = DDEConfig(
            name="multi_delay_comb",
            tau_max=3.0,
            T=15.0,
            n_grid=512,
            state_dim=1,
            param_names=param_names,
            param_ranges=param_ranges,
            requires_positive=False,
        )
        super().__init__(config)

    def sample_params(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample with incommensurate delays via golden-ratio spacing."""
        a = rng.uniform(-2.0, -0.1)

        params = {"a": a}

        # 70% negative feedback for stability
        for i in range(self.n_delays):
            if rng.random() < 0.7:
                params[f"b{i}"] = -np.abs(rng.uniform(0, 1.0))
            else:
                params[f"b{i}"] = rng.uniform(-1.0, 1.0)

        # Golden-ratio-based delay spacing for incommensurability
        phi_gr = (1 + np.sqrt(5)) / 2
        tau_base = rng.uniform(0.2, 0.6)
        for i in range(self.n_delays):
            tau_val = tau_base * (phi_gr ** i)
            # Add small random perturbation
            tau_val += rng.uniform(-0.05, 0.05)
            tau_val = np.clip(tau_val, 0.1, 3.0)
            params[f"tau{i}"] = tau_val

        return params

    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        dx = params["a"] * x[0]
        for i in range(self.n_delays):
            x_tau_i = x_delayed[f"tau{i}"][0]
            dx += params[f"b{i}"] * x_tau_i
        return np.array([dx])

    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params[f"tau{i}"] for i in range(self.n_delays)]


class StiffVdPDDE(DDEFamily):
    """
    Stiff Van der Pol oscillator with delayed feedback.

    x'(t) = v(t)
    v'(t) = mu*(1 - x^2)*v - x + kappa*x(t-tau)

    mu in [10, 50] produces relaxation oscillations with sharp transitions
    (fast jumps between slow manifold branches). These sharp features in time
    create high-frequency content that stresses FNO spectral truncation.

    Parameters: mu in [10, 50], kappa in [-1.0, 1.0], tau in [0.1, 2.0]
    """

    def __init__(self):
        config = DDEConfig(
            name="stiff_vdp",
            tau_max=2.0,
            T=30.0,
            n_grid=1024,
            state_dim=2,
            param_names=["mu", "kappa", "tau"],
            param_ranges={
                "mu": (10.0, 50.0),
                "kappa": (-1.0, 1.0),
                "tau": (0.1, 2.0),
            },
            requires_positive=False,
        )
        # Mark as stiff for solver selection
        config.stiff = True
        super().__init__(config)

    def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
            params: Dict[str, float]) -> np.ndarray:
        mu, kappa = params["mu"], params["kappa"]
        x_val, v_val = x[0], x[1]
        x_tau = x_delayed["tau"][0]

        dx = v_val
        dv = mu * (1.0 - x_val**2) * v_val - x_val + kappa * x_tau
        return np.array([dx, dv])

    def get_delays(self, params: Dict[str, float]) -> List[float]:
        return [params["tau"]]

    def sample_history(self, rng: np.random.Generator, t_hist: np.ndarray,
                       n_fourier: int = 5) -> np.ndarray:
        """Sample consistent history for stiff VdP: v = dx/dt."""
        L = self.config.tau_max

        c0 = rng.uniform(-1, 1)
        ak = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)
        bk = rng.uniform(-1, 1, size=n_fourier) / np.arange(1, n_fourier + 1)

        x_hist = np.full_like(t_hist, c0)
        v_hist = np.zeros_like(t_hist)

        for k in range(1, n_fourier + 1):
            omega = 2 * np.pi * k / L
            x_hist += ak[k-1] * np.cos(omega * t_hist)
            x_hist += bk[k-1] * np.sin(omega * t_hist)
            v_hist += -ak[k-1] * omega * np.sin(omega * t_hist)
            v_hist += bk[k-1] * omega * np.cos(omega * t_hist)

        return np.stack([x_hist, v_hist], axis=-1)


# Registry of all DDE families
DDE_FAMILIES = {
    "linear2": Linear2DDE,
    "hutch": HutchDDE,
    "mackey_glass": MackeyGlassDDE,
    "vdp": VdPDDE,
    "predator_prey": PredatorPreyDDE,
    "dist_uniform": DistUniformDDE,
    "dist_exp": DistExpDDE,
    "neutral": NeutralDDE,
    "chaotic_mg": ChaoticMackeyGlassDDE,
    "forced_duffing": ForcedDelayDuffing,
    "multi_delay_comb": MultiDelayComb,
    "stiff_vdp": StiffVdPDDE,
}


def get_family(name: str) -> DDEFamily:
    """Get a DDE family by name."""
    if name not in DDE_FAMILIES:
        raise ValueError(f"Unknown DDE family: {name}. Available: {list(DDE_FAMILIES.keys())}")
    return DDE_FAMILIES[name]()
