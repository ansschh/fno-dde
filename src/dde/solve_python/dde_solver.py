"""
Pure Python DDE Solver using scipy

A simple method-of-steps solver for DDEs with constant delays.
For production use, consider Julia's DifferentialEquations.jl.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DDESolution:
    """Container for DDE solution."""
    t: np.ndarray  # Time points
    y: np.ndarray  # Solution values, shape (n_times, state_dim)
    success: bool
    message: str


class DDESolver:
    """
    Simple DDE solver using method of steps with scipy's ODE solvers.
    
    This implementation:
    - Uses cubic spline interpolation for history
    - Advances solution in steps of size tau_min
    - Tracks solution history for delayed evaluations
    """
    
    def __init__(
        self,
        rhs: Callable,
        delays: List[float],
        history_func: Callable[[float], np.ndarray],
        t_span: Tuple[float, float],
        params: Dict[str, float],
        rtol: float = 1e-6,
        atol: float = 1e-9,
        stiff: bool = False,
    ):
        """
        Initialize DDE solver.

        Args:
            rhs: Right-hand side function f(t, y, y_delayed, params)
            delays: List of delay values
            history_func: Function phi(t) for t <= 0
            t_span: (t0, tf) integration interval
            params: Dictionary of parameters
            rtol, atol: Solver tolerances
            stiff: Use implicit (Radau) solver for stiff problems
        """
        self.rhs = rhs
        self.delays = sorted(delays) if delays else []
        self.tau_min = min(delays) if delays else 1.0  # Default step for ODEs
        self.tau_max = max(delays) if delays else 0.0
        self.history_func = history_func
        self.t_span = t_span
        self.params = params
        self.rtol = rtol
        self.atol = atol
        self.stiff = stiff
        self.method = 'Radau' if stiff else 'RK45'
        
        # Storage for solution history (for interpolation)
        self._t_history: List[np.ndarray] = []
        self._y_history: List[np.ndarray] = []
        self._interpolators: List[CubicSpline] = []
    
    def _get_delayed_value(self, t: float, delay: float) -> np.ndarray:
        """Get solution value at t - delay using interpolation."""
        t_delayed = t - delay
        
        if t_delayed <= 0:
            return self.history_func(t_delayed)
        
        # Find appropriate interpolator
        for i, (t_seg, interp) in enumerate(zip(self._t_history, self._interpolators)):
            if t_seg[0] <= t_delayed <= t_seg[-1]:
                return interp(t_delayed)
        
        # If we're at the boundary, use the last point
        if len(self._y_history) > 0:
            return self._y_history[-1][-1]
        
        return self.history_func(0)
    
    def _make_ode_rhs(self):
        """Create ODE right-hand side for current step.

        Supports arbitrary numbers of named delays by scanning param keys
        for any key starting with 'tau'.
        """
        # Pre-compute the delay name → value mapping once
        delay_map = self._build_delay_map()

        def ode_rhs(t, y):
            y_delayed = {}

            if not self.delays:
                return self.rhs(t, y, y_delayed, self.params)

            for name, delay_val in delay_map.items():
                y_delayed[name] = self._get_delayed_value(t, delay_val)

            return self.rhs(t, y, y_delayed, self.params)

        return ode_rhs

    def _build_delay_map(self) -> Dict[str, float]:
        """Build mapping from delay names to delay values.

        Handles:
        - Single delay: "tau" in params → {"tau": value}
        - Two delays: "tau1", "tau2" in params → {"tau1": v1, "tau2": v2}
        - N delays: "tau0".."tauN" in params → {"tau0": v0, ..., "tauN": vN}
        """
        # Collect all tau-prefixed param keys
        tau_keys = sorted([k for k in self.params if k.startswith("tau")])

        if len(tau_keys) == 0:
            return {}

        if len(tau_keys) == 1 and "tau" in tau_keys:
            return {"tau": self.params["tau"]}

        # Multiple delays — map name → value
        return {k: self.params[k] for k in tau_keys}
    
    def solve(self, n_points: int = 256, y_clip: float = 100.0) -> DDESolution:
        """
        Solve the DDE using method of steps with early termination.
        
        Args:
            n_points: Number of output points
            y_clip: Maximum allowed amplitude (early termination if exceeded)
            
        Returns:
            DDESolution with time points and solution values
        """
        t0, tf = self.t_span
        
        # Get initial condition from history at t=0
        y0 = self.history_func(0.0)
        
        # Determine step intervals (at least tau_min apart)
        step_size = self.tau_min
        n_steps = int(np.ceil((tf - t0) / step_size))
        
        all_t = [np.array([t0])]
        all_y = [y0.reshape(1, -1)]
        
        current_t = t0
        current_y = y0
        
        # Early termination event: stop when |y| exceeds y_clip
        def blowup_event(t, y):
            return y_clip - np.max(np.abs(y))
        blowup_event.terminal = True
        blowup_event.direction = -1
        
        for step in range(n_steps):
            step_end = min(current_t + step_size, tf)
            
            # Create ODE for this step
            ode_rhs = self._make_ode_rhs()
            
            # Solve ODE on this interval with early termination
            sol = solve_ivp(
                ode_rhs,
                (current_t, step_end),
                current_y,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=True,
                events=blowup_event,
            )
            
            # Check for early termination due to blowup
            if sol.t_events is not None and len(sol.t_events[0]) > 0:
                return DDESolution(
                    t=np.concatenate(all_t),
                    y=np.vstack(all_y),
                    success=False,
                    message=f"Early termination: amplitude exceeded {y_clip} at t={sol.t_events[0][0]:.3f}"
                )
            
            if not sol.success:
                return DDESolution(
                    t=np.concatenate(all_t),
                    y=np.vstack(all_y),
                    success=False,
                    message=f"Solver failed at step {step}: {sol.message}"
                )
            
            # Store solution for this segment
            t_segment = sol.t[1:]  # Exclude first point (already stored)
            y_segment = sol.y[:, 1:].T
            
            all_t.append(t_segment)
            all_y.append(y_segment)
            
            # Update history
            self._t_history.append(sol.t)
            self._y_history.append(sol.y.T)
            
            # Create interpolator for this segment
            if len(sol.t) >= 4:
                interp = CubicSpline(sol.t, sol.y.T)
            else:
                # Linear interpolation for short segments
                interp = CubicSpline(sol.t, sol.y.T, bc_type='natural')
            self._interpolators.append(interp)
            
            # Update for next step
            current_t = step_end
            current_y = sol.y[:, -1]
            
            if current_t >= tf:
                break
        
        # Combine all segments
        t_full = np.concatenate(all_t)
        y_full = np.vstack(all_y)
        
        # Interpolate to uniform grid
        t_uniform = np.linspace(t0, tf, n_points)
        
        # Build full interpolator
        full_interp = CubicSpline(t_full, y_full)
        y_uniform = full_interp(t_uniform)
        
        return DDESolution(
            t=t_uniform,
            y=y_uniform,
            success=True,
            message="Success"
        )


def solve_dde(
    family,
    params: Dict[str, float],
    history: np.ndarray,
    t_hist: np.ndarray,
    T: float,
    n_points: int = 256,
) -> DDESolution:
    """
    Convenience function to solve a DDE given a family and parameters.
    
    Args:
        family: DDEFamily instance
        params: Parameter dictionary
        history: History values on t_hist, shape (n_hist, state_dim)
        t_hist: History time points (should end at 0)
        T: Final time
        n_points: Number of output points
        
    Returns:
        DDESolution
    """
    # Create history function from data
    if history.ndim == 1:
        history = history.reshape(-1, 1)
    
    history_interp = CubicSpline(t_hist, history)
    
    def history_func(t):
        if t < t_hist[0]:
            return history[0]
        elif t > t_hist[-1]:
            return history[-1]
        return history_interp(t)
    
    # Get delays
    delays = family.get_delays(params)
    
    # Detect stiff flag from family config
    stiff = getattr(family.config, 'stiff', False)

    # Create solver
    solver = DDESolver(
        rhs=family.rhs,
        delays=delays,
        history_func=history_func,
        t_span=(0.0, T),
        params=params,
        stiff=stiff,
    )

    return solver.solve(n_points)
