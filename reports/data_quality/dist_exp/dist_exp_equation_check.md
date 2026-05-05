# dist_exp Equation Verification (Step 1)

## Confirmed: dist_exp DOES use x(t-τ)

### Python Implementation (`src/dde/families.py:461-473`)

```python
def rhs(self, t: float, x: np.ndarray, x_delayed: Dict[str, np.ndarray],
        params: Dict[str, float]) -> np.ndarray:
    r, K, lam, tau = params["r"], params["K"], params["lam"], params["tau"]
    x_val, z_val = x[0], x[1]
    x_tau = x_delayed["tau"][0]  # ← DELAY TERM
    
    # Normalization constant for finite window
    C = (1.0 - np.exp(-lam * tau)) / lam
    
    dx = r * x_val * (1.0 - z_val / K)
    dz = -lam * z_val + (x_val - np.exp(-lam * tau) * x_tau) / C  # ← USES x(t-τ)
    
    return np.array([dx, dz])
```

### Julia Implementation (`src/dde/solve_julia/families.jl:182-188`)

```julia
function rhs_dist_exp!(du, u, h, p, t)
    x, z = u[1], u[2]
    x_τ = h(p, t - p.τ)[1]  # ← DELAY TERM
    C = (1.0 - exp(-p.λ * p.τ)) / p.λ
    du[1] = p.r * x * (1.0 - z / p.K)
    du[2] = -p.λ * z + (x - exp(-p.λ * p.τ) * x_τ) / C  # ← USES x(t-τ)
end
```

## The Problem: Regime Analysis

The z equation is:
```
z'(t) = -λz(t) + (x(t) - exp(-λτ)·x(t-τ)) / C
```

where `C = (1 - exp(-λτ))/λ`

### When θ = λτ >> 1:
- exp(-λτ) ≈ 0
- C ≈ 1/λ
- z'(t) ≈ -λz(t) + λx(t)  ← **NO DELAY DEPENDENCE**

### Current Parameter Ranges:
- λ ∈ [0.2, 6.0]
- τ ∈ [0.5, 2.0]
- **θ = λτ ∈ [0.1, 12.0]**

Most samples have θ >> 3, so exp(-θ) < 5% and the delay effectively disappears.

## Fix Required

Constrain θ = λτ ∈ [0.3, 2.5] to ensure:
- exp(-0.3) ≈ 0.74 (strong delay dependence)
- exp(-2.5) ≈ 0.08 (still non-negligible)

---
*Generated: 2024-12-29*
