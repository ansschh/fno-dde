#!/usr/bin/env python3
"""Quick test to verify dist_exp theta constraint fix."""
import numpy as np
import sys
sys.path.insert(0, "src")
from dde.families import DDE_FAMILIES

f = DDE_FAMILIES["dist_exp"]()
rng = np.random.default_rng(42)

thetas = []
for _ in range(1000):
    p = f.sample_params(rng)
    theta = p["lam"] * p["tau"]
    thetas.append(theta)

thetas = np.array(thetas)
print(f"Theta range: [{thetas.min():.3f}, {thetas.max():.3f}]")
print(f"Theta mean: {thetas.mean():.3f}, median: {np.median(thetas):.3f}")
print(f"In [0.3, 2.5]: {100*((thetas >= 0.3) & (thetas <= 2.5)).mean():.1f}%")
print(f"exp(-theta) median: {np.median(np.exp(-thetas)):.3f}")
