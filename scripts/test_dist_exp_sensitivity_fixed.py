#!/usr/bin/env python3
"""
Test dist_exp delay sensitivity with the FIXED theta-constrained sampling.

This generates fresh samples (not from old data) to verify the fix works.
"""
import numpy as np
import sys
sys.path.insert(0, "src")

from dde.families import DDE_FAMILIES
from dde.solve_python.dde_solver import solve_dde

def test_sensitivity():
    print("="*70)
    print("Testing dist_exp delay sensitivity with FIXED theta constraint")
    print("="*70)
    
    family = DDE_FAMILIES["dist_exp"]()
    rng = np.random.default_rng(42)
    
    tau_low, tau_high = 0.5, 1.8
    n_samples = 100
    sensitivities = []
    thetas = []
    
    for i in range(n_samples):
        # Sample with new theta-constrained method
        params = family.sample_params(rng)
        theta = params["lam"] * params["tau"]
        thetas.append(theta)
        
        # Generate history
        t_hist = np.linspace(-family.config.tau_max, 0, 64)
        phi = family.sample_history(rng, t_hist)
        
        try:
            # Solve with tau_low
            params_low = params.copy()
            params_low["tau"] = tau_low
            sol_low = solve_dde(family, params_low, phi, t_hist, T=15.0, n_points=256)
            
            # Solve with tau_high
            params_high = params.copy()
            params_high["tau"] = tau_high
            sol_high = solve_dde(family, params_high, phi, t_hist, T=15.0, n_points=256)
            
            # Compute relative difference
            y_low = sol_low.y
            y_high = sol_high.y
            
            diff = np.linalg.norm(y_low - y_high) / (np.linalg.norm(y_low) + 1e-10)
            sensitivities.append(diff)
            
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue
    
    sensitivities = np.array(sensitivities)
    thetas = np.array(thetas)
    
    print(f"\nSamples tested: {len(sensitivities)}")
    print(f"\nTheta (λτ) distribution:")
    print(f"  Range: [{thetas.min():.3f}, {thetas.max():.3f}]")
    print(f"  Mean: {thetas.mean():.3f}, Median: {np.median(thetas):.3f}")
    print(f"  In [0.3, 2.5]: {100*((thetas >= 0.3) & (thetas <= 2.5)).mean():.1f}%")
    
    print(f"\nDelay sensitivity (τ: {tau_low} → {tau_high}):")
    print(f"  Mean: {sensitivities.mean():.4f} ± {sensitivities.std():.4f}")
    print(f"  Median: {np.median(sensitivities):.4f}")
    print(f"  p25: {np.percentile(sensitivities, 25):.4f}")
    print(f"  p75: {np.percentile(sensitivities, 75):.4f}")
    print(f"  Range: [{sensitivities.min():.4f}, {sensitivities.max():.4f}]")
    
    # Pass criteria
    median_pass = np.median(sensitivities) >= 0.10  # 10% threshold
    p25_pass = np.percentile(sensitivities, 25) >= 0.05  # 5% for p25
    
    print(f"\n--- PASS CRITERIA ---")
    print(f"  Median ≥ 10%: {'✓ PASS' if median_pass else '✗ FAIL'} ({100*np.median(sensitivities):.1f}%)")
    print(f"  p25 ≥ 5%:     {'✓ PASS' if p25_pass else '✗ FAIL'} ({100*np.percentile(sensitivities, 25):.1f}%)")
    
    overall = median_pass and p25_pass
    print(f"\n  OVERALL: {'✓ PASS' if overall else '✗ FAIL'}")
    
    return overall

if __name__ == "__main__":
    test_sensitivity()
