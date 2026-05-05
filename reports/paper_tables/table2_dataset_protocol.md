# Table 2: Dataset Protocol & Splits

| Family | N_train | N_val | N_test | T | dt_out | tau_max | ID tau range | OOD-delay tau | OOD-history | OOD-horizon |
|--------|--------:|------:|-------:|--:|-------:|--------:|--------------|---------------|-------------|-------------|
| Hutchinson | 8,000 | 1,000 | 2,000 | 15.0 | ~0.06 | 2.0 | [0.1, 2.0] | tau > 1.3 | Spline phi | T=20 |
| Linear2 | 8,000 | 1,000 | 2,000 | 15.0 | ~0.06 | 2.0 | [0.1, 2.0] x 2 | max(tau) > 1.3 | Spline phi | T=20 |
| Van der Pol | 8,000 | 1,000 | 2,000 | 15.0 | ~0.06 | 2.0 | [0.1, 2.0] | tau > 1.3 | Spline phi | T=20 |
| DistUniform | 8,000 | 1,000 | 2,000 | 15.0 | ~0.06 | 2.0 | [0.1, 2.0] | tau > 1.3 | Spline phi | T=20 |
| DistExp | 8,000 | 1,000 | 2,000 | 15.0 | ~0.06 | 2.0 | [0.1, 2.0] | tau > 1.3 | Spline phi | T=20 |

**Notes:**
- All families use 256 output grid points (dt ~ 0.059)
- History grid: 64 points on [-tau_max, 0]
- OOD-history: Spline-interpolated history functions (vs piecewise-linear ID)
- OOD-horizon: Extended prediction horizon T=20 (vs T=15 ID)

**Location:** Main paper or Appendix