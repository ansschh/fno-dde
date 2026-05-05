# DDE Family Difficulty Ranking (v2)

Updated after dist_exp fix (theta constraint applied).

## ID Test Performance (relL2 median)

| Rank | Family | ID Median | ID p95 | Notes |
|------|--------|-----------|--------|-------|
| 1 | **dist_exp** | 0.0216 | 0.1255 | FIXED v2: now delay-influenced |
| 2 | hutch | 0.049 | 0.19 | Single discrete delay |
| 3 | dist_uniform | 0.086 | 0.441 | Distributed delay (uniform kernel) |
| 4 | vdp | 0.297 | 0.888 | Oscillatory, 2D state |
| 5 | linear2 | 0.58 | 1.68 | Two interacting delays |

## OOD Gaps Summary

| Family | OOD-delay | OOD-history | OOD-horizon |
|--------|-----------|-------------|-------------|
| hutch | **6.7x** | - | - |
| linear2 | 1.5x | - | - |
| vdp | 1.11x | 4.36x | 4.16x |
| dist_uniform | 1.83x | **7.02x** | 2.52x |
| dist_exp | 1.17x | 1.44x | 0.83x |

## Key Changes from v1

### dist_exp Fix

| Metric | v1 (INVALID) | v2 (FIXED) |
|--------|--------------|------------|
| θ = λτ range | [0.1, 12.0] | **[0.5, 1.8]** |
| exp(-θ) median | 3.45% | **32.7%** |
| Delay sensitivity | 1.87% ✗ | **11.8% ✓** |
| OOD-delay gap | 0.96x (fake) | **1.17x** (real) |

The v1 dist_exp was NOT a valid delay benchmark because when θ >> 3, the term 
`exp(-λτ)·x(t-τ)` becomes negligible, making the system effectively an ODE.

## Interpretation

1. **dist_exp remains easiest** but now with valid delay dynamics
2. **linear2 remains hardest** - two interacting delays create complex dynamics
3. **OOD-delay gaps now make sense**: hutch (6.7x) >> dist_uniform (1.8x) > dist_exp (1.2x) > vdp (1.1x)
4. **OOD-history is challenging**: dist_uniform (7x), vdp (4x) show models memorize history patterns
5. **dist_exp_v2 is well-behaved**: reasonable gaps across all OOD splits

---
*Generated: 2024-12-29 | Baseline-All-5 v2*
