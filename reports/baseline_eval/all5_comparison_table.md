# Baseline-All-5 Complete Comparison

**Model**: Small FNO (93k params) | **Training**: 50k samples | **Metric**: relL2_orig

## ID Test Performance

| Family | Median | P95 | Mean±Std | State Dim |
|--------|--------|-----|----------|-----------|
| **dist_exp** | **0.041** | **0.101** | 0.049±0.039 | 2 |
| dist_uniform | 0.086 | 0.441 | 0.144±0.139 | 2 |
| hutch | 0.131 | 0.588 | 0.200±0.190 | 1 |
| vdp | 0.297 | 0.888 | 0.365±0.325 | 2 |
| linear2 | 0.574 | 1.391 | 0.667±0.516 | 1 |

## OOD Generalization Gaps (Median ratio vs ID)

| Family | OOD-delay | OOD-hole | OOD-history | OOD-horizon |
|--------|-----------|----------|-------------|-------------|
| dist_exp | **0.96x** | 1.04x | 3.76x | 3.06x |
| dist_uniform | 1.83x | **0.74x** | **7.02x** | 2.52x |
| hutch | 1.49x | 1.62x | 1.50x | **2.84x** |
| vdp | 1.11x | 1.00x | 4.36x | 4.16x |
| linear2 | **0.86x** | 0.99x | 1.43x | 1.89x |

## Full OOD Results (Median relL2_orig)

| Family | ID | OOD-delay | OOD-hole | OOD-history | OOD-horizon |
|--------|-----|-----------|----------|-------------|-------------|
| dist_exp | 0.041 | 0.039 | 0.043 | 0.155 | 0.126 |
| dist_uniform | 0.086 | 0.157 | 0.064 | 0.603 | 0.217 |
| hutch | 0.131 | 0.195 | 0.212 | 0.196 | 0.371 |
| vdp | 0.297 | 0.330 | 0.299 | 1.297 | 1.238 |
| linear2 | 0.574 | 0.496 | 0.567 | 0.821 | 1.084 |

## Key Findings

### ID Performance Ranking (easiest to hardest)
1. **dist_exp** (0.041) - Exponential kernel smooths dynamics
2. **dist_uniform** (0.086) - Uniform averaging moderately smooth
3. **hutch** (0.131) - Classic single-delay Hutchinson
4. **vdp** (0.297) - Nonlinear oscillator with delay coupling
5. **linear2** (0.574) - Two delays create complex interference

### OOD Robustness Insights

- **OOD-delay**: dist_exp shows NO degradation (0.96x), linear2 actually improves (0.86x)
- **OOD-hole**: Interpolation is generally easier than extrapolation
- **OOD-history**: Largest gaps across all families (1.4-7x) - model overfits to Fourier histories
- **OOD-horizon**: Temporal extrapolation challenging for oscillatory families (vdp=4.16x)

### Surprising Results
- linear2 is hardest ID but has best OOD-history robustness (1.43x)
- dist_exp has best ID AND best delay robustness
- vdp shows 4x gaps on both OOD-history and OOD-horizon

## Visualization Files

Each family has visualization in `reports/data_viz/{family}/`:
- `trajectories_id.png` - 20 sample trajectories
- `histograms_params.png` - Parameter distributions  
- `ood_compare_*.png` - ID vs OOD distribution comparisons
- `phaseplot_vdp.png` - Phase portrait (vdp only)

---
Generated: 2025-12-29
Protocol: configs/baseline_protocol.yaml
Tag: baseline_all5_frozen
