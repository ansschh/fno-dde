# Split Audit Report: DIST_UNIFORM

Generated: 2025-12-30T06:39:19.957201

## Summary Table

| Split | N | τ Range | L2 Norm (median) | Amplitude (median) | Roughness (median) |
|-------|---|---------|------------------|--------------------|--------------------|
| id_train | 50000 | - | 37.95 | 1.60 | 0.0067 |
| id_val | 2000 | - | 37.78 | 1.60 | 0.0067 |
| id_test | 2000 | - | 37.68 | 1.58 | 0.0067 |
| ood_delay | 2000 | - | 40.24 | 2.46 | 0.0067 |
| ood_history | 2000 | - | 47.29 | 2.63 | 0.0361 |
| ood_horizon | - | NOT FOUND | - | - | - |

## OOD vs ID Comparisons

| Split | L2 Norm Ratio | Amplitude Ratio | Roughness Ratio |
|-------|---------------|-----------------|-----------------|
| id_train | 1.007 | 1.014 | 0.999 |
| id_val | 1.003 | 1.014 | 0.995 |
| ood_delay | 1.068 | 1.561 | 1.001 |
| ood_history | 1.255 | 1.667 | 5.397 |

**Interpretation:**
- L2 Norm Ratio < 1.0 means OOD trajectories have *lower* energy → potentially easier
- Amplitude Ratio < 1.0 means OOD has smaller excursions → potentially easier
- Roughness Ratio > 1.0 means OOD history is rougher → potentially harder for model

## Delay Distribution Details
