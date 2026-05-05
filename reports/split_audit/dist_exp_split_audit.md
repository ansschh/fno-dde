# Split Audit Report: DIST_EXP

Generated: 2025-12-30T06:39:21.433258

## Summary Table

| Split | N | τ Range | L2 Norm (median) | Amplitude (median) | Roughness (median) |
|-------|---|---------|------------------|--------------------|--------------------|
| id_train | 50000 | - | 24.23 | 0.91 | 0.0089 |
| id_val | 2000 | - | 24.10 | 0.93 | 0.0089 |
| id_test | 2000 | - | 24.44 | 0.93 | 0.0089 |
| ood_delay | 2000 | - | 25.91 | 1.47 | 0.0089 |
| ood_history | 2000 | - | 25.90 | 1.15 | 0.0385 |
| ood_horizon | - | NOT FOUND | - | - | - |

## OOD vs ID Comparisons

| Split | L2 Norm Ratio | Amplitude Ratio | Roughness Ratio |
|-------|---------------|-----------------|-----------------|
| id_train | 0.992 | 0.976 | 0.999 |
| id_val | 0.986 | 0.995 | 0.997 |
| ood_delay | 1.060 | 1.579 | 0.994 |
| ood_history | 1.060 | 1.233 | 4.315 |

**Interpretation:**
- L2 Norm Ratio < 1.0 means OOD trajectories have *lower* energy → potentially easier
- Amplitude Ratio < 1.0 means OOD has smaller excursions → potentially easier
- Roughness Ratio > 1.0 means OOD history is rougher → potentially harder for model

## Delay Distribution Details
