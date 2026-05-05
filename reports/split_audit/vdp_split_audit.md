# Split Audit Report: VDP

Generated: 2025-12-30T06:39:18.445449

## Summary Table

| Split | N | τ Range | L2 Norm (median) | Amplitude (median) | Roughness (median) |
|-------|---|---------|------------------|--------------------|--------------------|
| id_train | 50000 | - | 52.77 | 8.85 | 0.2322 |
| id_val | 2000 | - | 52.11 | 8.70 | 0.2316 |
| id_test | 2000 | - | 53.21 | 9.04 | 0.2346 |
| ood_delay | 2000 | - | 51.79 | 9.09 | 0.2330 |
| ood_history | 2000 | - | 50.19 | 7.83 | 0.0756 |
| ood_horizon | - | NOT FOUND | - | - | - |

## OOD vs ID Comparisons

| Split | L2 Norm Ratio | Amplitude Ratio | Roughness Ratio |
|-------|---------------|-----------------|-----------------|
| id_train | 0.992 | 0.979 | 0.989 |
| id_val | 0.979 | 0.962 | 0.987 |
| ood_delay | 0.973 | 1.006 | 0.993 |
| ood_history | 0.943 | 0.866 | 0.322 |

**Interpretation:**
- L2 Norm Ratio < 1.0 means OOD trajectories have *lower* energy → potentially easier
- Amplitude Ratio < 1.0 means OOD has smaller excursions → potentially easier
- Roughness Ratio > 1.0 means OOD history is rougher → potentially harder for model

## Delay Distribution Details
