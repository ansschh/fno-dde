# Split Audit Report: HUTCH

Generated: 2025-12-28T22:56:45.435156

## Summary Table

| Split | N | τ Range | L2 Norm (median) | Amplitude (median) | Roughness (median) |
|-------|---|---------|------------------|--------------------|--------------------|
| id_train | 50000 | [0.10, 2.00] | 27.02 | 1.82 | 0.0157 |
| id_val | 2000 | [0.10, 2.00] | 26.38 | 1.73 | 0.0157 |
| id_test | 2000 | [0.10, 2.00] | 27.32 | 1.83 | 0.0158 |
| ood_delay | 2000 | [1.30, 2.00] | 42.17 | 7.12 | 0.0158 |
| ood_delay_hole | 2000 | [0.90, 1.10] | 26.75 | 2.37 | 0.0157 |
| ood_history | 2000 | [0.10, 2.00] | 26.97 | 1.62 | 0.0112 |

## OOD vs ID Comparisons

| Split | L2 Norm Ratio | Amplitude Ratio | Roughness Ratio |
|-------|---------------|-----------------|-----------------|
| id_train | 0.989 | 0.997 | 0.997 |
| id_val | 0.966 | 0.945 | 0.998 |
| ood_delay | 1.544 | 3.895 | 0.999 |
| ood_delay_hole | 0.979 | 1.297 | 0.996 |
| ood_history | 0.987 | 0.886 | 0.707 |

**Interpretation:**
- L2 Norm Ratio < 1.0 means OOD trajectories have *lower* energy → potentially easier
- Amplitude Ratio < 1.0 means OOD has smaller excursions → potentially easier
- Roughness Ratio > 1.0 means OOD history is rougher → potentially harder for model

## Delay Distribution Details

### id_train

- τ: mean=1.045, std=0.546, median=1.041

### id_val

- τ: mean=1.037, std=0.545, median=1.046

### id_test

- τ: mean=1.029, std=0.541, median=1.006

### ood_delay

- τ: mean=1.659, std=0.206, median=1.660

### ood_delay_hole

- τ: mean=1.000, std=0.058, median=1.000

### ood_history

- τ: mean=1.045, std=0.551, median=1.047
