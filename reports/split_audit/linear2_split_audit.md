# Split Audit Report: LINEAR2

Generated: 2025-12-28T22:56:54.893819

## Summary Table

| Split | N | τ Range | L2 Norm (median) | Amplitude (median) | Roughness (median) |
|-------|---|---------|------------------|--------------------|--------------------|
| id_train | 50000 | max∈[0.11, 2.00] | 2.26 | 0.83 | 0.0159 |
| id_val | 2000 | max∈[0.12, 2.00] | 2.17 | 0.81 | 0.0157 |
| id_test | 2000 | max∈[0.16, 2.00] | 2.21 | 0.83 | 0.0157 |
| ood_delay | 2000 | max∈[1.31, 2.00] | 3.86 | 1.14 | 0.0159 |
| ood_delay_hole | 2000 | max∈[0.90, 1.10] | 2.53 | 0.88 | 0.0158 |
| ood_history | 2000 | max∈[0.15, 2.00] | 2.14 | 0.78 | 0.0168 |

## OOD vs ID Comparisons

| Split | L2 Norm Ratio | Amplitude Ratio | Roughness Ratio |
|-------|---------------|-----------------|-----------------|
| id_train | 1.023 | 1.004 | 1.010 |
| id_val | 0.981 | 0.981 | 1.002 |
| ood_delay | 1.746 | 1.373 | 1.012 |
| ood_delay_hole | 1.143 | 1.059 | 1.004 |
| ood_history | 0.967 | 0.938 | 1.068 |

**Interpretation:**
- L2 Norm Ratio < 1.0 means OOD trajectories have *lower* energy → potentially easier
- Amplitude Ratio < 1.0 means OOD has smaller excursions → potentially easier
- Roughness Ratio > 1.0 means OOD history is rougher → potentially harder for model

## Delay Distribution Details

### id_train

- τ1: mean=1.052, range=[0.100, 2.000]
- τ2: mean=1.053, range=[0.100, 2.000]
- max(τ1,τ2): mean=1.368
- |τ1-τ2|: mean=0.632, std=0.447

### id_val

- τ1: mean=1.036, range=[0.101, 2.000]
- τ2: mean=1.039, range=[0.103, 1.996]
- max(τ1,τ2): mean=1.362
- |τ1-τ2|: mean=0.649, std=0.451

### id_test

- τ1: mean=1.063, range=[0.101, 1.999]
- τ2: mean=1.044, range=[0.102, 1.997]
- max(τ1,τ2): mean=1.374
- |τ1-τ2|: mean=0.641, std=0.445

### ood_delay

- τ1: mean=1.653, range=[1.300, 2.000]
- τ2: mean=1.650, range=[1.300, 2.000]
- max(τ1,τ2): mean=1.768
- |τ1-τ2|: mean=0.233, std=0.164

### ood_delay_hole

- τ1: mean=1.002, range=[0.900, 1.100]
- τ2: mean=0.999, range=[0.900, 1.100]
- max(τ1,τ2): mean=1.033
- |τ1-τ2|: mean=0.066, std=0.047

### ood_history

- τ1: mean=1.056, range=[0.104, 1.999]
- τ2: mean=1.040, range=[0.102, 1.998]
- max(τ1,τ2): mean=1.370
- |τ1-τ2|: mean=0.643, std=0.456
