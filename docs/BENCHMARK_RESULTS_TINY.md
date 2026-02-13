# DDE-FNO Benchmark Results: Tiny Dataset Validation

> **Date:** December 27, 2024  
> **Purpose:** Validate data generation pipeline and benchmark infrastructure on minimal datasets

---

## Overview

This document summarizes the results of running the complete benchmark pipeline on tiny datasets (64 train / 16 val / 16 test samples) for the 5 core DDE families. The goal was to verify:

1. Data generation pipeline works correctly
2. Quality control checks pass
3. Label fidelity meets acceptance criteria
4. Dataset diversity is non-trivial

---

## 1. Dataset Generation

### 1.1 Families Tested

| Family | Description | State Dim | Delays | Positive |
|--------|-------------|-----------|--------|----------|
| `linear2` | Linear DDE with 2 discrete delays | 1 | τ₁, τ₂ | No |
| `hutch` | Hutchinson (delayed logistic) | 1 | τ | Yes |
| `vdp` | Van der Pol with delayed feedback | 2 | τ | No |
| `dist_uniform` | Distributed delay (uniform kernel) | 2 | τ | Yes |
| `dist_exp` | Distributed delay (exponential kernel) | 2 | None (ODE) | Yes |

### 1.2 Generation Configuration

```yaml
n_train: 64
n_val: 16
n_test: 16
shard_size: 64
T: 20.0          # Solution horizon
dt_out: 0.05     # Output time step
tau_max: 2.0     # Maximum delay
seed: 42
solver: Python (scipy-based fallback)
```

### 1.3 Terminal Output: Dataset Generation

#### linear2
```
$ python src/datasets/generate_python.py linear2 --n_train=64 --n_val=16 --n_test=16 --shard_size=64 --output_dir=data_tiny

============================================================
Generating dataset: linear2 (Python solver)
============================================================
  tau_max=2.0, T=20.0, dt_out=0.05
  train=64, val=16, test=16

Generating train (64 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|
  Failures: {'amplitude_exceeded:58361547.41': 1, 'amplitude_exceeded:1664417008470.98': 1, ...}

Generating val (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|
  Failures: {'amplitude_exceeded:183.90': 1, 'amplitude_exceeded:1664417008470.98': 1, 
             'amplitude_exceeded:221428978.39': 1, 'amplitude_exceeded:313.66': 1, 
             'amplitude_exceeded:357.37': 1, 'amplitude_exceeded:9113.03': 1, 
             'amplitude_exceeded:41103.35': 1, 'amplitude_exceeded:707332.84': 1, 
             'amplitude_exceeded:71630.61': 1}

Generating test (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|
  Failures: {'amplitude_exceeded:8214.85': 1, 'amplitude_exceeded:779237183.10': 1, 
             'amplitude_exceeded:659640816.85': 1, 'amplitude_exceeded:3282075949.56': 1, 
             'amplitude_exceeded:518681.20': 1, 'amplitude_exceeded:22862659.25': 1, 
             'amplitude_exceeded:2029770357.82': 1, 'amplitude_exceeded:1934274468684.71': 1, 
             'amplitude_exceeded:610938039964405888.00': 1, 'amplitude_exceeded:82939782078558.33': 1, 
             'amplitude_exceeded:865245391714.49': 1}

============================================================
Dataset generation complete!
Manifest: data_tiny\linear2\manifest.json
============================================================
```

**Observation:** Many amplitude_exceeded failures (values up to 1e18) due to exponential growth in unstable linear DDEs. This is expected behavior - the QC correctly rejects unstable trajectories.

#### hutch
```
$ python src/datasets/generate_python.py hutch --n_train=64 --n_val=16 --n_test=16 --shard_size=64 --output_dir=data_tiny

============================================================
Generating dataset: hutch (Python solver)
============================================================
  tau_max=2.0, T=20.0, dt_out=0.05
  train=64, val=16, test=16

Generating train (64 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating val (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating test (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|
  Failures: {'amplitude_exceeded:129.29': 1}

============================================================
Dataset generation complete!
Manifest: data_tiny\hutch\manifest.json
============================================================
```

**Observation:** Clean generation with only 1 rejection in test set.

#### vdp
```
$ python src/datasets/generate_python.py vdp --n_train=64 --n_val=16 --n_test=16 --shard_size=64 --output_dir=data_tiny

============================================================
Generating dataset: vdp (Python solver)
============================================================
  tau_max=2.0, T=20.0, dt_out=0.05
  train=64, val=16, test=16

Generating train (64 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating val (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating test (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

============================================================
Dataset generation complete!
Manifest: data_tiny\vdp\manifest.json
============================================================
```

**Observation:** 100% clean generation, no failures.

#### dist_uniform
```
$ python src/datasets/generate_python.py dist_uniform --n_train=64 --n_val=16 --n_test=16 --shard_size=64 --output_dir=data_tiny

============================================================
Generating dataset: dist_uniform (Python solver)
============================================================
  tau_max=2.0, T=20.0, dt_out=0.05
  train=64, val=16, test=16

Generating train (64 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating val (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating test (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

============================================================
Dataset generation complete!
Manifest: data_tiny\dist_uniform\manifest.json
============================================================
```

**Observation:** 100% clean generation.

#### dist_exp (FAILED initially, then FIXED)

**First attempt (FAILED):**
```
$ python src/datasets/generate_python.py dist_exp --n_train=64 --n_val=16 --n_test=16 --shard_size=64 --output_dir=data_tiny

============================================================
Generating dataset: dist_exp (Python solver)
============================================================
  tau_max=2.0, T=20.0, dt_out=0.05
  train=64, val=16, test=16

Generating train (64 samples)...
  Warning: Only generated 0/64 samples for shard 0
  Failures: {'solver_failed': 640}

Generating val (16 samples)...
  Warning: Only generated 0/16 samples for shard 0
  Failures: {'solver_failed': 160}

Generating test (16 samples)...
  Warning: Only generated 0/16 samples for shard 0
  Failures: {'solver_failed': 160}

============================================================
Dataset generation complete!
Manifest: data_tiny\dist_exp\manifest.json
============================================================
```

**Root cause:** `dist_exp` has NO discrete delays (it's a pure ODE with auxiliary variable). The solver crashed on `min([])` when trying to compute `tau_min`.

**Fix applied to `src/dde/solve_python/dde_solver.py`:**
```python
# Before:
self.tau_min = min(delays)

# After:
self.tau_min = min(delays) if delays else 1.0  # Default step for ODEs
```

**Second attempt (SUCCESS):**
```
$ python src/datasets/generate_python.py dist_exp --n_train=64 --n_val=16 --n_test=16 --shard_size=64 --output_dir=data_tiny

============================================================
Generating dataset: dist_exp (Python solver)
============================================================
  tau_max=2.0, T=20.0, dt_out=0.05
  train=64, val=16, test=16

Generating train (64 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating val (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

Generating test (16 samples)...
  Shard 0: 100%|████████████████████████████████████████████████████████████████████████████|

============================================================
Dataset generation complete!
Manifest: data_tiny\dist_exp\manifest.json
============================================================
```

### 1.4 Generation Summary

| Family | Train | Val | Test | Failures | Notes |
|--------|-------|-----|------|----------|-------|
| `linear2` | 64 | 16 | 16 | ~50% amplitude exceeded | Expected for unbounded linear DDE |
| `hutch` | 64 | 16 | 16 | 1 | Clean generation |
| `vdp` | 64 | 16 | 16 | 0 | Clean generation |
| `dist_uniform` | 64 | 16 | 16 | 0 | Clean generation |
| `dist_exp` | 64 | 16 | 16 | 0 | Clean (after ODE fix) |

### 1.5 Output Artifacts

```
data_tiny/
├── linear2/
│   ├── train/shard_000.npz
│   ├── val/shard_000.npz
│   ├── test/shard_000.npz
│   └── manifest.json
├── hutch/
│   └── ...
├── vdp/
│   └── ...
├── dist_uniform/
│   └── ...
└── dist_exp/
    └── ...
```

Each `.npz` shard contains:
- `t_hist`: History time grid [-τ_max, 0]
- `t_out`: Output time grid [0, T]
- `phi`: History function values (B, N_hist, d_state)
- `y`: Solution trajectories (B, N_out, d_state)
- `params`: Parameter vectors
- `lags`: Delay values

---

## 2. Dataset Quality Benchmarks

### 2.1 Terminal Output: QC Benchmarks

#### hutch
```
$ python src/benchmarks/run_benchmarks.py --data_dir data_tiny --family hutch --output_dir reports_tiny --run_dataset

============================================================
Dataset Benchmarks: hutch
============================================================

1. Solver Health Stats...
  No solver health log found

2. Label Fidelity (Python solver)...
  Median rel L2: 5.32e-08
  95th percentile: 4.54e-07
  Pass: {'median_lt_1e3': True, 'p95_lt_1e2': True}

3. Residual Benchmark...
  Mean residual: 5.40e-03
  Max residual (p95): 7.14e-01

4. Diversity Metrics...
  Amplitude range: 5.995 ± 11.958
  Oscillations (mean): 1.9
  Frac with zero oscillations: 31.25%

Results saved to: reports_tiny\hutch_dataset_benchmarks.json

Benchmarks complete!
```

#### linear2
```
$ python src/benchmarks/run_benchmarks.py --data_dir data_tiny --family linear2 --output_dir reports_tiny --run_dataset

============================================================
Dataset Benchmarks: linear2
============================================================

1. Solver Health Stats...
  No solver health log found

2. Label Fidelity (Python solver)...
  Median rel L2: 3.50e-08
  95th percentile: 4.57e-07
  Pass: {'median_lt_1e3': True, 'p95_lt_1e2': True}

3. Residual Benchmark...
  Mean residual: 1.84e-01
  Max residual (p95): 2.76e+00

4. Diversity Metrics...
  Amplitude range: 3.622 ± 8.472
  Oscillations (mean): 1.5
  Frac with zero oscillations: 48.44%

Results saved to: reports_tiny\linear2_dataset_benchmarks.json

Benchmarks complete!
```

#### vdp
```
$ python src/benchmarks/run_benchmarks.py --data_dir data_tiny --family vdp --output_dir reports_tiny --run_dataset

============================================================
Dataset Benchmarks: vdp
============================================================

1. Solver Health Stats...
  No solver health log found

2. Label Fidelity (Python solver)...
  Median rel L2: 1.55e-06
  95th percentile: 9.66e-06
  Pass: {'median_lt_1e3': True, 'p95_lt_1e2': True}

3. Residual Benchmark...
  (skipped - RHS not implemented for vdp)

4. Diversity Metrics...
  Amplitude range: 4.541 ± 1.199
  Oscillations (mean): 2.0
  Frac with zero oscillations: 23.44%

Results saved to: reports_tiny\vdp_dataset_benchmarks.json

Benchmarks complete!
```

#### dist_uniform
```
$ python src/benchmarks/run_benchmarks.py --data_dir data_tiny --family dist_uniform --output_dir reports_tiny --run_dataset

============================================================
Dataset Benchmarks: dist_uniform
============================================================

1. Solver Health Stats...
  No solver health log found

2. Label Fidelity (Python solver)...
  Median rel L2: 1.12e-08
  95th percentile: 2.96e-08
  Pass: {'median_lt_1e3': True, 'p95_lt_1e2': True}

3. Residual Benchmark...
  (skipped - RHS not implemented for dist_uniform)

4. Diversity Metrics...
  Amplitude range: 0.848 ± 0.596
  Oscillations (mean): 1.7
  Frac with zero oscillations: 56.25%

Results saved to: reports_tiny\dist_uniform_dataset_benchmarks.json

Benchmarks complete!
```

#### dist_exp
```
$ python src/benchmarks/run_benchmarks.py --data_dir data_tiny --family dist_exp --output_dir reports_tiny --run_dataset

============================================================
Dataset Benchmarks: dist_exp
============================================================

1. Solver Health Stats...
  No solver health log found

2. Label Fidelity (Python solver)...
  Median rel L2: 4.96e-10
  95th percentile: 1.78e-09
  Pass: {'median_lt_1e3': True, 'p95_lt_1e2': True}

3. Residual Benchmark...
  (skipped - RHS not implemented for dist_exp)

4. Diversity Metrics...
  Amplitude range: 0.861 ± 0.695
  Oscillations (mean): 0.8
  Frac with zero oscillations: 56.25%

Results saved to: reports_tiny\dist_exp_dataset_benchmarks.json

Benchmarks complete!
```

### 2.2 Solver Health

No solver health logs were generated (Python fallback doesn't produce detailed logs). With Julia backend, this would show:
- Acceptance rate
- Fallback ladder usage
- Timing percentiles
- Rejection taxonomy

### 2.3 Residual Check Summary

Physics-based verification: computes `R(t) = dy/dt - f(t, y, y_delayed)` using numerical differentiation.

| Family | Mean Residual | Max Residual (p95) | Status |
|--------|--------------|-------------------|--------|
| `linear2` | 1.84e-1 | 2.76e+0 | ⚠️ Higher (expected for stiff) |
| `hutch` | 5.40e-3 | 7.14e-1 | ✅ Good |
| `vdp` | — | — | (RHS not implemented) |
| `dist_uniform` | — | — | (RHS not implemented) |
| `dist_exp` | — | — | (RHS not implemented) |

**Note:** Residual RHS implementations exist only for `linear2`, `hutch`, and `mackey_glass`. Others skipped.

### 2.4 Diversity Metrics Summary

Ensures dataset isn't trivial (all flat trajectories or immediate equilibrium).

| Family | Amplitude Range | Oscillations (mean) | Frac Zero Osc | Status |
|--------|----------------|--------------------|--------------| -------|
| `linear2` | 3.62 ± 8.47 | 1.5 | 48% | ✅ |
| `hutch` | 6.00 ± 11.96 | 1.9 | 31% | ✅ |
| `vdp` | 4.54 ± 1.20 | 2.0 | 23% | ✅ Best |
| `dist_uniform` | 0.85 ± 0.60 | 1.7 | 56% | ⚠️ Lower diversity |
| `dist_exp` | 0.86 ± 0.70 | 0.8 | 56% | ⚠️ Lower diversity |

**Observations:**
- VdP has the most consistent dynamics (oscillator behavior)
- Distributed delay families show lower amplitude variation (closer to equilibrium)
- Linear2 has high variance due to exponential growth/decay

---

## 3. Label Fidelity Benchmark

Compares "fast" solver output vs "reference" solver output to ensure training labels are numerically accurate.

### 3.1 Pass Criteria

- **Median relL2 < 1e-3** (labels are accurate)
- **95th percentile relL2 < 1e-2** (no catastrophic outliers)

### 3.2 Terminal Output: Label Fidelity

#### hutch
```
$ python src/benchmarks/label_fidelity.py --family hutch --n_samples 100 --output_dir reports_label_fidelity

Running label fidelity benchmark for hutch...

Results:
  Samples evaluated: 100
  Median rel L2: 6.65e-08
  Mean rel L2: 1.41e-07
  95th percentile: 5.19e-07
  Pass (median < 1e-3): True
  Pass (p95 < 1e-2): True

Results saved to: reports_label_fidelity\hutch_label_fidelity.json
```

#### linear2
```
$ python src/benchmarks/label_fidelity.py --family linear2 --n_samples 100 --output_dir reports_label_fidelity

Running label fidelity benchmark for linear2...

Results:
  Samples evaluated: 100
  Median rel L2: 3.20e-08
  Mean rel L2: 1.66e-07
  95th percentile: 6.41e-07
  Pass (median < 1e-3): True
  Pass (p95 < 1e-2): True

Results saved to: reports_label_fidelity\linear2_label_fidelity.json
```

#### vdp
```
$ python src/benchmarks/label_fidelity.py --family vdp --n_samples 100 --output_dir reports_label_fidelity

Running label fidelity benchmark for vdp...

Results:
  Samples evaluated: 100
  Median rel L2: 1.33e-06
  Mean rel L2: 3.15e-06
  95th percentile: 1.03e-05
  Pass (median < 1e-3): True
  Pass (p95 < 1e-2): True

Results saved to: reports_label_fidelity\vdp_label_fidelity.json
```

#### dist_uniform
```
$ python src/benchmarks/label_fidelity.py --family dist_uniform --n_samples 100 --output_dir reports_label_fidelity

Running label fidelity benchmark for dist_uniform...

Results:
  Samples evaluated: 100
  Median rel L2: 1.12e-08
  Mean rel L2: 1.47e-08
  95th percentile: 3.08e-08
  Pass (median < 1e-3): True
  Pass (p95 < 1e-2): True

Results saved to: reports_label_fidelity\dist_uniform_label_fidelity.json
```

#### dist_exp
```
$ python src/benchmarks/label_fidelity.py --family dist_exp --n_samples 100 --output_dir reports_label_fidelity

Running label fidelity benchmark for dist_exp...

Results:
  Samples evaluated: 100
  Median rel L2: 4.96e-10
  Mean rel L2: 7.26e-10
  95th percentile: 2.02e-09
  Pass (median < 1e-3): True
  Pass (p95 < 1e-2): True

Results saved to: reports_label_fidelity\dist_exp_label_fidelity.json
```

### 3.3 Results Summary Table

| Family | Median relL2 | Mean relL2 | p95 relL2 | Median Pass | p95 Pass |
|--------|-------------|-----------|----------|-------------|----------|
| `linear2` | **3.20e-8** | 1.66e-7 | 6.41e-7 | ✅ | ✅ |
| `hutch` | **6.65e-8** | 1.41e-7 | 5.19e-7 | ✅ | ✅ |
| `vdp` | **1.33e-6** | 3.15e-6 | 1.03e-5 | ✅ | ✅ |
| `dist_uniform` | **1.12e-8** | 1.47e-8 | 3.08e-8 | ✅ | ✅ |
| `dist_exp` | **4.96e-10** | 7.26e-10 | 2.02e-9 | ✅ | ✅ |

### 3.4 Analysis

All families **PASS** label fidelity criteria by comfortable margins:

- **Best:** `dist_exp` (1e-10) — ODE-only, no delay interpolation errors
- **Good:** `dist_uniform`, `linear2`, `hutch` (1e-8)
- **Acceptable:** `vdp` (1e-6) — 2D system with more complex dynamics

VdP has slightly higher error due to:
- 2D state space
- Nonlinear oscillatory dynamics
- More sensitive to numerical precision

**All families are well below the 1e-3 threshold**, indicating high-quality training labels.

---

## 4. Bugs Fixed During Testing

### 4.1 Family Naming Mismatch

**Issue:** Python and Julia used different family names.

**Fix:** Standardized to:
```python
DDE_FAMILIES = {
    "linear2": Linear2DDE,
    "hutch": HutchDDE,
    "mackey_glass": MackeyGlassDDE,
    "vdp": VdPDDE,
    "predator_prey": PredatorPreyDDE,
    "dist_uniform": DistUniformDDE,
    "dist_exp": DistExpDDE,
}
```

### 4.2 ODE Handling for dist_exp

**Issue:** `dist_exp` has no discrete delays (pure ODE auxiliary form), causing solver to fail on `min([])`.

**Fix:** Modified `dde_solver.py`:
```python
self.tau_min = min(delays) if delays else 1.0  # Default step for ODEs
```

### 4.3 Residual Check Duplicate Time Points

**Issue:** Interpolation failed with "Expect x to not have duplicates" when `t_hist[-1] ≈ t[0] ≈ 0`.

**Fix:** Added deduplication in `residual_check.py`:
```python
if np.abs(t_hist[-1] - t[0]) < 1e-10:
    t_full = np.concatenate([t_hist[:-1], t])
```

---

## 5. Output Files

### 5.1 Dataset Benchmarks
```
reports_tiny/
├── linear2_dataset_benchmarks.json
├── hutch_dataset_benchmarks.json
├── vdp_dataset_benchmarks.json
├── dist_uniform_dataset_benchmarks.json
└── dist_exp_dataset_benchmarks.json
```

### 5.2 Label Fidelity
```
reports_label_fidelity/
├── linear2_label_fidelity.json
├── hutch_label_fidelity.json
├── vdp_label_fidelity.json
├── dist_uniform_label_fidelity.json
└── dist_exp_label_fidelity.json
```

---

## 6. Conclusions

### ✅ Pipeline Validated

1. **Data generation** works for all 5 core families
2. **Quality control** catches amplitude explosions appropriately
3. **Label fidelity** passes with large margins (all < 1e-5, threshold 1e-3)
4. **Diversity** shows non-trivial dynamics across families

### ⚠️ Areas for Attention

1. **linear2** has high rejection rate — consider tightening parameter ranges
2. **Distributed delay families** show lower diversity — may need to expand parameter ranges
3. **Residual RHS** not implemented for vdp/dist families — add for complete physics verification

### 🔜 Ready for Next Steps

The pipeline is validated and ready to:
1. Scale to full dataset (1000+ samples)
2. Train FNO models
3. Run model evaluation benchmarks
4. Execute ablation studies

---

## Appendix: Commands Used

```bash
# Step 2: Generate tiny datasets
python src/datasets/generate_python.py linear2 --n_train=64 --n_val=16 --n_test=16 --output_dir=data_tiny
python src/datasets/generate_python.py hutch --n_train=64 --n_val=16 --n_test=16 --output_dir=data_tiny
python src/datasets/generate_python.py vdp --n_train=64 --n_val=16 --n_test=16 --output_dir=data_tiny
python src/datasets/generate_python.py dist_uniform --n_train=64 --n_val=16 --n_test=16 --output_dir=data_tiny
python src/datasets/generate_python.py dist_exp --n_train=64 --n_val=16 --n_test=16 --output_dir=data_tiny

# Step 3: QC benchmarks
python src/benchmarks/run_benchmarks.py --data_dir data_tiny --family hutch --output_dir reports_tiny --run_dataset
# (repeated for each family)

# Step 4: Label fidelity
python src/benchmarks/label_fidelity.py --family hutch --n_samples 100 --output_dir reports_label_fidelity
# (repeated for each family)
```
