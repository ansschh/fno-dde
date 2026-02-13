# DDE-FNO Implementation Summary

> **Fourier Neural Operators for Delay Differential Equations**
> 
> Complete implementation of a benchmark suite for learning DDE solution operators using FNOs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [DDE Families](#3-dde-families)
4. [Data Generation Pipeline](#4-data-generation-pipeline)
5. [Dataset Quality Benchmarks](#5-dataset-quality-benchmarks)
6. [Model Architecture](#6-model-architecture)
7. [Baseline Models](#7-baseline-models)
8. [Training System](#8-training-system)
9. [Evaluation & Metrics](#9-evaluation--metrics)
10. [Ablation Studies](#10-ablation-studies)
11. [Paper LaTeX Files](#11-paper-latex-files)
12. [Scripts & CLI Tools](#12-scripts--cli-tools)
13. [Configuration System](#13-configuration-system)

---

## 1. Project Overview

This project implements an operator learning framework for Delay Differential Equations (DDEs) using Fourier Neural Operators (FNOs). The key contribution is treating the DDE initial value problem as an operator learning task:

```
G: (φ, θ) → x(·) on [0, T]
```

Where:
- `φ` is the history function on `[-τ_max, 0]`
- `θ` are parameters (rates, delays, etc.)
- `x(t)` is the solution trajectory

### Key Features

- **7 DDE families** spanning discrete and distributed delays
- **Robust Julia-based solver** with 4-tier fallback ladder
- **Sharded dataset format** for resumable, parallel generation
- **Comprehensive quality control** (solver health, label fidelity, residuals, diversity)
- **Multiple OOD test splits** (delay, params, history, horizon, resolution)
- **Baseline models** for comparison (Naive, TCN, MLP)
- **Ablation study infrastructure** (capacity, encoding, loss weighting)

---

## 2. Directory Structure

```
dde-fno/
├── paper/                              # LaTeX paper files
│   ├── main.tex                        # Main document
│   ├── sections/                       # Individual sections (8 files)
│   └── references.bib                  # Bibliography (10 entries)
│
├── configs/
│   ├── dataset_v1.yaml                 # Master dataset configuration
│   ├── family_*.yaml                   # Per-family training configs (5 files)
│   └── ablations/                      # Ablation study configs
│       ├── fno_capacity.yaml           # Modes/width/depth sweep
│       ├── input_encoding.yaml         # Channel ablations
│       └── loss_weighting.yaml         # Temporal weighting schemes
│
├── src/
│   ├── dde/                            # DDE definitions & solvers
│   │   ├── families.py                 # Python family definitions
│   │   ├── solve_python/               # scipy-based solver
│   │   └── solve_julia/                # Julia SciML solver (recommended)
│   │
│   ├── datasets/                       # Data loading & generation
│   │   ├── sharded_dataset.py          # PyTorch Dataset for shards
│   │   ├── build_dataset_sharded.py    # Python wrapper for Julia
│   │   ├── generate_python.py          # Pure Python fallback
│   │   └── visualize.py                # Data visualization
│   │
│   ├── models/                         # Neural network architectures
│   │   ├── fno1d.py                    # FNO1D implementation
│   │   └── baselines.py                # Naive, TCN, MLP baselines
│   │
│   ├── train/                          # Training scripts
│   │   ├── train_fno_sharded.py        # Main training script
│   │   ├── eval.py                     # Evaluation utilities
│   │   └── ablations.py                # Ablation utilities
│   │
│   ├── benchmarks/                     # Benchmark implementations
│   │   ├── solver_health.py            # Solver statistics
│   │   ├── label_fidelity.py           # Fast vs reference comparison
│   │   ├── residual_check.py           # Physics-based QC
│   │   ├── diversity_metrics.py        # Dataset diversity
│   │   ├── model_metrics.py            # Model evaluation metrics
│   │   ├── ood_splits.py               # OOD test generation
│   │   ├── reproducibility.py          # Hash-based verification
│   │   ├── run_benchmarks.py           # Benchmark runner
│   │   └── eval_comprehensive.py       # Full evaluation script
│   │
│   └── utils/                          # Utilities
│       ├── config.py                   # Configuration loading
│       └── logging.py                  # Logging utilities
│
├── scripts/                            # CLI scripts
│   ├── quick_test.py                   # Pipeline verification
│   ├── run_experiment.py               # End-to-end runner
│   ├── run_ablation_sweep.py           # Ablation study runner
│   └── run_full_benchmark.py           # Complete benchmark pipeline
│
├── data/                               # Generated data (sharded format)
│   └── {family}/
│       ├── train/shard_*.npz
│       ├── val/shard_*.npz
│       ├── test/shard_*.npz
│       └── manifest.json
│
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 3. DDE Families

### 3.1 Family Definitions

Seven DDE families are implemented, covering discrete and distributed delays:

| # | Family | Equation | State Dim | Delays |
|---|--------|----------|-----------|--------|
| 1 | **Linear2** | `x'(t) = ax(t) + b₁x(t-τ₁) + b₂x(t-τ₂)` | 1 | 2 |
| 2 | **Hutchinson** | `x'(t) = rx(t)(1 - x(t-τ)/K)` | 1 | 1 |
| 3 | **Mackey-Glass** | `x'(t) = βx(t-τ)/(1+x(t-τ)ⁿ) - γx(t)` | 1 | 1 |
| 4 | **Van der Pol** | `x'=v, v'=μ(1-x²)v - x + κx(t-τ)` | 2 | 1 |
| 5 | **Predator-Prey** | `x'=x(α-βy(t-τ₁)), y'=y(-δ+γx(t-τ₂))` | 2 | 2 |
| 6 | **Dist. Uniform** | `x'=rx(1-m/K), m=(1/τ)∫x ds` | 2 | 1 |
| 7 | **Dist. Exponential** | `x'=rx(1-z/K), z=(1/C)∫exp(-λs)x ds` | 2 | 1 |

### 3.2 Parameter Ranges

Each family has configurable parameter ranges in `configs/dataset_v1.yaml`:

```yaml
families:
  hutch:
    param_ranges:
      r: [0.5, 3.0]
      K: [0.5, 2.0]
      tau: [0.1, 2.0]
    requires_positive: true
    y_clip: 100.0
```

### 3.3 Python Implementation

**File: `src/dde/families.py`**

```python
class HutchinsonDDE(DDEFamily):
    def rhs(self, t, x, x_delayed, params):
        r, K = params["r"], params["K"]
        x_tau = x_delayed["tau"]
        return np.array([r * x[0] * (1 - x_tau[0] / K)])
```

### 3.4 Julia Implementation

**File: `src/dde/solve_julia/families.jl`**

```julia
function rhs_hutch!(du, u, h, p, t)
    x = u[1]
    x_τ = h(p, t - p.τ)[1]
    du[1] = p.r * x * (1.0 - x_τ / p.K)
end
```

---

## 4. Data Generation Pipeline

### 4.1 Design Principles

1. **Sharded Storage** - Each shard is independent, enabling resume and parallelization
2. **Robust Solver** - 4-tier fallback ladder for difficult parameter combinations
3. **Quality Control** - Every sample checked for finite values, bounds, positivity, continuity
4. **Reproducibility** - Seeds, configs, and git commit tracked in manifest

### 4.2 Sharded Format

Each `.npz` shard contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `t_hist` | `(N_hist,)` | History time grid `[-τ_max, 0]` |
| `t_out` | `(N_out,)` | Output time grid `[0, T]` |
| `phi` | `(B, N_hist, d_hist)` | History function values |
| `y` | `(B, N_out, d_state)` | Solution trajectories |
| `params` | `(B, P)` | Parameter vectors |
| `lags` | `(B, L)` | Delay values |
| `attempts` | `(B,)` | Which solver succeeded |
| `meta_json` | string | Solver settings, seed, timestamp |

### 4.3 Julia Solver with Fallback Ladder

**File: `src/dde/solve_julia/solver.jl`**

```julia
attempts = [
    (alg = MethodOfSteps(Tsit5()),           dtmax = Inf),      # Fast
    (alg = MethodOfSteps(Vern6()),           dtmax = Inf),      # Accurate
    (alg = MethodOfSteps(Rosenbrock23()),    dtmax = Inf),      # Stiff
    (alg = MethodOfSteps(Tsit5(); constrained=true), dtmax = min_lag/5),  # Constrained
]
```

**Discontinuity Handling:**
```julia
function make_breakpoint_stops(lags, T)
    stops = [k*τ for τ in lags for k in 1:floor(Int, T/τ)]
    return sort!(unique!(stops))
end
```

### 4.4 History Function Sampling

**File: `src/dde/solve_julia/history.jl`**

Three methods implemented:

1. **Fourier Series** (default):
```julia
φ(t) = c₀ + Σₖ (aₖcos(2πkt/L) + bₖsin(2πkt/L))
```

2. **Cubic Spline** (OOD test):
```julia
function sample_spline_history(rng, t_hist; n_knots=5)
    knot_t = linspace(t_hist[1], t_hist[end], n_knots)
    knot_y = randn(rng, n_knots)
    return CubicSpline(knot_t, knot_y)(t_hist)
end
```

3. **Van der Pol Consistent** (x and v=dx/dt):
```julia
function sample_vdp_history(rng, t_hist)
    # x from Fourier, v = dx/dt analytically
    v[j] = Σₖ -aₖω sin(ωt) + bₖω cos(ωt)
end
```

### 4.5 Quality Control

**File: `src/dde/solve_julia/qc.jl`**

Checks performed:
- **Finite**: No NaN/Inf values
- **Bounded**: `max|y| < y_clip` (family-specific)
- **Positive**: `min(y) ≥ -1e-6` (for positive families)
- **Continuous**: `|φ(0) - y(0)| < tol`

### 4.6 Usage

**Julia (recommended):**
```bash
julia --project=src/dde/solve_julia src/dde/solve_julia/dataset_generator.jl hutch \
    --n_train=1000 --n_val=100 --n_test=100 --output_dir=data
```

**Python wrapper:**
```bash
python src/datasets/build_dataset_sharded.py hutch --n_train=1000 --verify
```

**Python fallback (slower):**
```bash
python src/datasets/generate_python.py hutchinson --n_train=800
```

---

## 5. Dataset Quality Benchmarks

### 5.1 Solver Health Statistics

**File: `src/benchmarks/solver_health.py`**

Tracks per sample:
- `retcode` (Success/Failure)
- `attempt_id` (1-4 fallback ladder)
- `wall_time`
- `max_state`, `min_state`
- `rejection_reason`

Computes:
- **Acceptance rate**: `accepted / attempted`
- **Fallback distribution**: % solved by each attempt
- **Timing**: p50, p95, p99 solve time
- **Rejection taxonomy**: Pie chart of failure reasons

### 5.2 Label Fidelity Benchmark

**File: `src/benchmarks/label_fidelity.py`**

Compares fast solver vs reference solver:

```python
# Fast: Tsit5, reltol=1e-6, abstol=1e-8
# Reference: Vern9, reltol=1e-9, abstol=1e-11

rel_l2 = ||y_fast - y_ref||₂ / ||y_ref||₂
```

**Pass Criteria:**
- Median rel L2 < 1e-3
- 95th percentile < 1e-2

### 5.3 Residual Benchmark

**File: `src/benchmarks/residual_check.py`**

Physics-based verification without re-solving:

```python
R(tᵢ) ≈ (x(tᵢ₊₁) - x(tᵢ₋₁)) / (2Δt) - f(tᵢ, x(tᵢ), x(tᵢ-τ), ...)
```

Catches:
- Wrong lag indexing
- Wrong units/scaling
- Interpolation bugs

### 5.4 Diversity Metrics

**File: `src/benchmarks/diversity_metrics.py`**

Ensures dataset isn't trivial:

| Metric | Description |
|--------|-------------|
| `amplitude_range` | max - min over trajectory |
| `n_oscillations` | Zero crossings / 2 |
| `settling_time` | Time to reach steady state |
| `dominant_freq` | FFT peak frequency |
| `history_roughness` | RMS of dφ/dt |

### 5.5 Reproducibility Verification

**File: `src/benchmarks/reproducibility.py`**

```python
def compute_shard_hash(shard_path):
    """SHA256 hash of arrays for regression test."""
    
def run_reproducibility_test(data_dir, family, manifest_path):
    """Verify same seed → same data."""
```

---

## 6. Model Architecture

### 6.1 FNO1D

**File: `src/models/fno1d.py`**

```
Input (B, L, C_in)
    ↓
Lifting Layer (Linear: C_in → width)
    ↓
┌─────────────────────────────────────┐
│  Fourier Layer (×n_layers)          │
│  ├── Spectral Conv (FFT → R·F(v) → IFFT) │
│  ├── Pointwise Conv (1×1)           │
│  └── Activation (GELU)              │
└─────────────────────────────────────┘
    ↓
Projection Layers (Linear: width → C_out)
    ↓
Output (B, L, C_out)
```

**Spectral Convolution:**
```python
class SpectralConv1d(nn.Module):
    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft[:, :, :self.modes] = einsum("bim,iom->bom", x_ft[:,:,:self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=x.size(-1))
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `modes` | 16 | Fourier modes retained |
| `width` | 64 | Hidden channel dimension |
| `n_layers` | 4 | Number of Fourier layers |
| `activation` | GELU | Nonlinearity |
| `padding` | 8 | For non-periodic boundaries |

### 6.2 Input Encoding (Combined Grid)

Input on grid `t ∈ [-τ_max, T]` with `N = N_hist + N_out` points:

| Channel | Shape | Description |
|---------|-------|-------------|
| 0..d-1 | `(N, d)` | History signal: φ(t) for t≤0, 0 for t>0 |
| d | `(N, 1)` | Mask: 1 for t≤0, 0 for t>0 |
| d+1 | `(N, 1)` | Normalized time: (t - t_min) / (t_max - t_min) |
| d+2..d+1+P | `(N, P)` | Parameters broadcast across time |

**Total input channels:** `d + 1 + 1 + P`

### 6.3 Variants

- **FNO1d**: Standard architecture
- **FNO1dResidual**: With residual connections and LayerNorm between blocks

---

## 7. Baseline Models

**File: `src/models/baselines.py`**

### 7.1 Naive Baseline

Predicts constant continuation:
```python
ŷ(t) = φ(0) for all t > 0
```

Sets a performance floor; catches trivial datasets.

### 7.2 TCN (Temporal Convolutional Network)

```python
class TCN(nn.Module):
    # Dilated causal convolutions
    # dilation = 2^layer for exponential receptive field
    # 6 layers, kernel_size=3, hidden=64
```

Strong non-operator baseline.

### 7.3 MLP Baseline

```python
class MLPBaseline(nn.Module):
    # Flattens input, 4 hidden layers
    # Only works for fixed sequence length
```

### 7.4 Linear ODE Baseline

Ignores delay terms entirely:
```python
# x' = f_θ(x, params)  [no delayed terms]
# Euler integration
```

---

## 8. Training System

### 8.1 Training Script

**File: `src/train/train_fno_sharded.py`**

**Features:**
- Masked MSE loss (only on future t > 0)
- AdamW optimizer with cosine LR schedule
- Gradient clipping
- Early stopping with patience
- TensorBoard logging
- Checkpoint saving (best + periodic)

**Loss Function:**
```python
def masked_mse_loss(pred, target, mask):
    mask = mask.unsqueeze(-1)  # (B, L, 1)
    sq_error = (pred - target) ** 2
    return (sq_error * mask).sum() / (mask.sum() * pred.shape[-1])
```

### 8.2 Relative L2 Metric

```python
def relative_l2_error(pred, target, mask):
    diff_norm = sqrt(sum((pred - target)² * mask))
    target_norm = sqrt(sum(target² * mask))
    return diff_norm / target_norm
```

### 8.3 Usage

```bash
python src/train/train_fno_sharded.py \
    --config configs/family_hutchinson.yaml \
    --data_dir data \
    --output_dir outputs \
    --device cuda
```

---

## 9. Evaluation & Metrics

### 9.1 Model Metrics

**File: `src/benchmarks/model_metrics.py`**

| Metric | Description |
|--------|-------------|
| `rel_l2` | Relative L2 error (mean, median, std, p95, max) |
| `mse` | Mean squared error |
| `max_error` | Maximum absolute error |
| `error_at_t0` | History adherence |
| `positivity` | Fraction negative, min predicted |
| `spectral_error` | L2 distance between FFT magnitudes |
| `error_vs_time` | Mean + p90 error curve over time |

### 9.2 OOD Test Splits

**File: `src/benchmarks/ood_splits.py`**

| Split | Description |
|-------|-------------|
| `test_id` | In-distribution (same ranges as train) |
| `test_ood_delay` | τ held out (e.g., train [0.1,1.3], test [1.3,2.0]) |
| `test_ood_params` | Parameters held out (corners) |
| `test_ood_history` | Different history generator (spline vs Fourier) |
| `test_horizon` | Extended T (e.g., train T=10, test T=20) |
| `test_resolution` | Different dt_out |

### 9.3 Comprehensive Evaluation

**File: `src/benchmarks/eval_comprehensive.py`**

```bash
python src/benchmarks/eval_comprehensive.py \
    --checkpoint outputs/best_model.pt \
    --family hutch \
    --data_dir data \
    --output_dir eval_results
```

Evaluates on all available splits, computes all metrics, saves JSON results.

---

## 10. Ablation Studies

### 10.1 Capacity Ablation

**File: `configs/ablations/fno_capacity.yaml`**

| Variant | Modes | Width | Layers | Est. Params |
|---------|-------|-------|--------|-------------|
| small_shallow | 8 | 32 | 4 | ~17K |
| medium_baseline | 16 | 64 | 4 | ~135K |
| medium_wide | 16 | 128 | 4 | ~530K |
| large_full | 32 | 128 | 6 | ~1.6M |

### 10.2 Input Encoding Ablation

**File: `configs/ablations/input_encoding.yaml`**

| Variant | Channels | Expected Outcome |
|---------|----------|------------------|
| `full` | history + mask + time + params | Best |
| `no_mask` | history + time + params | Slightly worse |
| `no_params` | history + mask + time | Poor on param-dependent families |
| `history_only` | history | Baseline, no param generalization |

### 10.3 Loss Weighting Ablation

**File: `configs/ablations/loss_weighting.yaml`**

| Variant | Weight Function | Use Case |
|---------|-----------------|----------|
| `uniform` | 1.0 | Baseline |
| `linear_increasing` | 0.5 + t/T | Emphasize late time |
| `two_stage` | uniform then late-focus | Often best for long trajectories |
| `segment` | [0.5, 0.75, 1.0, 1.5] | Granular control |

### 10.4 Ablation Runner

**File: `scripts/run_ablation_sweep.py`**

```bash
# Run capacity sweep
python scripts/run_ablation_sweep.py --family hutch --sweep capacity

# Run all ablations
python scripts/run_ablation_sweep.py --family hutch --sweep all
```

---

## 11. Paper LaTeX Files

### 11.1 Structure

**File: `paper/main.tex`**

```latex
\documentclass[11pt]{article}
\input{sections/introduction}
\input{sections/dde_background}
\input{sections/dde_numerics}
\input{sections/ml_literature}
\input{sections/fno_background}
\input{sections/benchmarks}
\input{sections/data_generation}
\input{sections/experiments}
\bibliography{references}
```

### 11.2 Sections

| File | Content |
|------|---------|
| `introduction.tex` | Problem statement, contributions |
| `dde_background.tex` | DDE theory, types of delays |
| `dde_numerics.tex` | Method of steps, discontinuity tracking |
| `ml_literature.tex` | Neural DDEs, PINNs, operator learning |
| `fno_background.tex` | FNO architecture, properties |
| `benchmarks.tex` | 5 DDE family descriptions |
| `data_generation.tex` | Operator formulation, encoding |
| `experiments.tex` | Model, training, metrics |

### 11.3 Bibliography

**File: `paper/references.bib`**

10 entries including:
- Li et al. (2020) - FNO
- Shampine & Thompson (2001) - DDE23
- Zhu et al. (2021) - Neural DDEs
- Stephany et al. (2024) - Learning delays
- Raissi et al. (2017) - PINNs

---

## 12. Scripts & CLI Tools

### 12.1 Quick Test

**File: `scripts/quick_test.py`**

Verifies all components work:
```bash
python scripts/quick_test.py
# Tests: family definitions, Python solver, dataset encoding, FNO model, loss functions
```

### 12.2 Run Experiment

**File: `scripts/run_experiment.py`**

End-to-end pipeline:
```bash
python scripts/run_experiment.py --family hutch --n_train 1000
# 1. Generate data
# 2. Train FNO
# 3. Evaluate
# 4. Visualize
```

### 12.3 Full Benchmark

**File: `scripts/run_full_benchmark.py`**

Complete benchmark suite:
```bash
python scripts/run_full_benchmark.py --family hutch --n_train 1000
# 1. Data generation
# 2. Dataset benchmarks
# 3. Train FNO
# 4. Model evaluation
# 5. Baseline comparison
```

---

## 13. Configuration System

### 13.1 Dataset Config

**File: `configs/dataset_v1.yaml`**

Master configuration with:
- Global settings (τ_max, T, dt_out, solver tolerances)
- Per-family parameter ranges
- OOD split definitions
- QC thresholds
- Train/val/test sizes

### 13.2 Training Configs

**Files: `configs/family_*.yaml`**

```yaml
family: hutchinson
batch_size: 32
epochs: 150
lr: 1e-3
model:
  modes: 16
  width: 64
  n_layers: 4
```

### 13.3 Config Loading

**File: `src/utils/config.py`**

```python
def load_config(config_path):
    """Load YAML config and merge with defaults."""
    
DEFAULT_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-3,
    'model': {'modes': 16, 'width': 64, 'n_layers': 4},
    ...
}
```

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Python source files | 25+ |
| Julia source files | 8 |
| LaTeX files | 10 |
| Config files | 8 |
| DDE families | 7 |
| Baseline models | 4 |
| Benchmark types | 5 |
| OOD split types | 5 |
| Ablation dimensions | 3 |

---

## Quick Start Commands

```bash
# 1. Setup
pip install -r requirements.txt
cd src/dde/solve_julia && julia --project=. -e "using Pkg; Pkg.instantiate()"

# 2. Verify installation
python scripts/quick_test.py

# 3. Generate data
python src/datasets/build_dataset_sharded.py hutch --n_train=1000 --verify

# 4. Run benchmarks
python scripts/run_full_benchmark.py --family hutch --n_train=1000

# 5. Evaluate trained model
python src/benchmarks/eval_comprehensive.py --checkpoint outputs/best_model.pt --family hutch
```

---

*Generated: December 27, 2024*
