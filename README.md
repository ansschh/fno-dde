# Fourier Neural Operators for Delay Differential Equations

This repository contains code for learning solution operators of delay differential equations (DDEs) using Fourier Neural Operators (FNOs).

## Project Structure

```
dde-fno/
├── paper/                          # LaTeX paper files
│   ├── main.tex                   # Main document
│   ├── sections/                  # Individual sections
│   └── references.bib             # Bibliography
│
├── data/
│   └── {family}/                  # Per-family sharded data
│       ├── train/shard_*.npz
│       ├── val/shard_*.npz
│       ├── test/shard_*.npz
│       └── manifest.json
│
├── src/
│   ├── dde/
│   │   ├── families.py            # Python DDE family definitions
│   │   ├── solve_python/          # Python DDE solver (scipy-based)
│   │   └── solve_julia/           # Julia SciML solver (recommended)
│   │       ├── families.jl        # Julia family definitions
│   │       ├── history.jl         # History sampling utilities
│   │       ├── solver.jl          # Robust solver with fallback ladder
│   │       ├── generate.jl        # Sample generator
│   │       ├── qc.jl              # Quality control checks
│   │       ├── dataset_generator.jl  # Sharded dataset writer
│   │       └── validation.jl      # Sanity tests
│   ├── datasets/
│   │   ├── sharded_dataset.py     # PyTorch Dataset for shards
│   │   ├── build_dataset_sharded.py  # Python wrapper for Julia
│   │   ├── generate_python.py     # Pure Python generator (fallback)
│   │   └── visualize.py           # Data visualization
│   ├── models/
│   │   └── fno1d.py               # FNO1D architecture
│   ├── train/
│   │   ├── train_fno_sharded.py   # Training script
│   │   └── eval.py                # Evaluation script
│   └── utils/
│       ├── config.py              # Configuration utilities
│       └── logging.py             # Logging utilities
│
├── scripts/
│   ├── run_experiment.py          # End-to-end experiment runner
│   └── quick_test.py              # Pipeline verification tests
│
└── configs/                        # Training configurations
    └── family_*.yaml
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Julia Setup (Recommended for Data Generation)

The Julia-based solver is **strongly recommended** for robust, accurate data generation.

1. Install Julia from https://julialang.org/downloads/
2. Setup the Julia environment:

```bash
cd src/dde/solve_julia
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Quick Start

### Option A: Full Pipeline (Recommended)

```bash
# Run complete experiment: generate data → train → evaluate
python scripts/run_experiment.py --family hutch --n_train 1000

# Use Python solver if Julia not available
python scripts/run_experiment.py --family linear --use_python_solver
```

### Option B: Step-by-Step

#### 1. Verify Setup

```bash
python scripts/quick_test.py
```

#### 2. Generate Dataset (Julia - Recommended)

```bash
# Direct Julia call
julia --project=src/dde/solve_julia src/dde/solve_julia/dataset_generator.jl hutch \
    --n_train=1000 --n_val=100 --n_test=100 --output_dir=data

# Or via Python wrapper
python src/datasets/build_dataset_sharded.py hutch --n_train=1000 --verify
```

#### 2b. Generate Dataset (Python Fallback)

```bash
python src/datasets/generate_python.py hutchinson --n_train=800
```

#### 3. Train Model

```bash
python src/train/train_fno_sharded.py \
    --config configs/family_hutchinson.yaml \
    --data_dir data \
    --output_dir outputs
```

#### 4. Visualize Data

```bash
python src/datasets/visualize.py hutch --data_dir data --output_dir plots
```

## DDE Families

### Family 1: Linear DDE (Two Delays)
```
x'(t) = a*x(t) + b₁*x(t-τ₁) + b₂*x(t-τ₂)
```
**Parameters:** a ∈ [-2, 1], b₁, b₂ ∈ [-2, 2], τ₁, τ₂ ∈ [0.1, 2.0]

### Family 2: Hutchinson (Delayed Logistic)
```
x'(t) = r * x(t) * (1 - x(t-τ) / K)
```
**Parameters:** r ∈ [0.5, 3.0], K ∈ [0.5, 2.0], τ ∈ [0.1, 2.0]  
**Constraint:** x > 0 (positive histories required)

### Family 3: Mackey-Glass
```
x'(t) = β * x(t-τ) / (1 + x(t-τ)^n) - γ * x(t)
```
**Parameters:** β ∈ [1, 4], γ ∈ [0.5, 3], τ ∈ [0.2, 2.0], n = 10 (fixed)  
**Constraint:** x > 0

### Family 4: Van der Pol with Delayed Feedback
```
x'(t) = v(t)
v'(t) = μ*(1 - x²)*v - x + κ*x(t-τ)
```
**Parameters:** μ ∈ [0.5, 3.0], κ ∈ [-2, 2], τ ∈ [0.1, 2.0]  
**Note:** 2D state with consistent (x, v=dx/dt) history

### Family 5: 2D Predator-Prey with Two Delays
```
x'(t) = x(t) * (α - β * y(t-τ₁))
y'(t) = y(t) * (-δ + γ * x(t-τ₂))
```
**Parameters:** α, β, γ, δ ∈ [0.5, 2.0], τ₁, τ₂ ∈ [0.1, 2.0]  
**Constraint:** x, y > 0

### Family 6: Distributed Delay - Uniform Kernel
```
x'(t) = r*x(t)*(1 - m(t)/K)
m(t) = (1/τ) ∫_{t-τ}^t x(s) ds
```
**Auxiliary form:** m'(t) = (x(t) - x(t-τ))/τ  
**Parameters:** r ∈ [0.5, 2.5], K ∈ [0.5, 2.0], τ ∈ [0.1, 2.0]

### Family 7: Distributed Delay - Exponential Kernel
```
x'(t) = r*x(t)*(1 - z(t)/K)
z(t) = (1/C) ∫_{t-τ}^t exp(-λ(t-s)) x(s) ds
```
**Auxiliary form:** z'(t) = -λz + (x - exp(-λτ)x(t-τ))/C  
**Parameters:** r ∈ [0.5, 2.5], K ∈ [0.5, 2.0], τ ∈ [0.1, 2.0], λ ∈ [0.5, 5.0]

## Operator Learning Formulation

The DDE initial value problem defines an operator:
```
G: (φ, θ) → x(·) on [0, T]
```
where:
- φ is the history function on [-τ_max, 0]
- θ are parameters
- x(t) is the solution trajectory

### Input Encoding

On a combined grid t ∈ [-τ_max, T]:
- **History signal**: φ(t) for t ≤ 0, else 0
- **Mask**: 1 for t ≤ 0, else 0
- **Time coordinate**: normalized t
- **Parameters**: broadcast as constant channels

### Output

Full solution x(t) on [-τ_max, T], with loss computed only on t ∈ [0, T].

## Data Generation Pipeline

### Key Design Principles

1. **Sharded Storage** - Resumable, parallelizable, no RAM issues
2. **Quality Control** - Finite check, bounds check, positivity, continuity at t=0
3. **Solver Fallback Ladder**:
   - Fast: `MethodOfSteps(Tsit5())`
   - Accurate: `MethodOfSteps(Vern6())`
   - Stiff: `MethodOfSteps(Rosenbrock23())`
   - Constrained: `MethodOfSteps(Tsit5(); constrained=true)`

4. **Discontinuity Handling** - `tstops` at multiples of τ
5. **Reproducible** - Seeds, configs, manifest tracking

### Shard Format

Each `.npz` shard contains:
- `t_hist`: (N_hist,) history time grid
- `t_out`: (N_out,) output time grid  
- `phi`: (B, N_hist, d_hist) history values
- `y`: (B, N_out, d_state) solutions
- `params`: (B, P) parameters
- `lags`: (B, L) delay values
- `meta_json`: solver settings, seed, timestamp

## Model Architecture

FNO1D with:
- Spectral convolution in time dimension
- 4-6 Fourier layers
- Width 64-128 channels
- 16-32 Fourier modes
- GELU activation

### Input Encoding (Combined Grid)

| Channel | Description |
|---------|-------------|
| 0..d-1 | History signal (φ for t≤0, 0 for t>0) |
| d | Mask (1 for t≤0, 0 for t>0) |
| d+1 | Normalized time |
| d+2..d+1+P | Parameters (broadcast) |

### Loss

MSE computed only on future region (t > 0):
```
L = (1/|T|) Σ_{t∈[0,T]} |ŷ(t) - y(t)|²
```

## Validation Tests

Run before generating large datasets:

```bash
julia --project=src/dde/solve_julia src/dde/solve_julia/validation.jl
```

Tests include:
- Linear DDE → ODE reduction (b₁=b₂=0)
- VdP DDE → ODE reduction (κ=0)
- History continuity at t=0
- Solver accuracy comparison (fast vs reference)

## Citation

If you use this code, please cite:

```bibtex
@article{li2020fno,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Li, Zongyi and others},
  journal={arXiv preprint arXiv:2010.08895},
  year={2020}
}
```

## License

MIT License
