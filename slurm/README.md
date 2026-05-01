# Caltech SLURM σ-stability sweep — instructions

Run the σ-stability sweep (60 cells: 4 σ × 5 fams × 3 seeds) on the Caltech
HPC cluster while RunPod handles the other sweeps.

## Prerequisites

- Caltech HPC account (`atiwari2`) with SLURM access
- SSH config entry for `caltech-hpc` (already in your `~/.ssh/config`)
- Duo 2FA cached (or be ready to authenticate)

## Step-by-step

### 1. SSH to Caltech once to cache 2FA

```bash
ssh caltech-hpc
# (Duo prompt — accept on phone)
exit
```

### 2. From local repo, deploy code + data

```bash
cd "A:/dde research/dde-fno"
bash slurm/deploy_to_caltech.sh
```

This bundles `src/`, `scripts/`, `configs/`, `slurm/` plus the 5 dist-kernel
data shards into `~/dde-fno/` on Caltech HPC. Total transfer ~600 MB,
takes 2-5 minutes.

### 3. SSH back to Caltech and submit the sweep

```bash
ssh caltech-hpc
cd ~/dde-fno
bash slurm/launch_caltech.sh
```

This:
- Creates a Python venv and installs PyTorch + dependencies
- Runs a quick build-model smoke check (verifies σ-bound code is in place)
- Submits a SLURM array job: 60 cells, 1 GPU each, 12h time limit per cell

Output:
```
Submitted SLURM array job: 12345678 (60 tasks)
```

### 4. Monitor

```bash
squeue -u $USER                          # see running cells
sacct -j 12345678                         # per-cell status
tail -f slurm_logs/12345678_0.out         # live training output
find outputs/sigma_*/raw -name test_results.json | wc -l    # completed
```

When complete, you'll have:
```
outputs/sigma_0.5/raw/{family}/clean/lemo_pc_nd/s{seed}/test_results.json
outputs/sigma_0.7/raw/{family}/clean/lemo_pc_nd/s{seed}/test_results.json
outputs/sigma_0.9/raw/{family}/clean/lemo_pc_nd/s{seed}/test_results.json
outputs/sigma_0.99/raw/{family}/clean/lemo_pc_nd/s{seed}/test_results.json
```
plus `best_model.pt`, `history.json`, `per_frame.json` per cell.

### 5. Pull results back

```bash
ssh caltech-hpc
cd ~/dde-fno
tar czf sigma_sweep_results.tar.gz outputs/sigma_* slurm_logs/
exit

# from local:
scp caltech-hpc:~/dde-fno/sigma_sweep_results.tar.gz "A:/dde research/dde-fno/bundles/"
```

## What's in this directory

- `sigma_sweep.sbatch` — SLURM array job file (60 cells)
- `deploy_to_caltech.sh` — local-side: bundles + scp's code + data
- `launch_caltech.sh` — Caltech-side: setups env + submits SLURM job
- `README.md` — this file

## What this gives you

The full **σ-stability frontier** (4 σ values + the σ=None baseline you already have on the laptop bundle):
- σ=None (unconstrained): 0.0123 mean test relL2 (from existing v2 sweep)
- σ=0.5: tightest constraint (Lipschitz ≤ 0.5)
- σ=0.7: moderate constraint
- σ=0.9: loose constraint
- σ=0.99: very loose (near unconstrained)

This is the empirical proxy for **Theorem T2 (σ-Lipschitz operator-norm bound)**
since each ckpt's spectral kernel passes through the per-mode SVD projection
that bounds `‖K[:,:,m]‖_op ≤ σ` per mode.

Plot via `paper/PLOTS_AND_TABLES_PLAN.md` figure B-4-style: x-axis = σ-target,
y-axis = (test relL2, certified Lipschitz, peak rollout norm). Combined with
the v2 unconstrained baseline gives the **σ-frontier** (accuracy vs stability
tradeoff figure for the paper's stability-theorem section).
