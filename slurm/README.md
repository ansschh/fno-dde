# Caltech SLURM σ-stability sweep — instructions

Run the σ-stability sweep (60 cells: 4 σ × 5 fams × 3 seeds) on the Caltech
HPC cluster while RunPod handles the other sweeps.

## Prerequisites

- Caltech HPC account (`atiwari2`) with SLURM access
- SSH config entry for `caltech-hpc` (already in your `~/.ssh/config`)
- Duo 2FA cached (or be ready to authenticate)

## Step-by-step (git-clone workflow — recommended)

This avoids the WSL bash dependency.  All steps run from PowerShell on
the laptop or in an SSH session on Caltech.

### 1. SSH to Caltech once to cache 2FA

```bash
ssh caltech-hpc
# (Duo prompt — accept on phone)
```

### 2. On Caltech: git clone the repo

```bash
cd ~
git clone https://github.com/ansschh/fno-dde.git dde-fno
cd dde-fno
mkdir -p data_dde_pde slurm_logs
exit
```

### 3. From PowerShell on the laptop: scp the 5 dist-kernel data shards

Data isn't tracked in git (it's ~600 MB).  From a PowerShell prompt:

```powershell
cd "A:\dde research\dde-fno"
scp -r data_dde_pde\dist_exp_rd_2d caltech-hpc:~/dde-fno/data_dde_pde/
scp -r data_dde_pde\dist_gaussian_rd_2d caltech-hpc:~/dde-fno/data_dde_pde/
scp -r data_dde_pde\dist_gamma_rd_2d caltech-hpc:~/dde-fno/data_dde_pde/
scp -r data_dde_pde\dist_uniform_rd_2d caltech-hpc:~/dde-fno/data_dde_pde/
scp -r data_dde_pde\dist_powerlaw_rd_2d caltech-hpc:~/dde-fno/data_dde_pde/
```

(or one shot: `scp -r data_dde_pde\dist_*_rd_2d caltech-hpc:~/dde-fno/data_dde_pde/`
if your shell expands the glob.  PowerShell does, cmd doesn't.)

Total transfer ~600 MB, takes 2-5 minutes.

### 4. SSH back to Caltech and submit the sweep

```bash
ssh caltech-hpc
cd ~/dde-fno
bash slurm/launch_caltech.sh
```

## Step-by-step (legacy bash-deploy workflow — only for Linux/macOS or WSL)

If you have a working bash environment with native scp:

```bash
cd "A:/dde research/dde-fno"
bash slurm/deploy_to_caltech.sh
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

When complete, every cell directory
`outputs/sigma_${σ}/raw/${family}/clean/lemo_pc_nd/s${seed}/` will have:

- `best_model.pt`            — checkpoint (model state + config)
- `history.json`             — per-epoch: train_loss, val_rel_l2, grad_norm,
                                weight_norm, op_norm_max (raw kernel σ-max
                                — proves σ-projection binding when > σ),
                                lr, wall_per_epoch_s, peak_mem_gb
- `test_results.json`        — final test relL2 + sigma + final_op_norm_max
                                + n_epochs + peak_mem_gb_overall + config
- `per_frame.json`           — per-rollout-step relL2 (aggregate-then-sqrt + abs MSE + naive baseline)
- `viz_samples.npz`          — 4 (input, target, pred) tuples for figures
- `kernel_snapshot.npz`      — learned spectral kernel + FiLM γ/β
- `residuals.npz`            — per-sample relL2 + residual norms (full test set)
- `long_rollout.npz`         — 5-step chained autoregressive rollout:
                                pred_norm_per_step, peak_norm_per_chain,
                                final_norm_per_chain, rel_l2_step0_mean
                                (σ-stability divergence proxy)
- `fft_residual.npz`         — |FFT_t(yhat-y)|² energy per spectral mode
                                (where errors live in lag-spectrum)
- `equivariance.json`        — cyclic-shift (1, 4, 16) equivariance test
                                (T1 in the wild: equiv_shift_*_mean/std/max,
                                T1_pass flag, max_shift_err)

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
