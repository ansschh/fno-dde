# Paper context for external LLM — LEMO-PC figure-set design

## Paper summary

**LEMO-PC** is a Lag-Equivariant Memory Operator for delay-structured PDEs.
Key architectural components:

- **Cyclic-FFT lag convolution** along the lag axis (history window) — produces a per-lag-mode complex spectral kernel `K[in_ch, out_ch, mode]`.
- **Per-(out_channel, lag_mode) FiLM modulation** — scaling $\gamma$ and bias $\beta$ are predicted from physical parameters (delay $\tau$, etc.) and applied to the spectral coefficients before the inverse FFT.
- **2D spatial spectral conv** (FNO-style) per lag position.
- Optional **σ-Lipschitz projection**: per-mode SVD clamp on the spectral kernel forces $\|K[:,:,m]\|_{\mathrm{op}} \le \sigma$ at every mode (tight bound, not the loose elementwise one).

Two formal theorems are Lean-formalized (0 sorries, standard axioms only):

- **T1 — cyclic-group lag-equivariance.** $\mathrm{LEMO}(\rho_k x) = \rho_k \mathrm{LEMO}(x)$ for any cyclic shift $k$ along the lag axis.
- **T2 — σ-stability operator-norm bound.** With the σ-projection active, the operator-norm of the lag-block is bounded by $\sigma$ per spectral mode.

## Benchmarks

5 distributed-delay reaction-diffusion 2D families. Each family generates trajectories from a PDE with a distributed-delay term whose temporal kernel takes a different functional form:

| Family | Kernel shape |
|---|---|
| `dist_exp_rd_2d` | $K(s) \propto e^{-3s}$ |
| `dist_gaussian_rd_2d` | $K(s) \propto e^{-(s-0.3)^2/0.05}$ |
| `dist_gamma_rd_2d` | $K(s) \propto s^{1.5} e^{-3s}$ |
| `dist_uniform_rd_2d` | $K(s) = 1_{s<0.5}$ |
| `dist_powerlaw_rd_2d` | $K(s) \propto (s+0.05)^{-1.2}$ |

Common settings:
- 64×64 spatial grid
- $n_{\text{hist}} = 64$ history frames + $n_{\text{out}} = 64$ future frames
- $\Delta t = 0.01$
- 1000 train / 200 val / 200 test trajectories per family
- 1 state channel, params_dim = 1

3 input-perturbation regimes (target stays clean):
- **clean** (no perturbation)
- **lowres** (input downsampled by 2× then upsampled back)
- **noisy** (Gaussian noise, std = 0.05 × channel-std)

3 random seeds per (family, regime): 42, 123, 456.

Plus 3 single-delay 2D families (`mackey_glass`, `wright`, `hutchinson`) on the same grid.

Plus an APEBench arc (kolmogorov_2d, burgers_1d, burgers_3d, gray_scott_3d, decaying_turbulence_2d) with clean and residual-anchor input modes.

## Models compared

| Model | Role | Params (~) | Key feature |
|---|---|---|---|
| **LEMO-PC** (ours) | primary | 2.7M | cyclic-FFT lag + FiLM + spatial spectral |
| LEMO (no FiLM) | ablation | 2.4M | cyclic-FFT lag + spatial spectral, NO FiLM |
| FNO | baseline | 11M | 3D spectral over (lag, space) |
| Markov-FNO | baseline | 480k | 2D spatial-only spectral, fed last frame |
| Window-FNO | baseline | 480k | spatial spectral fed sliding window |
| MemNO | baseline | 58k | per-spatial 1D MemNO along lag |
| F-FNO | baseline | 105k | factorized FNO per-axis |
| UNet | context | 2M | 2D U-Net (excluded from headline) |

All trained with `width=64`, `n_layers=3`, `lag_modes=24`, `spatial_modes=12`, `residual_anchor=True`, batch=4, 200 epochs, cosine LR.

## Experiments (state of compute)

### Already done — data on laptop

| Sweep | Cells | Status |
|---|---|---|
| `dist_kernel_v2_p1` (LEMO + LEMO-PC) | 5 fams × 3 regimes × 3 seeds × 2 models = 90 | ✅ checkpoints + artifacts in `extracted/pod1/` |
| `dist_kernel_v2_p2` (FNO/Markov-FNO/Window-FNO/UNet) | 5 fams × 3 regimes × 3 seeds × 4 models = 180 | ✅ test relL2 in pod2 logs (test_results.json was truncated) |
| `layer5_final_sweep_p{1,2}` (single-delay) | 3 fams × 3 regimes × 3 seeds × 6 models = 162 | ✅ |
| `sweep_apebench`, `sweep_apebench_residual_clean` | various | ✅ |
| `sweep_lemo_scale` (burgers_3d width grid) | 3 widths × 3 seeds = 9 | ✅ |
| LDS sweep (data-only) | 5 families × ACF, R²_markov, R²_full, LDS | ✅ |

### Currently running

- **Pod 3 Phase A** (8×H100): MemNO + F-FNO baselines on dist-kernel families = 30 cells. ~50% done. ETA 60-90 min.
- **Pod 4 chain** (8×H100):
  - Phase A done: sample-efficiency curve at $N_{\text{train}} \in \{100, 250, 500, 1000\}$ × LEMO-PC, FNO × 3 seeds = 24 cells
  - Phase B running: lag-grid quantization at lag_modes ∈ {8, 12, 24, 32} × 3 seeds = 12 cells
  - Phase C queued: F-FNO at n_layers=12 (depth control)
  - Aux jobs (free GPUs 6, 7): capture pipeline + post-hoc analyses on the existing 45 LEMO-PC ckpts
- **Caltech HPC SLURM σ-sweep** queued: 4 σ × 5 fams × 3 seeds = 60 cells. Datagen running first; sweep starts ~6h from now.

### Per-cell artifacts available

For every LEMO-PC cell (45 cells × clean+lowres+noisy):

- `best_model.pt` — checkpoint with full config
- `history.json` — per-epoch train_loss, val_rel_l2, grad_norm, weight_norm, **op_norm_max** (raw spectral kernel σ_max — proves σ-projection binding when above target σ), lr, wall_per_epoch_s, peak_mem_gb
- `test_results.json` — final test relL2, sigma, final_op_norm_max, params, full config
- `per_frame.json` — per-rollout-step relL2 + naive last-frame-copy baseline
- `viz_samples.npz` — 4 (input, target, pred) trajectories for visualization, full T=128 frames each, 64×64 spatial
- `kernel_snapshot.npz` — every spectral kernel + every FiLM γ/β layer per block (3 blocks per model)
- `residuals.npz` — per-sample test rel_L2 + residual norms across the full test set
- `long_rollout.npz` — 5-step chained autoregressive rollout: peak_norm_per_chain, final_norm_per_chain, per-step norm trajectory (σ-stability divergence proxy)
- `fft_residual.npz` — $|\text{FFT}_t(\hat{u} - u)|^2$ energy per spectral lag mode
- `equivariance.json` — cyclic-shift T1 test at deployment with shifts $\in \{1, 4, 16\}$, includes `T1_pass` flag, `max_shift_err`
- (post-hoc only) `kernel_recovery.npz/json`, `cross_family_relL2.json`, `adv_fgsm.json`, `noise_sweep.json`

## Statistical methodology

For LEMO-PC vs each baseline, 45 paired cells (5 fams × 3 regimes × 3 seeds):
- Paired-permutation test (10000 perms)
- Bootstrap 95% CI on improvement ratio (10000 resamples)
- Hedges *g* paired (small-sample-corrected $d_z = \bar{(b-a)} / \mathrm{std}(b-a, \mathrm{ddof}=1)$)
- Holm-Bonferroni correction across the 4 primary comparisons

## Headline numbers

| Comparison | % impr | 95% CI | Hedges *g* | $p$-value | $n$ |
|---|---:|---:|---:|---:|---:|
| LEMO-PC vs FNO | 69.3 | [66.2, 72.2] | 5.08 | $<10^{-4}$ | 45 |
| LEMO-PC vs Markov-FNO | 79.8 | [77.7, 81.8] | 5.90 | $<10^{-4}$ | 45 |
| LEMO-PC vs Window-FNO | 80.0 | [77.9, 82.0] | 6.01 | $<10^{-4}$ | 45 |
| LEMO-PC vs LEMO (no FiLM) | 94.8 | [94.2, 95.3] | 23.43 | $<10^{-4}$ | 45 |
| LEMO-PC vs UNet | 10.0 | [3.4, 16.0] | 0.45 | $4.2\times 10^{-3}$ | 45 (excluded from headline) |

## What I want from you

I want a **8–12 figure final set** for the paper. The figures must convey information that **a table or one-sentence summary cannot**.

Specifically I'm rejecting these "non-figures":
- Bar charts of model rel-L2 (info is in tables)
- Pareto-frontier scatters with 5 points
- Hedges *g* forest plots (one number per row)
- Per-regime box plots when the spread is small
- Single-number heatmaps (correlation = 0.97, Jaccard = 0.78)
- Curves that overlap perfectly across families (auto-correlation, FFT residual energy)
- Distribution histograms of training metrics

I want figures that:
1. **Show actual PDE field data** (initial condition / ground truth / prediction triptychs).
2. **Have multiple things going on** in one figure (annotations, multiple panels, overlays).
3. **Cannot be summarized in a sentence** because their structure conveys information.
4. **Match the visual standard** of the NO-paper-Figure-4 (Liu-Schiaffini et al. 2024) or SFNO Figure 3 quality.
5. **Are mechanism-revealing** rather than just performance-reporting.

Examples of figures I'd consider for inclusion:

- **Per-family failure-mode gallery**: For each family, find the *easiest* and *hardest* test trajectory by relL2; show GT and prediction side-by-side at $t=0, T/2, T$. Tells the reader where LEMO-PC succeeds and fails *visually*.
- **Cyclic-shift equivariance demo**: For one input $x$, render $\rho_k x$ and $\mathrm{LEMO}(\rho_k x)$ for $k \in \{0, 4, 16, 32\}$ side by side. The output panels should look *visually identical* up to a cyclic shift — that's the theorem in pictures.
- **Long-horizon stability**: With and without σ-projection, render the predicted field at $t = 64, 128, 256, 512, 1024$. The unconstrained version should visibly diverge; the constrained version should stay bounded. Single most direct theorem-empirical bridge.
- **Distributed-delay kernel inversion**: for each of the 5 families, plot in one figure: (a) ground-truth simulator kernel (analytic shape), (b) LEMO-PC's learned time-domain kernel (irFFT of spectral weights), (c) the spectral-domain learned kernel as a heatmap. Overlay them carefully — shows the architecture *learned the right physics*.
- **Adversarial perturbation visualization**: input $x$, FGSM-perturbed input $x + \epsilon \nabla_x L$, $\epsilon \in \{0, 0.01, 0.05\}$, predicted field $\hat{y}(x)$, $\hat{y}(x+\epsilon \nabla)$, $|\hat{y}(x+\epsilon \nabla) - \hat{y}(x)|$. Shows the σ-Lipschitz robustness claim *as a picture*.
- **Multi-block kernel composition**: LEMO-PC has 3 sequential lag-spectral blocks; show each block's learned spectral kernel separately + their composition — what each block contributes.
- **Conserved-quantity tracking**: $\|u(t)\|_{L^2}$ vs $\|\hat{u}(t)\|_{L^2}$ as a function of $t$ for several test trajectories; if model preserves the right conservation law, lines overlap.
- **Sample-efficiency curve as field rather than scalar**: at each $N_{\text{train}} \in \{100, 250, 500, 1000\}$, render the prediction at $t = T$ for the same trajectory. Watch the prediction quality improve visually.
- **σ-vs-accuracy tradeoff visualization**: at each $\sigma \in \{0.5, 0.7, 0.9, 0.99, \infty\}$, render the test field at $t=T$. Shows the accuracy-stability tradeoff *as physics not as a curve*.
- **Per-channel spectral filter response**: LEMO-PC has 64 output channels each modulated by FiLM; render the magnitude spectrum each channel selects from input. Shows specialization across channels (frequency selectivity).
- **Receptive-field saliency**: $\partial \hat{u}(t_{\text{out}}) / \partial u(t_{\text{in}})$ for $t_{\text{in}} \in \text{history}$. Shows which past frames the prediction at $t_{\text{out}}$ depends on most. If LEMO-PC truly learned the distributed-delay structure, the saliency should peak around the kernel center.
- **Dist-kernel difficulty vs LEMO-PC advantage scatter**: x = LDS (lag-dependence statistic), y = % improvement of LEMO-PC over the best baseline on that family. If our hypothesis is right (LEMO-PC's gain comes from delay-awareness), points should align.

Use the per-cell artifact list above to decide what's feasible. Don't propose figures that need data we don't have. For each suggestion:

- **What to plot**: the actual content
- **Why visual not table**: what does this convey that a table/sentence cannot?
- **Source data**: which artifact provides the input
- **Layout sketch**: 2 columns × 5 rows? sequence of frames? matrix?

Be ruthless. 12 figures max. Prioritize *mechanism* over *performance reporting*.
