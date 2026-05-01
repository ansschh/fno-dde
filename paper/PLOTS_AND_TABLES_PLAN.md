# Paper Plotting Plan — 2D-first, intentionally maximal

Generated 2026-04-29.  The paper's **main focus is 2D DDE-PDE benchmarks**;
the 1D continuous-τ / σ-stability / fair-1D-DDE story (per ADVICE.txt) is
deferred because no 1D LEMO checkpoints exist.

**Realistic paper pillars:**
1. Theorem + Lean formalization — `A:/dde research/lean/LEMO/`
2. **2D DDE-PDE benchmark (dist-kernel + single-delay)** — pods 1+2
3. APEBench / residual-anchor / scaling negative results — pod 1
4. 1D FNO + naive baseline anchor (small appendix, FNO-only)

ADVICE.txt's full figure menu is **ported to 2D** wherever it makes sense.
Each section below covers the 2D analog.

**Workflow:**
1. While the v2 sweep + follow-up phases run, capture pipeline saves
   per-frame relL2, naive baselines, kernel snapshots, residuals, viz
   samples per cell (`scripts/capture_paper_artifacts.py`).
2. Bundle pods → laptop (`scripts/bundle_pods_for_download.sh`).
3. Plot offline using captured JSONs/NPZs.

**Status legend:** ✅ have · 🟢 will-have (post-hoc capture / Phase E) ·
🟡 partial · 🔴 missing · 📜 historical (locate)

---

## Top-15 priority list (paper-final, 2D-first)

| # | Figure | Status | Source |
|---|--------|--------|--------|
| 1 | dist-kernel mean ± std heatmap (5 fams × 6 models × 3 regimes) | 🟢 | v2 sweep, both pods |
| 2 | LEMO_PC vs UNet head-to-head ratio per dataset/regime | 🟢 | v2 + Phase A param-match UNet |
| 3 | Per-frame relL2 vs rollout step (LEMO vs UNet vs naive copy), per dataset | 🟢 | post-hoc capture |
| 4 | LEMO_ND ablation: ~35× loss vs LEMO_PC (per-mode FiLM matters) | 🟢 | v2 |
| 5 | E_orbit (Metric M3) per model/dataset — theorem T1 verification | 🟢 | Phase E2 |
| 6 | Lag-shift OOD (τ outside training range) per model | 🟢 | Phase E1 |
| 7 | Param-efficiency Pareto frontier on dist-kernel | 🟢 | v2 |
| 8 | Single-delay Layer-5 heatmap (mackey/wright/hutch × 6 models × 3 regimes) | ✅ | pod 1 `layer5_final_sweep_p1` + pod 2 `layer5_final_sweep_p2` |
| 9 | Layer-4 single-seed leaderboard heatmap (6 fams × 6 models) | ✅ | pod 1 `layer4_audit` |
| 10 | APEBench clean-vs-residual delta heatmap | ✅ | pod 1 `sweep_apebench*` |
| 11 | Cyclic-pad jump statistic vs residual benefit scatter | 🟢 | derive from raw data + sweep results |
| 12 | Scaling curve burgers_3d (LEMO width vs error) | ✅ | pod 1 `sweep_lemo_scale` |
| 13 | Trajectory + error-map gallery per dataset (LEMO_PC vs UNet vs target) | 🟢 | post-hoc capture (`viz_samples.npz`) |
| 14 | Learned LEMO lag-spectrum + FiLM γ/β heatmap per dataset | 🟢 | post-hoc capture (`kernel_snapshot.npz`) |
| 15 | Theorem-to-experiment matrix + Lean axiom footprint | 🟢 | Lean repo + hand-curate |

---

## A. Theory-verification figures (2D analogs)

ADVICE.txt's section A was built around 1D continuous-τ.  The 2D analogs
that we CAN produce against v2 dist-kernel ckpts:

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| A1 | Test error vs τ-distance from training distribution (lag-shift OOD curve) | 🟢 | Phase E1 — `eval_lag_shift_ood.py` |
| A2 | Theorem-T1 cyclic-shift error histogram (E_orbit per shift k) | 🟢 | Phase E2 — `eval_equivariance.py` |
| A3 | Performance ratio LEMO_PC / UNet per family/regime | 🟢 | v2 + Phase A |
| A4 | Seed-wise boxplot for relL2 by family/regime/model | 🟢 | v2 (3 seeds) |
| A5 | Family × regime × model error heatmap | 🟢 | v2 |
| A6 | Calibration scatter: predicted vs target field magnitude (per dataset) | 🟢 | post-hoc capture |
| A7 | Residual histogram per model per dataset | 🟢 | `residuals.npz` |
| A8 | Residual vs τ (kernel center / decay rate) per dataset | 🟢 | post-hoc + dataset metadata |
| A9 | Learned 2D lag kernel K_θ heatmap (lag × output channel) | 🟢 | `kernel_snapshot.npz` |
| A10 | Operator slices: predicted future field as kernel parameter τ varies | 🟢 | post-hoc + dataset re-eval |
| A11 | Train-set τ density vs test error curve | 🟢 | derive from manifest + per-cell relL2 |
| A12 | Effective covering radius (τ-grid spacing) vs error | 🟢 | derive |
| A13 | Equivariance residual histogram under random cyclic shifts | 🟢 | Phase E2 |
| A14 | Orbit-constancy plot: same example, dense shift sweep | 🟢 | extend Phase E2 to denser shifts |
| A15 | Causal LEMO (T1') vs full LEMO_PC comparison | 🟡 | requires phase_d (deferred) |
| A16 | Lag-grid quantization scaling: error vs L (lag length) | 🔴 | needs new sweep with varying L |
| A17 | Theorem-to-empirics composite (T1 verification + OOD generalization) | 🟢 | composes A1, A2, A13 |

## B. Stability and σ-theorem figures (2D, requires σ-sweep)

We have v2 ckpts at σ=null (unconstrained).  σ-stability empirics need a
small additional sweep across σ values — out of current 10h budget but
small enough to do later.  Listing for completeness.

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| B1 | Final rollout norm vs σ (autoregressive 2D rollout) | 🔴 | needs σ-sweep |
| B2 | Peak rollout norm vs σ | 🔴 | same |
| B3 | Blow-up fraction across seeds vs σ | 🔴 | same |
| B4 | Test error vs σ on dist-kernel | 🔴 | same |
| B5 | Accuracy–stability frontier scatter | 🔴 | same |
| B6 | Per-step rollout norm curve over horizon, all σ | 🔴 | same |
| B7 | Per-step rollout error curve, all σ | 🔴 | same |
| B8 | Rollout phase portrait stable vs unstable | 🔴 | same |
| B9 | Histogram of learned spectral magnitudes by σ | 🔴 | extract from σ-sweep ckpts |
| B10 | Stability gap plot: log10 final norm by model | 🔴 | σ-sweep |
| B11 | Empirical-σ distribution from current v2 ckpts (σ=null) | 🟢 | extract from `kernel_snapshot.npz` |
| B12 | Theoretical σ bound vs measured operator norm (current ckpts) | 🟢 | derive |

## C. 2D DDE-PDE benchmark figures (THE MAIN CHAPTER)

ADVICE.txt's section C (1D parametric DDE) ported to 2D:

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| C1 | Family × regime × model heatmap (relL2 mean ± std) | 🟢 | v2 sweep (5 dist-kernel + 3 single-delay families) |
| C2 | Rank heatmap (1=best, 6=worst per cell) | 🟢 | derive |
| C3 | Win-count bar chart over cells | 🟢 | derive |
| C4 | Per-family grouped bar chart, ID test | 🟢 | v2 |
| C5 | Per-family grouped bar chart, lag-shift OOD | 🟢 | Phase E1 |
| C6 | Per-family grouped bar chart, regime corruption (lowres / noisy) | 🟢 | v2 |
| C7 | Per-cell top-3 margin plot | 🟢 | derive |
| C8 | Seed-wise dot plot per cell | 🟢 | v2 (3 seeds) |
| C9 | Family difficulty: best achieved error per family/regime | 🟢 | derive |
| C10 | Model robustness: std across seeds per family/regime | 🟢 | derive |
| C11 | ID vs OOD scatter per model | 🟢 | v2 + Phase E1 |
| C12 | OOD degradation ratio per model and family | 🟢 | derive |
| C13 | Param count vs error per model | 🟢 | v2 ckpts |
| C14 | Wall-clock vs error per model | 🟢 | v2 `test_results.json` |
| C15 | Param-efficiency Pareto frontier | 🟢 | derive |
| C16 | Wall-clock Pareto frontier | 🟢 | derive |
| C17 | Per-family representative trajectory overlays (true vs top-3 preds) | 🟢 | post-hoc capture (`viz_samples.npz`) |
| C18 | Error vs delay parameter (kernel mean τ) within each family | 🟢 | post-hoc — bin test set by τ |
| C19 | Error vs rollout horizon (per-frame relL2 curve) | 🟢 | `per_frame.json` |
| C20 | Error vs spatial resolution (clean vs lowres) | 🟢 | v2 |
| C21 | LEMO_PC learned lag spectrum / FiLM modulation viz per dataset | 🟢 | `kernel_snapshot.npz` |
| C22 | FiLM γ/β heatmap (out-channel × lag-mode) per dataset | 🟢 | `kernel_snapshot.npz` |
| C23 | LEMO_PC vs LEMO_ND ablation bar (~35× loss without per-mode FiLM) | 🟢 | v2 |
| C24 | Per-sample residual correlation matrix (LEMO_PC, UNet, FNO, naive) | 🟢 | `residuals.npz` |
| C25 | Hardest-decile Jaccard heatmap between models | 🟢 | derive |
| C26 | Phase-amplitude error decomposition per family (FFT-domain residual) | 🟢 | extend post-hoc |
| C27 | LEMO_PC vs UNet head-to-head ratio (forest plot) | 🟢 | v2 + Phase A |
| C28 | Naive-copy baseline relL2 per family (sanity) | 🟢 | post-hoc capture |
| C29 | Architecture-vs-task-fit heatmap (which model wins each cell) | 🟢 | derive |
| C30 | Single-delay (mackey/wright/hutch) leaderboard heatmap | ✅ | `layer5_final_sweep_p*` |
| C31 | Single-delay vs dist-kernel error gap (per model) | 🟢 | combine |
| C32 | Causal LEMO vs LEMO_PC comparison (if phase_d ran) | 🟡 | deferred |
| C33 | dist-kernel kernel-shape ablation: exp / Gaussian / gamma / uniform / power-law | 🟢 | v2 |

## D. LDS / mechanism figures (2D, data-driven — survives ckpt loss)

LDS ("lag-dependence statistic") is computed on data + naive predictors,
no ML ckpts needed.  Apply to 2D dist-kernel families.

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| D1 | LDS bar chart for 5 dist-kernel families + 3 single-delay families | 🟢 | rerun `scripts/lds_sweep.py` adapted for 2D |
| D2 | LDS vs LEMO advantage scatter (LEMO_PC over UNet) | 🟢 | derive |
| D3 | LDS vs best baseline type (local-conv vs spectral vs lag-equivariant) | 🟢 | derive |
| D4 | LDS vs OOD degradation ratio | 🟢 | derive |
| D5 | LDS stratified boxplots: weakly-lag-dependent vs lag-dominant | 🟢 | derive |
| D6 | Best horizon for LDS estimation per family | 🟢 | LDS sweep |
| D7 | R²_now vs R²_full scatter per family × horizon | 🟢 | LDS sweep |
| D8 | LDS bootstrap CIs over random subsets | 🟢 | LDS sweep |
| D9 | Mechanism summary plot: LDS, LEMO gain, winner color, regime shape | 🟢 | composite |

## E. APEBench / residual-anchor / scaling / negative results (2D + 3D)

Pre-existing on pod 1.  Use as the "honest negative results" arc.

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| E1 | Clean APEBench leaderboard heatmap | ✅ | pod 1 `sweep_apebench*` |
| E2 | Residual-anchor leaderboard heatmap | ✅ | pod 1 `sweep_apebench_residual_clean` |
| E3 | Residual delta heatmap (improvement per model/dataset) | ✅ | derive |
| E4 | Cyclic-pad jump magnitude bar chart by dataset | 🟢 | compute on raw shards |
| E5 | Residual benefit vs cyclic-pad jump scatter | 🟢 | derive |
| E6 | Per-frame relL2 burgers_3d clean vs residual (LEMO, UNet) | 🟢 | post-hoc on those ckpts |
| E7 | t=0 relL2 vs overall relL2 scatter | 🟢 | post-hoc |
| E8 | Naive-copy vs LEMO vs UNet at t=0 and t=horizon | 🟢 | post-hoc |
| E9 | Scaling curve burgers_3d: width vs error for LEMO_PC_ND | ✅ | pod 1 `sweep_lemo_scale` |
| E10 | Param count vs burgers_3d error including scaling sweep | ✅ | derive |
| E11 | Residual-anchor helps/hurts chart incl. burgers_1d counterexample | ✅ | derive |
| E12 | APEBench positive/negative summary (where spectral wins, where UNet wins) | 🟢 | hand-curate from E1/E2 |

## F. Training-dynamics / debugging figures

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| F1 | Train/val loss curves for flagship cells | ✅ | `history.json` per cell |
| F2 | Best-checkpoint epoch histogram by architecture | ✅ | derive from `history.json` |
| F3 | Learning curve overlay: LEMO_PC vs UNet on hardest family | ✅ | derive |
| F4 | Kernel magnitude histogram at end-of-training | 🟢 | `kernel_snapshot.npz` |
| F5 | FiLM γ/β distributions across datasets/regimes | 🟢 | same |
| F6 | Feature norm through depth (representative batch) | 🟢 | post-hoc forward-hook eval |
| F7 | Activation cascade plot LEMO vs UNet | 🟢 | post-hoc forward-hook eval |
| F8 | Gradient norm per epoch | 🔴 | not logged this round |
| F9 | Weight norm per epoch | 🔴 | not logged this round |

## G. 1D anchor (small appendix — what we do have)

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| G1 | FNO-vs-naive 1D benchmark anchor table | ✅ | `reports/baselines/comparison_table.md` |
| G2 | 1D OOD-delay degradation per family (FNO only) | ✅ | `reports/baseline_eval/baseline_all5_summary_v2.json` |
| G3 | linear2 phase-amplitude error decomposition | ✅ | `reports/linear2_diagnosis.json` |
| G4 | Per-family error-vs-time curve (FNO, hutch + linear2) | ✅ | `reports/baseline_eval/figs/*_error_vs_time.json` |
| G5 | Difficulty ranking 5 families | ✅ | derive |

## H. Formalization / theory-artifact figures

| # | Figure | Status | Source data |
|---|--------|--------|-------------|
| H1 | Theorem dependency graph (DAG) | 🟢 | hand-curate from Lean repo |
| H2 | Formalization coverage chart | 🟢 | Lean audit |
| H3 | Lean file size / theorem count bar chart | 🟢 | derive |
| H4 | Axiom footprint table | 🟢 | `lake env lean --print-axioms` |
| H5 | Theorem-to-experiment evidence matrix | 🟢 | hand-curate |
| H6 | Assumption-ledger map | 🟢 | hand-curate |

---

## Tables

### Core main-paper tables

| # | Table | Status | Source |
|---|-------|--------|--------|
| T1 | Theorem summary (theorem, assumption, Lean status, empirical proxy) | 🟢 | hand-curate |
| T2 | 2D DDE-PDE main leaderboard (5 dist-kernel × 6 models × 3 regimes, mean±std) | 🟢 | v2 + Phase A |
| T3 | Single-delay 2D leaderboard (3 fams × 6 models × 3 regimes, mean±std) | ✅ | layer5 |
| T4 | Param counts and wall-clock per 2D model | 🟢 | v2 |
| T5 | Param-efficiency table (params, error, params/error) | 🟢 | derive |
| T6 | Per-frame relL2 summary: t=0, ¼, ½, ¾, 1 of horizon | 🟢 | `per_frame.json` |
| T7 | Naive-copy baseline table per dataset | 🟢 | post-hoc |
| T8 | LEMO_PC vs UNet head-to-head table (effect size + sign) | 🟢 | v2 + Phase A |
| T9 | OOD degradation table (ID/OOD ratio per model/family) | 🟢 | Phase E1 |
| T10 | E_orbit (Metric M3) table per model/family | 🟢 | Phase E2 |

### Appendix benchmark tables

| # | Table | Status | Source |
|---|-------|--------|--------|
| T11 | Full 2D all-model all-cell raw results (all 3 seeds) | 🟢 | v2 |
| T12 | Rank-only version | 🟢 | derive |
| T13 | Architecture ablation table (LEMO_PC vs LEMO_ND) | 🟢 | v2 |
| T14 | LDS table for 2D families × horizons | 🟢 | LDS sweep adapted |
| T15 | Mechanism table per non-win cell | 🟢 | hand-curate from C24-C26 |
| T16 | Hardest-decile Jaccard among top models | 🟢 | post-hoc |
| T17 | Per-sample residual correlation among top models | 🟢 | post-hoc |
| T18 | APEBench clean vs residual table | ✅ | pod 1 |
| T19 | Residual benefit table with cyclic-gap statistic | 🟢 | derive |
| T20 | Scaling table on burgers_3d | ✅ | pod 1 |
| T21 | Phase-amplitude decomposition table | 🟢 | extend post-hoc |
| T22 | Corruption-regime table (clean / lowres / noisy) | 🟢 | v2 |
| T23 | Cross-regime robustness drop table | 🟢 | derive |
| T24 | Causal LEMO vs LEMO_PC comparison table (if phase_d ran) | 🟡 | deferred |
| T25 | Kernel-shape (exp/gauss/gamma/unif/powerlaw) ablation table | 🟢 | v2 |
| T26 | Single-delay vs dist-kernel error-gap table per model | 🟢 | combine |
| T27 | Causal vs full LEMO_PC equivariance comparison (if phase_d ran) | 🟡 | deferred |

### "Audit and honesty" tables

| # | Table | Status | Source |
|---|-------|--------|--------|
| T28 | Known limitations table (where LEMO loses, why) | 🟢 | hand-curate (note: lowres regime) |
| T29 | Benchmark suitability table (APEBench/PDEBench vs DDE-PDE) | 🟢 | hand-curate |
| T30 | Architecture-to-task fit table | 🟢 | hand-curate |
| T31 | Negative results table | 🟢 | hand-curate |
| T32 | Data-generation audit table | ✅ | `gen_dde_pde_data.py` + manifests |
| T33 | Per-benchmark metadata table (train/val/test, resolution, τ-laws, regimes) | ✅ | manifests |
| T34 | Phase completion table (which experiments are 1-seed / 3-seed / regime-complete) | 🟢 | hand-curate |
| T35 | Reproducibility table (config, seeds, output-dir, ckpt names, JSON schema) | 🟢 | hand-curate |
| T36 | Training stability table (divergence count, NaN count) | 🟡 | partial; from logs |
| T37 | Formalization status table | 🟢 | Lean repo |
| T38 | 1D FNO+naive anchor table | ✅ | local |

---

## Pre-paper data captures (must run BEFORE pod teardown)

All run on existing checkpoints — no retraining required.  Wired into
`scripts/followup_sweep.sh phase_capture` (chained after Phase E auto-launch).

1. **Per-frame relL2 + naive-copy baseline** for every cell in
   `dist_kernel_v2_p{1,2}`, `layer5_final_sweep_p{1,2}`, `layer4_audit`,
   `followup_a_unet_w64`.  Output: `per_frame.json` per cell.
2. **Trajectory predictions for visualization** — 4 sample (input, target,
   pred) tuples per cell.  Output: `viz_samples.npz` per cell.
3. **Learned kernel snapshots** — `weights_time` (Causal LEMO if phase_d ran),
   `weights` (spectral lag), FiLM γ/β.  Output: `kernel_snapshot.npz` per LEMO cell.
4. **Per-sample residuals** — for hardest-decile + correlation analysis.
   Output: `residuals.npz` per cell.
5. **E_orbit metric** — Phase E2 (`eval_equivariance.py`).
6. **OOD lag-shift** — Phase E1 (`eval_lag_shift_ood.py`).
7. **LDS per family** — rerun `scripts/lds_sweep.py` adapted to 2D.
8. **Cyclic-pad jump statistic per dataset** — compute on raw shards.

## Pod-to-laptop bundle

`scripts/bundle_pods_for_download.sh` tarballs:
- `outputs/{dist_kernel_v2_p*, layer5_final_sweep_p*, layer4_audit,
  sweep_lemo_scale, sweep_apebench, sweep_apebench_residual_clean,
  ab_residual, eq_orbit, lag_shift_ood, followup_*}/`
- `data_dde_pde/*/manifest.json` + `data_apebench/*/manifest.json`
- `reports/`

Excludes raw shards (regenerable from `gen_dde_pde_data.py`).
`--no-ckpts` flag drops `best_model.pt` files for ~70% size reduction.

Estimated bundle: ~25–35 GB total across both pods (~7 GB with --no-ckpts).

```
bash scripts/bundle_pods_for_download.sh
```

---

## Cut-decision rule

When everything is generated, decide cuts using this rule:
- **Main paper:** figures whose *absence* would weaken a positive claim.
- **Appendix:** figures that strengthen credibility but aren't load-bearing.
- **Audit/debug bundle:** everything else, posted as supplementary on
  GitHub/OSF.  Reviewers can audit anything they want without bloating the PDF.

The 2D dist-kernel + single-delay + APEBench arc is more than enough to
sustain a paper without the 1D theorem track.  ADVICE.txt's "freeze around
strongest mature story" advice still applies — just ported to 2D.
