# Sensitivity sweeps — design doc (Round 3, fixable issue [3])

This document specifies four modest sensitivity sweeps requested by the
NeurIPS reviewer panel under `REVIEW_PUNCH_LIST.md` "FIXABLE ISSUES [3]":

> Run modest sensitivity sweeps (lag grid size, number of modes, FiLM rank,
> history length; β in the weighted norm).

We bundle "lag grid size" and "history length" into a single sweep
(`n_hist` controls both, since the lag axis spans the history window) — see
sweep S1 below. The four sweeps are:

| #  | axis swept                  | grid                        | scope                                      | status         |
|----|-----------------------------|-----------------------------|--------------------------------------------|----------------|
| S1 | lag-grid / history length   | `n_hist ∈ {16,32,64,128}`   | LEMO_PC, dist_exp, clean, 3 seeds          | in flight (Pod 4 Phase B)|
| S2 | spectral spatial modes      | `spatial_modes ∈ {4,8,12,16,24}` | LEMO_PC + FNO, 5 fams, clean, 3 seeds | new launch (`launch_sensitivity_modes.sh`) |
| S3 | FiLM rank (`film_hidden`)   | `{16,32,64,128}`            | LEMO_PC, 5 fams, clean, 3 seeds            | new launch (`launch_sensitivity_film_rank.sh`) |
| S4 | β in weighted norm          | `{0.5, 0.8, 0.95, 0.99}`    | post-hoc on σ-sweep cells                  | analysis only (no retrain) |

All headline LEMO_PC defaults (so a sweep cell that hits the headline grid
should reproduce the headline number to within seed noise): `width=64`,
`n_layers=3`, `lag_modes=24`, `spatial_modes=12`, `film_hidden=64`,
`residual_anchor=True`, `batch=4`, `epochs=200`, cosine LR, Adam, lr=1e-3.

---

## S1 — Lag-grid size / history length (Pod 4 Phase B, in flight)

**Question.** How does test rel-L^2 vary as the lag-grid resolution
(`n_hist`) is changed? The cyclic-FFT lag layer's effective frequency
resolution is $1/n_{\text{hist}}$; the model can in principle exploit
finer-grained delay structure as `n_hist` grows, at the cost of more compute
and more wrap-around contamination near the boundary.

**Sweep grid.** `n_hist ∈ {16, 32, 64, 128}`, with `n_out` matched to
`n_hist` (so the lag axis and the prediction horizon both scale).

- Family: `dist_exp_rd_2d` (single representative; the LEMO_PC headline gain
  is uniform across the five distributed-kernel families to within
  $g\approx 5$, so a single family suffices for the sensitivity reading).
- Regime: `clean` (regime is orthogonal to lag-grid; verified by the
  headline 3-regime equivalence).
- Seeds: `42, 123, 456` (same as headline).
- Models: `lemo_pc_nd` only (the question is about LEMO's lag layer; the
  baselines either ignore lag (Markov) or treat it as a 3D spectral axis,
  which has its own different modes-vs-error story).
- Hyperparams: headline defaults except `n_hist` and `n_out` change;
  `lag_modes = min(24, n_hist // 2)` to respect Nyquist on small grids.

**Cell count.** $4 \times 1 \times 1 \times 3 = 12$ cells.

**Status.** This sweep is **already running on Pod 4 (Phase B)**. The
launcher reuses the existing data-gen pipeline (`gen_dde_pde_data.py`
`--n_hist <X>`) per cell, then trains via
`train_apebench_smoke.py`. Output land at
`outputs/sensitivity_lag_grid/n_hist_{N}/...`.

**Pre-registered prediction.** Test rel-L^2 should *decrease* monotonically
in `n_hist` from 16 to 64, then plateau or slightly increase at 128 due to
spectral over-fitting on the longer cyclic axis. The covering-radius lower
bound (\Cref{cor:augmentation-lower-bound}) does not directly apply here
(this is grid resolution, not augmentation), but a similar tightening
should be visible.

---

## S2 — Number of spectral modes (`spatial_modes`)

**Question.** How does the spectral truncation along the **spatial** axes
affect LEMO_PC accuracy? (We separately sweep `lag_modes` in the deferred
`run_deferred_sweeps.sh phase_3` lag-modes sweep — that is a *different*
truncation along the lag axis.)

**Sweep grid.** `spatial_modes ∈ {4, 8, 12, 16, 24}`. The
`train_apebench_smoke.py` already passes `min(spatial_modes, S//2)` so 24 is
clipped to 16 on a $64\times 64$ grid (Nyquist), which is fine — gives us
five distinct cells {4, 8, 12, 16, 16-clipped} with the headline at 12.

- Families: all five distributed-kernel families
  (`dist_{exp,gaussian,gamma,uniform,powerlaw}_rd_2d`) so we can read off
  whether spectral truncation interacts with the kernel family.
- Regime: `clean` only.
- Seeds: `42, 123, 456`.
- Models: `lemo_pc_nd` and `fno_nd` (so we can check whether the FNO
  baseline has a different optimal truncation, which would confirm the
  fairness of the headline `spatial_modes=12` choice).
- Hyperparams: headline defaults; only `spatial_modes` varies.

**Cell count.** $5 \text{ modes} \times 5 \text{ fams} \times 2 \text{ models} \times 3 \text{ seeds} = 150$ cells.

**Compute estimate.** Per-cell wall: ~5 min on H100 (matches reproduce_headline.sh
`train_one` 50 min for 200 epochs — but at width=64, n_hist=64 batch=8,
deferred-sweep workers run ~5 min/cell). $150 / 24 = 6.25$ groups
$\times 5 \min \approx 30 \min$ wall on an 8×H100 pod (24 workers, 3
cells/GPU per the standard config).

**Pre-registered prediction.** Both models should plateau at
`spatial_modes ≥ 8`. If LEMO_PC plateaus *below* the FNO plateau (i.e.,
LEMO_PC at 4 modes already beats FNO at 16 modes), that's mechanism evidence
that the lag-equivariant inductive bias does the work and the spatial
spectral capacity is cheap.

**Launcher.** `launch_sensitivity_modes.sh`.

---

## S3 — FiLM rank (`film_hidden`)

**Question.** The FiLM modulator (`FiLMLagSpectralND.film_net`) is a
2-layer MLP $\mathrm{params\_dim} \to \mathrm{film\_hidden} \to
2 \cdot \mathrm{out\_channels} \cdot \mathrm{lag\_modes}$ with
`film_hidden=64` in the headline. The reviewer panel asked whether the
result depends on this hidden width (a proxy for "FiLM rank"). The 1st
linear has shape $(\mathrm{params\_dim} \times \mathrm{film\_hidden})$ and
the 2nd linear has shape $(\mathrm{film\_hidden} \times 2 \cdot
\mathrm{out\_channels} \cdot \mathrm{lag\_modes})$; small `film_hidden`
imposes a low-rank bottleneck on the conditioning signal.

**Sweep grid.** `film_hidden ∈ {16, 32, 64, 128}`.

- Families: all five distributed-kernel families.
- Regime: `clean` only.
- Seeds: `42, 123, 456`.
- Model: `lemo_pc_nd` only (the FiLM head only exists in LEMO_PC).
- Hyperparams: headline defaults; only `film_hidden` varies.

**Cell count.** $4 \times 5 \times 1 \times 3 = 60$ cells.

**Compute estimate.** $60 / 24 = 2.5$ groups $\times 5 \min \approx 13 \min$
wall on an 8×H100 pod with 24 workers.

**Pre-registered prediction.** No significant change in test rel-L^2 across
`film_hidden ∈ {32, 64, 128}`; modest degradation at 16 if the parameter
space is too narrow to encode the per-family kernel. This would confirm
that the headline `film_hidden=64` is over-parameterised but not
load-bearing — the gain comes from the equivariant spectral lag kernel
itself, not from FiLM rank.

**Implementation notes.** The trainer (`scripts/train_apebench_smoke.py`)
does *not* currently expose `--film_hidden` as a CLI argument; the
constructor reads `model_cfg.get("film_hidden", 64)` in
`src/models/lemo_pc_nd.py::create_lemo_pc_nd`. The launcher
(`launch_sensitivity_film_rank.sh`) sets `film_hidden` via a one-line patch
to the trainer that adds the argparse arg and threads it into
`config["model"]["film_hidden"]`. The patch is reversible (touches only the
trainer's CLI surface, not the model code) and is idempotent — running the
launcher twice is safe.

**Launcher.** `launch_sensitivity_film_rank.sh`.

---

## S4 — β-rate in the weighted norm (post-hoc analysis)

**Question.** Cor.~\ref{cor:lemo-sigma} (architectural contraction) bounds
the rollout norm of $\mathrm{LEMO}_\sigma$ in the $\beta$-weighted history
norm $\|H\|_\beta = \sum_{j=0}^{n-1} \beta^j \|h_{n-1-j}\|$ for a decay rate
$\beta \in (0,1)$. The certified contraction constant is
$\sigma \cdot \beta^{-1}$, which is $<1$ iff $\sigma < \beta$. The
reviewer asked us to sweep $\beta$ to characterise how the weighted-norm
choice affects the certified frontier.

**Sweep grid.** $\beta \in \{0.5, 0.8, 0.95, 0.99\}$.

- This is a **post-hoc analysis** on the existing σ-sweep checkpoints
  produced by `run_deferred_sweeps.sh phase_2` (currently running on the
  Caltech HPC cluster). For each $(\sigma, \beta, \text{family},
  \text{seed})$ we measure:
    - the per-mode operator norm $\max_m \|K[:,:,m]\|_{\mathrm{op}}$
      (which is invariant under $\beta$ — it's a property of the kernel)
    - the **measured** rollout-norm growth in the $\beta$-weighted norm at
      each rollout step; this *does* depend on $\beta$
    - the certified bound $\sigma \cdot \beta^{-1}$ and whether the
      measured growth respects it.
- The σ-sweep is already running and supplies all the checkpoints we
  need; no retraining is required.

**Cell count.** Post-hoc: 4 betas $\times$ (existing 60 σ-cells) = 240
analysis points, but each is a forward-pass evaluation, not a training
run, so the marginal compute is negligible (<1 GPU-hour total for all 240).

**Pre-registered prediction.** For each $\sigma$ value, the certified bound
$\sigma\beta^{-1}$ is binding iff $\beta < \sigma$. When $\beta = \sigma$
the bound is exactly $1$ (marginally contractive); when $\beta > \sigma$
the bound is $<1$ (strictly contractive). The measured rollout-norm growth
should respect the certified bound at all $(\sigma, \beta)$, with slack
that grows as $\beta - \sigma$ grows (the bound is conservative for
non-adversarial inputs).

**Output.** A 2-D grid figure (rows = $\sigma$, columns = $\beta$) showing
measured-vs-certified rollout-norm ratio, plus a 1-line certificate-table
in the paper appendix. Implemented in `scripts/analysis_beta_post_hoc.py`
(to be added once the σ-sweep returns).

---

## Total compute budget

| sweep | cells | wall-clock (8×H100, 24 workers) | status |
|-------|-------|----------------------------------|--------|
| S1 lag-grid    | 12  | ~1h (data gen + train)         | in flight (Pod 4) |
| S2 modes       | 150 | ~30 min                         | new |
| S3 FiLM rank   | 60  | ~15 min                         | new |
| S4 β post-hoc  | 240 evals | <1h forward-pass        | post σ-sweep |
| **total new**  | **210 train + 240 evals** | **~1h GPU-pod time** | |

This fits comfortably inside the 15h GPU budget rule
(`feedback_no_impulsive_kills_15h_budget.md`). Smoke test first
(per `feedback_smoke_test_first.md`): one cell of S2 (LEMO_PC,
dist_exp_rd_2d, clean, seed 42, spatial_modes=12) should reproduce the
headline rel-L^2 ~0.012 and per-epoch wall <50s before launching the full
S2/S3 sweep.
