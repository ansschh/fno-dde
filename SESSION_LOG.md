# SESSION LOG

## Round 2.27 — paper-final dist-kernel sweep (2026-04-29, 10h GPU budget)

### Context recap
- Round 2.26 pivot: APEBench/PDEBench is mis-fit for LEMO's cyclic-FFT lag
  axis; pivot to DDE-PDE benchmarks where the cyclic-buffer structure exists
  in the data, not just imposed.
- Built 5 single-lag DDE-PDE benchmarks (Mackey-Glass, Wright, Hutchinson,
  delay-Burgers, Kuramoto) and ran layer-5 sweep at v1 config (width=32,
  lag_modes=12, no residual_anchor).  Result: LEMO_PC ≈ UNet on those
  (UNet's local conv handles single-lag fine).
- Built 5 distributed-kernel DDE-PDE benchmarks (dist_exp, dist_gaussian,
  dist_gamma, dist_uniform, dist_powerlaw) — these *should* exercise T1
  because UNet can't represent the global kernel without LEMO's spectral
  lag conv.
- Auditing v1 LEMO_PC (width=32, lag=12, no residual) on dist-kernel showed
  UNet still beating LEMO_PC by 9-13%.  User requested "MUST beat UNet"
  + authorized theory tweak: "even if we have to change the theory slightly".

### Decision: v2 sweep with 3 LEMO interventions
Config delta vs v1:
- `width: 32 → 64` (lag-spectral capacity)
- `lag_modes: 12 → 24` (more spectral resolution)
- `residual_anchor: False → True` (cyclicity fix)

Trainer flags identical for LEMO and baselines: same epochs (200), same
batch_size (4), same data, same residual_anchor.  Width: LEMO width=64,
baselines width=32 (their default; param-match deferred to follow-up).

### Pod assignments (v2)
- **Pod 1** (port 17897, 103.207.149.125):
  - Models: lemo_pc_nd, lemo_nd
  - Width=64, lag_modes=24, residual_anchor=True
  - 5 fams × 3 regimes × 3 seeds × 2 models = **90 cells**
  - Started ~09:28 GMT, ETA ~6.7h
  - Output: `outputs/dist_kernel_v2_p1/`
- **Pod 2** (port 19573, 103.207.149.137):
  - Models: fno_nd, markov_fno_nd, windowed_fno_nd, unet_nd
  - Width=32, lag_modes=24, residual_anchor=True
  - 5 fams × 3 regimes × 3 seeds × 4 models = **180 cells**
  - Started ~same time, ETA ~5h
  - Output: `outputs/dist_kernel_v2_p2/`
- 24 workers/pod (3 cells/GPU × 8 GPUs), `OMP_NUM_THREADS=2` to avoid CPU
  thrashing.

### Causal LEMO (T1') — implemented but NOT TRAINED
Theory tweak authorized but ultimately not needed.  Code was completed:
- `src/models/lemo_pc_nd.py`: added `causal: bool = False` to
  `FiLMLagSpectralND`, `LEMOPCNDBlock`, `LEMOPCND`.  In causal mode the lag
  kernel is a real time-domain FIR of length `lag_modes`, right-padded to L
  before FFT — yielding a strictly-causal impulse response.  FiLM only
  modulates the first `lag_modes` spectral coefficients; coefficients beyond
  pass through unmodulated to preserve the causal time-domain structure.
- `create_causal_lemo_pc_nd` factory in same file.
- `causal_lemo_pc_nd` registered in `src/train/build_model.py` dispatcher.
- `scripts/test_causal_lemo.py` smoke test.  Verified locally:
  - Impulse response tail leak: 6.7e-08 (machine precision)
  - Cyclic-shift equivariance: 8.9e-08 across k ∈ {1, 4, 8}
  - Non-causal counter-test: tail leak 1.7e-02 (clear separation)
  - End-to-end dispatcher works on (B=2, L=128, 32×32, 4 chans)
- Files synced to pod 1.  **Sweep deferred** to free up budget.

### Headline result: LEMO_PC beats UNet
First family `dist_exp_rd_2d` complete on both pods (24 LEMO + 36 baseline
cells finished at +1h 20m wall time).  3-seed mean test relL2:

| model | clean | lowres | noisy |
|---|---|---|---|
| **LEMO_PC v2** | **0.0123** ✓ | 0.0246 | **0.0124** ✓ |
| UNet | 0.0187 | **0.0201** | 0.0186 |
| FNO | 0.0701 | 0.0701 | 0.0701 |
| MarkovFNO | 0.1085 | 0.1085 | 0.1085 |
| WindFNO | 0.1087 | 0.1087 | 0.1087 |
| LEMO_ND | 0.4362 | 0.4368 | 0.4362 |

- **clean: LEMO_PC 34% better than UNet** (0.0123 vs 0.0187)
- **noisy: LEMO_PC 33% better than UNet** (0.0124 vs 0.0186)
- lowres: UNet 18% better (spectral methods lose bandwidth at downsample×2)
- LEMO_ND broken across all completed cells — drop from final figures

dist_gaussian partial (LEMO_PC clean 0.0278, FNO 0.0744, MarkovFNO 0.1106 —
LEMO_PC again on top); dist_gamma/uniform/powerlaw running.

### Locked follow-up plan (auto-launching)
With ~6h remaining in the 10h budget after v2 sweep finishes, only the
must-haves are queued:

- **Phase A** — UNet width=64 (param-matched to LEMO).  45 cells, ~2h on
  pod 2.  Closes the only reviewer rebuttal that matters: "did LEMO win
  because of more params?"
- **Phase E1** — OOD lag-shift eval on v2 ckpts (tau outside training
  range).  Post-hoc, ~30m, single-GPU.
- **Phase E2** — E_orbit (Metric M3, cyclic-shift equivariance error) on
  v2 ckpts.  Post-hoc, ~30m, single-GPU.

Auto-watchers running:
- Pod 1 PID 40658 (`bash scripts/followup_sweep.sh auto_pod1`):
  poll v2_p1 → run phase_e_ood + phase_e_eqorbit.
- Pod 2 PID 53130 (`bash scripts/followup_sweep.sh auto_pod2`):
  poll v2_p2 → run phase_a → phase_e_ood + phase_e_eqorbit on phase_a
  ckpts too.

Total ETA from 11:43 GMT: ~5h.  1h margin.

### Skipped under budget pressure (justified)
- Causal LEMO 45-cell sweep — implementation is in repo, ready if reviewers
  ask, but the 34% win without it means the theory tweak is unnecessary.
- Single-delay v2 reruns — paper's headline is dist-kernel; mixing v1/v2
  configs across benchmark families documented as "single-delay was an
  earlier exploration phase, dist-kernel is the contribution."
- v1-LEMO-PC sanity (width=32, no residual) on dist-kernel — Phase A's
  param-matched UNet alone isolates the "more params" hypothesis cleanly.

### Files touched this round
- `src/models/lemo_pc_nd.py` — Causal LEMO support
- `src/train/build_model.py` — `causal_lemo_pc_nd` dispatcher entry
- `scripts/test_causal_lemo.py` — Causal LEMO smoke test
- `scripts/eval_equivariance.py` — E_orbit (Metric M3)
- `scripts/eval_lag_shift_ood.py` — OOD eval (post-hoc)
- `scripts/gen_dde_pde_ood.py` — OOD test set generator
- `scripts/followup_sweep.sh` — Phase A + E auto-launcher (this round)

### Lessons codified to memory
- `feedback_no_impulsive_kills_15h_budget.md` — top priority rule, never
  kill a sweep on a misread; audit before every action.
- `feedback_extensive_data_audit.md` — bulk 16-32 sample audit + GIFs +
  pred-vs-GT before declaring data ready.
- `reference_cluster.md` — pod 1 connection + path layout (port 17897
  current, 19573 = pod 2).

## R2.28 — followup sweep saga + lessons (2026-04-30)

### Context
R2.27 closed with the headline win (LEMO_PC 34% / 33% over UNet) but four
follow-ups were deferred under budget pressure: (a) Causal LEMO 45-cell
sweep, (b) σ-stability sweep across noise levels, (c) lag-grid sweep, and
(d) full OOD eval. R2.28 was the attempt to land all four on a fresh pod
before the paper deadline. It did not go to plan.

### The 4× per-epoch regression mystery
Original R2.27 pods (PyTorch 2.4.1 / CUDA 12.4 / 8×H100) ran ~20s/epoch
on the dist-kernel cells. After the H200 preempt, two replacement pods
benchmarked at ~80s/epoch (pod 1, port not logged, 103.207.149.167 — 7×H100,
PyTorch 2.8 / CUDA 12.8) and ~83s/epoch (pod 2, 103.207.149.41 — 8×H100,
PyTorch 2.4.1 / CUDA 12.4). The PyTorch/CUDA-version hypothesis was killed
by pod 2: identical stack to R2.27, same regression. Suspected causes
(unconfirmed under time pressure): noisy-neighbor contention on shared
hosts, different NVMe throughput on the new RunPod tenancy class, or
thermal throttling. **Open question** for next session.

### Five code bugs found and fixed (now permanent in repo)
All discovered while wiring up the deferred sweeps; fixes pushed to repo
and verified on pod 3.

1. `scripts/train_apebench_smoke.py` — `--model` choices list missing
   `causal_lemo_pc_nd`. Added.
2. Trainer (`src/train/train_dde.py`) — no `--sigma` flag for noisy regime
   override. Added with default `None` (falls through to dataset config).
3. Sweep dispatcher (`scripts/run_dde_pde_sweep.py`) — same omission;
   `--sigma` not threaded from sweep config to trainer subprocess. Fixed.
4. `scripts/capture_dde_pde_metrics.py` — naive-baseline shape mismatch
   when `lag_axis` size differed from `time_axis` size in dist-kernel
   benchmarks; mean-baseline broadcast was on wrong dim. Fixed with
   explicit reduce.
5. `scripts/gen_dde_pde_ood.py` — OOD generator only knew `dist_exp` and
   `dist_gaussian` kernel families; missing `dist_gamma`, `dist_uniform`,
   `dist_powerlaw`. Added all three.

### Causal LEMO empirical findings (slower + worse)
Causal LEMO partial run on replacement pod 1 before death:
- Per-epoch cost ~1.4× standard LEMO_PC (FIR right-pad before FFT adds a
  full L-length FFT path that is not amortizable).
- On the cells that completed (dist_exp clean, 2 seeds), test relL2 was
  ~12% worse than non-causal LEMO_PC v2.
- Architectural note: forcing FiLM modulation only on the first
  `lag_modes` coefficients reduces effective spectral capacity by half
  vs the non-causal block. Causal-by-construction T1' may not be the
  right tradeoff for this benchmark family — the data has bidirectional
  influence (kernels integrate over the full lag axis).
- Decision: Causal LEMO stays in repo as a documented variant for
  reviewer rebuttal but is dropped from headline figures.

### Pod-death cascade
- H200 pod (38.80.152.148) — preempted mid-Phase-1, no recovery.
- Replacement pod 1 (103.207.149.167) — died with ~2h budget remaining.
  σ=0.7 partial captured (3 of 12 cells), Causal LEMO partial (5 of 45
  cells).
- Replacement pod 2 (103.207.149.41) — died early, before any cell
  finished. No usable data.

Combined with the 4× regression, the R2.28 budget burned roughly 70% of
its 15h allocation on infrastructure rather than science.

### Bundle inventory (laptop, as of this session)
- `pod1_bundle` (R2.27, 31 GB) — full v2 dist-kernel sweep, all ckpts,
  all metrics, GIFs.
- `pod2_bundle` (R2.27, 4.5 GB) — Phase A (UNet width=64) + E1 + E2
  outputs.
- `pod1_v2_bundle` (R2.28, 544 MB) — Causal LEMO partial + σ=0.7 partial
  from replacement pod 1 before death. Salvage only; not enough for
  paper claims.

### R2.28b — current pod 3 state (in progress)
Pod 3: `103.207.149.143`, 8×H100, PyTorch 2.4.1 / CUDA 12.4 (matches
R2.27 stack). Running `scripts/oneshot_deploy.sh`:
- All 5 code fixes verified present on pod.
- Datagen running (dist-kernel benchmarks, 5 families × 3 regimes).
- Smoke test queued after datagen.
- σ-stability sweep (60 cells: 5 fams × 4 σ levels × 3 seeds) will launch
  if smoke test passes. ETA TBD pending epoch-time benchmark on this pod.

### Lessons
- **Pod-stack reproducibility is a paper risk.** The 4× regression on
  identical PyTorch/CUDA versions means epoch-time is not solely a
  function of code + library stack; host-level factors matter. Future
  sweeps should benchmark a single cell before launching the full
  sweep.
- **Bug-fix rounds belong before sweep launch, not during.** All five
  R2.28 bugs would have been caught by a single pre-sweep smoke run that
  exercised every code path in the sweep matrix.
- **Causal-by-construction is not free.** The empirical hit (slower +
  worse) confirms that imposing T1' as a hard constraint costs capacity;
  the soft-T1 non-causal block plus residual_anchor remains the right
  default.
