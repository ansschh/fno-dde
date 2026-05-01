# BUGS AND GOTCHAS — LEMO Project

A consolidated record of every bug, gotcha, infrastructure failure, methodology issue, and performance regression encountered during the LEMO research saga (R2.26 → R2.28b). For each entry: what it was, how it manifested, root cause (if known), fix applied (or workaround), file/line where the fix lives, and lesson learned.

---

## SECTION 1 — Code Bugs

### 1.1 `--model` choices list missing `causal_lemo_pc_nd`
- **What:** The smoke-test harness rejected the new `causal_lemo_pc_nd` model name as an unknown choice.
- **How it manifested:** `scripts/train_apebench_smoke.py` argparse exited non-zero before any training started; smoke-test launches for the causal variant failed instantly.
- **Root cause:** When `causal_lemo_pc_nd` was added to `src/train/build_model.py` dispatcher in R2.27, the corresponding `--model` choices enum in `train_apebench_smoke.py` was not updated. Argparse `choices=` enforcement caught it.
- **Fix:** Added `causal_lemo_pc_nd` to the `--model` choices list.
- **File:** `scripts/train_apebench_smoke.py` (R2.28 fix, permanent in repo)
- **Lesson:** When registering a new model in the dispatcher, grep ALL scripts for the existing model-name set and add the new entry everywhere. A single source-of-truth model registry would prevent this entirely.

### 1.2 No `--sigma` flag in trainer for noisy-regime override
- **What:** Trainer had no CLI knob to override the noise level used to perturb data into the "noisy" regime; it always fell through to dataset config.
- **How it manifested:** σ-stability sweep (5 fams × 4 σ levels × 3 seeds = 60 cells) couldn't sweep across noise levels — every cell trained at the dataset-config default σ.
- **Root cause:** Original trainer signature predates per-cell σ override. The "noisy" regime was hardcoded in the dataset/regime config, not threaded through the CLI.
- **Fix:** Added `--sigma` flag with default `None` (falls through to dataset config when unset, overrides when provided).
- **File:** `src/train/train_dde.py` (R2.28 fix)
- **Lesson:** Any axis you intend to sweep over MUST be a CLI flag. Even if it looks like a dataset-side concern, a sweep dispatcher cannot vary it without an explicit hook.

### 1.3 Sweep dispatcher missing `--sigma` plumbing
- **What:** Even after the trainer accepted `--sigma`, the sweep dispatcher didn't thread it from sweep config to subprocess args.
- **How it manifested:** Sweep config listed `sigma: [0.1, 0.3, 0.5, 0.7]` but every subprocess launched with default σ — silent no-op.
- **Root cause:** Two-layer plumbing oversight: when 1.2 was fixed, the dispatcher (which constructs the `python train_dde.py …` argv) wasn't updated to include `--sigma <value>` from the cell config.
- **Fix:** Added explicit `--sigma` thread-through in dispatcher's argv construction.
- **File:** `scripts/run_dde_pde_sweep.py` (R2.28 fix)
- **Lesson:** End-to-end smoke a single cell of any new sweep axis BEFORE launching the full sweep. A 5-minute single-cell test would have caught this immediately.

### 1.4 Naive-baseline shape mismatch in `capture_dde_pde_metrics.py`
- **What:** When computing the mean-baseline relL2 reference, the broadcast was on the wrong dim for dist-kernel benchmarks where `lag_axis` size ≠ `time_axis` size.
- **How it manifested:** Naive-baseline numbers were nonsense (sometimes < the model's own relL2, breaking the "relative-to-baseline" interpretation); broadcast shape errors in numpy raised `ValueError` on some families.
- **Root cause:** Original metric capture assumed `lag_axis == time_axis` (square layout from the original 1D DDE benchmarks). Dist-kernel 2D benchmarks have different `lag_axis` and `time_axis` sizes; the implicit broadcast aligned on the wrong dim.
- **Fix:** Added explicit `axis=` reduction in the mean-baseline computation; checked shapes before broadcast.
- **File:** `scripts/capture_dde_pde_metrics.py` (R2.28 fix)
- **Lesson:** Never rely on numpy implicit broadcast for cross-dataset code. Make every reduction axis explicit; assert shapes at function entry.

### 1.5 OOD generator missing 3 of 5 dist-kernel families
- **What:** `gen_dde_pde_ood.py` only knew `dist_exp` and `dist_gaussian`; the other three (`dist_gamma`, `dist_uniform`, `dist_powerlaw`) were absent.
- **How it manifested:** OOD eval (Phase E1) ran on only 2 of 5 families — paper's lag-shift OOD claim would have been incomplete had this not been caught.
- **Root cause:** Generator was bootstrapped from a 2-family prototype and the family list wasn't updated when 3 more dist-kernel benchmarks were added.
- **Fix:** Added kernel-family branches for `dist_gamma`, `dist_uniform`, `dist_powerlaw` mirroring the existing two.
- **File:** `scripts/gen_dde_pde_ood.py` (R2.28 fix)
- **Lesson:** When the benchmark family count grows, audit ALL downstream scripts (gen, eval, viz, OOD, post-hoc) for hardcoded family lists. A `KERNEL_FAMILIES` constant in a shared config module would centralize this.

---

## SECTION 2 — Infrastructure Gotchas

### 2.1 H200 pod preempted mid-Phase-1 (no recovery)
- **What:** A RunPod H200 instance (38.80.152.148) was preempted in the middle of the R2.28 Phase-1 sweep. No checkpoint recovery; no usable data salvaged from that pod.
- **How it manifested:** SSH connection died; `nvidia-smi` over reconnection attempts returned host-unreachable; 100% of in-flight cells lost.
- **Root cause:** Spot/preemptible instance class. RunPod can reclaim H200 pods at any time without notice.
- **Fix:** None (no fix possible mid-flight). Workaround: switched to non-H200 pods (replacement H100s) for retries.
- **Lesson:** Treat any cluster pod as ephemeral. (a) Checkpoint EVERY epoch, not just at the end. (b) Stream partial results to local laptop continuously (not as a final tar). (c) Never schedule a sweep on a preemptible instance class without an automatic resume-from-ckpt path.

### 2.2 Replacement pod 1 died with ~2h budget remaining
- **What:** Replacement pod 1 (103.207.149.167, 7×H100, PyTorch 2.8 / CUDA 12.8) died unexpectedly during R2.28 follow-up.
- **How it manifested:** SSH unreachable; only partial cells completed (3 of 12 σ=0.7 cells; 5 of 45 Causal LEMO cells).
- **Root cause:** Unknown — possibly host-level issue, network blip, or RunPod-side restart. No diagnostic logs preserved.
- **Fix/Workaround:** Salvaged the 544 MB of partial results into `pod1_v2_bundle`; partial data documented as "salvage only, not enough for paper claims."
- **Lesson:** Continuous streaming of bundles to the laptop (rsync every N minutes) is mandatory. Do NOT rely on a single end-of-sweep tar; pod death between sweep finish and tar download = total loss.

### 2.3 Replacement pod 2 died early — zero usable data
- **What:** Replacement pod 2 (103.207.149.41, 8×H100, PyTorch 2.4.1 / CUDA 12.4 — identical stack to R2.27) died before any cell completed.
- **How it manifested:** Pod ssh connection lost; no completed cells at all.
- **Root cause:** Unknown — same cluster-side instability.
- **Fix/Workaround:** Switched to a third replacement pod 3 (103.207.149.143).
- **Lesson:** Do NOT commit a 4-hour sweep to a pod you haven't run a 5-minute single-cell test on. Always verify the pod can finish at least one cell end-to-end before launching the full matrix.

### 2.4 `pkill -9 -f <pattern>` matches the bash session running it
- **What:** A `pkill -9 -f <pattern>` invocation can match the bash process executing the pkill itself if `<pattern>` appears in its own argv.
- **How it manifested:** Self-kills mid-session — script terminates before the kill loop finishes; subsequent commands in the same shell session don't run.
- **Root cause:** `pkill -f` matches against the FULL command line of every process, including the shell that's running pkill. If the pattern is a substring of the script name or args, the shell self-kills.
- **Fix/Workaround:** Use specific PIDs (`kill <pid>`) instead of pattern-based pkill. If pattern-based is required, use `pkill -9 -f <pattern> --inverse-of-self` style or filter `$$` out: `pkill -9 -f <pattern> $(pgrep -f <pattern> | grep -v $$)`.
- **Lesson:** Pattern-based kills are a footgun. Prefer recording PIDs at launch time and killing by PID. When you must use pattern, exclude `$$` and document it.

### 2.5 Network filesystem slowness on RunPod tenancy
- **What:** Some RunPod tenancy classes have notably slower NVMe / network FS than others.
- **How it manifested:** Datagen writes that took 2 minutes on R2.27 pods took 8+ minutes on some replacement pods. Suspected contributor to the 4× per-epoch regression (see §4.1).
- **Root cause:** RunPod tenancy class differences (shared vs dedicated NVMe, host neighbor count). Not directly observable from inside the pod.
- **Fix/Workaround:** None confirmed. Mitigation: re-tar from scratch on each new pod rather than rsync; benchmark NVMe write throughput at pod-up time.
- **Lesson:** Pod selection matters. When two pods nominally have the same GPUs and same software stack, IO and noisy-neighbor differences can still produce 2-4× wall-clock differences. Always benchmark.

### 2.6 Python deps frequently missing on fresh pods (matplotlib, h5py, deepxde)
- **What:** Common deps not in default RunPod image; cells fail silently with import errors that don't surface in dispatcher summary.
- **How it manifested:** Cells finished with rc != 0 in well under expected wall time; logs showed `ModuleNotFoundError: No module named 'h5py'` etc.
- **Root cause:** RunPod base image is minimal. Each new pod needs explicit dep install.
- **Fix:** `pip install matplotlib h5py deepxde --break-system-packages` proactively in the deploy script.
- **File:** `scripts/oneshot_deploy.sh` (proactive dep install pattern)
- **Lesson:** Always run a dep-check + install step BEFORE the first sweep on any fresh pod. Log to `feedback_no_impulsive_kills_15h_budget.md` reflects this is a repeat offense.

---

## SECTION 3 — Methodology Issues

### 3.1 Per-frame relL2 broken when the target is small
- **What:** Per-frame relative L2 metric `||pred - gt|| / ||gt||` blows up or becomes meaningless when the target frame norm is near zero (e.g., decay benchmarks, frames near steady state).
- **How it manifested:** A few frames in dist-kernel benchmarks had near-zero `||gt||` (especially in low-amplitude regimes); per-frame relL2 spiked to absurd values, dragging the mean up; "uniform 0.92" looking results were sometimes just one bad frame.
- **Root cause:** The relative-error denominator vanishes; division by ~0 explodes.
- **Fix/Workaround:** Aggregate over the full trajectory (||pred - gt||_F over T,X,Y / ||gt||_F over T,X,Y) instead of per-frame; for per-frame breakdown, use absolute MSE alongside relL2 and inspect the denominator.
- **Lesson:** Always include a non-relative metric (MSE, MAE) alongside relative metrics. Audit raw `test_results.json` for any individual sample/frame where `|gt|` is near zero before trusting aggregate relative numbers.

### 3.2 APEBench / PDEBench mis-fit for LEMO's cyclic-FFT lag axis (R2.26 pivot)
- **What:** APEBench/PDEBench benchmarks impose a cyclic-buffer structure on the lag axis that's not actually present in the data — LEMO's structural prior assumes the cyclic structure exists in the data, not just in the input layout.
- **How it manifested:** R2.26 sweep showed LEMO_PC ≈ UNet on these benchmarks; spectral lag conv didn't help.
- **Root cause:** APEBench data is non-DDE in nature; the "lag axis" is an artificial reshape, not a physical delay. UNet's local convolution handles a single artificial-lag context fine.
- **Fix/Workaround:** Pivot to native DDE-PDE benchmarks (Mackey-Glass, Wright, Hutchinson, delay-Burgers, Kuramoto, then 5 dist-kernel families) where the cyclic-buffer structure exists in the underlying physics.
- **Lesson:** A method's structural prior must match the data's structural prior. If the prior is "lag axis is cyclic," only benchmarks with genuine cyclic delay dynamics will exercise the prior. Test on one canary benchmark before committing to a benchmark family.

### 3.3 Single-lag DDE-PDE benchmarks fail to exercise T1 either
- **What:** Even after pivoting to genuine DDE-PDEs (Mackey-Glass etc.), single-lag versions (one fixed τ) failed to differentiate LEMO_PC from UNet.
- **How it manifested:** R2.26 layer-5 sweep at v1 config (width=32, lag_modes=12, no residual_anchor) showed LEMO_PC ≈ UNet on single-lag benchmarks.
- **Root cause:** A single fixed lag is a local operation in the lag axis — UNet's local conv handles it without needing the spectral lag conv. The structural advantage of LEMO only shows when the kernel is global (distributed across all lags).
- **Fix/Workaround:** Built 5 distributed-kernel benchmarks (`dist_exp`, `dist_gaussian`, `dist_gamma`, `dist_uniform`, `dist_powerlaw`) where the kernel integrates over the full lag axis.
- **Lesson:** A method's headline claim ("better at delay equations") is too coarse. The right benchmark is the one that requires the structural feature unique to your method — global kernel integration, not local single-lag lookup. Pick benchmarks adversarially.

### 3.4 v1-LEMO_PC underconfigured to win on dist-kernel
- **What:** R2.27 audit at v1 config (width=32, lag_modes=12, no residual_anchor) showed UNet still beating LEMO_PC by 9-13% on dist-kernel benchmarks.
- **How it manifested:** Dist-kernel was supposed to be where LEMO wins; v1 numbers said otherwise.
- **Root cause:** v1 lag-spectral capacity was insufficient for the dist-kernel families' kernel complexity; residual_anchor was needed to fix cyclicity numerical drift.
- **Fix:** v2 config delta — `width: 32 → 64`, `lag_modes: 12 → 24`, `residual_anchor: False → True`. Headline result: LEMO_PC v2 beats UNet by 34% (clean) and 33% (noisy) on `dist_exp_rd_2d`.
- **Lesson:** Capacity tuning matters. A 9-13% deficit at v1 became a 34% advantage at v2 with a 2× width and 2× spectral resolution change. Don't conclude "method doesn't work" until you've tuned capacity to a reasonable point.

### 3.5 No-unfair-favors discipline (residual_anchor must apply to all baselines)
- **What:** Any intervention that helps LEMO must be applied identically to all baselines.
- **How it manifested:** v2 config turned on `residual_anchor=True` for LEMO and ALL baselines (FNO, MarkovFNO, WindFNO, UNet) — not just LEMO.
- **Root cause:** Reviewer-credibility risk. Residual_anchor is a generic spectral fix; if applied only to LEMO, the comparison is rigged.
- **Fix:** Trainer flags identical for LEMO and baselines (same epochs=200, batch_size=4, data, residual_anchor). Width remains the only differentiator (LEMO 64, baselines 32 — addressed by Phase A param-matched UNet@64).
- **File:** Sweep configs in `outputs/dist_kernel_v2_p1/` and `outputs/dist_kernel_v2_p2/`; Phase A in `scripts/followup_sweep.sh`
- **Lesson:** Codified in `feedback_no_unfair_lemo_favors.md`. Honest framing matters more than maximizing apparent win. Residual helped FNO_ND (17.8%) MORE than LEMO_PC_ND (7.6%) on Kolmogorov 2D — so it's partly a "general spectral-method fix"; document this honestly.

### 3.6 Param-match audit (LEMO 64 vs baseline 32) — Phase A
- **What:** Headline win was at LEMO width=64 vs baselines width=32. Reviewer rebuttal: "did LEMO win because of more params?"
- **How it manifested:** Reviewer-rebuttal risk preempted; Phase A queued to run UNet at width=64 (param-matched).
- **Root cause:** Width difference left an open hypothesis space.
- **Fix:** Phase A — UNet width=64, 45 cells, ~2h on pod 2. Closes the param-count rebuttal.
- **File:** `scripts/followup_sweep.sh` (Phase A section)
- **Lesson:** Anticipate the obvious rebuttal and run the param-matched control before paper submission. Reviewers WILL ask.

### 3.7 Causal-by-construction (T1') costs capacity
- **What:** Forcing FiLM modulation only on the first `lag_modes` coefficients (causal mode) reduces effective spectral capacity by half vs the non-causal block.
- **How it manifested:** Causal LEMO partial run on R2.28 replacement pod 1: per-epoch ~1.4× standard LEMO_PC (extra L-length FFT path); on completed cells (dist_exp clean, 2 seeds), test relL2 was ~12% worse than non-causal LEMO_PC v2.
- **Root cause:** Causal FIR right-pad before FFT adds a non-amortizable L-length FFT path. The data has bidirectional influence (kernels integrate over the full lag axis), so causal-by-construction is a bad inductive bias here.
- **Fix/Workaround:** Causal LEMO stays in repo as a documented variant for reviewer rebuttal but is dropped from headline figures. Soft-T1 non-causal block + residual_anchor remains the default.
- **File:** `src/models/lemo_pc_nd.py` (`causal: bool = False` param), `src/train/build_model.py` (`causal_lemo_pc_nd` dispatcher), `scripts/test_causal_lemo.py` (smoke test)
- **Lesson:** Hard architectural constraints cost capacity. If T1' is satisfied empirically by soft constraint + residual_anchor, don't pay for hard T1'. Decide constraint vs prior empirically, not dogmatically.

---

## SECTION 4 — Performance Regressions

### 4.1 The 4× per-epoch regression mystery (UNRESOLVED)
- **What:** Original R2.27 pods ran ~20s/epoch on dist-kernel cells. R2.28 replacement pods benchmarked at ~80-83s/epoch — a 4× regression.
- **How it manifested:**
  - R2.27 pods (PyTorch 2.4.1 / CUDA 12.4 / 8×H100): ~20s/epoch.
  - Replacement pod 1 (PyTorch 2.8 / CUDA 12.8 / 7×H100): ~80s/epoch.
  - Replacement pod 2 (PyTorch 2.4.1 / CUDA 12.4 / 8×H100 — IDENTICAL stack to R2.27): ~83s/epoch.
- **Root cause:** UNKNOWN. The PyTorch/CUDA-version hypothesis was killed by pod 2 (identical stack, same regression). Suspected causes (unconfirmed under time pressure):
  - Noisy-neighbor contention on shared hosts.
  - Different NVMe throughput on the new RunPod tenancy class.
  - Thermal throttling.
  - Network FS slowness (see §2.5).
- **Fix/Workaround:** None — root cause not identified. Mitigation: pod 3 (103.207.149.143, PyTorch 2.4.1 / CUDA 12.4) selected to match R2.27 stack; epoch-time benchmark to be run before sweep launch.
- **Lesson:** Open question for next session. Pod-stack reproducibility is a paper-risk: epoch-time is not solely a function of code + library stack; host-level factors matter. Going forward: benchmark a single cell BEFORE launching the full sweep on any new pod. If epoch-time differs by >2× from prior pods, switch pods.

### 4.2 Causal LEMO ~1.4× per-epoch slower than non-causal LEMO_PC
- **What:** Causal mode adds an L-length FFT path that's not amortizable.
- **How it manifested:** R2.28 partial Causal LEMO run showed 1.4× wall-clock per epoch vs non-causal at the same width/modes.
- **Root cause:** Causal FIR right-pad before FFT yields a strictly-causal impulse response — but the right-pad means the FFT is over the full L (vs non-causal's `lag_modes`-truncated FFT).
- **Fix/Workaround:** None at the architectural level — this is a fundamental cost of causal-by-construction. Practical fix: drop Causal LEMO from headline; keep as rebuttal-only variant.
- **File:** `src/models/lemo_pc_nd.py` `FiLMLagSpectralND` causal branch.
- **Lesson:** Architectural costs accumulate. Always benchmark a new architectural variant on a single cell before scheduling 45 cells of it.

---

## SECTION 5 — Best Practices Learned

### 5.1 Smoke-test before sweep launch
- **Rule:** Run a single end-to-end cell (datagen → train 1 epoch → metric capture → eval) BEFORE launching the full sweep matrix. If any step fails, fix it locally before burning 24 workers × N hours.
- **Source:** All 5 R2.28 bugs (§1.1-1.5) would have been caught by a single pre-sweep smoke run.
- **Codified:** `scripts/oneshot_deploy.sh` — datagen → smoke test → sweep, in that order.

### 5.2 Re-tar from scratch on each new pod
- **Rule:** Don't rsync data between pods or rely on cross-pod NFS. Re-generate or re-tar data directly on the target pod.
- **Source:** §2.5 (network FS slowness on some tenancies); cross-pod rsync was unreliable.
- **Codified:** Deploy script always runs datagen locally on the pod.

### 5.3 Stream results continuously, not as end-of-sweep tar
- **Rule:** rsync partial results from pod → laptop every N minutes. If pod dies between sweep finish and final tar, you lose everything.
- **Source:** §2.1 (H200 preempt — total loss); §2.2 (pod 1 death — partial salvage of 544 MB).
- **Codified:** Streaming rsync watcher in `scripts/followup_sweep.sh` (auto_pod1, auto_pod2 watchers).

### 5.4 Maximize GPU usage — 24 workers / 8 GPUs minimum
- **Rule:** Always dispatch at 3+ cells/GPU. 1-cell-per-GPU underutilizes H100s by 60-70%.
- **Source:** `feedback_max_gpu_usage.md`. R2.27 used 24 workers/pod with `OMP_NUM_THREADS=2`.
- **Verify after launch:** `nvidia-smi` should show all 8 GPUs at 30-100% util AND 18-30GB memory.

### 5.5 Audit before answering — verify numbers, dynamics, data, code
- **Rule:** Before reporting any result, verify (a) numbers in raw `test_results.json`, (b) training loss actually descended (read log, not just final metric), (c) data didn't silently fail (sample trajectory, NaN check, regime perturbation differs between clean/lowres/noisy), (d) comparison is fair (matched epochs, params, data, seeds).
- **Source:** `feedback_audit_before_answer.md`. Past saves:
  - `markov_fno_nd` "uniform 0.92" was actually severe underparameterization (42k params).
  - U-Net beats LEMO on burgers_3d at matched 5.5M-vs-5.7M params.
  - Residual_anchor "always helps" was wrong — HURTS 3.4× on burgers_1d.

### 5.6 Extensive data + visualization audit BEFORE declaring data ready
- **Rule:** 16-32 random sample renders + diagnostic histograms + GIFs + pred-vs-GT side-by-side. NOT 1-3 cherry-picked trajectories. Visualize at the actual training time horizon, not at long-term saturation.
- **Source:** `feedback_extensive_data_audit.md`. Past failures:
  - Killed Layer 5 sweep based on misread audit PNGs at T=20 (saturation) when training was at T=1.27.
  - Declared APEBench 5x5 sweep "LEMO wins 5/5" without UNet baseline.
- **Tool:** `scripts/viz_bulk_audit.py` (16 samples × 6 frames + final_std + ||u(T)-u(0)||/||u(0)|| histograms).

### 5.7 No unfair LEMO favors — interventions apply to all baselines
- **Rule:** Any data-presentation, training-protocol, or hyperparameter change that helps LEMO must be applied identically to all baselines. Run A/B with the change applied to ALL baselines; report gain per baseline.
- **Source:** `feedback_no_unfair_lemo_favors.md`. User's exact words: "I do not want to give special favours to LEMO so that it performs well."
- **Codified:** v2 sweep applied `residual_anchor=True` to LEMO AND all baselines uniformly.

### 5.8 Never kill a sweep on misread — audit first
- **Rule:** A failed cell with rc!=0 is NOT a reason to kill — investigate the cell's log first. Always verify launch is healthy: `head -3 sweep.log` shows expected job count, `ps -ef | grep train | wc -l` matches `n_workers`, `nvidia-smi` shows expected utilization.
- **Source:** `feedback_no_impulsive_kills_15h_budget.md`. Past mistakes:
  - Killed Layer 5 sweep on misread audit PNGs (~30min recovery).
  - Killed Layer 4 8-worker sweep to relaunch with 24 workers (~10min).
- **Top priority rule:** "Speed is secondary to correctness. A 30-minute audit that prevents a 4-hour mistake is ALWAYS worth it."

### 5.9 Install deps proactively on every fresh pod
- **Rule:** `pip install matplotlib h5py deepxde --break-system-packages` on every new pod BEFORE the first sweep. Don't assume base image has them.
- **Source:** §2.6. Repeat offense — h5py and matplotlib silently missing in past sweeps.
- **Codified:** Top of `scripts/oneshot_deploy.sh`.

### 5.10 Centralize family/model lists — avoid hardcoded enums
- **Rule:** When the benchmark family count or model count grows, audit ALL downstream scripts (gen, eval, viz, OOD, post-hoc, smoke-test argparse) for hardcoded lists. Prefer a shared `KERNEL_FAMILIES` / `MODEL_REGISTRY` constant.
- **Source:** §1.1 (model choices missing causal_lemo_pc_nd), §1.5 (OOD generator missing 3 of 5 families).
- **TODO (deferred):** Refactor to a single registry module; both bugs were duplicate-info violations.

### 5.11 Make every sweep axis a CLI flag
- **Rule:** Any axis you intend to sweep over (σ, lag, width, etc.) MUST be a CLI flag on the trainer AND threaded through the sweep dispatcher.
- **Source:** §1.2, §1.3 (σ flag missing on both layers).
- **Lesson:** Two-layer plumbing — fix at trainer AND dispatcher in the same commit. End-to-end smoke a single cell with the new flag before launching the sweep.

### 5.12 Benchmark epoch-time before launching a multi-hour sweep on any new pod
- **Rule:** On any new pod, run a single cell first and verify epoch-time is within 2× of expectation. If not, switch pods (or diagnose noisy-neighbor / NVMe / thermal).
- **Source:** §4.1 (4× per-epoch regression burned ~70% of R2.28 budget on infrastructure rather than science).
- **Top finding:** Pod-stack reproducibility is a paper risk — epoch-time depends on host-level factors beyond library versions.

---

## Append-only log conventions

When adding a new entry:
1. Append to the appropriate section (1–5) at the bottom.
2. Use the existing template: What / How it manifested / Root cause / Fix / File / Lesson.
3. Cross-reference related entries with `§N.M`.
4. If the entry codifies a new feedback rule, also add the rule to `~/.claude/projects/A--dde-research/memory/feedback_<name>.md`.
