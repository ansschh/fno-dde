# Final Research Plan — DDE-PDE Benchmark Suite + 5-Layer Audit + Complete Sweep

**Status:** plan-of-record for the FINAL phase of the LEMO research project.
After this sweep completes, we write the paper from the resulting numbers.

---

## Section 0 — Goal

Produce a reviewer-bulletproof empirical case for LEMO_PC_ND on
problems where its lag-equivariance inductive bias is the natural fit:
delay-differential equations coupled to spatial PDEs (DDE-PDE).

Reviewer attacks we anticipate and pre-empt:

| Attack | Defense |
|---|---|
| "Did you cherry-pick datasets?" | 6 benchmarks spanning biology / chemistry / fluid / oscillator domains, each justified independently. |
| "Are your simulators correct?" | Convergence studies, conservation laws, comparison to known reference solutions per benchmark. |
| "Is the comparison fair?" | All baselines see the same data, same regime perturbations, same epoch budget, same eval protocol; reported with mean ± std over multiple seeds. |
| "Is your model just bigger?" | HP scan reports per-baseline parameter-matched results, plus a scaling curve showing where each baseline saturates. |
| "Can you handle low-data / OOD / noise?" | 3 corruption regimes (clean / lowres / noisy) AND OOD lag-shift tests AND ablation studies. |
| "Does the architecture even exercise the lag-equivariance theorem?" | The DDE-PDE benchmarks include explicit u(t-τ) terms or distributed-delay kernels; the cyclic-buffer structure is mathematically built-in, not imposed. |
| "What about local methods like UNet?" | UNet is in the baseline pool. We expect UNet to be competitive but no longer dominant on properly-delay-structured problems. If it still wins on some cells, we report that honestly. |

---

## Section 1 — DDE-PDE Benchmark Suite

### Design principles

Each benchmark must satisfy:
1. **Mathematical specification** — explicit PDE form, BCs, ICs, parameter ranges.
2. **Delay relevance** — explicit memory term (delayed argument or distributed-delay kernel) so that the lag axis genuinely couples past to present.
3. **Solver justification** — known-good numerical method, implementation provenance.
4. **Verification** — convergence study (mesh / dt refinement), conservation check, comparison to reference solutions where available.
5. **Trajectory diversity** — initial conditions sampled from a parameterized manifold; parameter coverage broad enough to span dynamic regimes.
6. **Train/val/test split** — IID over IC parameters; no temporal leakage within a trajectory.

### B1 — Mackey-Glass + diffusion (2D)

```
∂u/∂t = D ∇²u  +  β · u(x, t-τ) / (1 + u(x, t-τ)^n)  -  γ · u(x, t)
```

* Domain: periodic torus [0, 2π]² with N=64 grid.
* Parameters: D, β, n, γ, τ. Default β=2.0, n=10, γ=1.0, τ=2.0, D=0.1.
* Regimes: stable steady state (β·γ small), oscillatory, chaotic (β·n·τ large).
* Why delay-relevant: classical Mackey-Glass blood-cell model with explicit delay τ, well-studied chaos onset.
* Solver: method of steps + RK4 in time; spectral spatial discretization.
* Verification: reproduce known chaotic regime at β=2, n=10, τ=2 (lit value); convergence test at dt = {1/16, 1/32, 1/64} should converge below 1e-4 relL2 to reference.
* Sample size: 256 train + 64 val + 64 test trajectories at distinct random IC (vary β, τ, IC perturbation).

### B2 — Wright equation + diffusion (2D)

```
∂u/∂t = D ∇²u  -  α · u(x, t-τ) · (1 + u(x, t))
```

* Domain: torus [0, 2π]² with N=64.
* Parameters: D, α, τ. Default α=1.5, τ=π/(2α) ≈ 1.05 (near critical).
* Regimes: small α — exponential decay; α near π/(2τ) — Hopf bifurcation, oscillation; large α — chaotic.
* Why delay-relevant: Wright's conjecture is a textbook delay-equation; explicit u(t-τ) term.
* Solver: same as B1.
* Verification: Hopf bifurcation onset at α·τ = π/2 (must reproduce within 1%).
* Sample size: 256 + 64 + 64.

### B3 — Hutchinson's logistic + diffusion (2D)

```
∂u/∂t = D ∇²u  +  r · u(x, t) · (1  -  u(x, t-τ) / K)
```

* Domain: torus [0, 2π]² with N=64.
* Parameters: D, r, K, τ.
* Regimes: small rτ — convergence to K; rτ > π/2 — sustained oscillation.
* Why delay-relevant: classical population-dynamics model with maturation delay.
* Solver: same as B1.
* Verification: oscillation amplitude vs rτ matches Wright's analytical bound.
* Sample size: 256 + 64 + 64.

### B4 — Distributed-delay reaction-diffusion (2D and 3D)

```
∂u/∂t = D ∇²u  +  ∫_0^∞ K(s) · f(u(x, t-s)) ds
```

with kernel `K(s) = (1/τ) · exp(-s/τ)` (exponential), or `K(s) = (s/τ²) · exp(-s/τ)` (gamma).

* Reaction term `f(u) = u·(1 - u)` (logistic) or `f(u) = u² · (1 - u)` (cubic).
* Domain: 2D 64² and 3D 32³.
* Parameters: D, τ, kernel choice, f choice.
* Regimes: short τ — quasi-Markov; long τ — strong memory effect; cubic — pattern formation.
* Why delay-relevant: distributed-delay kernel integrates ENTIRE past horizon; cyclic-buffer structure is exact.
* Solver: method of steps with exponential-kernel update (Sundials-style implicit), trapezoidal in s; spectral in space.
* Verification: τ→0 limit should match standard reaction-diffusion (NL Schrödinger / Fisher); compare to closed-form linear stability.
* Sample size: 256 + 64 + 64 per dimension.

### B5 — Delayed-feedback Burgers (2D)

```
∂u/∂t + u · ∇u = ν ∇²u  +  α · (u(x, t-τ) - u_target(x))
```

where `u_target` is a target profile (sinusoidal or constant).

* Domain: torus [0, 2π]² with N=64.
* Parameters: ν, α, τ, u_target form.
* Regimes: α=0 — standard Burgers (shocks form); α moderate — feedback control to target; α large — instability and chaos.
* Why delay-relevant: closed-loop feedback control with delay; common in fluid control engineering.
* Solver: method of steps + spectral or finite-volume Burgers solver; track shock positions.
* Verification: α=0 must reproduce APEBench Burgers within 1% relL2; shock location convergence.
* Sample size: 256 + 64 + 64.

### B6 — Ring-coupled Kuramoto field (2D)

```
∂θ/∂t = ω(x)  +  K · ∫ G(x-y) · sin(θ(y, t-τ) - θ(x, t)) dy
```

* Domain: torus [0, 2π]² with N=64; coupling G is a Gaussian kernel.
* Parameters: ω distribution, K, τ, σ (kernel width).
* Regimes: K small — incoherent; K large — synchronization; τ large — chimera states.
* Why delay-relevant: ring-coupled oscillators with delay; phase variable θ has cyclic structure.
* Solver: method of steps + RK4 in time; spectral coupling integration.
* Verification: K_critical match Kuramoto's mean-field analytical result.
* Sample size: 256 + 64 + 64.

### Summary

```
B1 Mackey-Glass + diff (2D)          T=128, n_hist=64, n_out=64, 64×64
B2 Wright + diff (2D)                T=128, n_hist=64, n_out=64, 64×64
B3 Hutchinson + diff (2D)            T=128, n_hist=64, n_out=64, 64×64
B4 Dist-delay RD (2D)                T= 64, n_hist=32, n_out=32, 64×64
B4 Dist-delay RD (3D)                T= 32, n_hist=16, n_out=16, 32×32×32
B5 Delayed-fb Burgers (2D)           T= 64, n_hist=32, n_out=32, 64×64
B6 Ring Kuramoto (2D)                T=128, n_hist=64, n_out=64, 64×64
```

Total 7 datasets (B4 has 2D and 3D variants). Sized to ~1-3 GB each
(within disk budget). Generation time ~6-12h on cluster CPU.

---

## Section 2 — Baseline pool

ND-compatible baselines we run on every dataset:

```
1.  LEMO_PC_ND        (our main contribution)
2.  LEMO_ND           (lag-equivariant without per-mode FiLM)
3.  FNO_ND            (joint spectral conv)
4.  Markov_FNO_ND     (1-frame autoregressive)
5.  Windowed_FNO_ND   (K-frame autoregressive)
6.  UNet_ND           (local conv multi-scale)
7.  DeepONet_ND       (branch+trunk, ND-adapted)        [NEEDS adaptation]
8.  MemNO_ND          (FNO + sequence memory, ND)        [NEEDS adaptation]
9.  LocalNO_ND        (FNO + localized integral, ND)     [NEEDS adaptation]
10. NIE_ND            (Neural Integral Equation, ND)     [NEEDS adaptation]
11. ANIE_ND           (Attentional NIE, ND)              [NEEDS adaptation]
```

ND-adaptation work for 7-11 is part of Section 3 (audit layer 2).

---

## Section 3 — 5-Layer Audit Pyramid

Each layer is an HONEST full-fidelity replication. No smoke / no toy.
Same code paths, same hyperparameters, same eval protocol, same epoch
budget. Only the SCOPE shrinks per layer.

### Layer 1 — Single cell pipeline check (1 cell)

* 1 dataset (Mackey-Glass 2D), 1 baseline (LEMO_PC_ND), seed 42, 200 epochs.
* Verify: data loads, model dispatches, training descends, eval matches expected magnitude order.
* Pass: test_relL2 < 0.5 (any reasonable PDE emulator should beat this).
* Cluster wall: ~30 min.

### Layer 2 — Multi-baseline pipeline check (11 cells)

* 1 dataset (Mackey-Glass 2D), all 11 baselines, seed 42, 200 epochs.
* Verify: every baseline dispatches, no shape mismatches, no OOMs, all complete.
* Pass: every baseline produces test_results.json with finite numbers.
* If any baseline (esp. 7-11 ND-adapted) fails: fix and re-run that cell.
* Cluster wall: ~5h.

### Layer 3 — Multi-dataset pipeline check (7 cells)

* All 7 DDE-PDE datasets, 1 baseline (LEMO_PC_ND), seed 42, 200 epochs.
* Verify: data layout works for every dataset, training descends on every dataset.
* Pass: test_relL2 < 0.5 on every dataset.
* If any dataset fails (e.g., NaN in solver, loss explodes): regenerate or adjust solver.
* Cluster wall: ~6h.

### Layer 4 — Half-scale full sweep (77 cells)

* All 7 datasets × all 11 baselines × 1 seed × 200 epochs.
* Verify: cross-product works, no edge cases, no resource exhaustion at half scale.
* Aggregate into a 7×11 table to confirm the LEMO advantage is visible at 1 seed.
* Pass: aggregation runs without error, no completed cell has NaN, results are within an order of magnitude of expectations.
* Cluster wall: ~24-30h on 24 workers / 8 GPUs.

### Layer 5 — FINAL COMPLETE SWEEP

* All 7 datasets × all 11 baselines × 3 corruption regimes (clean / lowres / noisy) × 3 seeds × 200 epochs.
* PLUS: HP scan on 1 representative dataset (Mackey-Glass 2D) — 3 widths × 3 depths × 3 modes for LEMO_PC_ND only — to give the parameter-matched scaling story.
* PLUS: residual_anchor variant for the spectral-method cells (LEMO_PC, LEMO, FNO, Markov, Windowed) on each of 7 datasets clean regime × 3 seeds.

```
core sweep:   7 datasets × 11 baselines × 3 regimes × 3 seeds = 693 cells
HP scan:      27 cells × 3 seeds                              =  81 cells
residual:     5 spectral × 7 datasets × 3 seeds               = 105 cells
```

Total: **879 cells**. At ~60-90 min/cell on H100, 24 workers/8 GPUs:
~80-100 GPU-hours. Wall ~4-5 days on the cluster.

* Verify on completion: aggregator produces a clean per-dataset
  per-baseline best-of-{clean, residual} table, plus per-dataset
  rollout-error curves, plus HP-scaling curves.

---

## Section 4 — Order of operations

```
[NOW]      Wait for in-flight scaling sweep (9 cells) to finish
              [confirms 'scaling alone doesn't close 3D-fluid gap']
[+0]       Update SESSION_LOG round 2.26      [DONE]
[+0]       Write FINAL_RESEARCH_PLAN.md       [DONE]
[+1d]      Build DDE-PDE solver code (Section 1)
[+2d]      Generate sample data for B1, audit (NaN, conservation, viz)
[+2d]      Generate research-scale data for all 7 benchmarks
[+3d]      ND-adapt baselines 7-11 (DeepONet, MemNO, LocalNO, NIE, ANIE)
[+3d]      Audit layer 1 (1 cell)
[+3d]      Audit layer 2 (11 cells)
[+4d]      Audit layer 3 (7 cells)
[+5d]      Audit layer 4 (77 cells)
[+6d]      Audit layer 5 — FINAL SWEEP launches
[+10d]     Final sweep completes
[+11d]     Aggregate + write paper update
```

---

## Section 5 — Decision points and abort criteria

* **After layer 1**: if pipeline broken on a fundamental level, halt and fix.
* **After layer 2**: if 1+ baseline can't be ND-adapted in <1d, drop that baseline (and document).
* **After layer 3**: if 1+ benchmark has solver instability we can't fix, drop that benchmark.
* **After layer 4**: if the LEMO advantage is invisible at 1 seed (i.e., no clear separation), pause and audit DEEPLY before committing to layer 5 compute.
* **After layer 5**: aggregate, present to user, decide on final paper claims.

---

## Section 6 — Reviewer-anticipation checklist

For each result we report, we will have answered:

- [ ] Multi-seed mean ± std (3 seeds minimum)
- [ ] Parameter count per cell (matched-compute story)
- [ ] Train/val/test split (no leakage)
- [ ] Training-curve sanity (descent, plateau detection)
- [ ] Per-frame relL2 curve (not just aggregate)
- [ ] Naive last-frame baseline (must beat for legitimacy)
- [ ] Statistical significance (paired-t or bootstrap on per-sample relL2)
- [ ] HP scan (width / depth / modes) showing per-baseline scaling
- [ ] OOD test (lag-shift, parameter shift)
- [ ] Robustness (3 corruption regimes)
- [ ] Reproducibility (frozen seeds, versioned data, versioned code)
- [ ] Source code + data publicly released alongside paper
