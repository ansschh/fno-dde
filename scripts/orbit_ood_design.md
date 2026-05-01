# Lag-Shift Orbit OOD Experiment — Design Document

**Status**: design only (no GPU runs). Compute estimate at the end.
**Target risk**: Round-3 fatal issue [3] — "no controlled lag-shift/orbit OOD experiment tied to the stated theorems; kernel-shape variation does not test orbit transfer or the covering-radius lower bound."
**Theorems addressed**: `Cref{thm:ood-gap}` (orbit-extension gap), `Cref{cor:pac-gap}` (PAC sample-complexity factor-n), `Cref{cor:augmentation-lower-bound}` (covering-radius lower bound).
**Section file**: `A:/dde research/NeurIPS_LEMO/sections/experimental_design_and_evaluation_plan.tex` (new subsection: "Lag-shift orbit OOD test (planned)").

---

## 1. Experimental object

**Family.** `dist_exp_rd_2d` (distributed-delay reaction-diffusion with exponential kernel `K(s) = (1/tau) * exp(-s/tau)`). This is the cleanest of the five distributed-kernel families because the kernel mass is concentrated near `s=0` and the mass *location* (kernel center) is a single scalar — perfect for a controlled cyclic-shift action on the lag axis.

**Lag axis.** History length `n_hist = 64` at `dt = 0.01`. The lag axis is identified with `Z/64Z` in the discrete benchmark (Track B); cyclic shift `Shift_k h(j) = h((j-k) mod 64)`.

**Controlled variable.** `tau_shift in {0, 1, 2, ..., 63}` — an *integer* shift index applied to the kernel center on the lag axis. Concretely, the standard `dist_exp` kernel has weight at quadrature node `s_i = i*dt` of `K(s_i) = (1/tau)*exp(-s_i/tau)*dt` (then normalized). We define the *shifted* kernel by

  `K_k(s_i) = K(s_{(i-k) mod n_quad})`

i.e. by cyclically rotating the kernel-weight vector `(K_0, K_1, ..., K_{n_quad-1})` by `k` positions on the discrete quadrature grid. This is exactly the discrete `Shift_k` action of `Cref{thm:layer-equiv}` on the lag axis.

**Why this is not "kernel-shape variation".** The kernel's *shape* (exponential decay profile) is held fixed; only its discrete *position* on the lag grid changes. By construction, this generates the cyclic orbit of one kernel under `C_{n_quad}` action — the exact object `Cref{thm:ood-gap}` and `Cref{cor:augmentation-lower-bound}` are stated about.

---

## 2. Lag-shift action — formal definition

Let `n_quad = round(tau_max / dt) = round(0.4 / 0.01) = 40` for the default `dist_exp` (we set `tau = 0.1`, `tau_max = 4*tau = 0.4` to keep `n_quad < n_hist = 64` so the orbit fits inside the history window).

For each base trajectory `j` (sampled with random IC + random `A`, `D`), build a *family* of trajectories indexed by `tau_shift in {0, 1, ..., n_quad - 1} = {0, 1, ..., 39}`:

```
K_base[i] = (1/tau) * exp(-i*dt / tau)  for i in [0, n_quad)
K_base    /= sum(K_base * dt)            # normalize to mass 1

K_shifted[i] = K_base[(i - tau_shift) mod n_quad]
```

The simulator is the same for all `tau_shift` values: only the kernel-weight vector rotates. Initial conditions and `A`, `D` are fixed within an orbit (so all 40 trajectories in one orbit share the same IC and reaction parameters). The non-degeneracy condition (A13) of `Cref{cor:augmentation-lower-bound}` is satisfied by construction: the trajectory at `tau_shift=0` and at `tau_shift=20` differ uniformly (the active memory is at the nearest past vs at the middle past), and one verifies `lambda > 0` empirically per trajectory before declaring the orbit valid.

**Orbit cardinality.** Each base IC + parameter draw produces an orbit of size `|C_{n_quad}| = 40`. The orbit covers the cyclic group `Z/40Z` exactly.

---

## 3. Train / test split — orbit-aware

**Number of base trajectories**: `N_orbits = 256` (each yields 40 shifted copies, so the underlying full orbit-pool is `256 * 40 = 10240` trajectories).

**Training shifts** (8 representatives, evenly spaced):
```
S_train = {0, 5, 10, 15, 20, 25, 30, 35}
```
(every 5th index; covering radius `r(A) = 2` in cyclic distance).

**Held-out shifts** (32 unseen shifts):
```
S_test  = {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, ..., 39} \setminus S_train
```

So:
* `train` = 256 orbits × 8 shifts = **2048 trajectories**.
* `test_orbit` = 256 orbits × 32 shifts = **8192 trajectories**.

**Validation** is a held-out 32-orbit subset of the orbit pool with the *training* shifts (so `val_in_orbit` measures fitting; `test_orbit` measures orbit-transfer).

**Variant sweep on the train set** for the augmentation analysis: build sub-training sets that use only `m` shifts:

| label   | m  | training shifts (covering radius `r(A)` on `Z/40Z`) |
|---------|----|-----------------------------------------------------|
| m=1     |  1 | {0}                       (r = 20)                  |
| m=2     |  2 | {0, 20}                   (r = 10)                  |
| m=4     |  4 | {0, 10, 20, 30}           (r = 5)                   |
| m=8     |  8 | {0, 5, 10, 15, 20, 25, 30, 35}  (r = 2)           |
| m=16    | 16 | every 3rd      (closest to uniform; r = 1)          |
| m=32    | 32 | every shift except 8 held-out  (r ≈ 1)              |

The covering radius decreases as `r(A) ≈ n_quad / (2m)`, exactly the bound in `Cref{cor:augmentation-lower-bound}`.

---

## 4. Models

We train two architectures:

**LEMO-PC** (the lag-equivariant model). `model_class = lemo_pc_nd`, the same architecture used in the headline 45-cell sweep (`width=64`, `n_layers=3`, `lag_modes=24`, `spatial_modes=12`, `params_dim=1` for `A`). By construction the lag-conv kernel weights are shared across absolute lag indices (Theorem 1), so the output is exactly cyclic-equivariant in the lag axis.

**Per-lag MLP** (the non-equivariant baseline). Because no off-the-shelf "non-equivariant lag mixer" exists in the repo, we add a thin wrapper around the `mlp` (sequence-to-sequence) baseline that uses *per-lag-index* learnable weights (i.e., the absolute lag index breaks shift-equivariance). Specifically, instead of `BU(j) + sum_r A_r U(j-r) + b` with shared `A_r`, the per-lag MLP uses

  `(L U)(j) = B_j U(j) + W_j flatten(U)` for each absolute lag index `j`,

so each `j` gets its own dense map. This is the cleanest non-equivariant counterpart and corresponds to the unrestricted hypothesis class `F` in `Cref{thm:ood-gap}` part (ii). Parameter count is matched to LEMO-PC (~2.7M) by tuning the per-lag hidden dim. Implementation goes in `models/per_lag_mlp_nd.py` and dispatches via `model_class = "per_lag_mlp_nd"` in `build_model.py` (existing pattern, see `models/baselines_nd.py`).

---

## 5. Sweep matrix

Two arms:

**Arm A — equivariant.** `LEMO-PC, m=8 (full S_train)`, 3 seeds = **3 cells**. Test on `S_test` (32 unseen shifts). Theorem prediction: orbit-constant test error (i.e., test error ~= train error) regardless of which subset of shifts was used, because every trajectory in the orbit collapses to the same equivalence class under equivariance. Optionally we also evaluate the same 3 LEMO-PC checkpoints under each `m in {1,2,4,8,16,32}` augmentation budget — this is a *re-train per m* not a re-eval, so 6 m-values × 3 seeds = **18 cells**.

**Arm B — non-equivariant baseline.** `per_lag_mlp_nd, m in {1,2,4,8,16,32}`, 3 seeds = **6 × 3 = 18 cells**.

**Shared eval set.** Both arms test on the *same* `S_test` with the same 256 orbits.

**Cell budget.** 18 (LEMO-PC) + 18 (baseline) = **36 cells** + 6 cells for an "ideal-baseline" upper bound (per_lag_mlp at `m = full = 40` — no augmentation gap left) = **42 cells**, matching the user-specified 36 + 6 split.

---

## 6. Predicted results

### Theory predictions — exact

**LEMO-PC** (Arm A): test rel-`L^2` is **constant in `m`** (the augmentation budget). This is `Cref{thm:ood-gap}` part (i): every orbit's loss is constant under a `Shift_k`-equivariant model and a `Shift_k`-invariant loss. Empirically, finite-precision floats produce a sub-`1e-5` fluctuation; we predict effectively flat.

**Per-lag-MLP** (Arm B): test rel-`L^2` is **lower-bounded by `~0.5 * lambda - C_1 * r(A)`** by `Cref{cor:augmentation-lower-bound}`, which decays linearly in `r(A)`. Since `r(A) = n_quad / (2m) = 20/m`, and the empirical Lipschitz modulus `C_1` of the simulator and uniform separation `lambda` are estimable on a held-out batch, we predict an error curve that scales approximately as

  `err_baseline(m) ~ a / m  + err_floor`

Plotted on log-`m` × log-`err`, the per-lag-MLP curve has slope ~-1 down to the noise floor, while the LEMO curve is flat.

### Predicted plot

```
test rel-L^2 (log)
  10^0  +
        |  + per-lag-MLP, m=1
        |    |
  10^-1 |    +    + per-lag-MLP, m=2
        |         |
        |              + per-lag-MLP, m=4
  10^-2 |                   + per-lag-MLP, m=8
        |                        + per-lag-MLP, m=16
        |                             + per-lag-MLP, m=32 + per-lag-MLP, m=full
  10^-3 +========================== LEMO-PC (flat) =========================
        |
        +----+----+----+----+----+----+----+
            1    2    4    8   16   32   m  (log scale)
```

The horizontal LEMO-PC line is the *equivariant guarantee*; the slope of the per-lag-MLP curve is the *factor-n PAC gap*. The width of the gap at any `m` is the empirical sample-complexity advantage; the gap at `m = m_max` (the largest augmentation budget that still leaves >0 unseen shifts in `Z/40Z`) is the *orbit-extension lower bound* of `Cref{cor:augmentation-lower-bound}`.

---

## 7. Statistical protocol

* 3 seeds per cell (`{42, 123, 456}` matching the headline sweep).
* Paired comparison per `(m, seed)` cell: `LEMO-PC[m=8]` vs `per_lag_mlp[m]`.
* Reported metrics:
  * mean test rel-`L^2` per cell with bootstrap-95\% CI (10\,000 resamples)
  * paired-permutation `p`-value across 3 seeds at each `m`
  * fitted slope `d log(err) / d log(m)` for the per-lag-MLP curve, with bootstrap CI
  * residual `r(A)` covering radius for each `m`, computed from the actual train shifts.
* Sanity check: the LEMO-PC curve should not have slope distinguishable from 0 at p < 0.05; the per-lag-MLP curve should have slope distinguishable from 0 at p < 0.01.

---

## 8. Files produced

* `scripts/gen_orbit_ood_data.py` — generator. Produces `data_orbit_ood/dist_exp_rd_2d_orbit/{train,val,test}/shard_*.npz`. Also produces a JSON manifest of which `(orbit_id, tau_shift)` tuples are in each split.
* `scripts/launch_orbit_ood.sh` — sweep launcher. Generates data, trains 42 cells across 8 GPUs / 24 workers, then runs the orbit-OOD eval.
* `scripts/orbit_ood_design.md` — this file.
* `sections/experimental_design_and_evaluation_plan.tex` — new subsection "Lag-shift orbit OOD test (planned)".

Output directories: `outputs/orbit_ood_sweep/{lemo_pc_nd, per_lag_mlp_nd}/m{1,2,4,8,16,32}/s{42,123,456}/test_results.json`.

---

## 9. Compute estimate

Per-cell wall-clock at headline settings on H100:
* LEMO-PC: ~50 minutes / 200 epochs / 1 cell.
* per_lag_mlp: ~40 minutes / 200 epochs / 1 cell (smaller compute, fewer layers).

Sweep: 42 cells × 24 parallel workers / 8 GPUs (3 per GPU) ≈ 50min × ceil(42/24) = **~100 minutes wall-clock** on a single 8×H100 pod. Add data-gen wall (~30 min on CPU for 256 orbits × 40 shifts × 64-frame integration ≈ 10k trajectories) and audit wall (~10 min) → **~2.5 hours total wall-clock budget**.

GPU-hours: 42 cells × 0.75 h × 1 GPU = **~32 GPU-hours**. Within the 15-h-budget rule (this is one focused sweep, not the global budget cap; complements rather than replaces the in-progress sweeps).

---

## 10. Why this experiment is the right test

Reviewer concern: the headline sweep varies *which kernel family* (exp/gaussian/gamma/uniform/powerlaw) — these are different operators on disjoint orbits, so the comparison does not test orbit transfer.

This experiment fixes the kernel *family* and varies only the *position* on the lag orbit, so train and test points lie on the *same* orbit. By construction, an equivariant model has identical loss on train and test (Theorem 3.4 = `Cref{thm:ood-gap}` part (i)). A non-equivariant model can fit the training shifts arbitrarily well yet have lower-bounded error on the held-out shifts (`Cref{cor:augmentation-lower-bound}`). The plot of test error vs. augmentation budget `m` directly visualizes the predicted factor-`n` PAC gap of `Cref{cor:pac-gap}`.
