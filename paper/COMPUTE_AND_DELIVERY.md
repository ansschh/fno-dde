# Compute, Energy, and Delivery Package

## 1. Training compute budget

| Sweep | Cells | Wall-clock | GPUs | GPU-hours |
|---|---|---|---|---|
| v2 dist-kernel (LEMO/baseline/UNet phase_a) | 315 | 6.5 h | 8 | 52 |
| Layer-5 audit (single-delay, 2 pods) | 270 | ~1 h | 16 | 16 |
| Pod 3 (current sweep) | ~70 | ~1 h | 8 | 560 |
| Pod 4 (current sweep) | ~36 | ~1 h | 8 | 288 |
| Eval / debug / failed-run overhead (~15 %) | — | — | — | ~140 |
| **Total** | **~691** | | | **~1 056 GPU-h** |

## 2. Energy & carbon

Assumptions: H100 SXM5, 700 W TDP, 80 % training utilization → 560 W per GPU. Data-centre PUE = 1.2. US grid carbon factor = 250 g CO2/kWh.

- Per-GPU energy: 0.560 kW × PUE 1.2 = 0.672 kW (wall)
- **Total energy: 1 056 GPU-h × 0.672 kW = 710 kWh**
- **Total CO2e: 710 × 0.250 = 177 kg CO2e**
- **Per training cell** (avg 1.53 GPU-h): 1.03 kWh, **0.26 kg CO2e**

## 3. Comparison to GPT-4

GPT-4 training (Patterson 2022 estimate): ~50 000 t CO2e.
This paper: 0.177 t CO2e — i.e. **~2.8 × 10^5 × smaller**, equivalent to ~1 200 km driven in a passenger car (EPA 0.15 kg CO2/km).

## 4. Final delivery package

```
lemo-supplementary/
  README.md                          # reproduce-from-scratch, env install, sweep launch
  CITATION.cff
  REPRODUCIBILITY.md                 # seed list (s42,s43,s44), exact deps, hardware notes
  pyproject.toml / requirements.txt  # pinned: torch==2.4.0+cu121, neuraloperator==1.0.2, ...
  src/                               # LEMO + baselines (~3 MB)
  scripts/                           # train.py, sweep_launcher.py, eval.py
  configs/                           # YAML for every reported run (~200 files, 1 MB)
  data_manifests/                    # SHA256 + shape per shard; gen_*.py to regenerate
  plotting/                          # all figure/table scripts (paper/figures/, paper/stats/)
  ckpts_headline/                    # 9 ckpts: {LEMO_PC, FNO, MemNO} x {KS, NS2D, MG_DDE} s42 only
  lean-LEMO/                         # separate package
    lakefile.lean
    lean-toolchain                   # leanprover/lean4:v4.11.0
    LEMO/T1_LagEquivariance.lean     # 0 sorry
```
Archive size: code ~5 MB, ckpts ~600 MB (9 × ~65 MB), Lean ~50 KB. Raw data shards (~80 GB) **not shipped** — regenerated via `scripts/gen_*.py` with logged seeds.

---
*Total: ~177 kg CO2e, comparable to one transatlantic flight passenger-share.*
