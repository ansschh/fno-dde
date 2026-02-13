# DDE-FNO Benchmark Validation Checklist

**Purpose:** Definition of done before architecture changes. Answer each section with file paths + commands + 1-2 line interpretation.

**Families:** `hutch`, `linear2`, `vdp`, `dist_uniform`, `dist_exp`

---

## 0) One-Page Sanity: Current State

### 0.1 Git Tag
- [ ] Frozen baseline has a git tag
```bash
git describe --tags --always
```
**Expected:** Tag like `baseline-v2` or commit hash  
**Output file:** N/A (terminal output)

### 0.2 Baseline Protocol
- [ ] Canonical baseline protocol exists
```bash
cat configs/baseline_protocol.yaml
```
**Expected file:** `configs/baseline_protocol.yaml`

### 0.3 Freeze Manifest
- [ ] Single freeze manifest defines the benchmark pack
```bash
ls -la freeze_manifest_*.json
cat freeze_manifest_all5_v2.json
```
**Expected file:** `freeze_manifest_all5_v2.json`  
**Must include:** dataset paths, hashes, run IDs, config snapshots, commit hash

### 0.4 No Version Mixing
- [ ] dist_exp uses v2 data (not v1)
```bash
python -c "import json; m=json.load(open('freeze_manifest_all5_v2.json')); print(m['dist_exp']['data_path'])"
```
**Expected:** `data_baseline_v2/dist_exp` (NOT `data_baseline_v1`)

---

## 1) Benchmark Validity: Solving What We Claim

### 1A) Delay Actually Matters

#### 1A.1 Delay Sensitivity (dist_exp v2)
- [ ] Delay sensitivity report exists and passed
```bash
cat reports/data_quality/dist_exp_v2/delay_sensitivity.json
```
**Expected file:** `reports/data_quality/dist_exp_v2/delay_sensitivity.json`  
**Pass criteria:** median sensitivity >= 10%

#### 1A.2 Theta Distribution (dist_exp v2)
- [ ] theta = lambda * tau is constrained properly
```bash
cat reports/data_quality/dist_exp_v2/theta_distribution.json
```
**Expected:** theta in [0.5, 1.8], exp(-theta) median > 20%

#### 1A.3 RHS Uses Delay Term
- [ ] Code confirms x(t-tau) in RHS
```bash
grep -n "x\[.*delay\|history\|t-tau" src/dde/families.py
```
**Expected:** Line numbers showing delay term usage

### 1B) Distributed Delay Invariant

- [ ] dist_uniform auxiliary identity residual is small
```bash
cat reports/data_quality/dist_uniform/aux_identity.json
```
**Expected file:** `reports/data_quality/dist_uniform/aux_identity.json`  
**Pass criteria:** residual median < 1e-3

### 1C) VdP History Consistency

- [ ] VdP history consistency report exists
```bash
cat reports/data_quality/vdp/history_consistency.json
```
**Expected file:** `reports/data_quality/vdp/history_consistency.json`  
**Check:** RMS of v - dx/dt should not blow up on OOD-history

---

## 2) Dataset Generation Integrity

### 2A) Manifest Completeness

- [ ] Each family has manifest.json in all splits
```bash
for fam in hutch linear2 vdp dist_uniform dist_exp; do
  echo "=== $fam ==="
  ls data_baseline_v*/\$fam/manifest.json 2>/dev/null || echo "MISSING"
done
```
**Must include:** seed, solver backend, tolerances, param ranges, acceptance/rejection counts, normalization stats path

### 2B) No Leakage / Correct Splits

- [ ] Split audit report exists for all families
```bash
python scripts/audit_splits.py --families hutch linear2 vdp dist_uniform dist_exp
cat reports/split_audit/*.json
```
**Expected files:** `reports/split_audit/{family}_split_audit.json`  
**Check:** 
- ID test identical across experiments (hash match)
- OOD splits differ only by intended factor

### 2C) Reproducibility

- [ ] Same seed/config produces identical hashes
```bash
python scripts/repro_test.py --family hutch --seed 42
cat reports/repro/hutch_repro.txt
```
**Expected file:** `reports/repro/{family}_repro.txt`

---

## 3) Numerical Label Quality

### 3A) Label Fidelity (Fast vs Reference Solve)

- [ ] Label fidelity reports exist for all families
```bash
for fam in hutch linear2 vdp dist_uniform dist_exp; do
  echo "=== $fam ==="
  cat reports/data_quality/\$fam/label_fidelity.json 2>/dev/null || echo "MISSING"
done
```
**Expected files:** `reports/data_quality/{family}/label_fidelity.json`  
**Pass criteria:** median < 1e-3, p95 < 1e-2

### 3B) Residual Benchmark on Labels

- [ ] Residual reports and plots exist
```bash
for fam in hutch linear2 vdp dist_uniform dist_exp; do
  echo "=== $fam ==="
  cat reports/data_quality/\$fam/residual.json 2>/dev/null || echo "MISSING"
done
```
**Expected files:** 
- `reports/data_quality/{family}/residual.json`
- `reports/data_quality/{family}/residual_vs_time.png`

---

## 4) Training Correctness

### 4A) Masking + Metric Correctness

- [ ] Mask sanity verified (sum = n_out, zeros on history)
- [ ] relL2 computed in original space, future-only
- [ ] Normalization stats from ID train applied to all splits
```bash
python scripts/diagnostic_audit.py --families hutch linear2 vdp dist_uniform dist_exp
```
**Expected output:** All checks PASS

### 4B) Overfit Test

- [ ] 64-sample overfit achieves near-zero error
```bash
python scripts/overfit_test.py --family hutch --n_samples 64
cat reports/overfit/{family}_overfit.json
```
**Expected files:** `reports/overfit/{family}_overfit.json`  
**Pass criteria:** relL2_orig < 1e-2 after training

### 4C) Training Curves Sane

- [ ] Training curves exist for all families
```bash
ls reports/model_viz/*/*/training_curves.png
```
**Expected files:** `reports/model_viz/{family}/{run_id}/training_curves.png`  
**Check:** No divergence, val tracks train reasonably

---

## 5) Evaluation Robustness

### 5A) Required Splits Evaluated

- [ ] Metrics JSON exists for all splits
```bash
for split in id ood_delay ood_history ood_horizon; do
  echo "=== $split ==="
  python -c "import json; d=json.load(open('reports/model_viz/all5/baseline_all5_metrics_full.json')); print({k: d[k].get('$split', {}).get('median', 'MISSING') for k in ['hutch','linear2','vdp','dist_uniform','dist_exp']})"
done
```
**Expected:** All families have all 4 splits

### 5B) Tail Behavior (P95)

- [ ] Per-sample errors stored, not just aggregates
```bash
python -c "import json; d=json.load(open('reports/model_viz/all5/baseline_all5_metrics_full.json')); print('hutch per_sample count:', len(d['hutch']['id'].get('per_sample', [])))"
```
**Expected:** per_sample array with 2000 entries

### 5C) Residual on Predictions

- [ ] Physics residual computed on model predictions
```bash
cat reports/model_quality/{family}/pred_residual.json
```
**Expected files:** `reports/model_quality/{family}/pred_residual.json`  
**Priority:** HIGH if missing

---

## 6) Baselines Beyond FNO

### 6.1 Identity Baseline

- [ ] Identity baseline (y(t) = y(0)) computed
```bash
cat reports/model_viz/all5/identity_baseline_comparison.json
```
**Expected file:** `reports/model_viz/all5/identity_baseline_comparison.json`  
**Check:** FNO beats identity by significant margin (>2x)

### 6.2 Naive Baseline (Persistence)

- [ ] Naive baseline results for all families
```bash
cat reports/baselines/naive_baseline.json
```
**Expected file:** `reports/baselines/naive_baseline.json`

### 6.3 TCN Baseline

- [ ] TCN trained and evaluated on all families
```bash
cat reports/baselines/tcn_baseline.json
```
**Expected file:** `reports/baselines/tcn_baseline.json`

### 6.4 Comparison Table

- [ ] FNO vs TCN vs Naive comparison table exists
```bash
cat reports/baselines/comparison_table.md
```

---

## 7) Statistical Robustness (Multi-Seed)

- [ ] At least 3 seeds run for baseline FNO
```bash
for seed in 42 43 44; do
  echo "=== Seed $seed ==="
  ls outputs/baseline_v*/hutch_seed${seed}_*/best_model.pt 2>/dev/null || echo "MISSING"
done
```
**Expected:** 3 runs per family (minimum: hutch, linear2, vdp)

- [ ] Seed variance table generated
```bash
cat reports/paper_tables/table_seed_variance.md
```
**Expected format:** mean +/- std for ID median and p95

---

## 8) Visualization Package

### 8.1 All-5 Panel Figures (10 total)

- [ ] Error vs time panels (4 splits)
- [ ] Tail P95 panels (4 splits)
- [ ] Training loss panel
- [ ] Val relL2 panel
```bash
ls reports/model_viz/all5_panels/*.png | wc -l
```
**Expected:** 10 PNG files

### 8.2 Per-Family Diagnostic Pack

- [ ] ECDF of relL2
- [ ] Waterfall sorted error
- [ ] Success rate vs time
- [ ] Amplitude ratio pred/true
```bash
ls reports/model_viz/*/*/diagnostics/*.png | head -20
```
**Expected files per family:**
- `ecdf_relL2.png`
- `waterfall.png`
- `success_rate.png`
- `amplitude_ratio.png`

### 8.3 Reproducible from Arrays

- [ ] Curve data saved as NPZ (not stitched PNGs)
```bash
ls reports/model_viz/*/*/curves/*.npz | head -10
```

---

## 9) Paper Readiness: Tables

### 9.1 Family Definitions
- [ ] `reports/paper_tables/table1_family_definitions.md`

### 9.2 Dataset Protocol
- [ ] `reports/paper_tables/table2_dataset_protocol.md`

### 9.3 Data Quality (label fidelity, residual, acceptance)
- [ ] `reports/paper_tables/table3_data_quality.md` (TODO if missing)

### 9.4 Baseline Performance
- [ ] `reports/paper_tables/table3_baseline_performance.md`

### 9.5 OOD Gaps
- [ ] `reports/paper_tables/table5_ood_gaps.md`

### 9.6 Baselines Comparison
- [ ] `reports/paper_tables/table_baselines_comparison.md` (TODO if missing)

### 9.7 Seed Variance
- [ ] `reports/paper_tables/table_seed_variance.md` (TODO if missing)

### 9.8 Combined LaTeX
- [ ] `reports/paper_tables/table_combined.tex`

**Check:** All tables generated from scripts, not hand-edited
```bash
python scripts/benchmark_pack/generate_paper_tables.py
```

---

## 10) Red Flag Questions (Fast Bug Catch)

### 10.1 Overfit Sanity
- [ ] Can overfit 64 samples to near-zero error?
```bash
python scripts/overfit_test.py --family hutch --n_samples 64 --epochs 500
```
**If NO:** encoding/loss/normalization bug

### 10.2 Tau Shuffle Test
- [ ] Shuffling tau in batch worsens loss?
```bash
python scripts/tau_shuffle_test.py --family hutch
```
**If NO:** model may not use delay/params

### 10.3 Zero History Test
- [ ] Zeroing history input collapses performance?
```bash
python scripts/zero_history_test.py --family hutch
```
**If NO:** history not wired correctly

### 10.4 Zero Prediction Sanity
- [ ] Predicting zeros gives relL2 ~ 1 in normalized space?
```bash
python scripts/zero_pred_test.py --family hutch
```
**If NO:** metric bug

---

## Quick Status Summary

Run this to get current checklist status:
```bash
python scripts/benchmark_pack/checklist_status.py
```

---

## File Structure Reference

```
dde-fno/
├── configs/
│   └── baseline_protocol.yaml
├── freeze_manifest_all5_v2.json
├── data_baseline_v1/
│   └── {family}/manifest.json
├── data_baseline_v2/
│   └── dist_exp/manifest.json
├── outputs/baseline_v*/
│   └── {family}_seed{N}_{timestamp}/
│       ├── best_model.pt
│       ├── config.yaml
│       └── history.json
├── reports/
│   ├── data_quality/{family}/
│   │   ├── delay_sensitivity.json
│   │   ├── label_fidelity.json
│   │   ├── residual.json
│   │   └── *.png
│   ├── model_viz/
│   │   ├── all5/
│   │   │   ├── baseline_all5_metrics_full.json
│   │   │   └── identity_baseline_comparison.json
│   │   ├── all5_panels/*.png (10 figures)
│   │   └── {family}/{run_id}/
│   │       ├── curves/*.npz
│   │       └── diagnostics/*.png
│   ├── paper_tables/
│   │   ├── table*.md
│   │   └── table_combined.tex
│   ├── baselines/
│   │   ├── naive_baseline.json
│   │   ├── tcn_baseline.json
│   │   └── comparison_table.md
│   ├── split_audit/{family}_split_audit.json
│   ├── repro/{family}_repro.txt
│   └── overfit/{family}_overfit.json
└── scripts/
    ├── benchmark_pack/
    │   ├── all5_panels.py
    │   ├── save_curves.py
    │   ├── diagnostic_plots.py
    │   ├── generate_paper_tables.py
    │   ├── identity_baseline.py
    │   └── checklist_status.py
    ├── audit_splits.py
    ├── repro_test.py
    ├── overfit_test.py
    ├── tau_shuffle_test.py
    ├── zero_history_test.py
    └── zero_pred_test.py
```

---

**Last updated:** 2024-12-30
