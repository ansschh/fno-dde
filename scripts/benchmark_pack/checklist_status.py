#!/usr/bin/env python3
"""
Quick audit of checklist status - shows what exists vs what's missing.
"""
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
FAMILIES = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]

def check_file(path: str, description: str) -> tuple:
    """Check if file exists, return (status, path)."""
    full_path = ROOT / path
    exists = full_path.exists()
    return ("PASS" if exists else "MISSING", str(full_path))

def check_glob(pattern: str, min_count: int = 1) -> tuple:
    """Check if glob pattern matches at least min_count files."""
    matches = list(ROOT.glob(pattern))
    if len(matches) >= min_count:
        return ("PASS", f"{len(matches)} files found")
    return ("MISSING", f"Only {len(matches)} files (need {min_count})")

def main():
    results = {}
    
    print("="*70)
    print("DDE-FNO BENCHMARK CHECKLIST STATUS")
    print("="*70)
    
    # Section 0: Project State
    print("\n## 0) Project State")
    results["0.2 baseline_protocol"] = check_file("configs/baseline_protocol.yaml", "Baseline protocol")
    results["0.3 freeze_manifest"] = check_glob("freeze_manifest_*.json")
    
    for key, (status, detail) in results.items():
        if key.startswith("0"):
            print(f"  [{status}] {key}: {detail}")
    
    # Section 1: Benchmark Validity
    print("\n## 1) Benchmark Validity")
    results["1A.1 dist_exp_delay_sensitivity"] = check_file("reports/data_quality/dist_exp_v2/delay_sensitivity.json", "Delay sensitivity")
    results["1A.2 dist_exp_theta"] = check_file("reports/data_quality/dist_exp_v2/theta_distribution.json", "Theta distribution")
    results["1B dist_uniform_aux"] = check_file("reports/data_quality/dist_uniform/aux_identity.json", "Aux identity")
    results["1C vdp_history"] = check_file("reports/data_quality/vdp/history_consistency.json", "VdP history")
    
    for key, (status, detail) in results.items():
        if key.startswith("1"):
            print(f"  [{status}] {key}")
    
    # Section 2: Dataset Integrity
    print("\n## 2) Dataset Integrity")
    for fam in FAMILIES:
        data_dir = "data_baseline_v2" if fam == "dist_exp" else "data_baseline_v1"
        key = f"2A manifest_{fam}"
        results[key] = check_file(f"{data_dir}/{fam}/manifest.json", f"{fam} manifest")
        print(f"  [{results[key][0]}] {key}")
    
    results["2B split_audit"] = check_glob("reports/split_audit/*_split_audit.json", 5)
    results["2C repro"] = check_glob("reports/repro/*_repro.txt", 1)
    print(f"  [{results['2B split_audit'][0]}] 2B split_audit: {results['2B split_audit'][1]}")
    print(f"  [{results['2C repro'][0]}] 2C repro: {results['2C repro'][1]}")
    
    # Section 3: Label Quality
    print("\n## 3) Label Quality")
    for fam in FAMILIES:
        key = f"3A label_fidelity_{fam}"
        results[key] = check_file(f"reports/data_quality/{fam}/label_fidelity.json", f"{fam} label fidelity")
        print(f"  [{results[key][0]}] {key}")
    
    # Section 4: Training Correctness
    print("\n## 4) Training Correctness")
    results["4B overfit_tests"] = check_glob("reports/overfit/*_overfit.json", 1)
    results["4C training_curves"] = check_glob("reports/model_viz/*/*/training_curves.png", 5)
    for key in ["4B overfit_tests", "4C training_curves"]:
        print(f"  [{results[key][0]}] {key}: {results[key][1]}")
    
    # Section 5: Evaluation
    print("\n## 5) Evaluation")
    results["5A metrics_full"] = check_file("reports/model_viz/all5/baseline_all5_metrics_full.json", "Full metrics")
    results["5C pred_residual"] = check_glob("reports/model_quality/*/pred_residual.json", 1)
    print(f"  [{results['5A metrics_full'][0]}] 5A metrics_full")
    print(f"  [{results['5C pred_residual'][0]}] 5C pred_residual: {results['5C pred_residual'][1]}")
    
    # Check splits in metrics
    metrics_path = ROOT / "reports/model_viz/all5/baseline_all5_metrics_full.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        splits = ["id", "ood_delay", "ood_history", "ood_horizon"]
        for fam in FAMILIES:
            fam_data = metrics.get(fam, {})
            missing = [s for s in splits if s not in fam_data]
            if missing:
                print(f"    {fam}: MISSING splits {missing}")
            else:
                print(f"    {fam}: all 4 splits present")
    
    # Section 6: Baselines
    print("\n## 6) Baselines")
    results["6.1 identity_baseline"] = check_file("reports/model_viz/all5/identity_baseline_comparison.json", "Identity baseline")
    results["6.2 naive_baseline"] = check_file("reports/baselines/naive_baseline.json", "Naive baseline")
    results["6.3 tcn_baseline"] = check_file("reports/baselines/tcn_baseline.json", "TCN baseline")
    for key in ["6.1 identity_baseline", "6.2 naive_baseline", "6.3 tcn_baseline"]:
        print(f"  [{results[key][0]}] {key}")
    
    # Section 7: Multi-seed
    print("\n## 7) Multi-Seed")
    for seed in [42, 43, 44]:
        count = len(list(ROOT.glob(f"outputs/baseline_v*/*_seed{seed}_*/best_model.pt")))
        print(f"  Seed {seed}: {count}/5 families")
    
    # Section 8: Visualization
    print("\n## 8) Visualization")
    results["8.1 all5_panels"] = check_glob("reports/model_viz/all5_panels/*.png", 10)
    results["8.2 diagnostics"] = check_glob("reports/model_viz/*/*/diagnostics/*.png", 5)
    results["8.3 curve_npz"] = check_glob("reports/model_viz/*/*/curves/*.npz", 5)
    for key in ["8.1 all5_panels", "8.2 diagnostics", "8.3 curve_npz"]:
        print(f"  [{results[key][0]}] {key}: {results[key][1]}")
    
    # Section 9: Paper Tables
    print("\n## 9) Paper Tables")
    tables = [
        ("table1_family_definitions.md", "Family definitions"),
        ("table2_dataset_protocol.md", "Dataset protocol"),
        ("table3_baseline_performance.md", "Baseline performance"),
        ("table4_model_settings.md", "Model settings"),
        ("table5_ood_gaps.md", "OOD gaps"),
        ("table6_time_segments.md", "Time segments"),
        ("table_combined.tex", "Combined LaTeX"),
    ]
    for fname, desc in tables:
        key = f"9 {fname}"
        results[key] = check_file(f"reports/paper_tables/{fname}", desc)
        print(f"  [{results[key][0]}] {desc}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for s, _ in results.values() if s == "PASS")
    total = len(results)
    print(f"\n  {passed}/{total} checks passed ({100*passed/total:.0f}%)")
    
    missing = [(k, d) for k, (s, d) in results.items() if s == "MISSING"]
    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for k, d in missing:
            print(f"    - {k}")
    
    print()


if __name__ == "__main__":
    main()
