#!/usr/bin/env python3
"""
Generate comparison table: Naive vs FNO baselines.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

def main():
    # Load metrics
    fno_path = ROOT / "reports/model_viz/all5/baseline_all5_metrics_full.json"
    naive_path = ROOT / "reports/baselines/naive_baseline.json"
    identity_path = ROOT / "reports/model_viz/all5/identity_baseline_comparison.json"
    
    with open(fno_path) as f:
        fno = json.load(f)
    with open(naive_path) as f:
        naive = json.load(f)
    with open(identity_path) as f:
        identity = json.load(f)
    
    families = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]
    
    # Generate markdown table
    lines = [
        "# Baseline Comparison: FNO vs Naive (Persistence)",
        "",
        "| Family | Naive (Median) | FNO (Median) | Improvement | Naive (P95) | FNO (P95) | P95 Improve |",
        "|--------|---------------:|-------------:|------------:|------------:|----------:|------------:|",
    ]
    
    for fam in families:
        naive_med = naive[fam]["id"]["median"]
        naive_p95 = naive[fam]["id"]["p95"]
        fno_med = fno[fam]["id"]["median"]
        fno_p95 = fno[fam]["id"]["p95"]
        
        improve_med = naive_med / fno_med if fno_med > 0 else 0
        improve_p95 = naive_p95 / fno_p95 if fno_p95 > 0 else 0
        
        lines.append(
            f"| {fam} | {naive_med:.4f} | {fno_med:.4f} | {improve_med:.1f}x | "
            f"{naive_p95:.4f} | {fno_p95:.4f} | {improve_p95:.1f}x |"
        )
    
    lines.extend([
        "",
        "**Metric:** Relative L2 error in original space, future region only (with loss_mask).",
        "",
        "**Naive baseline:** Persistence predictor y(t) = y(0) for all t.",
        "",
        "**Interpretation:** FNO improvement factor shows how much better FNO is than simply predicting the initial condition.",
    ])
    
    # Save
    output_dir = ROOT / "reports/baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_table.md", "w") as f:
        f.write("\n".join(lines))
    
    print("\n".join(lines))
    print(f"\n✓ Saved to: {output_dir / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
