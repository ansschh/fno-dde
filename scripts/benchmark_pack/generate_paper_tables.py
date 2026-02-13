#!/usr/bin/env python3
"""
Generate paper-ready tables for DDE-FNO benchmark.

Tables generated:
1. Benchmark family definitions (Main paper)
2. Dataset protocol & splits (Main paper)
3. Baseline performance across families (Main paper)
4. Baseline model & training settings (Main/Appendix)
5. Time-segment error breakdown (Appendix)
6. OOD generalization gaps (Main paper)

Output: reports/paper_tables/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import json
from typing import Dict, List

FAMILY_ORDER = ["hutch", "linear2", "vdp", "dist_uniform", "dist_exp"]

# Family metadata
FAMILY_INFO = {
    "hutch": {
        "display": "Hutchinson",
        "equation": r"$\dot{x} = r x(t)(1 - x(t-\tau)/K)$",
        "state_dim": 1,
        "delay_type": "discrete",
        "n_delays": 1,
        "notes": "Positive, logistic growth",
    },
    "linear2": {
        "display": "Linear2",
        "equation": r"$\dot{x} = a_1 x(t-\tau_1) + a_2 x(t-\tau_2)$",
        "state_dim": 1,
        "delay_type": "discrete",
        "n_delays": 2,
        "notes": "Stability-sensitive",
    },
    "vdp": {
        "display": "Van der Pol",
        "equation": r"$\ddot{x} - \mu(1-x^2)\dot{x} + x = \gamma x(t-\tau)$",
        "state_dim": 2,
        "delay_type": "discrete",
        "n_delays": 1,
        "notes": "Oscillator, limit cycle",
    },
    "dist_uniform": {
        "display": "DistUniform",
        "equation": r"$\dot{x} = f(x, \frac{1}{\tau}\int_{t-\tau}^t x(s)ds)$",
        "state_dim": 2,
        "delay_type": "distributed",
        "n_delays": 1,
        "notes": "Uniform kernel, aux. ODE",
    },
    "dist_exp": {
        "display": "DistExp",
        "equation": r"$\dot{x} = f(x, \int K_\lambda(t-s) x(s) ds)$",
        "state_dim": 2,
        "delay_type": "distributed",
        "n_delays": 1,
        "notes": r"Exp kernel, $\theta=\lambda\tau \in [0.5, 1.8]$",
    },
}

MODEL_PATHS = {
    "dist_exp": "outputs/baseline_v2/dist_exp_seed42/dist_exp_seed42_20251229_065403",
    "hutch": "outputs/baseline_v1/hutch_seed42_20251228_131919",
    "linear2": "outputs/baseline_v1/linear2_seed42_20251228_142839",
    "vdp": "outputs/baseline_v1/vdp_seed42_20251229_020516",
    "dist_uniform": "outputs/baseline_v1/dist_uniform_seed42_20251229_030851",
}


def load_metrics() -> Dict:
    """Load metrics from the merged JSON."""
    metrics_path = Path("reports/model_viz/all5/baseline_all5_metrics_full.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def generate_table1_family_definitions(output_dir: Path):
    """Table 1: Benchmark family definitions."""
    lines = [
        "# Table 1: DDE Benchmark Family Definitions",
        "",
        "| Family | Equation | State dim | Delay type | # delays | Notes |",
        "|--------|----------|----------:|------------|----------|-------|",
    ]
    
    for family in FAMILY_ORDER:
        info = FAMILY_INFO[family]
        lines.append(
            f"| {info['display']} | {info['equation']} | {info['state_dim']} | "
            f"{info['delay_type']} | {info['n_delays']} | {info['notes']} |"
        )
    
    lines.extend(["", "**Location:** Main paper, Section 4 (Experimental Setup)"])
    
    with open(output_dir / "table1_family_definitions.md", "w") as f:
        f.write("\n".join(lines))
    print("  Saved: table1_family_definitions.md")


def generate_table2_dataset_protocol(output_dir: Path):
    """Table 2: Dataset protocol & splits."""
    lines = [
        "# Table 2: Dataset Protocol & Splits",
        "",
        "| Family | N_train | N_val | N_test | T | dt_out | tau_max | ID tau range | OOD-delay tau | OOD-history | OOD-horizon |",
        "|--------|--------:|------:|-------:|--:|-------:|--------:|--------------|---------------|-------------|-------------|",
    ]
    
    # Common values
    common = {
        "N_train": "8,000",
        "N_val": "1,000", 
        "N_test": "2,000",
        "T": "15.0",
        "dt_out": "~0.06",
        "tau_max": "2.0",
    }
    
    for family in FAMILY_ORDER:
        if family == "linear2":
            tau_range = "[0.1, 2.0] x 2"
            ood_delay = "max(tau) > 1.3"
        else:
            tau_range = "[0.1, 2.0]"
            ood_delay = "tau > 1.3"
        
        lines.append(
            f"| {FAMILY_INFO[family]['display']} | {common['N_train']} | {common['N_val']} | "
            f"{common['N_test']} | {common['T']} | {common['dt_out']} | {common['tau_max']} | "
            f"{tau_range} | {ood_delay} | Spline phi | T=20 |"
        )
    
    lines.extend([
        "",
        "**Notes:**",
        "- All families use 256 output grid points (dt ~ 0.059)",
        "- History grid: 64 points on [-tau_max, 0]",
        "- OOD-history: Spline-interpolated history functions (vs piecewise-linear ID)",
        "- OOD-horizon: Extended prediction horizon T=20 (vs T=15 ID)",
        "",
        "**Location:** Main paper or Appendix",
    ])
    
    with open(output_dir / "table2_dataset_protocol.md", "w") as f:
        f.write("\n".join(lines))
    print("  Saved: table2_dataset_protocol.md")


def generate_table3_baseline_performance(output_dir: Path, metrics: Dict):
    """Table 3: Baseline performance across families."""
    lines = [
        "# Table 3: Baseline FNO Performance (ID Test)",
        "",
        "| Family | Median relL2 | P90 relL2 | P95 relL2 | N samples |",
        "|--------|-------------:|----------:|----------:|----------:|",
    ]
    
    for family in FAMILY_ORDER:
        if family in metrics and metrics[family].get("id"):
            m = metrics[family]["id"]
            lines.append(
                f"| {FAMILY_INFO[family]['display']} | {m['median']:.4f} | "
                f"{m['p90']:.4f} | {m['p95']:.4f} | {m['n_samples']} |"
            )
        else:
            lines.append(f"| {FAMILY_INFO[family]['display']} | — | — | — | — |")
    
    lines.extend([
        "",
        "**Metric:** Relative L2 error in original (denormalized) space, future region only.",
        "",
        "**Location:** Main paper, Section 5 (Results)",
    ])
    
    with open(output_dir / "table3_baseline_performance.md", "w") as f:
        f.write("\n".join(lines))
    print("  Saved: table3_baseline_performance.md")


def generate_table4_model_settings(output_dir: Path):
    """Table 4: Baseline model & training settings."""
    lines = [
        "# Table 4: Baseline Model & Training Settings",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        "| **Architecture** | FNO1d-Residual |",
        "| Fourier modes | 12 |",
        "| Width (hidden dim) | 48 |",
        "| Layers | 3 |",
        "| Activation | GELU |",
        "| Dropout | 0.1 |",
        "| **Training** | |",
        "| Optimizer | AdamW |",
        "| Learning rate | 1e-3 |",
        "| LR scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |",
        "| Weight decay | 1e-4 |",
        "| Batch size | 32 |",
        "| Max epochs | 150 |",
        "| Early stopping | patience=20 |",
        "| **Normalization** | |",
        "| Input/output | Per-channel mean/std from training set |",
        "| Applied to | All splits (train stats only) |",
        "",
        "**Total parameters:** ~93k",
        "",
        "**Location:** Main paper or Appendix",
    ]
    
    with open(output_dir / "table4_model_settings.md", "w") as f:
        f.write("\n".join(lines))
    print("  Saved: table4_model_settings.md")


def generate_table5_ood_gaps(output_dir: Path, metrics: Dict):
    """Table 5: OOD generalization gaps."""
    lines = [
        "# Table 5: OOD Generalization Gaps",
        "",
        "| Family | ID median | OOD-delay | Gap | OOD-history | Gap | OOD-horizon | Gap |",
        "|--------|----------:|----------:|----:|------------:|----:|------------:|----:|",
    ]
    
    for family in FAMILY_ORDER:
        if family not in metrics or not metrics[family].get("id"):
            lines.append(f"| {FAMILY_INFO[family]['display']} | — | — | — | — | — | — | — |")
            continue
        
        id_med = metrics[family]["id"]["median"]
        row = f"| {FAMILY_INFO[family]['display']} | {id_med:.4f}"
        
        for split in ["ood_delay", "ood_history", "ood_horizon"]:
            if metrics[family].get(split):
                ood_med = metrics[family][split]["median"]
                gap = ood_med / (id_med + 1e-10)
                row += f" | {ood_med:.4f} | {gap:.1f}×"
            else:
                row += " | — | —"
        
        row += " |"
        lines.append(row)
    
    lines.extend([
        "",
        "**Gap:** Ratio of OOD median to ID median (higher = worse generalization).",
        "",
        "**Key findings:**",
        "- OOD-delay shows largest gaps (extrapolation to unseen delay values)",
        "- OOD-history moderate gaps (different history function class)",
        "- OOD-horizon often < 1× (shorter relative prediction, easier)",
        "",
        "**Location:** Main paper, Section 5",
    ])
    
    with open(output_dir / "table5_ood_gaps.md", "w") as f:
        f.write("\n".join(lines))
    print("  Saved: table5_ood_gaps.md")


def generate_table6_time_segments(output_dir: Path, metrics: Dict):
    """Table 6: Time-segment error breakdown."""
    lines = [
        "# Table 6: Time-Segment Error Breakdown (ID Test)",
        "",
        "Time segments: Early [0, 3.75], Mid [3.75, 11.25], Late [11.25, 15]",
        "",
        "| Family | Early median | Mid median | Late median | Late/Early ratio |",
        "|--------|-------------:|-----------:|------------:|-----------------:|",
    ]
    
    # Load error curves for time-segment analysis
    for family in FAMILY_ORDER:
        curves_path = Path(f"reports/model_viz/{family}") 
        # Find the run directory
        run_dirs = list(curves_path.glob("*/curves/error_vs_time_id.npz"))
        
        if run_dirs:
            data = np.load(run_dirs[0])
            t = data["t"]
            median = data["median"]
            
            # Segment indices (assuming T=15, 256 points)
            n = len(t)
            early_end = n // 4
            late_start = 3 * n // 4
            
            early_med = np.mean(median[:early_end])
            mid_med = np.mean(median[early_end:late_start])
            late_med = np.mean(median[late_start:])
            ratio = late_med / (early_med + 1e-10)
            
            lines.append(
                f"| {FAMILY_INFO[family]['display']} | {early_med:.4f} | "
                f"{mid_med:.4f} | {late_med:.4f} | {ratio:.2f}× |"
            )
        else:
            lines.append(f"| {FAMILY_INFO[family]['display']} | — | — | — | — |")
    
    lines.extend([
        "",
        "**Interpretation:**",
        "- Late/Early ratio > 1 indicates error drift (accumulation over time)",
        "- Higher ratio = more temporal extrapolation difficulty",
        "",
        "**Location:** Appendix",
    ])
    
    with open(output_dir / "table6_time_segments.md", "w") as f:
        f.write("\n".join(lines))
    print("  Saved: table6_time_segments.md")


def generate_combined_latex(output_dir: Path, metrics: Dict):
    """Generate combined LaTeX table for paper."""
    lines = [
        "% Combined performance table for paper",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Baseline FNO Performance on DDE Benchmark}",
        "\\label{tab:baseline_performance}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Family & State & Delay & ID Med & ID P95 & OOD-delay & OOD-history \\\\",
        "\\midrule",
    ]
    
    for family in FAMILY_ORDER:
        info = FAMILY_INFO[family]
        if family in metrics and metrics[family].get("id"):
            m = metrics[family]["id"]
            id_med = f"{m['median']:.3f}"
            id_p95 = f"{m['p95']:.3f}"
            
            ood_d = "—"
            if metrics[family].get("ood_delay"):
                gap = metrics[family]["ood_delay"]["median"] / (m["median"] + 1e-10)
                ood_d = f"{gap:.1f}×"
            
            ood_h = "—"
            if metrics[family].get("ood_history"):
                gap = metrics[family]["ood_history"]["median"] / (m["median"] + 1e-10)
                ood_h = f"{gap:.1f}×"
        else:
            id_med = id_p95 = ood_d = ood_h = "—"
        
        lines.append(
            f"{info['display']} & {info['state_dim']} & {info['delay_type'][:4]} & "
            f"{id_med} & {id_p95} & {ood_d} & {ood_h} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    with open(output_dir / "table_combined.tex", "w") as f:
        f.write("\n".join(lines))
    print("  Saved: table_combined.tex")


def main():
    output_dir = Path("reports/paper_tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating paper tables to: {output_dir}\n")
    
    # Load metrics
    metrics = load_metrics()
    
    # Generate all tables
    generate_table1_family_definitions(output_dir)
    generate_table2_dataset_protocol(output_dir)
    generate_table3_baseline_performance(output_dir, metrics)
    generate_table4_model_settings(output_dir)
    generate_table5_ood_gaps(output_dir, metrics)
    generate_table6_time_segments(output_dir, metrics)
    generate_combined_latex(output_dir, metrics)
    
    print(f"\n✓ Generated 7 table files in: {output_dir}")


if __name__ == "__main__":
    main()
