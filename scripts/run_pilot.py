#!/usr/bin/env python3
"""
CLI wrapper for the pilot protocol.

Runs a small-scale experiment (generate → train → evaluate) for a DDE/PDE
family to validate it before committing to large-scale runs.

Usage:
    python scripts/run_pilot.py neutral
    python scripts/run_pilot.py burgers --n_train 800 --epochs 30
    python scripts/run_pilot.py --all-dde
    python scripts/run_pilot.py --all-pde
    python scripts/run_pilot.py --all
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path
_project_root = Path(__file__).resolve().parent.parent / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def get_all_families(category: str = "all"):
    """Return a list of family names."""
    families = []
    if category in ("all", "dde"):
        try:
            from dde.families import DDE_FAMILIES
            families.extend(DDE_FAMILIES.keys())
        except ImportError:
            pass
    if category in ("all", "pde"):
        try:
            from pde.families import PDE_FAMILIES, _import_all_families
            _import_all_families()
            families.extend(PDE_FAMILIES.keys())
        except ImportError:
            pass
    return families


def main():
    parser = argparse.ArgumentParser(
        description="Run pilot experiments for DDE/PDE families.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "families", nargs="*", default=[],
        help="Family names to pilot (e.g., neutral burgers).",
    )
    parser.add_argument("--all", action="store_true", help="Pilot all families.")
    parser.add_argument("--all-dde", action="store_true", help="Pilot all DDE families.")
    parser.add_argument("--all-pde", action="store_true", help="Pilot all PDE families.")
    parser.add_argument("--n_train", type=int, default=1600, help="Training samples (default: 1600).")
    parser.add_argument("--n_val", type=int, default=200, help="Validation samples (default: 200).")
    parser.add_argument("--n_test", type=int, default=200, help="Test samples (default: 200).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--data_dir", default=None, help="Data dir (temp if not set).")
    parser.add_argument("--output_dir", default=None, help="Output dir (temp if not set).")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep temp directories.")
    parser.add_argument("--save-results", default=None,
                        help="Path to save results JSON (e.g., reports/pilot_results.json).")
    args = parser.parse_args()

    # Determine families to run
    families = list(args.families)
    if args.all:
        families = get_all_families("all")
    elif args.all_dde:
        families.extend(get_all_families("dde"))
    elif args.all_pde:
        families.extend(get_all_families("pde"))

    if not families:
        parser.error("Specify at least one family or use --all / --all-dde / --all-pde.")

    # Remove duplicates preserving order
    seen = set()
    unique = []
    for f in families:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    families = unique

    from utils.pilot import run_pilot

    all_results = {}
    for family in families:
        print(f"\n{'='*60}")
        print(f"  PILOT: {family}")
        print(f"{'='*60}")

        result = run_pilot(
            family=family,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            cleanup=not args.no_cleanup,
        )
        all_results[family] = result

    # Summary table
    print(f"\n\n{'='*80}")
    print("PILOT RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Family':<20} {'Status':<10} {'QC%':<8} {'relL2 med':<12} {'relL2 p95':<12} {'Verdict'}")
    print(f"{'-'*80}")

    for fam, res in all_results.items():
        status = "PASS" if res.get("success") else "FAIL"
        qc = f"{res.get('qc_pass_rate', 0):.0%}" if res.get("qc_pass_rate") else "N/A"
        med = f"{res.get('test_rel_l2_median', -1):.4f}" if res.get("test_rel_l2_median") else "N/A"
        p95 = f"{res.get('test_rel_l2_p95', -1):.4f}" if res.get("test_rel_l2_p95") else "N/A"
        verdict = res.get("message", "")[:30]
        print(f"{fam:<20} {status:<10} {qc:<8} {med:<12} {p95:<12} {verdict}")

    print(f"{'='*80}")

    # Save results
    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to Python types for JSON serialization
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        with open(out_path, "w") as f:
            json.dump(_clean(all_results), f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
