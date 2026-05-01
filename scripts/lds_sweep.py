"""Sweep horizon_steps for LDS metric across families."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from compute_lds import compute_lds_for_family

for family in ("dist_exp", "vdp"):
    print(f"\n=== {family} ===")
    for horizon in (1, 3, 5, 10, 20, 50):
        r = compute_lds_for_family("data_baseline_v2", family, max_samples=2000,
                                    horizon_steps=horizon)
        r2n = r.get("r2_now", float("nan"))
        r2f = r.get("r2_history", float("nan"))
        lds = r.get("lds", float("nan"))
        print(f"  horizon={horizon:3d}: R2_now={r2n:.3f}  R2_full={r2f:.3f}  LDS={lds:.3f}")
