"""Print FiLM diagnostic stats from validation runs."""
import json, sys
from pathlib import Path

fams = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/film_fix_val")
print(f"{'family':<22} {'γ.std':>8} {'β.std':>8} {'γ.mean':>8} {'input_n':>8} {'test_L2':>8}")
print("-" * 70)
for fam in fams:
    p = out_root / fam / "clean" / "lemo_pc_nd" / "s42" / "history.json"
    if not p.exists():
        print(f"{fam:<22} (no history.json)")
        continue
    h = json.load(open(p))
    g_std = h.get("film_gamma_std_pre_tanh", [0.0])[-1]
    b_std = h.get("film_beta_std_pre_tanh", [0.0])[-1]
    g_mean = h.get("film_gamma_mean_pre_tanh", [0.0])[-1]
    in_n = h.get("film_input_norm", [0.0])[-1]
    t_l2 = h.get("test_rel_l2", 0.0)
    print(f"{fam:<22} {g_std:>8.4f} {b_std:>8.4f} {g_mean:>8.4f} {in_n:>8.4f} {t_l2:>8.4f}")
