"""V05 between/within variance ratio per state_dict key.

Loads all 5 families × 3 seeds (15 checkpoints) and computes:
   between-family variance / within-family variance
per parameter key. High ratio => component carries family-specific signal.
"""
import os
import numpy as np
import torch

FAMS = ["dist_exp_rd_2d", "dist_gamma_rd_2d", "dist_gaussian_rd_2d",
        "dist_powerlaw_rd_2d", "dist_uniform_rd_2d"]
SHORT = {"dist_exp_rd_2d": "exp", "dist_gamma_rd_2d": "gamma",
         "dist_gaussian_rd_2d": "gauss", "dist_powerlaw_rd_2d": "powerlaw",
         "dist_uniform_rd_2d": "uniform"}
SEEDS = [42, 123, 456]
BASE_PT = r"A:/dde research/dde-fno/extracted_lemo_pc/outputs/dist_kernel_v2_p1/raw"

# 15 state dicts
sds = {}
for f in FAMS:
    sds[f] = {}
    for s in SEEDS:
        p = os.path.join(BASE_PT, f, f"clean/lemo_pc_nd/s{s}/best_model.pt")
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            sds[f][s] = {k: v.detach().cpu().numpy().astype(np.complex64 if torch.is_complex(v) else np.float32)
                         for k, v in sd.items() if hasattr(v, "detach")}
        else:
            print(f"MISSING {p}")

# For each key, build a (F, S, D) tensor where D = flattened size.
keys = list(sds[FAMS[0]][SEEDS[0]].keys())
print(f"\n{'key':<55s}  {'shape':<22s}  {'between':>10s}  {'within':>10s}  {'B/W':>9s}  {'B/W (cplx)':>12s}")
results = []
for k in keys:
    arrs = []
    has_all = True
    for f in FAMS:
        row = []
        for s in SEEDS:
            if s not in sds[f]:
                has_all = False; break
            v = sds[f][s][k]
            row.append(v.ravel())
        if not has_all: break
        arrs.append(row)
    if not has_all:
        continue
    A = np.array(arrs)  # (F, S, D)
    if A.dtype == np.complex64 or A.dtype == np.complex128:
        # use abs for variance
        A_real = np.concatenate([A.real, A.imag], axis=-1)
    else:
        A_real = A
    fam_means = A_real.mean(axis=1)                # (F, D)
    grand_mean = A_real.mean(axis=(0, 1))          # (D,)
    # Between-family variance: var across family means, weighted by S
    SS_between = ((fam_means - grand_mean[None]) ** 2).sum() * len(SEEDS)
    # Within-family variance: var of seeds within each family
    SS_within = ((A_real - fam_means[:, None, :]) ** 2).sum()
    df_between = (len(FAMS) - 1) * A_real.shape[-1]
    df_within = len(FAMS) * (len(SEEDS) - 1) * A_real.shape[-1]
    var_b = SS_between / max(df_between, 1)
    var_w = SS_within / max(df_within, 1)
    ratio = var_b / max(var_w, 1e-20)
    results.append((k, A.shape, var_b, var_w, ratio))

# Sort by ratio descending — highest ratio = most family-specific
results.sort(key=lambda r: -r[4])
print("\n" + "=" * 100)
print("RANKED: Between-family / Within-family variance ratio")
print("(High ratio => family-specific. Low (~1) => seed noise dominates.)")
print("=" * 100)
print(f"\n{'key':<55s}  {'shape':<22s}  {'var_btw':>10s}  {'var_wth':>10s}  {'ratio':>10s}")
for k, sh, b, w, r in results:
    print(f"  {k:<55s}  {str(sh):<22s} {b:>10.4e}  {w:>10.4e}  {r:>10.2f}")
