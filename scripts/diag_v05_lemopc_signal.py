"""V05 diagnostic: where does LEMO-PC's family-specific signal live?

Components inspected:
  1. K     - base spectral lag kernel (in, out, M)  cfloat
  2. β     - additive FiLM bias (per-sample, per-(out, mode))
  3. γ     - multiplicative FiLM gain
  4. K_eff - γ⊙K + β (per-sample)
  5. A_spat - spatial FNO weights (in, out, kx, ky) cfloat
  6. film_net.0/2 weights and biases
  7. best_model.pt - 1×1 channel-mix B and lift/heads/norms

For each component we compute:
  - per-family time-domain "kernel signature" where appropriate
  - pairwise CosSim across the 5 families
  - between-family / within-family variance ratio (when seeds available)
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMS = ["dist_exp_rd_2d", "dist_gamma_rd_2d", "dist_gaussian_rd_2d",
        "dist_powerlaw_rd_2d", "dist_uniform_rd_2d"]
SHORT = {"dist_exp_rd_2d": "exp", "dist_gamma_rd_2d": "gamma",
         "dist_gaussian_rd_2d": "gauss", "dist_powerlaw_rd_2d": "powerlaw",
         "dist_uniform_rd_2d": "uniform"}
BASE = r"A:/dde research/dde-fno/extracted/pod1/outputs/dist_kernel_v2_p1/raw"
BASE_PT = r"A:/dde research/dde-fno/extracted_lemo_pc/outputs/dist_kernel_v2_p1/raw"
OUT_DIR = r"A:/dde research/dde-fno/reports/V05_diag"
os.makedirs(OUT_DIR, exist_ok=True)
TMP = OUT_DIR  # save plots here (writable on Windows)


def cossim(a, b, eps=1e-12):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        # use complex inner product
        num = np.real(np.vdot(a, b))
        den = np.linalg.norm(a) * np.linalg.norm(b)
    else:
        num = float(a @ b)
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / max(den, eps)


def pairwise_cossim(vecs):
    n = len(vecs)
    M = np.eye(n)
    for i in range(n):
        for j in range(n):
            M[i, j] = cossim(vecs[i], vecs[j])
    return M


def load_snap(fam, seed=42):
    p = os.path.join(BASE, fam, f"clean/lemo_pc_nd/s{seed}/kernel_snapshot.npz")
    return np.load(p)


def compute_film(snap, block, params):
    """Run film_net forward to get gamma, beta for given params (B, 3)."""
    W0 = snap[f"blocks.{block}.A_lag.film_net.0.weight"]  # (64, 3)
    b0 = snap[f"blocks.{block}.A_lag.film_net.0.bias"]    # (64,)
    W2 = snap[f"blocks.{block}.A_lag.film_net.2.weight"]  # (3072, 64)
    b2 = snap[f"blocks.{block}.A_lag.film_net.2.bias"]    # (3072,)
    h = params @ W0.T + b0[None]                          # (B, 64)
    # GELU
    h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
    out = h @ W2.T + b2[None]                             # (B, 3072)
    OC, M = 64, 24
    gamma = out[:, : OC * M].reshape(-1, OC, M)
    beta = out[:, OC * M:].reshape(-1, OC, M)
    return gamma, beta


def k_to_time(K_complex, n_time=46):
    """K shape (in, out, M) cfloat. Returns avg |K(t)| of shape (n_time,) by
    irfft. Here we follow `kernel_recovery.npz`'s convention."""
    in_, out_, M = K_complex.shape
    # zero-pad to (n_time//2 + 1) modes
    n_modes = n_time // 2 + 1
    pad = np.zeros((in_, out_, n_modes), dtype=K_complex.dtype)
    use_M = min(M, n_modes)
    pad[..., :use_M] = K_complex[..., :use_M]
    Kt = np.fft.irfft(pad, n=n_time, axis=-1)  # (in, out, n_time)
    return np.mean(np.abs(Kt), axis=(0, 1))   # (n_time,)


def beta_to_time(beta, n_time=46):
    """beta shape (B, OC=64, M=24). Returns time-domain shape per sample, then
    avg |β(t)| across (B, OC). Convention: pad to n_modes and irfft."""
    B, OC, M = beta.shape
    n_modes = n_time // 2 + 1
    use_M = min(M, n_modes)
    # treat beta as real -> complex by Re=beta, Im=0 (matches what active gets
    # added in the model code)
    pad = np.zeros((B, OC, n_modes), dtype=np.complex64)
    pad[..., :use_M] = beta[..., :use_M].astype(np.complex64)
    bt = np.fft.irfft(pad, n=n_time, axis=-1)
    return np.mean(np.abs(bt), axis=(0, 1)), bt


# -------------------------------------------------------------------------
# 1. Load all 5 families, all 3 blocks, plus param vectors per family
# -------------------------------------------------------------------------
print("=" * 70)
print("Loading 5 families' kernel_snapshot.npz + viz_samples params")
print("=" * 70)

snaps = {f: load_snap(f, 42) for f in FAMS}

# Get param vectors per family (mean of viz_samples params)
params_per_fam = {}
for f in FAMS:
    vp = os.path.join(BASE, f, "clean/lemo_pc_nd/s42/viz_samples.npz")
    d = np.load(vp)
    p = d["input"][:, 0, 0, 0, -3:]  # (4, 3)
    params_per_fam[f] = p
    print(f"  {SHORT[f]:9s} params (n={p.shape[0]}): mean={p.mean(0)} std={p.std(0)}")

# -------------------------------------------------------------------------
# 2. K (base lag kernel) — across families
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Component 1: BASE LAG KERNEL K  (in, out, M)")
print("=" * 70)
K_per_fam_blk = {}
for f in FAMS:
    snap = snaps[f]
    K_per_fam_blk[f] = []
    for blk in range(3):
        Kr = snap[f"blocks.{blk}.A_lag.weights__re"]
        Ki = snap[f"blocks.{blk}.A_lag.weights__im"]
        K_per_fam_blk[f].append(Kr + 1j * Ki)
print("\nK[block 0] L2 norm per family:")
for f in FAMS:
    print(f"  {SHORT[f]:9s}  {np.linalg.norm(K_per_fam_blk[f][0]):.4f}")
print("\nPairwise CosSim of K[block 0] flattened:")
M_K = pairwise_cossim([K_per_fam_blk[f][0] for f in FAMS])
print("            " + "  ".join(f"{SHORT[f]:>9s}" for f in FAMS))
for i, f in enumerate(FAMS):
    print(f"  {SHORT[f]:9s} " + "  ".join(f"{M_K[i,j]:9.4f}" for j in range(len(FAMS))))

# -------------------------------------------------------------------------
# 3. β (additive FiLM bias) computed at REAL family params
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Component 2: β (additive FiLM bias) at family-specific params")
print("=" * 70)

beta_per_fam = {f: [] for f in FAMS}
gamma_per_fam = {f: [] for f in FAMS}
beta_t_per_fam = {f: [] for f in FAMS}   # time-domain |β(t)| per block

for f in FAMS:
    p = params_per_fam[f]  # (n, 3)
    snap = snaps[f]
    for blk in range(3):
        g, b = compute_film(snap, blk, p)  # (n, 64, 24) each
        gamma_per_fam[f].append(g)
        beta_per_fam[f].append(b)
        b_t_avg, _ = beta_to_time(b)
        beta_t_per_fam[f].append(b_t_avg)

print("\n|β| stats per family (block 0, mean across n samples):")
for f in FAMS:
    b = beta_per_fam[f][0]
    print(f"  {SHORT[f]:9s}  mean|β|={np.mean(np.abs(b)):.5f}  max|β|={np.max(np.abs(b)):.5f}  "
          f"|β|/|K|≈{np.linalg.norm(b)/np.linalg.norm(K_per_fam_blk[f][0]):.3f}")

print("\n|γ| stats per family (block 0, mean across n samples):")
for f in FAMS:
    g = gamma_per_fam[f][0]
    print(f"  {SHORT[f]:9s}  mean γ={np.mean(g):+.5f}  std γ={np.std(g):.5f}  "
          f"mean|γ|={np.mean(np.abs(g)):.5f}")

# -------------------------------------------------------------------------
# Pairwise CosSim of β across families (mean across samples per family)
# -------------------------------------------------------------------------
print("\nPairwise CosSim of mean β per family (block 0):")
beta_means_blk0 = [beta_per_fam[f][0].mean(0) for f in FAMS]   # each (64, 24)
M_beta = pairwise_cossim(beta_means_blk0)
print("            " + "  ".join(f"{SHORT[f]:>9s}" for f in FAMS))
for i, f in enumerate(FAMS):
    print(f"  {SHORT[f]:9s} " + "  ".join(f"{M_beta[i,j]:9.4f}" for j in range(len(FAMS))))

print("\nPairwise CosSim of mean β per family (block 1):")
beta_means_blk1 = [beta_per_fam[f][1].mean(0) for f in FAMS]
M_beta_b1 = pairwise_cossim(beta_means_blk1)
for i, f in enumerate(FAMS):
    print(f"  {SHORT[f]:9s} " + "  ".join(f"{M_beta_b1[i,j]:9.4f}" for j in range(len(FAMS))))

print("\nPairwise CosSim of mean β per family (block 2):")
beta_means_blk2 = [beta_per_fam[f][2].mean(0) for f in FAMS]
M_beta_b2 = pairwise_cossim(beta_means_blk2)
for i, f in enumerate(FAMS):
    print(f"  {SHORT[f]:9s} " + "  ".join(f"{M_beta_b2[i,j]:9.4f}" for j in range(len(FAMS))))

# -------------------------------------------------------------------------
# Pairwise CosSim of time-domain |β(t)| per family
# -------------------------------------------------------------------------
print("\nPairwise CosSim of time-domain |β(t)| (block 0, irfft to t):")
M_betat = pairwise_cossim([beta_t_per_fam[f][0] for f in FAMS])
print("            " + "  ".join(f"{SHORT[f]:>9s}" for f in FAMS))
for i, f in enumerate(FAMS):
    print(f"  {SHORT[f]:9s} " + "  ".join(f"{M_betat[i,j]:9.4f}" for j in range(len(FAMS))))

# -------------------------------------------------------------------------
# 4. K_eff = γ * K + β  (per-sample) - compare time-domain shape across fams
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Component 3: K_eff = γ⊙K + β (per-sample effective lag kernel)")
print("=" * 70)

# Use mean over n samples per family. K_eff has shape (in=64, out=64, M=24).
# γ, β are (n, OC=64, M=24). K is (in, out, M). γ broadcasts across `in`, β
# is added per (out, M) — but in the model β is broadcast across `in` too in
# the spectral domain (each out channel's modes get the same shift regardless
# of which in-channel contributed). Actually β is added to active = K @ x_hat,
# so β shape is (B, OC, M) with no in-axis. For comparing magnitudes we form
# K_mod = K * mean_gamma  (per (out, M) gain)  then |β_eff(t)| separately.
K_eff_t_per_fam = {}
for f in FAMS:
    K = K_per_fam_blk[f][0]  # (64, 64, 24)
    g = gamma_per_fam[f][0].mean(0)  # (64, 24) avg γ across n samples
    b = beta_per_fam[f][0].mean(0)   # (64, 24)
    # γ broadcasts to (in, out, M) by replicating across in axis
    K_mod = K * g[None, :, :]
    # add β contribution: the additive part isn't a function of `in`, so we
    # treat the effective per-output-channel "mode response" as
    # diag-by-in projection. Simpler: report the time-domain shape by
    # averaging across (in,out) of K_mod and treat β as an additive offset to
    # OC's spectrum.
    Kt_mod = k_to_time(K_mod, n_time=46)
    bt_avg, _ = beta_to_time(beta_per_fam[f][0])
    K_eff_t_per_fam[f] = (Kt_mod, bt_avg, Kt_mod + bt_avg)

print("\nPairwise CosSim of avg time-domain (γ⊙K + β)(t)  [block 0]:")
M_keff = pairwise_cossim([K_eff_t_per_fam[f][2] for f in FAMS])
print("            " + "  ".join(f"{SHORT[f]:>9s}" for f in FAMS))
for i, f in enumerate(FAMS):
    print(f"  {SHORT[f]:9s} " + "  ".join(f"{M_keff[i,j]:9.4f}" for j in range(len(FAMS))))

# -------------------------------------------------------------------------
# 5. Spatial FNO weights — A_spat
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Component 4: A_spat (spatial FNO spectral weights)")
print("=" * 70)
A_spat_per_fam = {f: [] for f in FAMS}
for f in FAMS:
    snap = snaps[f]
    for blk in range(3):
        Ar = snap[f"blocks.{blk}.A_spat.weights__re"]
        Ai = snap[f"blocks.{blk}.A_spat.weights__im"]
        A_spat_per_fam[f].append(Ar + 1j * Ai)

print("\nA_spat[block 0] Frobenius norm:")
for f in FAMS:
    print(f"  {SHORT[f]:9s}  {np.linalg.norm(A_spat_per_fam[f][0]):.4f}")
print("\nPairwise CosSim of A_spat[block 0] flattened:")
M_aspat = pairwise_cossim([A_spat_per_fam[f][0] for f in FAMS])
print("            " + "  ".join(f"{SHORT[f]:>9s}" for f in FAMS))
for i, f in enumerate(FAMS):
    print(f"  {SHORT[f]:9s} " + "  ".join(f"{M_aspat[i,j]:9.4f}" for j in range(len(FAMS))))

# -------------------------------------------------------------------------
# 6. FiLM net weights themselves (W0, b0, W2, b2)
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Component 5: FiLM net weights (the per-family parameter encoder)")
print("=" * 70)

for which in ["weight", "bias"]:
    for layer in [0, 2]:
        key = f"blocks.0.A_lag.film_net.{layer}.{which}"
        vecs = [snaps[f][key] for f in FAMS]
        norms = [np.linalg.norm(v) for v in vecs]
        M_w = pairwise_cossim(vecs)
        offdiag = M_w[~np.eye(5, dtype=bool)]
        print(f"\n {key}: shape={vecs[0].shape} | norms={[f'{n:.3f}' for n in norms]}")
        print(f"   off-diag CosSim range: [{offdiag.min():.4f}, {offdiag.max():.4f}]  mean={offdiag.mean():.4f}")

# -------------------------------------------------------------------------
# 7. Full state_dict from best_model.pt
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Component 6: Full state_dict from best_model.pt (incl. 1×1 conv B)")
print("=" * 70)
try:
    import torch
    sd_per_fam = {}
    for f in FAMS:
        pt = os.path.join(BASE_PT, f, "clean/lemo_pc_nd/s42/best_model.pt")
        if os.path.exists(pt):
            ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            sd_per_fam[f] = {k: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                             for k, v in sd.items() if hasattr(v, "shape") or hasattr(v, "detach")}
        else:
            print(f"  {f}: best_model.pt not found at {pt}")

    if sd_per_fam:
        print("\nstate_dict keys (from gauss):")
        for k, v in sd_per_fam[FAMS[2]].items():
            print(f"  {k:55s} {tuple(v.shape)} {v.dtype}")

        # Pairwise CosSim of every key across families
        print("\nPairwise CosSim across 5 families per state_dict key (off-diag mean / range):")
        rows = []
        for k in sd_per_fam[FAMS[0]].keys():
            try:
                vs = [sd_per_fam[f][k] for f in FAMS if k in sd_per_fam[f]]
                if len(vs) < 5: continue
                if vs[0].size < 4: continue
                M_k = pairwise_cossim(vs)
                offdiag = M_k[~np.eye(5, dtype=bool)]
                rows.append((k, vs[0].shape, float(offdiag.mean()),
                             float(offdiag.min()), float(offdiag.max()),
                             float(np.mean([np.linalg.norm(v) for v in vs]))))
            except Exception as e:
                pass

        # Sort by mean cossim ascending — lower => more family-specific
        rows.sort(key=lambda r: r[2])
        print(f"\n{'key':<55s}  {'shape':<22s} {'cs_mean':>9s}  {'cs_min':>8s}  {'cs_max':>8s}  {'avg|w|':>9s}")
        for k, sh, m, lo, hi, n in rows:
            print(f"  {k:<55s}  {str(sh):<22s} {m:>9.4f}  {lo:>8.4f}  {hi:>8.4f}  {n:>9.3f}")
except Exception as e:
    print(f"  ERROR loading state_dict: {e}")
    import traceback; traceback.print_exc()

# -------------------------------------------------------------------------
# 8. Plots
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Plots")
print("=" * 70)

# Plot 1: Heatmap of pairwise CosSim per component (block 0)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
mats = [
    ("Base K (block 0)", M_K),
    ("β [time-dom] (block 0)", M_betat),
    ("γ⊙K + β [time-dom]", M_keff),
    ("A_spat (block 0)", M_aspat),
]
for ax, (name, M) in zip(axes, mats):
    im = ax.imshow(M, vmin=-0.2, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[f] for f in FAMS], rotation=45)
    ax.set_yticks(range(5)); ax.set_yticklabels([SHORT[f] for f in FAMS])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.6 else "black", fontsize=8)
    ax.set_title(name)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("V05: Pairwise family CosSim — which component carries family-specific signal?")
plt.tight_layout()
plt.savefig(os.path.join(TMP, "V05_diag_cossim_heatmaps.png"), dpi=120)
plt.close()
print(f"  saved {os.path.join(TMP, 'V05_diag_cossim_heatmaps.png')}")

# Plot 2: Time-domain |β(t)| per family (block 0,1,2) overlayed with GT
try:
    kr_path_per_fam = {f: os.path.join(BASE, f, "clean/lemo_pc_nd/s42/kernel_recovery.npz")
                       for f in FAMS}
    kr = {f: np.load(p) for f, p in kr_path_per_fam.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = plt.cm.tab10(range(5))
    for blk, ax in enumerate(axes):
        for ci, f in enumerate(FAMS):
            ax.plot(beta_t_per_fam[f][blk], color=colors[ci],
                    label=f"{SHORT[f]} |β(t)|", lw=1.5)
        # GT (same shape used for all families's K)
        for ci, f in enumerate(FAMS):
            ax.plot(kr[f]["K_gt"] / np.max(np.abs(kr[f]["K_gt"])) * np.max(beta_t_per_fam[f][blk]),
                    color=colors[ci], ls="--", lw=0.7, alpha=0.5)
        ax.set_title(f"Block {blk}: |β(t)| per family (solid)  vs  GT shape (dashed, rescaled)")
        ax.set_xlabel("t (lag)")
        if blk == 0: ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(TMP, "V05_diag_beta_time_per_family.png"), dpi=120)
    plt.close()
    print(f"  saved {os.path.join(TMP, 'V05_diag_beta_time_per_family.png')}")
except Exception as e:
    print(f"  beta plot error: {e}")

# Plot 3: γ * K + β  effective time-domain kernel per family vs GT
try:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for ci, f in enumerate(FAMS):
        Kt_mod, bt_avg, eff = K_eff_t_per_fam[f]
        ax.plot(eff, color=colors[ci], lw=1.8, label=f"{SHORT[f]} γK+β")
        gt = kr[f]["K_gt"]
        gtn = gt * np.max(eff) / max(np.max(np.abs(gt)), 1e-12)
        ax.plot(gtn, color=colors[ci], ls="--", lw=0.9, alpha=0.6)
    ax.set_title("Block 0 effective kernel: γ⊙K + β  (per family)  vs GT (dashed)")
    ax.set_xlabel("t (lag)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(TMP, "V05_diag_keff_per_family.png"), dpi=120)
    plt.close()
    print(f"  saved {os.path.join(TMP, 'V05_diag_keff_per_family.png')}")
except Exception as e:
    print(f"  keff plot error: {e}")

print("\nDONE")
