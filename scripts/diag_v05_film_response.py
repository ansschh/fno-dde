"""V05 diagnostic 3: per-sample β response and proposed V05 figure.

Q: Does β (additive FiLM bias) actually vary with sample params, or is it
   essentially constant per family?

If FiLM weights are near zero (which they are: norms 0.000-0.142), then β
is dominated by film_net.2.bias and is essentially CONSTANT across samples.
That makes the 'family-specific β' story collapse: β is per-family but the
intra-family variation is tiny — which is consistent with the within-family
variance ratio table showing ratio < 1.

Plot:
  (1) Per-sample β L2 norm vs param-vector L2 norm — slope ≈ 0 means FiLM is dead
  (2) Per-family mean β(t) overlay
  (3) Heatmap of CosSim of K_eff vs GT(family) — does the network's
      effective kernel for sample s of family F actually look like F's GT?
"""
import os
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
TMP = r"A:/dde research/dde-fno/reports/V05_diag"
os.makedirs(TMP, exist_ok=True)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def compute_film(snap, block, params):
    W0 = snap[f"blocks.{block}.A_lag.film_net.0.weight"]
    b0 = snap[f"blocks.{block}.A_lag.film_net.0.bias"]
    W2 = snap[f"blocks.{block}.A_lag.film_net.2.weight"]
    b2 = snap[f"blocks.{block}.A_lag.film_net.2.bias"]
    h = gelu(params @ W0.T + b0[None])
    out = h @ W2.T + b2[None]
    OC, M = 64, 24
    g = out[:, : OC * M].reshape(-1, OC, M)
    b = out[:, OC * M:].reshape(-1, OC, M)
    return g, b


# ---------------------------------------------------------------
# Question: how much does β change across samples within a family?
# ---------------------------------------------------------------
print("Within-family β variation across samples:")
print("=" * 80)
for f in FAMS:
    snap = np.load(os.path.join(BASE, f, "clean/lemo_pc_nd/s42/kernel_snapshot.npz"))
    vp = np.load(os.path.join(BASE, f, "clean/lemo_pc_nd/s42/viz_samples.npz"))
    p_real = vp["input"][:, 0, 0, 0, -3:]                 # (n=4, 3) real samples
    # also probe with synthetic params spanning a wide range
    np.random.seed(0)
    p_synth = np.random.randn(64, 3).astype(np.float32) * 1.5
    p_all = np.concatenate([p_real, p_synth])
    g, b = compute_film(snap, 0, p_all)                   # (n_all, 64, 24)
    b_const = b.mean(axis=0)                               # the "average" β
    devs = b - b_const[None]                               # (n_all, 64, 24)
    # ratio: variation across samples / mean abs of β
    var_per_sample = np.linalg.norm(devs.reshape(devs.shape[0], -1), axis=1)
    mean_norm = np.linalg.norm(b_const)
    print(f"  {SHORT[f]:9s}  |β_const|={mean_norm:.4f}  "
          f"|β-β_const|: mean={var_per_sample.mean():.4f} max={var_per_sample.max():.4f} "
          f"  ratio = {var_per_sample.mean()/max(mean_norm,1e-9):.2%}")

# ---------------------------------------------------------------
# K_eff vs GT pairwise CosSim (does effective kernel match family GT?)
# ---------------------------------------------------------------
print("\nK_eff (γ⊙K + β) time-domain  vs  GT[family]  CosSim  (off-diag = mismatch test):")
print("=" * 80)

K_eff_per_fam = {}
GT_per_fam = {}
for f in FAMS:
    snap = np.load(os.path.join(BASE, f, "clean/lemo_pc_nd/s42/kernel_snapshot.npz"))
    vp = np.load(os.path.join(BASE, f, "clean/lemo_pc_nd/s42/viz_samples.npz"))
    kr = np.load(os.path.join(BASE, f, "clean/lemo_pc_nd/s42/kernel_recovery.npz"))
    p = vp["input"][:, 0, 0, 0, -3:]
    Kr = snap["blocks.0.A_lag.weights__re"]
    Ki = snap["blocks.0.A_lag.weights__im"]
    K = Kr + 1j * Ki                                       # (64,64,24)
    g, b = compute_film(snap, 0, p)
    g_avg = g.mean(0); b_avg = b.mean(0)                   # (64,24) each
    # K_eff in spectral domain. Build per-(out, mode) effective spectral kernel
    # by averaging K across in axis (paper convention) then * γ + β.
    K_avg_in = K.mean(axis=0)                              # (64,24)
    K_eff_spec = K_avg_in * g_avg + b_avg                  # (64,24)
    # irfft to time domain (length 46 to match GT)
    n_time = 46
    n_modes = n_time // 2 + 1
    pad = np.zeros((64, n_modes), dtype=np.complex64)
    pad[..., :min(24, n_modes)] = K_eff_spec[..., :min(24, n_modes)].astype(np.complex64)
    Kt = np.fft.irfft(pad, n=n_time, axis=-1)              # (64, 46)
    K_eff_t = np.mean(np.abs(Kt), axis=0)                  # (46,)
    K_eff_per_fam[f] = K_eff_t
    GT_per_fam[f] = kr["K_gt"]

print(f"{'  ':12s}" + "  ".join(f"GT_{SHORT[g]:>8s}" for g in FAMS))
for fi, f in enumerate(FAMS):
    row = f"{SHORT[f]:9s}  "
    for g in FAMS:
        a = K_eff_per_fam[f]; b = GT_per_fam[g]
        cs = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        row += f"{cs:>10.3f}"
    print("Keff_" + row)

# ---------------------------------------------------------------
# V05 PROPOSAL FIGURE: 3-panel
#   Panel A: K_t_avg_abs (the "collapsed mode-1 sinusoid" plot from the paper)
#   Panel B: GT kernels per family (the truth they're SUPPOSED to recover)
#   Panel C: K_eff = γ⊙K + β  per family (best honest reconstruction)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = plt.cm.tab10(range(5))

# Panel A: K (the published "recovered kernel" plot) - collapsed
ax = axes[0]
for ci, f in enumerate(FAMS):
    kr = np.load(os.path.join(BASE, f, "clean/lemo_pc_nd/s42/kernel_recovery.npz"))
    K = kr["K_t_avg_abs"]
    ax.plot(K / max(K.max(), 1e-9), color=colors[ci], lw=1.5, label=SHORT[f])
ax.set_title("(A) Published K(t) — base lag kernel,\n |K|_avg time-domain (NORMALIZED)\nALL FAMILIES COLLAPSE")
ax.set_xlabel("t (lag)")
ax.legend(fontsize=8, loc="best")

# Panel B: GT kernels per family
ax = axes[1]
for ci, f in enumerate(FAMS):
    kr = np.load(os.path.join(BASE, f, "clean/lemo_pc_nd/s42/kernel_recovery.npz"))
    K = kr["K_gt"]
    ax.plot(K / max(K.max(), 1e-9), color=colors[ci], lw=1.5, label=SHORT[f])
ax.set_title("(B) Ground-truth distributed-delay\nkernel K_GT(t) per family\n(DIVERSE)")
ax.set_xlabel("t (lag)")
ax.legend(fontsize=8, loc="best")

# Panel C: K_eff per family — the honest "what the model actually applies"
ax = axes[2]
for ci, f in enumerate(FAMS):
    K = K_eff_per_fam[f]
    ax.plot(K / max(K.max(), 1e-9), color=colors[ci], lw=1.5, label=SHORT[f])
ax.set_title("(C) Effective kernel K_eff = γ⊙K + β\n(per family, normalized)\nSTILL MOSTLY COLLAPSED")
ax.set_xlabel("t (lag)")
ax.legend(fontsize=8, loc="best")

plt.suptitle("V05 PROPOSAL: LEMO-PC does NOT recover the per-family DDE kernel — published K is a mode-1 artifact,\nFiLM γ≈0 and β-modulation is too weak to differentiate families", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(TMP, "V05_diag_proposal_3panel.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"\nsaved {os.path.join(TMP, 'V05_diag_proposal_3panel.png')}")

# Also: per-family CosSim heatmap of K_eff vs GT (cross-family confusion matrix)
fig, ax = plt.subplots(figsize=(6, 5))
M = np.zeros((5, 5))
for fi, f in enumerate(FAMS):
    for gi, g in enumerate(FAMS):
        a = K_eff_per_fam[f]; b = GT_per_fam[g]
        M[fi, gi] = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[f] for f in FAMS], rotation=45)
ax.set_yticks(range(5)); ax.set_yticklabels([SHORT[f] for f in FAMS])
ax.set_xlabel("GT kernel from family")
ax.set_ylabel("K_eff trained on family")
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                color="white" if M[i, j] < 0.6 else "black")
ax.set_title("V05: K_eff(family) vs GT(family) confusion matrix\nDiagonal does NOT dominate — K_eff is family-agnostic")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(TMP, "V05_diag_confusion.png"), dpi=120)
plt.close()
print(f"saved {os.path.join(TMP, 'V05_diag_confusion.png')}")
