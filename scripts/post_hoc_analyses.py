"""Post-hoc analyses on existing checkpoints (no retraining required).

Runs five analyses that directly back paper claims:
  1. Kernel recovery   — extract spectral kernel from LEMO_PC ckpt, FFT to time
                          domain, compare to ground-truth distributed-delay kernel.
                          (Tests T1-equivariance + recovery of true delay structure.)
  2. Cross-family transfer — each ckpt evaluated on every other family's test
                              shard.  Measures inductive bias across delay families.
  3. Adversarial FGSM — gradient-based input perturbation on test inputs;
                         measure rel-L2 vs perturbation magnitude.
                         Tests the σ-bound's robustness claim.
  4. Noise sweep      — eval at increasing input noise levels {0, 0.05, 0.1, 0.2, 0.5}.
  5. Single-delay 2D   — same per-frame + rollout treatment on existing
                         single-delay (mackey_glass_2d) ckpts (subset).

Usage:
    python3 scripts/post_hoc_analyses.py \\
        --layer_root outputs/dist_kernel_v2_p1 \\
        --data_dir data_dde_pde \\
        [--analyses kernel,transfer,fgsm,noise]   # comma list, default = all
        [--device cuda]                            # default = auto

Outputs (per cell directory):
  - kernel_recovery.npz
  - cross_family_relL2.json   (one per ckpt; aggregated to a single matrix later)
  - adv_fgsm.npz
  - noise_sweep.json

Aggregator (run after per-cell pass):
  python3 scripts/post_hoc_analyses.py --aggregate --layer_root <root>
    -> writes paper/figures/F09_kernel_recovery.{pdf,png}
    -> writes paper/figures/F10_cross_family.{pdf,png}
    -> writes paper/figures/F11_adversarial.{pdf,png}
    -> writes paper/figures/F12_noise_sweep.{pdf,png}
    -> writes paper/tables/T06_kernel_recovery_relL2.tex
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
        "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]


# -------------------- shared loader --------------------

def load_cell(ckpt_path: Path, data_dir: str, family: str, device: str):
    """Load model + matching test/train loaders.

    Reads regime, noise_std, downsample_factor, residual_anchor from the
    saved training config (with safe fallbacks for older ckpts) so the
    eval-time loader matches the training-time loader.  Mismatches here
    silently produce wrong relL2 numbers."""
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    parts = ckpt_path.parts
    regime_from_path = parts[-4] if len(parts) >= 4 else "clean"
    ra = bool(cfg.get("residual_anchor", False))
    regime = cfg.get("regime", regime_from_path)
    noise_std = float(cfg.get("noise_std", 0.05))
    downsample_factor = int(cfg.get("downsample_factor", 2))
    train_loader, _, test_loader = create_apebench_dataloaders(
        data_dir, family, batch_size=4,
        regime=regime, noise_std=noise_std, downsample_factor=downsample_factor,
        residual_anchor=ra, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, test_loader, train_loader, cfg


# -------------------- 1. kernel recovery --------------------

def kernel_recovery(model, family: str, device: str) -> dict:
    """Extract LEMO_PC spectral kernel, FFT to time domain, return arrays.

    The model's first FiLMLagSpectralND layer holds the (in, out, modes) complex
    weights.  We compute time-domain kernel K_t = irfft(K_f) then take the
    absolute amplitude across (in, out) channels averaged: |K_t(t)|.

    The ground-truth distributed-delay kernel for each family has known shape
    (e.g. exponential, gaussian, etc.).  We return both for comparison.
    """
    out = {}
    # Find first complex spectral kernel.
    K = None
    for mod in model.modules():
        w = getattr(mod, "weights", None)
        if w is None or not torch.is_complex(w) or w.dim() != 3:
            continue
        K = w.detach().cpu().numpy()
        break
    if K is None:
        return {}
    # Spectral -> time domain.
    # K shape: (in_ch, out_ch, modes).  Pad to L=2*(modes-1) → real-valued kernel.
    in_ch, out_ch, m = K.shape
    L = 2 * (m - 1) if m > 1 else m
    # Use rfft inverse to recover real kernel.
    K_t = np.fft.irfft(K, n=L, axis=-1)            # (in, out, L)
    # Average abs amplitude across channel pairs.
    K_amp = np.abs(K_t).mean(axis=(0, 1))           # (L,)
    # Also keep the channel-channel max so we can plot heatmap.
    K_amp_per_pair = np.abs(K_t).max(axis=-1)       # (in, out)
    out["K_t_avg_abs"] = K_amp.astype(np.float32)
    out["K_amp_per_pair"] = K_amp_per_pair.astype(np.float32)
    out["family"] = family
    # Synthetic ground-truth distributed kernels (normalized to unit area):
    t = np.arange(L) / max(L - 1, 1)
    if family.startswith("dist_exp"):
        gt = np.exp(-3 * t)  # rate 3
    elif family.startswith("dist_gaussian"):
        gt = np.exp(-((t - 0.3) ** 2) / 0.05)
    elif family.startswith("dist_gamma"):
        gt = (t ** 1.5) * np.exp(-3 * t)
    elif family.startswith("dist_uniform"):
        gt = (t < 0.5).astype(np.float32)
    elif family.startswith("dist_powerlaw"):
        gt = (t + 0.05) ** (-1.2)
    else:
        gt = np.zeros_like(t)
    if gt.sum() > 0:
        gt = gt / gt.sum()
    out["K_gt"] = gt.astype(np.float32)
    # Cosine similarity between learned and GT.
    a = K_amp / (np.linalg.norm(K_amp) + 1e-12)
    b = gt / (np.linalg.norm(gt) + 1e-12)
    out["cosine_similarity"] = float((a * b).sum())
    return out


# -------------------- 2. cross-family transfer --------------------

def _read_manifest(data_dir: str, family: str) -> dict:
    """Returns manifest dict or {} if missing."""
    p = Path(data_dir) / family / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _fingerprint_family(data_dir: str, family: str) -> dict:
    """Shape fingerprint that the model's lift / output / params layers
    depend on.  Two families are transfer-compatible iff their fingerprints
    are equal."""
    m = _read_manifest(data_dir, family)
    return {
        "spatial_shape":  tuple(m.get("spatial_shape", ())),
        "spatial_dims":   int(m.get("spatial_dims", -1)),
        "num_channels":   int(m.get("num_channels", -1)),
        "n_hist":         int(m.get("n_hist", -1)),
        "n_out":          int(m.get("n_out", -1)),
        "params_dim":     int(m.get("params_dim", -1)),
    }


def cross_family_transfer(model, ckpt_family: str, data_dir: str,
                           device: str, n_max_samples: int = 64,
                           residual_anchor: bool = True) -> dict:
    """Evaluate ckpt on every dist-kernel family's test shard whose data
    fingerprint matches the source family.  Skips incompatible families
    (different spatial shape, channel count, n_hist, n_out, params_dim);
    silently passing mismatched-shape input to the model would either
    crash with a dim error or — worse — produce silently-wrong predictions
    when shapes happen to coincide.

    `residual_anchor` MUST match the training-time setting of the source
    ckpt — the model was trained on residual-anchored input and will
    behave incorrectly on raw input.  Default True (matches v2/sigma).

    Returns dict with rel_l2 per compatible target family + a 'skipped'
    list documenting incompatible families and the reason.
    """
    from datasets.apebench_dataset import create_apebench_dataloaders
    src_fp = _fingerprint_family(data_dir, ckpt_family)
    out = {"ckpt_family": ckpt_family, "rel_l2": {}, "skipped": [],
           "source_fingerprint": src_fp}
    for tgt_fam in FAMS:
        if (Path(data_dir) / tgt_fam / "manifest.json").exists() is False:
            out["skipped"].append({"family": tgt_fam, "reason": "no manifest"})
            continue
        tgt_fp = _fingerprint_family(data_dir, tgt_fam)
        if tgt_fam != ckpt_family and tgt_fp != src_fp:
            mismatches = {k: (src_fp[k], tgt_fp[k]) for k in src_fp
                          if src_fp[k] != tgt_fp[k]}
            out["skipped"].append({"family": tgt_fam,
                                    "reason": "fingerprint mismatch",
                                    "diff": mismatches})
            continue
        try:
            _, _, test_loader = create_apebench_dataloaders(
                data_dir, tgt_fam, batch_size=8, residual_anchor=residual_anchor, seed=42)
        except Exception as e:
            out["skipped"].append({"family": tgt_fam,
                                    "reason": f"loader error: {e}"})
            continue
        model.eval()
        rels = []
        seen = 0
        try:
            with torch.no_grad():
                for batch in test_loader:
                    if seen >= n_max_samples:
                        break
                    x = batch["input"].to(device).float()
                    y = batch["target"].to(device).float()
                    mask = batch["loss_mask"].to(device).float()
                    # Belt-and-suspenders: confirm input channel count matches
                    # what the model expects (model.lift.in_features for LEMO).
                    expected_in = getattr(getattr(model, "lift", None),
                                            "in_features", None)
                    if expected_in is not None and x.shape[-1] != expected_in:
                        raise ValueError(
                            f"input channels {x.shape[-1]} != model.lift.in_features {expected_in}")
                    yhat = model(x)
                    n_spatial = y.dim() - 3
                    mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
                    diff_sq = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim())))
                    tgt_sq = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))) + 1e-12
                    r = torch.sqrt(diff_sq / tgt_sq).cpu().numpy()
                    rels.extend(r.tolist())
                    seen += y.shape[0]
        except Exception as e:
            out["skipped"].append({"family": tgt_fam,
                                    "reason": f"forward error: {e}"})
            continue
        if rels:
            out["rel_l2"][tgt_fam] = float(np.mean(rels))
    return out


# -------------------- 3. adversarial FGSM --------------------

def adversarial_fgsm(model, test_loader, device,
                     epsilons=(0.0, 0.001, 0.005, 0.01, 0.02, 0.05),
                     n_max_samples: int = 16) -> dict:
    """Gradient-based FGSM perturbation on input state channels (only).

    For each ε, perturbs input by ε * sign(grad of loss wrt input) and
    measures rel-L2 on the test set.  Tests if σ-bounded model resists
    perturbation linearly in ε."""
    out = {"epsilons": list(epsilons), "rel_l2_per_eps": []}
    for eps in epsilons:
        rels = []
        seen = 0
        for batch in test_loader:
            if seen >= n_max_samples:
                break
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            n_spatial = y.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            n_state_ch = y.shape[-1]
            # Perturb only state channels of x.
            x_state = x[..., :n_state_ch].detach().clone().requires_grad_(True)
            x_aux = x[..., n_state_ch:].detach()
            x_full = torch.cat([x_state, x_aux], dim=-1)
            yhat = model(x_full)
            diff_sq = ((yhat - y) ** 2 * mask_bc).sum()
            if eps > 0:
                grad = torch.autograd.grad(diff_sq, x_state)[0]
                x_state_perturbed = (x_state + eps * grad.sign()).detach()
            else:
                x_state_perturbed = x_state.detach()
            x_full2 = torch.cat([x_state_perturbed, x_aux], dim=-1)
            with torch.no_grad():
                yhat2 = model(x_full2)
                diff_sq2 = ((yhat2 - y) ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim())))
                tgt_sq2 = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))) + 1e-12
                r = torch.sqrt(diff_sq2 / tgt_sq2).cpu().numpy()
                rels.extend(r.tolist())
            seen += y.shape[0]
        out["rel_l2_per_eps"].append(float(np.mean(rels)) if rels else float("nan"))
    return out


# -------------------- 4. noise sweep --------------------

def noise_sweep(model, test_loader, device,
                noise_levels=(0.0, 0.05, 0.1, 0.2, 0.5),
                n_max_samples: int = 16,
                rng_seed: int = 0) -> dict:
    """Eval at fixed Gaussian-noise levels added to input state channels."""
    out = {"noise_levels": list(noise_levels), "rel_l2_per_noise": []}
    rng = np.random.default_rng(rng_seed)
    for sigma in noise_levels:
        rels = []
        seen = 0
        with torch.no_grad():
            for batch in test_loader:
                if seen >= n_max_samples:
                    break
                x = batch["input"].to(device).float()
                y = batch["target"].to(device).float()
                mask = batch["loss_mask"].to(device).float()
                n_state_ch = y.shape[-1]
                if sigma > 0:
                    eps_t = torch.from_numpy(
                        rng.normal(0, sigma, size=tuple(x[..., :n_state_ch].shape))
                    ).float().to(device)
                    x_state_n = x[..., :n_state_ch] + eps_t
                    x = torch.cat([x_state_n, x[..., n_state_ch:]], dim=-1)
                yhat = model(x)
                n_spatial = y.dim() - 3
                mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
                diff_sq = ((yhat - y) ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim())))
                tgt_sq = (y ** 2 * mask_bc).sum(dim=tuple(range(1, y.dim()))) + 1e-12
                r = torch.sqrt(diff_sq / tgt_sq).cpu().numpy()
                rels.extend(r.tolist())
                seen += y.shape[0]
        out["rel_l2_per_noise"].append(float(np.mean(rels)) if rels else float("nan"))
    return out


# -------------------- per-cell driver --------------------

def process_cell(ckpt_path: Path, data_dir: str, family: str,
                 device: str, analyses: set) -> dict:
    cell_dir = ckpt_path.parent
    summary = {}
    try:
        model, test_loader, _, cfg = load_cell(ckpt_path, data_dir, family, device)
    except Exception as e:
        return {"error": f"load failed: {e}"}
    if "kernel" in analyses and not (cell_dir / "kernel_recovery.npz").exists():
        try:
            res = kernel_recovery(model, family, device)
            if res:
                np.savez_compressed(cell_dir / "kernel_recovery.npz", **{
                    k: v for k, v in res.items() if isinstance(v, np.ndarray)})
                # write scalars to json.
                scalars = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
                (cell_dir / "kernel_recovery.json").write_text(json.dumps(scalars, indent=2))
                summary["kernel_cosine"] = scalars.get("cosine_similarity", None)
        except Exception as e:
            summary["kernel_err"] = str(e)
    if "transfer" in analyses and not (cell_dir / "cross_family_relL2.json").exists():
        try:
            ra = bool(cfg.get("residual_anchor", False))
            res = cross_family_transfer(model, family, data_dir, device,
                                         residual_anchor=ra)
            (cell_dir / "cross_family_relL2.json").write_text(json.dumps(res, indent=2))
            summary["cross_family_n"] = len(res.get("rel_l2", {}))
        except Exception as e:
            summary["transfer_err"] = str(e)
    if "fgsm" in analyses and not (cell_dir / "adv_fgsm.json").exists():
        try:
            res = adversarial_fgsm(model, test_loader, device)
            (cell_dir / "adv_fgsm.json").write_text(json.dumps(res, indent=2))
            summary["fgsm_eps_n"] = len(res.get("epsilons", []))
        except Exception as e:
            summary["fgsm_err"] = str(e)
    if "noise" in analyses and not (cell_dir / "noise_sweep.json").exists():
        try:
            res = noise_sweep(model, test_loader, device)
            (cell_dir / "noise_sweep.json").write_text(json.dumps(res, indent=2))
            summary["noise_levels_n"] = len(res.get("noise_levels", []))
        except Exception as e:
            summary["noise_err"] = str(e)
    return summary


# -------------------- aggregator --------------------

def aggregate_and_plot(layer_root: Path):
    """Read per-cell results and produce paper figures + tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPO / "paper" / "figures"
    tab_dir = REPO / "paper" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # F09 kernel recovery
    cosines_per_fam = {f: [] for f in FAMS}
    K_amp_traces = {f: [] for f in FAMS}
    K_gt_traces = {f: [] for f in FAMS}
    for f in (Path(layer_root)).rglob("kernel_recovery.json"):
        d = json.loads(f.read_text())
        fam = d.get("family", "")
        if fam in cosines_per_fam:
            cosines_per_fam[fam].append(d.get("cosine_similarity", float("nan")))
        npz = f.with_suffix(".npz")
        if npz.exists():
            arr = np.load(npz)
            if fam in K_amp_traces:
                K_amp_traces[fam].append(arr["K_t_avg_abs"])
                K_gt_traces[fam].append(arr["K_gt"])
    if any(K_amp_traces[f] for f in FAMS):
        fig, axes = plt.subplots(1, len(FAMS), figsize=(3.0 * len(FAMS), 3.5),
                                  sharey=True)
        if len(FAMS) == 1:
            axes = [axes]
        for ax, fam in zip(axes, FAMS):
            traces = K_amp_traces[fam]
            gt = K_gt_traces[fam]
            if not traces:
                ax.set_visible(False)
                continue
            t = np.arange(len(traces[0]))
            arr = np.stack([tr[:len(t)] for tr in traces], axis=0)
            arr = arr / (arr.max(axis=-1, keepdims=True) + 1e-12)
            gt_arr = np.stack([g[:len(t)] for g in gt], axis=0)
            gt_arr = gt_arr / (gt_arr.max(axis=-1, keepdims=True) + 1e-12)
            ax.plot(t, arr.mean(axis=0), color="#d62728", lw=1.5, label="learned")
            ax.fill_between(t, arr.mean(axis=0) - arr.std(axis=0),
                             arr.mean(axis=0) + arr.std(axis=0),
                             color="#d62728", alpha=0.2)
            ax.plot(t, gt_arr.mean(axis=0), color="black", lw=1.0,
                    linestyle="--", label="ground truth")
            cos = np.nanmean(cosines_per_fam[fam]) if cosines_per_fam[fam] else float("nan")
            ax.set_title(f"{fam.replace('_rd_2d', '')}\nCosSim = {cos:.2f}")
            ax.set_xlabel("lag t")
            ax.grid(linestyle="--", alpha=0.4)
        axes[0].set_ylabel("normalized |kernel|")
        axes[0].legend(fontsize=8)
        fig.suptitle("Recovery of distributed-delay kernel from learned spectral kernel")
        fig.tight_layout()
        out = fig_dir / "F09_kernel_recovery.pdf"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"), dpi=150)
        plt.close(fig)
        print(f"  -> {out.name}")
    # F10 cross-family
    matrix = {fs: {} for fs in FAMS}
    for f in Path(layer_root).rglob("cross_family_relL2.json"):
        d = json.loads(f.read_text())
        src = d.get("ckpt_family")
        if src not in matrix:
            continue
        for tgt, v in d.get("rel_l2", {}).items():
            matrix[src].setdefault(tgt, []).append(v)
    if any(matrix[s] for s in FAMS):
        M = np.full((len(FAMS), len(FAMS)), np.nan)
        for i, src in enumerate(FAMS):
            for j, tgt in enumerate(FAMS):
                vals = matrix[src].get(tgt, [])
                if vals:
                    M[i, j] = np.mean(vals)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(M, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(FAMS)))
        ax.set_xticklabels([f.replace("_rd_2d", "").replace("dist_", "") for f in FAMS],
                           rotation=30, ha="right")
        ax.set_yticks(range(len(FAMS)))
        ax.set_yticklabels([f.replace("_rd_2d", "").replace("dist_", "") for f in FAMS])
        ax.set_xlabel("test family")
        ax.set_ylabel("train family")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("test relL2")
        for i in range(len(FAMS)):
            for j in range(len(FAMS)):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v > 0.05 else "black", fontsize=8)
        ax.set_title("Cross-family transfer (LEMO-PC)")
        fig.tight_layout()
        out = fig_dir / "F10_cross_family.pdf"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"), dpi=150)
        plt.close(fig)
        print(f"  -> {out.name}")
    # F11_adversarial + F12_noise_sweep DROPPED (2026-05-03):
    # both were single-model (LEMO-PC only) and have been replaced by the
    # multi-model F11_robustness.{pdf,png} (1x2: FGSM left + Gaussian right)
    # generated by scripts/make_F11_robustness_combined.py.
    # T06 kernel-recovery cosine table
    if any(cosines_per_fam[f] for f in FAMS):
        s = [r"\begin{tabular}{lcr}",
             r"\toprule",
             r"Family & Cosine similarity (mean $\pm$ std) & $n$ \\",
             r"\midrule"]
        for fam in FAMS:
            cs = [c for c in cosines_per_fam[fam] if not np.isnan(c)]
            if cs:
                s.append(f"{fam.replace('_rd_2d','').replace('dist_','')} & "
                         f"{np.mean(cs):.3f} $\\pm$ {np.std(cs):.3f} & {len(cs)} \\\\")
            else:
                s.append(f"{fam.replace('_rd_2d','').replace('dist_','')} & -- & 0 \\\\")
        s += [r"\bottomrule",
              r"\end{tabular}"]
        out = tab_dir / "T06_kernel_recovery_cosine.tex"
        out.write_text("\n".join(s))
        print(f"  -> {out.name}")


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_root", required=True,
                    help="e.g. outputs/dist_kernel_v2_p1 (or extracted/.../dist_kernel_v2_p1)")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--analyses", default="kernel,transfer,fgsm,noise",
                    help="comma list: kernel,transfer,fgsm,noise")
    ap.add_argument("--aggregate", action="store_true",
                    help="skip per-cell pass, just aggregate existing results to figures/tables")
    args = ap.parse_args()

    layer = Path(args.layer_root)
    if args.aggregate:
        print(f"[post-hoc] aggregating from {layer}")
        aggregate_and_plot(layer)
        return

    analyses = set(a.strip() for a in args.analyses.split(",") if a.strip())
    ckpts = sorted(layer.glob("raw/**/best_model.pt"))
    print(f"[post-hoc] {len(ckpts)} ckpts under {layer}; analyses = {sorted(analyses)}")
    if not ckpts:
        print("  (no checkpoints found — nothing to do)")
        return
    for c in ckpts:
        parts = c.parts
        family = parts[-5]
        cell = "/".join(parts[-5:-1])
        msg = process_cell(c, args.data_dir, family, args.device, analyses)
        print(f"  {cell}: {msg}")
    print("\n[post-hoc] running aggregator...")
    aggregate_and_plot(layer)


if __name__ == "__main__":
    main()
