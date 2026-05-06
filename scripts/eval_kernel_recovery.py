"""Universal kernel-recovery evaluation across delay-aware models.

For every checkpoint of a delay-aware model class, extract the impulse-response
shape of its learned temporal kernel and compute cosine similarity with the
analytic ground-truth distributed-delay kernel for that family.

Supported model classes:
  - lemo_pc_nd                 spectral lag conv (complex weights -> IRFFT)
  - causal_smooth_lemo_pc_nd   same body as lemo_pc_nd
  - nide_nd                    real time-domain FIR via K_time
  - ndde_nd                    K discrete delays via raw_taus -> binned histogram
  - s4_nd                      SSM impulse response via S4DKernel.kernel(L)

Output per cell: kernel_recovery_universal.json next to best_model.pt with
keys {family, model, seed, kernel_amp, gt_kernel, cosine_similarity}.

Usage:
  python scripts/eval_kernel_recovery.py --roots extracted/h100_pull_2026_05_05 \
      outputs/a_fix_runpod outputs/film_ablation_runpod \
      --regimes clean --models lemo_pc_nd causal_smooth_lemo_pc_nd \
      nide_nd ndde_nd s4_nd
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

KERNEL_LEN_DEFAULT = 64  # impulse-response length to recover
DELAYED_AWARE_MODELS = {
    "lemo_pc_nd", "causal_smooth_lemo_pc_nd",
    "nide_nd", "ndde_nd", "s4_nd",
}


def gt_kernel(family: str, L: int) -> np.ndarray:
    """Analytic ground-truth distributed-delay kernel, normalized to unit area.
    Shape conventions match `post_hoc_analyses.py::analyze_kernel_recovery`.
    """
    t = np.arange(L) / max(L - 1, 1)
    if family.startswith("dist_exp"):
        gt = np.exp(-3 * t)
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
    return gt.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float((a * b).sum())


def extract_kernel_lemo_pc(model, L: int) -> np.ndarray:
    """LEMO-PC / causal-smooth: complex spectral kernel -> IRFFT to time domain.
    Average abs amplitude across (in, out) channel pairs."""
    K = None
    for mod in model.modules():
        w = getattr(mod, "weights", None)
        if w is not None and torch.is_complex(w) and w.dim() == 3:
            K = w.detach().cpu().numpy()
            break
    if K is None:
        return None
    in_ch, out_ch, m = K.shape
    Lk = 2 * (m - 1) if m > 1 else m
    K_t = np.fft.irfft(K, n=Lk, axis=-1)
    amp = np.abs(K_t).mean(axis=(0, 1))  # (Lk,)
    if Lk != L:
        # Resample to length L by simple zero-pad / truncate.
        if Lk < L:
            amp = np.concatenate([amp, np.zeros(L - Lk, dtype=amp.dtype)])
        else:
            amp = amp[:L]
    return amp.astype(np.float32)


def extract_kernel_nide(model, L: int) -> np.ndarray:
    """NIDE-ND: real K_time parameter (in_ch, out_ch, lag_modes)."""
    K = None
    for name, p in model.named_parameters():
        if name.endswith(".K_time"):
            K = p.detach().cpu().numpy()
            break
    if K is None:
        return None
    in_ch, out_ch, lag_modes = K.shape
    amp = np.abs(K).mean(axis=(0, 1))  # (lag_modes,)
    if lag_modes < L:
        amp = np.concatenate([amp, np.zeros(L - lag_modes, dtype=amp.dtype)])
    else:
        amp = amp[:L]
    return amp.astype(np.float32)


def extract_kernel_ndde(model, L: int) -> np.ndarray:
    """NDDE-ND: K discrete delays via raw_taus -> sigmoid * max_tau.
    Build kernel as a normalized histogram over the lag axis."""
    raw_taus = None
    n_total = None
    for name, p in model.named_parameters():
        # Match both top-level "raw_taus" and nested "<wrapper>.raw_taus".
        if name == "raw_taus" or name.endswith(".raw_taus"):
            raw_taus = p.detach().cpu().numpy()
    # Try buffers too (n_total may be registered as a buffer).
    for name, b in model.named_buffers():
        if "n_total" in name or "max_tau" in name:
            v = b.detach().cpu().numpy()
            if v.ndim == 0:
                n_total = float(v)
                break
    if raw_taus is None:
        return None
    if n_total is None:
        n_total = float(L)  # fallback
    p_taus = 1.0 / (1.0 + np.exp(-raw_taus))  # sigmoid
    delays = p_taus * max(n_total - 1, 1)  # in [0, n_total-1]
    # Histogram into L bins on [0, L-1].
    amp = np.zeros(L, dtype=np.float32)
    for d in delays:
        # Linear interpolation between two adjacent bins.
        d = float(np.clip(d, 0, L - 1))
        lo = int(np.floor(d))
        hi = min(lo + 1, L - 1)
        w = d - lo
        amp[lo] += (1 - w)
        amp[hi] += w
    if amp.sum() > 0:
        amp = amp / amp.sum()
    return amp


def extract_kernel_s4(model, L: int) -> np.ndarray:
    """S4-ND: SSM impulse response via the S4DKernel block."""
    # Find first module that exposes a `kernel(L)` method matching the
    # S4D pattern (log_dt + log_A_real + B_real + C_real present).
    for mod in model.modules():
        if (hasattr(mod, "log_dt") and hasattr(mod, "log_A_real") and
                hasattr(mod, "C_real") and hasattr(mod, "kernel") and callable(mod.kernel)):
            with torch.no_grad():
                k = mod.kernel(L)  # (M, L) real
            k_np = k.detach().cpu().numpy()
            amp = np.abs(k_np).mean(axis=0)  # (L,)
            return amp.astype(np.float32)
    return None


EXTRACTORS = {
    "lemo_pc_nd":               extract_kernel_lemo_pc,
    "causal_smooth_lemo_pc_nd": extract_kernel_lemo_pc,
    "nide_nd":                  extract_kernel_nide,
    "ndde_nd":                  extract_kernel_ndde,
    "s4_nd":                    extract_kernel_s4,
}


def evaluate_one(ckpt_path: Path, data_dir: str, L: int, device: str) -> dict:
    """Load one ckpt, extract kernel, return record dict."""
    from datasets.apebench_dataset import create_apebench_dataloaders
    from train.build_model import build_model

    parts = ckpt_path.parts
    family = parts[-5]; regime = parts[-4]; model_name = parts[-3]; seed = parts[-2]
    if model_name not in EXTRACTORS:
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ra = bool(cfg.get("residual_anchor", False))
    _, _, test_loader = create_apebench_dataloaders(
        data_dir, family, batch_size=2, residual_anchor=ra, seed=42)
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    extract = EXTRACTORS[model_name]
    amp = extract(model, L)
    if amp is None:
        return None
    gt = gt_kernel(family, L)
    cos = cosine(amp, gt)
    return {
        "family": family, "regime": regime, "model": model_name, "seed": seed,
        "kernel_amp": amp.tolist(),
        "gt_kernel": gt.tolist(),
        "cosine_similarity": cos,
        "kernel_length": L,
    }


def emit_table(records: list, out_path: Path):
    """LaTeX table with rows = models, columns = families, cells = mean +/- std."""
    FAMS = ["dist_exp_rd_2d", "dist_gaussian_rd_2d", "dist_gamma_rd_2d",
            "dist_uniform_rd_2d", "dist_powerlaw_rd_2d"]
    FAM_COLS = ["Exp", "Gauss", "Gamma", "Uniform", "Power"]
    MODEL_ROWS = [
        ("lemo_pc_nd",                "LEMO-PC"),
        ("causal_smooth_lemo_pc_nd",  "LEMO-PC (causal smooth)"),
        ("nide_nd",                   "NIDE"),
        ("ndde_nd",                   "NDDE"),
        ("s4_nd",                     "S4"),
    ]

    by_cell = defaultdict(list)  # (model, fam) -> [cosine sims]
    for r in records:
        by_cell[(r["model"], r["family"])].append(r["cosine_similarity"])

    # Find best per family for bolding.
    fam_best = {}
    for fam in FAMS:
        best_val = -1.0
        best_model = None
        for m, _ in MODEL_ROWS:
            vs = by_cell.get((m, fam), [])
            if vs and float(np.mean(vs)) > best_val:
                best_val = float(np.mean(vs))
                best_model = m
        fam_best[fam] = best_model

    lines = [
        r"\begin{tabular}{l " + "c " * len(FAMS) + "}",
        r"\toprule",
        r"Model & " + " & ".join(FAM_COLS) + r" \\",
        r"\midrule",
    ]
    for m, label in MODEL_ROWS:
        cells = []
        for fam in FAMS:
            vs = by_cell.get((m, fam), [])
            if not vs:
                cells.append("--")
            else:
                mu = float(np.mean(vs))
                sd = float(np.std(vs)) if len(vs) > 1 else 0.0
                txt = f"{mu:.3f} $\\pm$ {sd:.3f}"
                if fam_best[fam] == m:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--regimes", default="clean", help="comma-separated")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Filter to specific model classes (default = all delay-aware)")
    ap.add_argument("--length", type=int, default=KERNEL_LEN_DEFAULT)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing kernel_recovery_universal.json")
    ap.add_argument("--out_table", default="reports/T_kernel_recovery.tex")
    args = ap.parse_args()

    regimes = set(args.regimes.split(","))
    model_filter = set(args.models) if args.models else DELAYED_AWARE_MODELS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = []
    for root in args.roots:
        rp = Path(root)
        if not rp.is_absolute():
            rp = REPO / rp
        ckpts.extend(sorted(rp.rglob("best_model.pt")))
    print(f"[krec] device={device}  ckpts found={len(ckpts)}", flush=True)

    records = []
    for c in ckpts:
        parts = c.parts
        try:
            family = parts[-5]; regime = parts[-4]; model_name = parts[-3]; seed = parts[-2]
        except IndexError:
            continue
        if regime not in regimes or model_name not in model_filter:
            continue
        out_path = c.parent / "kernel_recovery_universal.json"
        if out_path.exists() and not args.force:
            try:
                rec = json.loads(out_path.read_text())
                records.append(rec)
                continue
            except Exception:
                pass
        try:
            rec = evaluate_one(c, args.data_dir, args.length, device)
        except Exception as e:
            print(f"  ERROR {c}: {e}", flush=True)
            continue
        if rec is None:
            continue
        out_path.write_text(json.dumps(rec))
        records.append(rec)
        print(f"  ok {rec['model']} {rec['family']} s{rec['seed']}: "
              f"cos={rec['cosine_similarity']:.3f}", flush=True)

    print(f"[krec] total records: {len(records)}", flush=True)
    out_table = Path(args.out_table)
    if not out_table.is_absolute():
        out_table = REPO / out_table
    out_table.parent.mkdir(parents=True, exist_ok=True)
    emit_table(records, out_table)
    print(f"[krec] wrote {out_table}", flush=True)


if __name__ == "__main__":
    main()
