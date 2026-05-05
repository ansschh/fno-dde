"""Long-horizon rollout with predicted SPATIAL FIELDS saved (for hero figure).

`eval_long_horizon.py` saves only norms; this script additionally saves
the full predicted field at each rollout step for a small number of cells
(default 1 chain, T=256), so we can render side-by-side panels of:

  left:  unconstrained baseline predicting at t=128 / t=256, field diverging
  right: LEMO-PC at the same horizon, field bounded and on-target

Output: per-cell `long_rollout_fields.npz` with arrays:
  pred_fields:   (n_chain, T, *spatial, C)   <-- the predicted field
  target_fields: (n_chain, T_gt, *spatial, C) where T_gt is what's available
  pred_norm:     (n_chain, T)  <-- ‖pred(t)‖_2 trajectory
  target_norm:   (n_chain, T_gt)
  base_input:    (n_chain, n_hist, *spatial, C)  <-- the seed history

Usage:
  python scripts/eval_long_rollout_fields.py \\
    --roots /workspace/dde-fno/extracted \\
    --models lemo_pc_nd noneq_film_nd lemo_bcorrect_nd \\
    --families dist_exp_rd_2d \\
    --regimes clean \\
    --seeds 42 \\
    --n_chain 1 --T 256 \\
    --data_dir data_dde_pde
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def parse_path(p: Path):
    parts = p.parts
    try:
        seed = parts[-2]; model = parts[-3]; reg = parts[-4]; fam = parts[-5]
    except IndexError:
        return None
    if not seed.startswith("s"):
        return None
    return fam, reg, model, seed


def autoregressive_rollout(model, base_history: torch.Tensor, n_chain: int,
                            n_hist: int, n_out: int, T: int, device):
    """Roll out autoregressively: each chain predicts the next n_out frames,
    then those frames become part of the next history. Stop when total saved
    frames reach T.

    base_history: (n_chain, n_hist, *spatial, C_total)  — full input including
                                                          aux (params/mask) channels
    Returns: pred_fields (n_chain, T, *spatial, C_state)
    """
    C_total = base_history.shape[-1]
    history = base_history.clone()  # (n_chain, n_hist, *, C_total)
    out = []
    saved = 0
    iteration = 0
    while saved < T:
        with torch.no_grad():
            yhat = model(history)  # (n_chain, n_out, *spatial, C_state)
        C_state = yhat.shape[-1]
        out.append(yhat.detach().cpu())
        saved += yhat.shape[1]
        # Build the next history: shift by n_out frames. Keep the aux channels
        # constant from base_history (params/mask don't evolve).
        if saved < T:
            new_state = yhat[:, -n_hist:].clone()  # last n_hist of pred frames
            # If yhat shorter than n_hist, mix with previous history.
            if new_state.shape[1] < n_hist:
                pad = history[:, new_state.shape[1]:, ..., :C_state]
                new_state = torch.cat([pad, new_state], dim=1)
            # Reattach aux channels from history.
            if C_total > C_state:
                aux = history[:, -n_hist:, ..., C_state:]
                history = torch.cat([new_state, aux], dim=-1)
            else:
                history = new_state
        iteration += 1
        if iteration > 50:
            break  # safety
    full = torch.cat(out, dim=1)  # (n_chain, total_pred, *, C_state)
    return full[:, :T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--models", nargs="*", required=True)
    ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--regimes", nargs="*", default=["clean"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--n_chain", type=int, default=1)
    ap.add_argument("--T", type=int, default=256, help="Total rollout steps to save")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = []
    for r in args.roots:
        ckpts.extend(Path(r).rglob("best_model.pt"))

    filtered = []
    for c in ckpts:
        meta = parse_path(c)
        if meta is None:
            continue
        fam, reg, mdl, seed = meta
        if mdl not in args.models:
            continue
        if fam not in args.families:
            continue
        if reg not in args.regimes:
            continue
        if int(seed.lstrip("s")) not in args.seeds:
            continue
        filtered.append(c)

    print(f"[long-fields] {len(filtered)} cells to process")
    for i, ckpt in enumerate(filtered, 1):
        out_path = ckpt.parent / "long_rollout_fields.npz"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(filtered)}] {ckpt.parent}: skip", flush=True)
            continue
        meta = parse_path(ckpt)
        fam, reg, mdl, seed = meta
        t0 = time.time()
        try:
            from datasets.apebench_dataset import create_apebench_dataloaders
            from train.build_model import build_model
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            cfg = ck["config"]
            noise_std = float(cfg.get("noise_std", 0.05))
            downsample_factor = int(cfg.get("downsample_factor", 2))
            ra = bool(cfg.get("residual_anchor", False))
            _, _, test_loader = create_apebench_dataloaders(
                args.data_dir, fam, batch_size=max(args.n_chain, 1),
                regime=reg, noise_std=noise_std,
                downsample_factor=downsample_factor,
                residual_anchor=ra, seed=42,
            )
            sample = next(iter(test_loader))
            in_ch = sample["input"].shape[-1]
            out_ch = sample["target"].shape[-1]
            n_total = sample["input"].shape[1]
            model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
            model.load_state_dict(ck["model_state_dict"])
            model = model.to(device).eval()

            base_input = sample["input"][: args.n_chain].to(device).float()
            target = sample["target"][: args.n_chain].to(device).float()  # (n_chain, T_gt, *, C_state)

            # Number of frames per forward pass (output length expected).
            n_out = int(cfg.get("n_out", target.shape[1]))
            n_hist = int(cfg.get("n_hist", base_input.shape[1]))

            pred_fields = autoregressive_rollout(
                model, base_input, args.n_chain, n_hist=n_hist,
                n_out=n_out, T=args.T, device=device,
            )  # CPU tensor (n_chain, T, *, C_state)

            # Norms.
            flat_p = pred_fields.flatten(2)  # (n_chain, T, spatial*C)
            pred_norm = flat_p.norm(dim=-1).numpy()
            flat_t = target.detach().cpu().flatten(2)
            target_norm = flat_t.norm(dim=-1).numpy()

            np.savez(out_path,
                     pred_fields=pred_fields.numpy(),
                     target_fields=target.cpu().numpy(),
                     pred_norm=pred_norm,
                     target_norm=target_norm,
                     base_input=base_input.cpu().numpy(),
                     n_hist=n_hist, n_out=n_out, T=args.T)
            print(f"[{i}/{len(filtered)}] {fam}/{reg}/{mdl}/{seed}: ok "
                  f"pred_T={pred_fields.shape[1]} pred_peak_norm={float(pred_norm.max()):.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(filtered)}] {fam}/{reg}/{mdl}/{seed}: FAIL "
                  f"{type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
