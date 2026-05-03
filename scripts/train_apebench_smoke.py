"""
APEBench smoke trainer: end-to-end pipeline check for ND models.

Loads APEBench-format shards (gen_apebench_data.py output), trains
one ND model for a configurable epoch count with masked MSE on the
future-segment portion, and reports test relative L2.

This is the minimal end-to-end pipeline before scaling to the full
multi-day Phase 4 sweep.

Usage:
    python3 scripts/train_apebench_smoke.py \\
        --data_dir data_apebench --family kolmogorov_2d \\
        --model lemo_pc_nd --epochs 50 --batch_size 4 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from datasets.apebench_dataset import create_apebench_dataloaders
from train.build_model import build_model


def evaluate(model, loader, device):
    model.eval()
    rel_l2_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device).float()
            y_target = batch["target"].to(device).float()
            # Denominator is always in the original (un-anchored) space so
            # residual_anchor and clean modes report apples-to-apples relL2.
            y_target_raw = batch.get("target_raw", batch["target"]).to(device).float()
            mask = batch["loss_mask"].to(device).float()
            y_pred = model(x)
            # mask shape (B, T); broadcast over spatial+channel.
            n_spatial = y_target.dim() - 3      # B, T, *spatial, C
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            diff = (y_pred - y_target) * mask_bc
            tgt = y_target_raw * mask_bc
            num = torch.sqrt((diff ** 2).sum(dim=tuple(range(1, diff.dim()))) + 1e-10)
            den = torch.sqrt((tgt ** 2).sum(dim=tuple(range(1, tgt.dim()))) + 1e-10)
            rel = num / den
            rel_l2_list.append(rel.cpu().numpy())
    return float(np.mean(np.concatenate(rel_l2_list)))


def _compute_weight_norm(model: torch.nn.Module) -> float:
    """Total L2 norm of trainable parameters (handles complex parameters)."""
    with torch.no_grad():
        sq = sum(float((p.detach().abs() ** 2).sum().item())
                 for p in model.parameters() if p.requires_grad)
    return float(sq ** 0.5)


def _compute_op_norm_max(model: torch.nn.Module) -> float:
    """Largest per-mode operator-norm sigma_max(K[:,:,m]) over all spectral
    lag layers (FiLMLagSpectralND.weights, complex tensor shape (in, out, modes)).
    Returns 0.0 if no such layers found.  Proves whether σ-projection is binding."""
    max_op = 0.0
    with torch.no_grad():
        for mod in model.modules():
            w = getattr(mod, "weights", None)
            if w is None or not torch.is_complex(w) or w.dim() != 3:
                continue
            Km = w.permute(2, 0, 1).contiguous()  # (modes, in, out)
            try:
                S = torch.linalg.svdvals(Km)
                cur = float(S.max().item())
                if cur > max_op:
                    max_op = cur
            except Exception:
                pass
    return float(max_op)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_apebench")
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", default="lemo_pc_nd",
                    choices=["lemo_nd", "lemo_pc_nd", "causal_lemo_pc_nd",
                              "per_lag_mlp_nd", "fno_nd", "memno_nd", "ffno_nd",
                              "markov_fno_nd", "windowed_fno_nd", "unet_nd",
                              "fno_film_nd", "noneq_film_nd", "lemo_bcorrect_nd",
                              "nide_nd", "ndde_nd", "s4_nd"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--lag_modes", type=int, default=8)
    ap.add_argument("--spatial_modes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/apebench_smoke")
    ap.add_argument("--regime", default="clean", choices=["clean", "lowres", "noisy"])
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--downsample_factor", type=int, default=2)
    ap.add_argument("--sigma", type=str, default="none")
    ap.add_argument("--num_train_samples", type=int, default=None,
                    help="Subsample train shard to this many trajectories "
                         "(for sample-efficiency curve).  Default = use all.")
    ap.add_argument("--residual_anchor", action="store_true",
                    help="Present input/target as residuals from u(t=n_hist-1). "
                         "Makes the lag-axis cyclic-FFT boundary continuous.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, test_loader = create_apebench_dataloaders(
        args.data_dir, args.family, batch_size=args.batch_size,
        regime=args.regime,
        noise_std=args.noise_std,
        downsample_factor=args.downsample_factor,
        seed=args.seed,
        residual_anchor=args.residual_anchor,
        num_train_samples=args.num_train_samples,
    )
    sample = next(iter(train_loader))
    in_channels = sample["input"].shape[-1]
    out_channels = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    spatial_shape = tuple(sample["input"].shape[2:-1])
    params_dim = sample["params"].shape[-1]
    print(f"data layout: input {tuple(sample['input'].shape)} -> target {tuple(sample['target'].shape)}")
    print(f"n_total (lag) = {n_total}, spatial = {spatial_shape}, in_ch = {in_channels}, out_ch = {out_channels}")

    spatial_modes = tuple(min(args.spatial_modes, s // 2) for s in spatial_shape)
    # FNO treats lag + spatial as one physical axes block; LEMO treats lag separately.
    fno_physical_shape = (n_total,) + spatial_shape
    fno_modes = (args.lag_modes,) + spatial_modes
    config = {
        "model_class": args.model,
        "residual_anchor":   bool(args.residual_anchor),
        "regime":            args.regime,
        "noise_std":         float(args.noise_std),
        "downsample_factor": int(args.downsample_factor),
        "model": {
            "length":         n_total,
            "spatial_shape":  list(spatial_shape),
            "spatial_modes":  list(spatial_modes),
            "lag_modes":      args.lag_modes,
            # For FNO: physical axes = (lag, *spatial), one mode per axis.
            "physical_shape": list(fno_physical_shape),
            "modes":          list(fno_modes),
            "width":          args.width,
            "n_layers":       args.n_layers,
            "kernel_hidden":  64,
            "params_dim":     params_dim,
            "sigma":          (None if args.sigma.lower() in ("none","null","") else float(args.sigma)),
            "activation":     "gelu",
        },
    }
    model = build_model(config, in_channels=in_channels,
                        out_channels=out_channels, length=n_total)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = model.to(device)
    print(f"model: {args.model}, params: {n_params:,}")

    # Separate FiLM params into a no-weight-decay group so the per-sample
    # modulation isn't driven to zero by L2 regularization. WD on FiLM (=1e-3)
    # was the dominant cause of FiLM nullification in the original sweep.
    film_params, other_params = [], []
    for n, p in model.named_parameters():
        if "film_net" in n:
            film_params.append(p)
        else:
            other_params.append(p)
    opt = torch.optim.Adam([
        {"params": other_params, "weight_decay": 1e-3},
        {"params": film_params,  "weight_decay": 0.0},
    ], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 1e-3)

    out_dir = Path(args.output_dir) / args.family / args.regime / args.model / f"s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": [], "val_rel_l2": [], "test_rel_l2": [],
        "grad_norm": [], "weight_norm": [], "op_norm_max": [],
        "lr": [], "wall_per_epoch_s": [], "peak_mem_gb": [],
        # FiLM diagnostic: per-epoch (gamma, beta) statistics across all
        # film_net layers. Detects FiLM nullification (γ→0, β→0).
        "film_gamma_mean_pre_tanh": [],
        "film_gamma_std_pre_tanh": [],
        "film_beta_std_pre_tanh": [],
        "film_input_norm": [],  # ||film_net.0.weight||_F to detect dead first layer
    }
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        grad_norm_acc = 0.0
        n_steps = 0
        epoch_t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            x = batch["input"].to(device).float()
            y_target = batch["target"].to(device).float()
            mask = batch["loss_mask"].to(device).float()
            y_pred = model(x)
            n_spatial = y_target.dim() - 3
            mask_bc = mask.view(*mask.shape, *((1,) * (n_spatial + 1)))
            diff = (y_pred - y_target) * mask_bc
            denom = mask_bc.sum().clamp_min(1.0) * y_target.shape[-1]
            loss = (diff ** 2).sum() / denom
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm_acc += float(gn) if gn is not None else 0.0
            opt.step()
            running_loss += loss.item()
            n_steps += 1
        sched.step()
        avg_loss = running_loss / n_steps
        val_rl2 = evaluate(model, val_loader, device)
        history["train_loss"].append(avg_loss)
        history["val_rel_l2"].append(val_rl2)
        history["grad_norm"].append(grad_norm_acc / max(n_steps, 1))
        history["weight_norm"].append(_compute_weight_norm(model))
        history["op_norm_max"].append(_compute_op_norm_max(model))
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        history["wall_per_epoch_s"].append(time.time() - epoch_t0)
        history["peak_mem_gb"].append(
            float(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0)
        # FiLM diagnostic: estimate γ/β stats by passing test-set params through film_net
        try:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                fp = sample_batch["params"].to(device).float()
                gammas, betas, in_norms = [], [], []
                for blk in getattr(model, "blocks", []):
                    if hasattr(blk, "A_lag") and hasattr(blk.A_lag, "film_net"):
                        film = blk.A_lag.film_net(fp)
                        OC = blk.A_lag.out_channels; M = blk.A_lag.lag_modes
                        gammas.append(film[:, :OC * M])
                        betas.append(film[:, OC * M:])
                        in_norms.append(blk.A_lag.film_net[0].weight.norm().item())
                if gammas:
                    g_all = torch.cat(gammas, dim=1)
                    b_all = torch.cat(betas, dim=1)
                    history["film_gamma_mean_pre_tanh"].append(float(g_all.mean()))
                    history["film_gamma_std_pre_tanh"].append(float(g_all.std()))
                    history["film_beta_std_pre_tanh"].append(float(b_all.std()))
                    history["film_input_norm"].append(float(np.mean(in_norms)))
                else:
                    history["film_gamma_mean_pre_tanh"].append(0.0)
                    history["film_gamma_std_pre_tanh"].append(0.0)
                    history["film_beta_std_pre_tanh"].append(0.0)
                    history["film_input_norm"].append(0.0)
        except Exception as _e:
            for k in ("film_gamma_mean_pre_tanh", "film_gamma_std_pre_tanh",
                      "film_beta_std_pre_tanh", "film_input_norm"):
                history[k].append(0.0)
        elapsed = time.time() - t0
        print(f"[ep {epoch+1:>3d}/{args.epochs}]  train_loss={avg_loss:.4e}  "
              f"val_relL2={val_rl2:.4f}  grad={history['grad_norm'][-1]:.2e}  "
              f"opK={history['op_norm_max'][-1]:.3f}  t={elapsed:.0f}s")
        if val_rl2 < best_val:
            best_val = val_rl2
            torch.save({"model_state_dict": model.state_dict(),
                         "config": config, "epoch": epoch + 1},
                        out_dir / "best_model.pt")

    # Final test eval with best.
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_rl2 = evaluate(model, test_loader, device)
    history["test_rel_l2"] = test_rl2
    print(f"\n=== FINAL test relL2 = {test_rl2:.4f} ===")

    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    final_op_norm = _compute_op_norm_max(model)
    json.dump({"family": args.family, "model": args.model,
                "test_rel_l2_mean": test_rl2,
                "best_val_rel_l2": best_val,
                "params": n_params,
                "wall_seconds": time.time() - t0,
                "sigma": (None if args.sigma.lower() in ("none","null","") else float(args.sigma)),
                "final_op_norm_max": final_op_norm,
                "n_epochs": args.epochs,
                "peak_mem_gb_overall": (max(history["peak_mem_gb"]) if history["peak_mem_gb"] else 0.0),
                "seed": args.seed,
                "regime": args.regime,
                "residual_anchor": bool(args.residual_anchor),
                "noise_std": float(args.noise_std),
                "downsample_factor": int(args.downsample_factor),
                "config": config["model"]},
              open(out_dir / "test_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
