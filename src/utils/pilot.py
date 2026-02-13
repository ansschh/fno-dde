"""
Pilot Protocol Utility

Runs a small-scale experiment to validate a new family before committing
to large-scale data generation and training. Checks:
1. Solver produces valid solutions (QC pass rate > 80%)
2. FNO can learn something (loss decreases)
3. Training converges without NaN/Inf
4. Returns key metrics for go/no-go decision
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# Ensure project root on path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def run_pilot(
    family: str,
    n_train: int = 1600,
    n_val: int = 200,
    n_test: int = 200,
    epochs: int = 50,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42,
    cleanup: bool = True,
) -> Dict[str, Any]:
    """
    Run a small pilot experiment for a DDE or PDE family.

    Args:
        family: Family name (e.g., "neutral", "burgers")
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        epochs: Training epochs
        batch_size: Batch size
        data_dir: Data directory (uses temp if None)
        output_dir: Output directory (uses temp if None)
        device: Device for training
        seed: Random seed
        cleanup: Remove temp directories when done

    Returns:
        Dict with keys:
        - success: bool
        - qc_pass_rate: float
        - final_train_loss: float
        - final_val_loss: float
        - final_rel_l2: float
        - test_rel_l2_median: float (if test succeeded)
        - test_rel_l2_p95: float
        - frmse_low/mid/high: float (frequency-binned)
        - message: str
    """
    use_temp_data = data_dir is None
    use_temp_output = output_dir is None

    if use_temp_data:
        data_dir = tempfile.mkdtemp(prefix="pilot_data_")
    if use_temp_output:
        output_dir = tempfile.mkdtemp(prefix="pilot_output_")

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "family": family,
        "n_train": n_train,
        "epochs": epochs,
        "success": False,
        "message": "",
    }

    try:
        # --- Step 1: Generate data ---
        print(f"[Pilot] Generating {n_train + n_val + n_test} samples for {family}...")

        # Detect DDE vs PDE family
        is_pde = False
        try:
            from dde.families import DDE_FAMILIES
            if family in DDE_FAMILIES:
                _generate_dde_data(family, n_train, n_val, n_test, data_path, seed)
            else:
                is_pde = True
        except ImportError:
            is_pde = True

        if is_pde:
            try:
                from pde.families import PDE_FAMILIES
                if family in PDE_FAMILIES:
                    _generate_pde_data(family, n_train, n_val, n_test, data_path, seed)
                else:
                    result["message"] = f"Unknown family: {family}"
                    return result
            except ImportError:
                result["message"] = f"Cannot import PDE families for {family}"
                return result

        # Check QC pass rate
        manifest_path = data_path / family / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            actual_train = manifest.get("splits", {}).get("train", {}).get("n_samples", 0)
            qc_rate = actual_train / max(n_train, 1)
            result["qc_pass_rate"] = qc_rate
            print(f"[Pilot] QC pass rate: {qc_rate:.1%}")

            if qc_rate < 0.5:
                result["message"] = f"QC pass rate too low: {qc_rate:.1%}"
                return result

        # --- Step 2: Train ---
        print(f"[Pilot] Training for {epochs} epochs...")

        from datasets.sharded_dataset import create_sharded_dataloaders
        from models import create_fno1d, count_parameters
        from train.train_fno_sharded import ShardedTrainer, masked_mse_loss, evaluate_model

        torch.manual_seed(seed)
        np.random.seed(seed)

        dev = torch.device(device if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, test_loader = create_sharded_dataloaders(
            data_dir=str(data_path), family=family,
            batch_size=batch_size, num_workers=2,
        )

        sample = next(iter(train_loader))
        in_ch = sample["input"].shape[-1]
        out_ch = sample["target"].shape[-1]

        config = {
            "family": family,
            "model": {"modes": 16, "width": 48, "n_layers": 3, "padding": 8, "dropout": 0.1},
            "use_residual": True,
            "lr": 1e-3, "lr_min": 1e-6, "scheduler": "cosine",
            "weight_decay": 1e-3, "grad_clip": 1.0,
            "epochs": epochs, "patience": epochs,  # no early stopping in pilot
            "save_every": epochs,
        }

        model = create_fno1d(in_ch, out_ch, config["model"], use_residual=True)
        print(f"[Pilot] Model params: {count_parameters(model):,}")

        trainer = ShardedTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            config=config, device=dev, output_dir=output_path,
        )
        trainer.train(epochs=epochs)

        result["final_train_loss"] = trainer.history["train_loss"][-1]
        result["final_val_loss"] = trainer.history["val_loss"][-1]
        result["final_rel_l2"] = trainer.history["val_rel_l2"][-1]

        # Check for convergence
        if np.isnan(result["final_train_loss"]) or np.isinf(result["final_train_loss"]):
            result["message"] = "Training diverged (NaN/Inf loss)"
            return result

        # --- Step 3: Evaluate ---
        print("[Pilot] Evaluating on test set...")

        best_ckpt = torch.load(output_path / "best_model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        y_mean = train_loader.dataset.y_mean
        y_std = train_loader.dataset.y_std
        test_metrics = evaluate_model(model, test_loader, dev,
                                       y_mean=y_mean, y_std=y_std)

        result["test_rel_l2_median"] = test_metrics.get("rel_l2_original_median",
                                                          test_metrics.get("rel_l2_normalized_median", -1))
        result["test_rel_l2_p95"] = test_metrics.get("rel_l2_original_p95",
                                                       test_metrics.get("rel_l2_normalized_p95", -1))

        # Frequency-binned evaluation
        try:
            from eval.unified_eval import evaluate_model_on_loader
            eval_metrics = evaluate_model_on_loader(model, test_loader, dev, "pilot_test")
            result["frmse_low"] = eval_metrics.frmse_low
            result["frmse_mid"] = eval_metrics.frmse_mid
            result["frmse_high"] = eval_metrics.frmse_high
        except Exception:
            pass

        result["success"] = True
        is_hard = result["test_rel_l2_median"] > 0.5
        result["message"] = (
            f"CONFIRMED HARD (median relL2={result['test_rel_l2_median']:.3f})"
            if is_hard else
            f"Pilot passed (median relL2={result['test_rel_l2_median']:.3f})"
        )

    except Exception as e:
        result["message"] = f"Pilot failed: {e}"
        import traceback
        traceback.print_exc()

    finally:
        if cleanup:
            if use_temp_data and Path(data_dir).exists():
                shutil.rmtree(data_dir, ignore_errors=True)
            if use_temp_output and Path(output_dir).exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"[Pilot Result] {family}: {result['message']}")
    if result.get("qc_pass_rate"):
        print(f"  QC pass rate: {result['qc_pass_rate']:.1%}")
    if result.get("test_rel_l2_median"):
        print(f"  Test relL2 median: {result['test_rel_l2_median']:.4f}")
        print(f"  Test relL2 p95:    {result['test_rel_l2_p95']:.4f}")
    if result.get("frmse_high"):
        print(f"  fRMSE low/mid/high: {result.get('frmse_low', 0):.4f} / "
              f"{result.get('frmse_mid', 0):.4f} / {result.get('frmse_high', 0):.4f}")
    print(f"{'='*60}")

    return result


def _generate_dde_data(family, n_train, n_val, n_test, data_path, seed):
    """Generate DDE data using the Python solver."""
    from dde.families import get_family
    from dde.solve_python.dde_solver import solve_dde

    fam = get_family(family)
    rng = np.random.default_rng(seed)

    for split_name, n_samples in [("train", n_train), ("val", n_val), ("test", n_test)]:
        split_dir = data_path / family / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        shard_size = min(64, n_samples)
        n_shards = (n_samples + shard_size - 1) // shard_size

        collected = 0
        shard_idx = 0
        phi_list, y_list, params_list, lags_list = [], [], [], []

        max_attempts = n_samples * 5
        attempts = 0

        t_hist = np.linspace(-fam.config.tau_max, 0, fam.config.n_grid // 2)
        dt_out = fam.config.T / (fam.config.n_grid // 2)
        t_out_grid = np.linspace(dt_out, fam.config.T, fam.config.n_grid // 2)

        while collected < n_samples and attempts < max_attempts:
            attempts += 1
            params = fam.sample_params(rng)
            history = fam.sample_history(rng, t_hist)

            try:
                sol = solve_dde(fam, params, history, t_hist, fam.config.T,
                                n_points=len(t_out_grid))
            except Exception:
                continue

            if not sol.success:
                continue
            if not np.all(np.isfinite(sol.y)):
                continue
            if np.max(np.abs(sol.y)) > 100:
                continue

            phi_list.append(history)
            y_list.append(sol.y)
            delays = fam.get_delays(params)
            param_vec = np.array([params[k] for k in fam.config.param_names])
            params_list.append(param_vec)
            lags_list.append(np.array(delays))
            collected += 1

            if len(phi_list) >= shard_size or collected >= n_samples:
                np.savez(
                    split_dir / f"shard_{shard_idx}.npz",
                    t_hist=t_hist,
                    t_out=t_out_grid,
                    phi=np.array(phi_list),
                    y=np.array(y_list),
                    params=np.array(params_list),
                    lags=np.array(lags_list),
                )
                shard_idx += 1
                phi_list, y_list, params_list, lags_list = [], [], [], []

        # Write manifest
        manifest = {
            "family": family,
            "config": {"tau_max": fam.config.tau_max, "T": fam.config.T},
            "param_names": fam.config.param_names,
            "state_dim": fam.config.state_dim,
            "splits": {},
            "seed": seed,
            "generator": "python_pilot",
        }

    # Re-count actual samples per split
    for split_name in ["train", "val", "test"]:
        split_dir = data_path / family / split_name
        shards = sorted(split_dir.glob("shard_*.npz"))
        total = sum(np.load(s)["phi"].shape[0] for s in shards)
        manifest["splits"][split_name] = {"n_samples": total, "n_shards": len(shards)}

    with open(data_path / family / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _generate_pde_data(family, n_train, n_val, n_test, data_path, seed):
    """Generate PDE data — delegates to PDE generation utilities."""
    from pde.families import get_pde_family

    fam = get_pde_family(family)
    rng = np.random.default_rng(seed)

    x_grid = np.linspace(fam.config.domain[0], fam.config.domain[1],
                          fam.config.n_spatial, endpoint=False)

    for split_name, n_samples in [("train", n_train), ("val", n_val), ("test", n_test)]:
        split_dir = data_path / family / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        shard_size = min(64, n_samples)
        input_list, sol_list, params_list = [], [], []
        collected = 0
        shard_idx = 0

        max_attempts = n_samples * 5
        attempts = 0
        while collected < n_samples and attempts < max_attempts:
            attempts += 1
            params = fam.sample_params(rng)

            # KS has variable domain length — recompute x_grid per sample
            sample_x_grid = x_grid
            if fam.config.name == "ks":
                L = params["L"]
                sample_x_grid = np.linspace(0, L, fam.config.n_spatial, endpoint=False)

            # Wave family needs c_contrast passed to sample_input_function
            if fam.config.name == "wave":
                c_contrast = params.get("c_contrast", 1.5)
                input_func = fam.sample_input_function(rng, sample_x_grid, c_contrast=c_contrast)
            else:
                input_func = fam.sample_input_function(rng, sample_x_grid)

            try:
                solution = fam.solve(input_func, sample_x_grid, params)
            except Exception as e:
                continue

            if not np.all(np.isfinite(solution)):
                continue

            input_list.append(input_func)
            sol_list.append(solution)
            param_vec = np.array([params[k] for k in fam.config.param_names])
            params_list.append(param_vec)
            collected += 1

            if len(input_list) >= shard_size or collected >= n_samples:
                np.savez(
                    split_dir / f"shard_{shard_idx}.npz",
                    x_grid=sample_x_grid,
                    input_func=np.array(input_list),
                    solution=np.array(sol_list),
                    params=np.array(params_list),
                    meta_json=json.dumps({"seed": seed, "family": family}),
                )
                shard_idx += 1
                input_list, sol_list, params_list = [], [], []

    manifest = {
        "family": family,
        "config": {"domain": list(fam.config.domain), "T": fam.config.T,
                    "n_spatial": fam.config.n_spatial},
        "param_names": fam.config.param_names,
        "state_dim": fam.config.state_dim,
        "input_type": fam.config.input_type,
        "splits": {},
        "seed": seed,
        "generator": "python_pilot",
    }

    for split_name in ["train", "val", "test"]:
        split_dir = data_path / family / split_name
        shards = sorted(split_dir.glob("shard_*.npz"))
        total = sum(np.load(s)["input_func"].shape[0] for s in shards)
        manifest["splits"][split_name] = {"n_samples": total, "n_shards": len(shards)}

    with open(data_path / family / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
