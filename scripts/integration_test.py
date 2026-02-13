#!/usr/bin/env python3
"""
End-to-end integration test for the DDE-FNO pipeline.

Runs the full pipeline (generate → train → evaluate → fRMSE → plots) with
tiny data sizes to verify everything works. Tests both a DDE and a PDE family.

Usage:
    python scripts/integration_test.py
    python scripts/integration_test.py --dde-family hutch --pde-family burgers
    python scripts/integration_test.py --skip-pde   # DDE only
"""

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

# Ensure project root on path
_project_root = Path(__file__).resolve().parent.parent / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class IntegrationTest:
    """Runs a single end-to-end integration test for one family."""

    def __init__(self, family: str, is_pde: bool, n_samples: int = 100,
                 epochs: int = 10, batch_size: int = 16, device: str = "cpu"):
        self.family = family
        self.is_pde = is_pde
        self.n_samples = n_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.data_dir = None
        self.output_dir = None
        self.errors = []

    def run(self) -> bool:
        """Run all stages. Returns True if all pass."""
        self.data_dir = tempfile.mkdtemp(prefix=f"inttest_{self.family}_data_")
        self.output_dir = tempfile.mkdtemp(prefix=f"inttest_{self.family}_out_")

        try:
            self._step_generate()
            self._step_load_data()
            self._step_train()
            self._step_evaluate()
            self._step_frmse()
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            traceback.print_exc()
        finally:
            # Cleanup temp dirs
            shutil.rmtree(self.data_dir, ignore_errors=True)
            shutil.rmtree(self.output_dir, ignore_errors=True)

        return len(self.errors) == 0

    def _step_generate(self):
        """Step 1: Generate data."""
        print(f"  [1/5] Generating {self.n_samples} samples...")
        data_path = Path(self.data_dir)

        if self.is_pde:
            from utils.pilot import _generate_pde_data
            _generate_pde_data(self.family, self.n_samples // 2,
                               self.n_samples // 4, self.n_samples // 4,
                               data_path, seed=42)
        else:
            from utils.pilot import _generate_dde_data
            _generate_dde_data(self.family, self.n_samples // 2,
                               self.n_samples // 4, self.n_samples // 4,
                               data_path, seed=42)

        # Verify manifest exists
        manifest_path = data_path / self.family / "manifest.json"
        assert manifest_path.exists(), f"Manifest not created at {manifest_path}"

        with open(manifest_path) as f:
            manifest = json.load(f)

        for split in ["train", "val", "test"]:
            n = manifest["splits"][split]["n_samples"]
            assert n > 0, f"Split {split} has 0 samples"

        print(f"        Manifest OK: {manifest['splits']}")

    def _step_load_data(self):
        """Step 2: Load data into DataLoader."""
        print("  [2/5] Loading data into DataLoader...")
        from datasets.sharded_dataset import create_sharded_dataloaders

        train_loader, val_loader, test_loader = create_sharded_dataloaders(
            data_dir=self.data_dir,
            family=self.family,
            batch_size=self.batch_size,
            num_workers=0,
        )

        sample = next(iter(train_loader))
        assert "input" in sample, "Missing 'input' key in batch"
        assert "target" in sample, "Missing 'target' key in batch"
        assert torch.all(torch.isfinite(sample["input"])), "Non-finite values in input"
        assert torch.all(torch.isfinite(sample["target"])), "Non-finite values in target"

        print(f"        input shape:  {sample['input'].shape}")
        print(f"        target shape: {sample['target'].shape}")

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader

    def _step_train(self):
        """Step 3: Train FNO model."""
        print(f"  [3/5] Training for {self.epochs} epochs...")
        from models import create_fno1d, count_parameters
        from train.train_fno_sharded import ShardedTrainer

        torch.manual_seed(42)
        dev = torch.device(self.device)

        sample = next(iter(self._train_loader))
        in_ch = sample["input"].shape[-1]
        out_ch = sample["target"].shape[-1]

        config = {
            "family": self.family,
            "model": {"modes": 8, "width": 16, "n_layers": 2, "padding": 4, "dropout": 0.0},
            "use_residual": True,
            "lr": 1e-3, "lr_min": 1e-6, "scheduler": "cosine",
            "weight_decay": 1e-3, "grad_clip": 1.0,
            "epochs": self.epochs, "patience": self.epochs,
            "save_every": self.epochs,
        }

        model = create_fno1d(in_ch, out_ch, config["model"], use_residual=True)
        n_params = count_parameters(model)
        print(f"        Model params: {n_params:,}")

        trainer = ShardedTrainer(
            model=model, train_loader=self._train_loader,
            val_loader=self._val_loader,
            config=config, device=dev, output_dir=Path(self.output_dir),
        )
        trainer.train(epochs=self.epochs)

        # Verify training converged (no NaN)
        final_loss = trainer.history["train_loss"][-1]
        assert np.isfinite(final_loss), f"Training diverged: loss={final_loss}"
        assert final_loss < trainer.history["train_loss"][0], "Loss did not decrease"

        print(f"        Final train loss: {final_loss:.6f}")
        print(f"        Final val relL2:  {trainer.history['val_rel_l2'][-1]:.4f}")

        self._model = model
        self._trainer = trainer

    def _step_evaluate(self):
        """Step 4: Evaluate on test set."""
        print("  [4/5] Evaluating on test set...")
        from train.train_fno_sharded import evaluate_model

        dev = torch.device(self.device)

        # Load best checkpoint if available
        best_ckpt_path = Path(self.output_dir) / "best_model.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=dev, weights_only=False)
            self._model.load_state_dict(ckpt["model_state_dict"])

        y_mean = self._train_loader.dataset.y_mean
        y_std = self._train_loader.dataset.y_std
        metrics = evaluate_model(self._model, self._test_loader, dev,
                                 y_mean=y_mean, y_std=y_std)

        # Check key metrics exist and are finite
        for key in ["mse_normalized", "rel_l2_normalized_median"]:
            assert key in metrics, f"Missing metric: {key}"
            assert np.isfinite(metrics[key]), f"Non-finite metric {key}={metrics[key]}"

        print(f"        Test MSE: {metrics['mse_normalized']:.6f}")
        print(f"        Test relL2 median: {metrics['rel_l2_normalized_median']:.4f}")

    def _step_frmse(self):
        """Step 5: Compute frequency-binned RMSE."""
        print("  [5/5] Computing fRMSE...")
        try:
            from eval.unified_eval import evaluate_model_on_loader

            dev = torch.device(self.device)
            eval_metrics = evaluate_model_on_loader(
                self._model, self._test_loader, dev, "integration_test"
            )

            print(f"        fRMSE low:  {eval_metrics.frmse_low:.4f}")
            print(f"        fRMSE mid:  {eval_metrics.frmse_mid:.4f}")
            print(f"        fRMSE high: {eval_metrics.frmse_high:.4f}")

            assert np.isfinite(eval_metrics.frmse_low), "fRMSE low is non-finite"
            assert np.isfinite(eval_metrics.frmse_mid), "fRMSE mid is non-finite"
            assert np.isfinite(eval_metrics.frmse_high), "fRMSE high is non-finite"
        except Exception as e:
            self.errors.append(f"fRMSE failed: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end integration test for DDE-FNO pipeline.",
    )
    parser.add_argument("--dde-family", default="hutch", help="DDE family to test (default: hutch).")
    parser.add_argument("--pde-family", default="burgers", help="PDE family to test (default: burgers).")
    parser.add_argument("--skip-dde", action="store_true", help="Skip DDE test.")
    parser.add_argument("--skip-pde", action="store_true", help="Skip PDE test.")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per split (default: 100).")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu).")
    args = parser.parse_args()

    results = {}

    if not args.skip_dde:
        print(f"\n{'='*60}")
        print(f"  INTEGRATION TEST: DDE ({args.dde_family})")
        print(f"{'='*60}")
        test = IntegrationTest(
            args.dde_family, is_pde=False,
            n_samples=args.n_samples, epochs=args.epochs, device=args.device,
        )
        passed = test.run()
        results[args.dde_family] = {"passed": passed, "errors": test.errors}
        status = "PASSED" if passed else "FAILED"
        print(f"\n  => DDE ({args.dde_family}): {status}")
        if test.errors:
            for e in test.errors:
                print(f"     ERROR: {e}")

    if not args.skip_pde:
        print(f"\n{'='*60}")
        print(f"  INTEGRATION TEST: PDE ({args.pde_family})")
        print(f"{'='*60}")
        test = IntegrationTest(
            args.pde_family, is_pde=True,
            n_samples=args.n_samples, epochs=args.epochs, device=args.device,
        )
        passed = test.run()
        results[args.pde_family] = {"passed": passed, "errors": test.errors}
        status = "PASSED" if passed else "FAILED"
        print(f"\n  => PDE ({args.pde_family}): {status}")
        if test.errors:
            for e in test.errors:
                print(f"     ERROR: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, res in results.items():
        status = "PASSED" if res["passed"] else "FAILED"
        print(f"  {name}: {status}")
        if not res["passed"]:
            all_passed = False

    if all_passed:
        print("\nAll integration tests PASSED.")
        sys.exit(0)
    else:
        print("\nSome integration tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
