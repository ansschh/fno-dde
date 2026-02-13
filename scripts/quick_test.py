"""
Quick Test Script

Generates a tiny dataset and trains briefly to verify the pipeline works.
Use this to check your setup before running full experiments.

Usage:
    python scripts/quick_test.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch


def test_family_definitions():
    """Test that all DDE families are properly defined."""
    print("Testing family definitions...")
    
    from dde.families import DDE_FAMILIES, get_family
    
    for name in DDE_FAMILIES.keys():
        family = get_family(name)
        rng = np.random.default_rng(42)
        
        # Test param sampling
        params = family.sample_params(rng)
        assert len(params) > 0, f"No params for {name}"
        
        # Test history sampling
        t_hist = np.linspace(-2.0, 0.0, 64)
        history = family.sample_history(rng, t_hist)
        assert history.shape[0] == 64, f"Bad history shape for {name}"
        
        # Test delays
        delays = family.get_delays(params)
        assert len(delays) > 0, f"No delays for {name}"
        
        print(f"  {name}: OK")
    
    print("Family definitions: PASS\n")


def test_python_solver():
    """Test the Python DDE solver."""
    print("Testing Python DDE solver...")
    
    from dde.families import get_family
    from dde.solve_python.dde_solver import solve_dde
    
    family = get_family("linear")
    rng = np.random.default_rng(42)
    
    t_hist = np.linspace(-2.0, 0.0, 64)
    history = family.sample_history(rng, t_hist)
    params = family.sample_params(rng)
    
    sol = solve_dde(
        family=family,
        params=params,
        history=history,
        t_hist=t_hist,
        T=5.0,
        n_points=100,
    )
    
    assert sol.success, f"Solver failed: {sol.message}"
    assert sol.y.shape[0] == 100, f"Wrong output shape: {sol.y.shape}"
    assert np.all(np.isfinite(sol.y)), "Non-finite values in solution"
    
    print(f"  Solved linear DDE: y range [{sol.y.min():.3f}, {sol.y.max():.3f}]")
    print("Python solver: PASS\n")


def test_dataset_encoding():
    """Test the dataset encoding for FNO."""
    print("Testing dataset encoding...")
    
    from dde.families import get_family
    from dde.solve_python.dde_solver import solve_dde
    
    family = get_family("hutchinson")
    rng = np.random.default_rng(42)
    
    tau_max = 2.0
    T = 10.0
    n_hist = 64
    n_out = 200
    
    t_hist = np.linspace(-tau_max, 0, n_hist)
    t_out = np.linspace(0, T, n_out)
    t_combined = np.concatenate([t_hist, t_out])
    
    history = family.sample_history(rng, t_hist)
    params = family.sample_params(rng)
    
    sol = solve_dde(family, params, history, t_hist, T, n_out)
    assert sol.success
    
    # Build FNO input encoding
    n_total = n_hist + n_out
    state_dim = family.config.state_dim
    n_params = len(family.config.param_names)
    
    # History signal
    hist_signal = np.zeros((n_total, state_dim))
    hist_signal[:n_hist] = history
    
    # Mask
    mask = np.zeros((n_total, 1))
    mask[:n_hist] = 1.0
    
    # Time
    t_norm = (t_combined - t_combined.min()) / (t_combined.max() - t_combined.min())
    t_channel = t_norm.reshape(-1, 1)
    
    # Params
    param_vec = np.array([params[name] for name in family.config.param_names])
    param_channel = np.tile(param_vec, (n_total, 1))
    
    # Combine
    input_tensor = np.concatenate([hist_signal, mask, t_channel, param_channel], axis=1)
    
    expected_channels = state_dim + 1 + 1 + n_params
    assert input_tensor.shape == (n_total, expected_channels), f"Bad input shape: {input_tensor.shape}"
    
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Expected channels: {expected_channels}")
    print("Dataset encoding: PASS\n")


def test_fno_model():
    """Test the FNO model forward pass."""
    print("Testing FNO model...")
    
    from models.fno1d import FNO1d, count_parameters
    
    batch_size = 4
    seq_length = 264  # 64 hist + 200 out
    in_channels = 6   # state(1) + mask(1) + time(1) + params(3)
    out_channels = 1
    
    model = FNO1d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=16,
        width=32,
        n_layers=4,
    )
    
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")
    
    # Forward pass
    x = torch.randn(batch_size, seq_length, in_channels)
    y = model(x)
    
    assert y.shape == (batch_size, seq_length, out_channels), f"Bad output shape: {y.shape}"
    assert torch.all(torch.isfinite(y)), "Non-finite values in output"
    
    # Backward pass
    loss = y.mean()
    loss.backward()
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print("FNO model: PASS\n")


def test_loss_functions():
    """Test the masked loss functions."""
    print("Testing loss functions...")
    
    from train.train_fno import masked_mse_loss, relative_l2_error
    
    batch_size = 4
    seq_length = 264
    n_hist = 64
    out_channels = 1
    
    pred = torch.randn(batch_size, seq_length, out_channels)
    target = torch.randn(batch_size, seq_length, out_channels)
    
    # Mask: 0 for history, 1 for future
    mask = torch.zeros(batch_size, seq_length)
    mask[:, n_hist:] = 1.0
    
    mse = masked_mse_loss(pred, target, mask)
    rel_l2 = relative_l2_error(pred, target, mask)
    
    assert mse.ndim == 0, "MSE should be scalar"
    assert rel_l2.ndim == 0, "Rel L2 should be scalar"
    assert mse > 0, "MSE should be positive"
    assert rel_l2 > 0, "Rel L2 should be positive"
    
    print(f"  MSE: {mse.item():.6f}")
    print(f"  Rel L2: {rel_l2.item():.4f}")
    print("Loss functions: PASS\n")


def run_all_tests():
    """Run all quick tests."""
    print("="*60)
    print("DDE-FNO Quick Test Suite")
    print("="*60 + "\n")
    
    tests = [
        test_family_definitions,
        test_python_solver,
        test_dataset_encoding,
        test_fno_model,
        test_loss_functions,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}\n")
            failed += 1
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
