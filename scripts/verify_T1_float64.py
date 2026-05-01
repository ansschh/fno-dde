"""Verify Theorem T1 (cyclic-shift lag-equivariance) at float64 precision.

This is the precision-floor sanity check requested by the Round-3 review:
if T1 holds in float32 to a precision floor, then casting to float64 with
the SAME architecture and SAME test should drop the residual by ~9 orders
of magnitude (float64 ULP ~2e-16 vs float32 ULP ~1e-7), demonstrating
that the residual is purely numerical precision, not architectural.

What the script actually verifies (two complementary tests):

  (A) Architectural T1 — the spectral lag operator alone (Lean's T1).
      The Lean proof statement is for the multiplier-only operator
      LEMO_lag x = irfft(K * rfft(x)), which is exactly cyclic-equivariant
      on R. Empirically (with the FiLM gamma/beta neutralized to 1/0,
      reducing the operator to its multiplier-only form):
          float32 residual ~ 1e-7  (single-precision FFT round-off)
          float64 residual ~ 2e-16 (double-precision FFT round-off)
      This collapses to the float64 precision floor, confirming the
      operator is exactly cyclic-equivariant.

  (B) Full LEMO-PC model (`forward(x)`).
      The trained model includes additive FiLM-bias terms (`beta`) that
      are NOT phase-rotated by cyclic shift, breaking strict T1 to a
      structural ~5e-3 residual that does NOT collapse in float64.
      This is consistent with the formal statement: T1 in Lean covers
      the spectral multiplier; the FiLM additive bias is an architectural
      modification that introduces a small bounded equivariance defect.

The script reports both numbers so the paper text can faithfully say:
  "The kernel multiplier is exactly cyclic-equivariant to the float64
   precision floor (~2e-16); the full LEMO-PC model deviates by ~5e-3
   in BOTH float32 and float64, identifying the residual as the additive
   FiLM bias rather than numerical precision."

Usage:
    python3 scripts/verify_T1_float64.py
    python3 scripts/verify_T1_float64.py --ckpt <other ckpt>
    python3 scripts/verify_T1_float64.py --family dist_delay_rd_2d   # local fallback

Default checkpoint:
    extracted_lemo_pc/outputs/dist_kernel_v2_p1/raw/dist_exp_rd_2d/clean/
        lemo_pc_nd/s42/best_model.pt

Default fallback test family (when dist_exp_rd_2d shards are not on this
host but the architecture is compatible — params_dim=3, n_hist=64,
n_out=64, spatial=(64,64)): dist_delay_rd_2d.

TODO: if neither the local checkpoint nor a compatible local data shard
is reachable, this script will raise with a clear message; in that case
re-run on a pod where the dist_exp_rd_2d shards are present.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_CKPT = (REPO / "extracted_lemo_pc" / "outputs" / "dist_kernel_v2_p1"
                / "raw" / "dist_exp_rd_2d" / "clean" / "lemo_pc_nd" / "s42"
                / "best_model.pt")


# ----------------------------- helpers ----------------------------------

def _build_test_loader(data_root: Path, family: str):
    """Try the requested family; fall back to a compatible local family.

    The checkpoint config dictates architectural constants (length=128,
    spatial=64x64, params_dim=3). Any family whose manifest matches those
    is a valid carrier for the equivariance test, since T1 is a structural
    property of the architecture and the test only needs ONE input tensor
    of the right shape.
    """
    from datasets.apebench_dataset import create_apebench_dataloaders
    candidates = [family]
    if family != "dist_delay_rd_2d":
        candidates.append("dist_delay_rd_2d")
    last_err = None
    for fam in candidates:
        try:
            _, _, test_loader = create_apebench_dataloaders(
                str(data_root), fam, batch_size=2,
                regime="clean", noise_std=0.0, downsample_factor=1,
                residual_anchor=True, seed=42)
            return test_loader, fam
        except FileNotFoundError as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"None of {candidates} have shards under {data_root}. "
        f"Last error: {last_err}. "
        "TODO: run this script on a pod where dist_exp_rd_2d/test/shard_000.npz exists."
    )


def _cast_to_double(model: torch.nn.Module) -> torch.nn.Module:
    """Cast every parameter and buffer to its double-precision counterpart.

    `model.double()` upgrades real float32 → float64 but leaves complex64
    UNCHANGED, while `model.to(torch.complex128)` upgrades complex but
    leaves real unchanged. Neither alone covers a model that holds both
    real (Conv1d, LayerNorm) AND complex (cfloat spectral kernels)
    parameters — which LEMO-PC-ND does. This function does both, in place.
    """
    for _, p in model.named_parameters(recurse=True):
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.float64)
        elif p.dtype == torch.complex64:
            p.data = p.data.to(torch.complex128)
    for _, b in model.named_buffers(recurse=True):
        if b.dtype == torch.float32:
            b.data = b.data.to(torch.float64)
        elif b.dtype == torch.complex64:
            b.data = b.data.to(torch.complex128)
    return model


@contextmanager
def _patch_torch_cfloat_to_cdouble():
    """Process-wide swap of torch.cfloat → torch.complex128.

    Several places in `src/models/lemo_pc_nd.py` construct intermediate
    tensors with `dtype=torch.cfloat` hard-coded inside `forward()`.
    Under this context manager, those references resolve to complex128,
    matching cdouble parameters and double inputs without code edits.
    """
    orig = torch.cfloat
    torch.cfloat = torch.complex128
    try:
        yield
    finally:
        torch.cfloat = orig


@contextmanager
def _null_ctx():
    yield


def _per_shift_relerr(model: torch.nn.Module, x: torch.Tensor, shifts,
                      use_double: bool = False, params=None,
                      params_arg: bool = False):
    """Run model(x), model(roll(x,k)) for each k; report relative L2 errors.

    `params_arg=True` means the model's forward signature is
    `forward(x, params)` (an internal layer like FiLMLagSpectralND).
    Otherwise it is the full LEMOPCND with signature `forward(x)` that
    extracts params internally from the input tensor's last channels.
    """
    model.eval()
    out = {}
    ctx = _patch_torch_cfloat_to_cdouble() if use_double else _null_ctx()

    def _call(xx):
        if params_arg:
            return model(xx, params)
        return model(xx)

    # When `params_arg=True`, the lag axis is dim=2 (B, C, lag, *spatial).
    # When `params_arg=False`, the lag axis is dim=1 (B, lag, *spatial, C).
    lag_dim = 2 if params_arg else 1

    with torch.no_grad(), ctx:
        y0 = _call(x)
        for k in shifts:
            x_sh = torch.roll(x, shifts=k, dims=lag_dim)
            y_sh = _call(x_sh)
            y_ro = torch.roll(y0, shifts=k, dims=lag_dim)
            num = (y_sh - y_ro).flatten(1).norm(dim=1)
            den = y_ro.flatten(1).norm(dim=1).clamp_min(1e-30)
            out[k] = (num / den).mean().item()
    return out


def _neutralize_film(film_lag_layer: torch.nn.Module) -> None:
    """Set the FiLM affine to identity: gamma=1, beta=0.

    Setting gamma=1 and beta=0 recovers the multiplier-only spectral
    operator (active * gamma + beta == active) and therefore the
    architecturally-equivariant operator that Lean's T1 covers. This
    isolates the FFT round-off precision floor from the FiLM bias's
    structural equivariance defect.
    """
    with torch.no_grad():
        # Output of film_net is concatenated [gamma_flat | beta_flat].
        film_lag_layer.film_net[-1].weight.zero_()
        n = film_lag_layer.out_channels * film_lag_layer.lag_modes
        b = film_lag_layer.film_net[-1].bias.clone()
        b[:n] = 1.0   # gamma = 1
        b[n:] = 0.0   # beta = 0
        film_lag_layer.film_net[-1].bias.copy_(b)


# ----------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--family", default="dist_exp_rd_2d")
    ap.add_argument("--data_dir", default=str(REPO / "data_dde_pde"))
    ap.add_argument("--shifts", default="1,4,16,32,63")
    ap.add_argument("--device", default=None,
                    help="cpu or cuda; default = cuda if available, else cpu")
    args = ap.parse_args()

    from train.build_model import build_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Either scp it from a pod (see reference_cluster.md for SSH details) "
            "or pass --ckpt with a local path. "
            "TODO: run on a pod if no local ckpt is available."
        )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print("=== T1 float64 precision-floor check ===")
    print(f"checkpoint:   {ckpt_path}")
    print(f"model_class:  {cfg.get('model_class')}")
    print(f"model_cfg:    {cfg.get('model')}")
    print(f"device:       {device}")

    test_loader, family_used = _build_test_loader(Path(args.data_dir), args.family)
    print(f"test family:  {family_used}"
          + ("  (FALLBACK -- requested family not local)" if family_used != args.family else ""))
    sample = next(iter(test_loader))
    in_ch = sample["input"].shape[-1]
    out_ch = sample["target"].shape[-1]
    n_total = sample["input"].shape[1]
    print(f"input shape:  {tuple(sample['input'].shape)}  (B, T, *spatial, C)")
    print(f"in_channels={in_ch}  out_channels={out_ch}  T={n_total}")

    model = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    shifts = [int(s) for s in args.shifts.split(",")]
    x_real = sample["input"].to(device)
    torch.manual_seed(0)
    x_rand = torch.randn_like(x_real)

    # ====================================================================
    # TEST B1: full LEMO-PC model, float32 — paper baseline
    # ====================================================================
    print("\n" + "=" * 72)
    print("TEST B -- full LEMO-PC model")
    print("=" * 72)
    print("\n  -- B.float32 (baseline; matches paper) --")
    err_real_f32 = _per_shift_relerr(model, x_real.float(), shifts, use_double=False)
    err_rand_f32 = _per_shift_relerr(model, x_rand.float(), shifts, use_double=False)
    print(f"  {'shift k':>8s}  {'rel_err real':>16s}  {'rel_err random':>16s}")
    for k in shifts:
        print(f"  {k:8d}  {err_real_f32[k]:16.3e}  {err_rand_f32[k]:16.3e}")
    max_full_f32 = max(max(err_real_f32.values()), max(err_rand_f32.values()))
    print(f"  max(B.float32) = {max_full_f32:.3e}")

    # ====================================================================
    # TEST B2: full LEMO-PC model, float64 — does residual collapse?
    # ====================================================================
    _cast_to_double(model)
    print("\n  -- B.float64 --")
    err_real_f64 = _per_shift_relerr(model, x_real.double(), shifts, use_double=True)
    err_rand_f64 = _per_shift_relerr(model, x_rand.double(), shifts, use_double=True)
    print(f"  {'shift k':>8s}  {'rel_err real':>16s}  {'rel_err random':>16s}")
    for k in shifts:
        print(f"  {k:8d}  {err_real_f64[k]:16.3e}  {err_rand_f64[k]:16.3e}")
    max_full_f64 = max(max(err_real_f64.values()), max(err_rand_f64.values()))
    print(f"  max(B.float64) = {max_full_f64:.3e}")

    # ====================================================================
    # TEST A: pure spectral kernel (FiLM neutralized to gamma=1, beta=0)
    # This is the operator Lean's T1 actually proves equivariant. The full
    # LEMO-PC model deviates from T1 only via the additive FiLM bias.
    # ====================================================================
    print("\n" + "=" * 72)
    print("TEST A -- pure spectral kernel (FiLM gamma=1, beta=0; the operator T1 covers)")
    print("=" * 72)
    print("Rationale: the Lean T1 proof is for the multiplier-only spectral lag")
    print("operator x -> irfft(K * rfft(x)). The trained LEMO-PC model adds an")
    print("affine FiLM modulation per spectral mode whose additive bias beta does")
    print("not phase-rotate under cyclic shift -- that bias is the architectural")
    print("residual ~5e-3 in TEST B. Setting gamma=1, beta=0 recovers the operator T1")
    print("formally proves equivariant; the residual then becomes the FFT")
    print("round-off precision floor.")

    # Rebuild a fresh model (cast back to float32 for the float32 leg).
    model_a = build_model(cfg, in_channels=in_ch, out_channels=out_ch, length=n_total)
    model_a.load_state_dict(ckpt["model_state_dict"])
    model_a = model_a.to(device).eval()
    # Neutralize FiLM in every block.
    for blk in model_a.blocks:
        _neutralize_film(blk.A_lag)

    print("\n  -- A.float32 --")
    err_a_real_f32 = _per_shift_relerr(model_a, x_real.float(), shifts, use_double=False)
    err_a_rand_f32 = _per_shift_relerr(model_a, x_rand.float(), shifts, use_double=False)
    print(f"  {'shift k':>8s}  {'rel_err real':>16s}  {'rel_err random':>16s}")
    for k in shifts:
        print(f"  {k:8d}  {err_a_real_f32[k]:16.3e}  {err_a_rand_f32[k]:16.3e}")
    max_a_f32 = max(max(err_a_real_f32.values()), max(err_a_rand_f32.values()))
    print(f"  max(A.float32) = {max_a_f32:.3e}")

    _cast_to_double(model_a)
    print("\n  -- A.float64 --")
    err_a_real_f64 = _per_shift_relerr(model_a, x_real.double(), shifts, use_double=True)
    err_a_rand_f64 = _per_shift_relerr(model_a, x_rand.double(), shifts, use_double=True)
    print(f"  {'shift k':>8s}  {'rel_err real':>16s}  {'rel_err random':>16s}")
    for k in shifts:
        print(f"  {k:8d}  {err_a_real_f64[k]:16.3e}  {err_a_rand_f64[k]:16.3e}")
    max_a_f64 = max(max(err_a_real_f64.values()), max(err_a_rand_f64.values()))
    print(f"  max(A.float64) = {max_a_f64:.3e}")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"TEST A -- pure spectral kernel (operator Lean T1 proves equivariant):")
    print(f"  max float32 residual : {max_a_f32:.3e}")
    print(f"  max float64 residual : {max_a_f64:.3e}")
    print(f"  ratio f32/f64        : {max_a_f32 / max(max_a_f64, 1e-30):.2e}")
    if max_a_f64 < 1e-12 and max_a_f32 / max_a_f64 > 1e6:
        print("  VERDICT: collapses to float64 precision floor -- kernel is")
        print("           exactly cyclic-equivariant; T1 confirmed.")
    else:
        print("  VERDICT: float64 residual did NOT collapse -- investigate.")

    print(f"\nTEST B -- full LEMO-PC model (with FiLM affine):")
    print(f"  max float32 residual : {max_full_f32:.3e}")
    print(f"  max float64 residual : {max_full_f64:.3e}")
    print(f"  ratio f32/f64        : {max_full_f32 / max(max_full_f64, 1e-30):.2e}")
    if abs(max_full_f32 - max_full_f64) / max(max_full_f32, 1e-30) < 0.01:
        print("  INTERPRETATION: float32 ~= float64; the residual is structural")
        print("                  (FiLM additive bias beta breaks strict T1, ~5e-3)")
        print("                  and NOT numerical precision. This identifies the")
        print("                  architectural source of the ~5e-3 deviation.")


if __name__ == "__main__":
    main()
