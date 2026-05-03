"""Run a single B5/P3 cell by global index.

Usage:
    python3 _run_b5_p3_cell.py <cell_idx> <gpu_id>

Mirrors `_run_offload_cell.py` but reads from `_b5_p3_cells.py::all_cells()`.
Idempotent skip on existing test_results.json.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def _arg_value(args, flag):
    try:
        i = args.index(flag)
        return args[i + 1]
    except (ValueError, IndexError):
        return None


def _result_path(cell_args):
    """Predict the location of test_results.json for a cell from its args."""
    args = list(cell_args)
    out_dir = _arg_value(args, "--output_dir")
    fam = _arg_value(args, "--family")
    reg = _arg_value(args, "--regime")
    mdl = _arg_value(args, "--model")
    seed = _arg_value(args, "--seed")
    if not all([out_dir, fam, reg, mdl, seed]):
        return None
    base = Path(out_dir)
    if not base.is_absolute():
        base = REPO / base
    return base / fam / reg / mdl / f"s{seed}" / "test_results.json"


if __name__ == "__main__":
    cell_idx = int(sys.argv[1])
    gpu_id = int(sys.argv[2])
    sys.path.insert(0, str(REPO / "scripts"))
    from _b5_p3_cells import all_cells
    cells = all_cells()
    if cell_idx >= len(cells):
        print(f"[run-cell] cell_idx {cell_idx} out of range (have {len(cells)})", file=sys.stderr)
        sys.exit(2)
    cell = cells[cell_idx]
    result = _result_path(cell["args"])
    if result is not None and result.exists() and result.stat().st_size > 0:
        print(f"[run-cell] cell_idx {cell_idx}: SKIP (already complete: {result})")
        sys.exit(0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("OMP_NUM_THREADS", "4")
    cmd = [sys.executable, "-u", str(REPO / "scripts" / "train_apebench_smoke.py")] + list(cell["args"])
    rc = subprocess.call(cmd, env=env, cwd=str(REPO))
    sys.exit(rc)
