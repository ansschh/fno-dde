"""Run a single fill-table-gaps cell by index. Mirrors `_run_a_fix_cell.py`."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    cell_idx = int(sys.argv[1])
    gpu_id = int(sys.argv[2])
    sys.path.insert(0, str(REPO / "scripts"))
    from _fill_table_gaps_cells import all_cells
    cells = all_cells()
    if cell_idx >= len(cells):
        print(f"[run-fill] cell_idx {cell_idx} out of range (have {len(cells)})", file=sys.stderr)
        sys.exit(2)
    cell = cells[cell_idx]
    args = cell["args"]
    out_idx = args.index("--output_dir")
    out_dir = args[out_idx + 1]
    fam = cell["fam"]; reg = cell["reg"]; seed = cell["seed"]; mdl = cell["model"]
    base = Path(out_dir)
    if not base.is_absolute():
        base = REPO / base
    result_path = base / fam / reg / mdl / f"s{seed}" / "test_results.json"
    if result_path.exists() and result_path.stat().st_size > 0:
        print(f"[run-fill] cell {cell_idx}: SKIP (already complete)")
        sys.exit(0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("OMP_NUM_THREADS", "4")
    cmd = [sys.executable, "-u", str(REPO / "scripts" / "train_apebench_smoke.py")] + list(args)
    rc = subprocess.call(cmd, env=env, cwd=str(REPO))
    sys.exit(rc)
