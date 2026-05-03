"""Run a single offload cell by global index.

Usage:
    python3 _run_offload_cell.py <cell_idx> <gpu_id>

Reads the cell at index `cell_idx` from the master cell list
(`_caltech_offload_cells.py:all_cells()`) and invokes
`train_apebench_smoke.py` with that cell's args list.  GPU is set via
CUDA_VISIBLE_DEVICES.

This dispatcher exists because the bash launcher's eval-with-quoting
pattern is fragile across edge cases; doing the dispatch in python with
subprocess.run + a real argv list is bullet-proof.
"""
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
    from _caltech_offload_cells import all_cells
    cells = all_cells()
    if cell_idx >= len(cells):
        print(f"[run-cell] cell_idx {cell_idx} out of range (have {len(cells)})", file=sys.stderr)
        sys.exit(2)
    cell = cells[cell_idx]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("OMP_NUM_THREADS", "4")
    cmd = [sys.executable, "-u", str(REPO / "scripts" / "train_apebench_smoke.py")] + list(cell["args"])
    rc = subprocess.call(cmd, env=env, cwd=str(REPO))
    sys.exit(rc)
