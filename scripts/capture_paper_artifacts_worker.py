"""Worker process for parallel capture pipeline.

Reads a list of ckpt paths from a file (one per line) and processes each
using `capture_paper_artifacts.process_cell`.  Pinned to a single GPU via
`CUDA_VISIBLE_DEVICES`.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.capture_paper_artifacts import process_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_list", required=True, help="Path to text file with one ckpt path per line")
    ap.add_argument("--data_dir", default="data_dde_pde")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--n_viz_samples", type=int, default=4)
    args = ap.parse_args()

    paths = [Path(line.strip()) for line in open(args.ckpt_list) if line.strip()]
    print(f"[gpu {args.gpu_id}] processing {len(paths)} ckpts")

    for c in paths:
        parts = c.parts
        family = parts[-5]
        cell = "/".join(parts[-5:-1])
        msg = process_cell(c, args.data_dir, family, "cuda", args.n_viz_samples)
        print(f"  {cell}: {msg}", flush=True)


if __name__ == "__main__":
    main()
