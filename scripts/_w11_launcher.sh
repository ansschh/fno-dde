#!/bin/bash
# W11 compute-matched FNO@400ep launcher: 8-way parallel across GPUs.
# 15 cells (5 fams × 3 seeds × clean) → 2 cells/GPU on average.
set -e
cd /workspace/dde-fno

N_GPUS="${N_GPUS:-8}"
mkdir -p logs/w11

# Generate cells JSON to a temp file (avoids bash escape hell)
CELLS_FILE=$(mktemp)
python3 scripts/_w11_compute_matched_cells.py --json > "$CELLS_FILE"
N_CELLS=$(python3 -c "import json; print(len(json.load(open('$CELLS_FILE'))))")
echo "[w11] $N_CELLS cells across $N_GPUS GPUs"

# Per-GPU shard files: list of cell indices
TMPDIR=$(mktemp -d)
for g in $(seq 0 $((N_GPUS - 1))); do
  shard_file="$TMPDIR/shard_$g.txt"
  > "$shard_file"
  for i in $(seq 0 $((N_CELLS - 1))); do
    if [ $((i % N_GPUS)) -eq $g ]; then
      echo $i >> "$shard_file"
    fi
  done
done

# Launch one worker per GPU
PIDS=()
for g in $(seq 0 $((N_GPUS - 1))); do
  shard="$TMPDIR/shard_$g.txt"
  if [ ! -s "$shard" ]; then continue; fi
  log="logs/w11/gpu${g}.log"
  CUDA_VISIBLE_DEVICES=$g \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python3 -u -c "
import json, subprocess, sys
from pathlib import Path
cells = json.load(open('$CELLS_FILE'))
shard_idxs = [int(line.strip()) for line in open('$shard') if line.strip()]
print(f'[gpu $g] processing {len(shard_idxs)} cells', flush=True)
for idx in shard_idxs:
    cell = cells[idx]
    args = cell['args']
    out_dir = args[args.index('--output_dir') + 1]
    fam = args[args.index('--family') + 1]
    seed = args[args.index('--seed') + 1]
    marker = Path(out_dir) / fam / 'clean' / 'fno_nd' / f's{seed}' / 'test_results.json'
    if marker.exists() and marker.stat().st_size > 0:
        print(f'[gpu $g] cell {idx}: SKIP (already done)', flush=True)
        continue
    print(f'[gpu $g] cell {idx}: starting {fam} s{seed}', flush=True)
    rc = subprocess.call(['python3', '-u', 'scripts/train_apebench_smoke.py'] + args)
    print(f'[gpu $g] cell {idx}: rc={rc}', flush=True)
print(f'[gpu $g] all cells done', flush=True)
" > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[w11] launched gpu $g (pid $!)"
done

echo "[w11] ${#PIDS[@]} workers running. Waiting..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "  worker pid $pid exit nonzero"
done
echo "=== W11 complete ==="
rm -rf "$TMPDIR" "$CELLS_FILE"
