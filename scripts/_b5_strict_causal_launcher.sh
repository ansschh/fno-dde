#!/bin/bash
# B5 strict-causal Toeplitz launcher: 15 cells across 8 GPUs.
#
# Each cell is LEMO-PC with --causal flag (FiLMLagSpectralND uses
# time-domain truncated FIR, not cyclic FFT). 100 epochs.
set -e
cd /workspace/dde-fno

mkdir -p logs/b5_strict_causal

CELLS_FILE=$(mktemp)
python3 scripts/_b5_strict_causal_cells.py --json > "$CELLS_FILE"
N_CELLS=$(python3 -c "import json; print(len(json.load(open('$CELLS_FILE'))))")
echo "[b5-strict] $N_CELLS cells"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
echo "[b5-strict] using GPUs: ${GPUS[@]}"

PIDS=()
for i in $(seq 0 $((N_CELLS - 1))); do
  g_idx=$((i % N_GPUS))
  g=${GPUS[$g_idx]}
  log="logs/b5_strict_causal/cell_${i}_gpu${g}.log"
  CUDA_VISIBLE_DEVICES=$g \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python3 -u -c "
import json, subprocess, sys
from pathlib import Path
cells = json.load(open('$CELLS_FILE'))
c = cells[$i]
args = c['args']
fam = args[args.index('--family') + 1]
seed = args[args.index('--seed') + 1]
out_dir = args[args.index('--output_dir') + 1]
marker = Path(out_dir) / fam / 'clean' / 'lemo_pc_nd' / f's{seed}' / 'test_results.json'
if marker.exists() and marker.stat().st_size > 0:
    print(f'[gpu $g cell $i] SKIP (already done)', flush=True)
    sys.exit(0)
print(f'[gpu $g cell $i] starting {fam} s{seed}', flush=True)
rc = subprocess.call(['python3', '-u', 'scripts/train_apebench_smoke.py'] + args)
print(f'[gpu $g cell $i] rc={rc}', flush=True)
" > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[b5-strict] launched cell $i on gpu $g (pid $!)"
done

echo "[b5-strict] ${#PIDS[@]} cells running. Waiting..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "  cell pid $pid exit nonzero"
done
echo "=== b5 strict-causal pilot done ==="
rm -f "$CELLS_FILE"

echo ""
echo "[b5-strict] results:"
for d in outputs/b5_strict_causal_runpod/raw/*/clean/lemo_pc_nd/s*; do
  if [ -f "$d/test_results.json" ]; then
    rel=$(python3 -c "import json; d=json.load(open('$d/test_results.json')); print(f'{d[\"test_rel_l2_mean\"]:.4f}')")
    echo "  $d: $rel"
  fi
done
