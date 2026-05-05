#!/bin/bash
# LEMO_σ-Lip pilot launcher: 3 cells across 3 GPUs (1 cell per GPU, parallel).
set -e
cd /workspace/dde-fno

mkdir -p logs/sigma_lip_pilot

CELLS_FILE=$(mktemp)
python3 scripts/_lemo_sigma_lip_cells.py --json > "$CELLS_FILE"
N_CELLS=$(python3 -c "import json; print(len(json.load(open('$CELLS_FILE'))))")
echo "[sigma-lip] $N_CELLS cells"

# Use GPUs 0, 1, 2 — these are likely free after W11 finishes.
# Caller can override with N_GPUS env if needed.
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
echo "[sigma-lip] using GPUs: ${GPUS[@]}"

PIDS=()
for i in $(seq 0 $((N_CELLS - 1))); do
  g_idx=$((i % N_GPUS))
  g=${GPUS[$g_idx]}
  log="logs/sigma_lip_pilot/cell_${i}_gpu${g}.log"
  CUDA_VISIBLE_DEVICES=$g \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python3 -u -c "
import json, subprocess, sys
cells = json.load(open('$CELLS_FILE'))
c = cells[$i]
args = c['args']
fam = args[args.index('--family') + 1]
seed = args[args.index('--seed') + 1]
print(f'[gpu $g cell $i] starting {fam} s{seed}', flush=True)
rc = subprocess.call(['python3', '-u', 'scripts/train_apebench_smoke.py'] + args)
print(f'[gpu $g cell $i] rc={rc}', flush=True)
" > "$log" 2>&1 &
  PIDS+=("$!")
  echo "[sigma-lip] launched cell $i on gpu $g (pid $!)"
done

echo "[sigma-lip] ${#PIDS[@]} cells running. Waiting..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "  cell pid $pid exit nonzero"
done
echo "=== sigma-lip pilot done ==="
rm -f "$CELLS_FILE"

# Quick result summary
echo ""
echo "[sigma-lip] results:"
for d in outputs/lemo_sigma_lip_pilot_runpod/raw/*/clean/lemo_sigma_lip_nd/s*; do
  if [ -f "$d/test_results.json" ]; then
    rel=$(python3 -c "import json; d=json.load(open('$d/test_results.json')); print(f'{d[\"test_rel_l2_mean\"]:.4f}')")
    echo "  $d: $rel"
  fi
done
