#!/bin/bash
# Per-pod launcher for the Caltech offload.
#
# Usage: bash _launch_caltech_offload.sh <START> <COUNT> <NGPU>
#   e.g. bash _launch_caltech_offload.sh 0 87 8
#
# Distributes cells [START, START+COUNT) cyclically across <NGPU> GPUs;
# each GPU runs its assigned cells SEQUENTIALLY, all GPUs in parallel.
#
# Cell schema (one JSON per line from --slice):
#   {"sweep": "...", "fam": "...", "reg": "...", "seed": 42,
#    "model": "...", "out_dir": "...", "extra_flags": [...]}
set -e
cd /workspace/dde-fno

START=${1:-0}
COUNT=${2:-50}
NGPU=${3:-8}
EPOCHS=${EPOCHS:-200}

mkdir -p train_logs/offload

# Snapshot the cell-list slice so we can read it without re-running python.
python3 scripts/_caltech_offload_cells.py --slice $START $COUNT \
  > train_logs/offload/cells_${START}_${COUNT}.jsonl

NCELLS=$(wc -l < train_logs/offload/cells_${START}_${COUNT}.jsonl)
echo "[launch] $NCELLS cells (indices $START..$((START+COUNT-1))) across $NGPU GPUs"

# Distribute cells cyclically across GPUs; each GPU runs its slice sequentially.
for gpu in $(seq 0 $((NGPU-1))); do
  (
    line=0
    while IFS= read -r cell_json; do
      if [ $((line % NGPU)) -eq $gpu ]; then
        # Parse JSON via python for robustness.
        eval "$(python3 - <<PYEOF
import json, shlex
c = json.loads('''$cell_json''')
print(f"SWEEP={shlex.quote(c['sweep'])}")
print(f"FAM={shlex.quote(c['fam'])}")
print(f"REG={shlex.quote(c['reg'])}")
print(f"SEED={c['seed']}")
print(f"MODEL={shlex.quote(c['model'])}")
print(f"OUT_DIR={shlex.quote(c['out_dir'])}")
print(f"EXTRA={shlex.quote(' '.join(c.get('extra_flags', [])))}")
PYEOF
)"
        log=train_logs/offload/${SWEEP}_${MODEL}_${FAM}_${REG}_s${SEED}.log
        global_idx=$((START + line))
        echo "[GPU $gpu] cell $global_idx ($SWEEP / $MODEL / $FAM / $REG / s$SEED) -> $log"
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 \
          python3 -u scripts/train_apebench_smoke.py \
            --family "$FAM" --model "$MODEL" --regime "$REG" \
            --noise_std 0.05 --downsample_factor 2 \
            --epochs $EPOCHS --batch_size 4 \
            --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
            --seed $SEED \
            --data_dir data_dde_pde \
            --output_dir "$OUT_DIR" \
            $EXTRA \
            > "$log" 2>&1 || echo "[GPU $gpu] cell $global_idx FAILED (continuing)"
        echo "[GPU $gpu] done cell $global_idx"
      fi
      line=$((line + 1))
    done < train_logs/offload/cells_${START}_${COUNT}.jsonl
    echo "[GPU $gpu] all cells done"
  ) &
done
wait
echo "[launch] all $NCELLS cells (indices $START..$((START+COUNT-1))) complete."
