#!/bin/bash
# One-shot bootstrap for a fresh RunPod instance for the Caltech offload.
#
# Usage: bash _bootstrap_runpod_offload.sh <START> <COUNT> <NGPU>
#
# Idempotent — safe to re-run if interrupted.
set -e

cd /workspace 2>/dev/null || cd /root
mkdir -p /workspace && cd /workspace

# 1. Clone or update repo (must be at HEAD with FiLM fix included).
if [ ! -d dde-fno/.git ]; then
  git clone https://github.com/ansschh/fno-dde.git dde-fno
fi
cd dde-fno
git fetch origin main
git reset --hard origin/main
git log --oneline -3

# Sanity: assert FiLM fix commit is reachable.
if ! git merge-base --is-ancestor 842be7c HEAD 2>/dev/null; then
  echo "[bootstrap] ERROR: FiLM fix commit 842be7c is not in the ancestry of HEAD. Aborting."
  exit 2
fi

# 2. Install Python deps (uses existing python3, no venv to keep it simple
# on RunPod where the system python is already 3.10+ with pytorch baked in).
pip install -q numpy scipy matplotlib h5py pyyaml tqdm einops 2>&1 | tail -3
python3 -c "import torch; print(f'[bootstrap] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, ngpu: {torch.cuda.device_count()}')"

# 3. Generate data if missing. The dist_*_rd_2d data is a few hundred MB
# total and takes ~25-30 min single-process; we parallelize one family per
# core for faster bring-up.
mkdir -p data_dde_pde
need_gen=0
for fam in dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d; do
  if [ ! -f data_dde_pde/$fam/manifest.json ] || [ ! -f data_dde_pde/$fam/test/shard_000.h5 ]; then
    need_gen=1
    break
  fi
done
if [ $need_gen -eq 1 ]; then
  echo "[bootstrap] generating dist_*_rd_2d data (parallel by family)..."
  # Use the SAME args as slurm/gen_data.sbatch (validated on Caltech).
  # Default dt=0.05 violates the stability check (sf=5.12 > 2.78); we MUST
  # pass --dt 0.01 and --T_total 1.28 (not the argparse default 16.0).
  for fam in dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d; do
    (
      python3 scripts/gen_dde_pde_data.py --family $fam \
        --num_train 1000 --num_val 200 --num_test 200 --num_points 64 \
        --T_total 1.28 --dt 0.01 --n_hist 64 --n_out 64 \
        --out_dir data_dde_pde \
        > data_gen_${fam}.log 2>&1 || echo "[gen] $fam FAILED"
    ) &
  done
  wait
  echo "[bootstrap] data gen done."
else
  echo "[bootstrap] data already present, skipping gen."
fi

# 4. Smoke test: 5-epoch single cell on GPU 0, verify per-epoch < 50s.
SMOKE_LOG=train_logs/offload/_smoke.log
mkdir -p train_logs/offload
echo "[bootstrap] running 5-epoch smoke test on GPU 0..."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python3 -u scripts/train_apebench_smoke.py \
  --family dist_exp_rd_2d --model lemo_pc_nd --regime clean \
  --noise_std 0.05 --downsample_factor 2 \
  --epochs 5 --batch_size 4 \
  --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
  --seed 42 \
  --data_dir data_dde_pde \
  --output_dir outputs/_smoke \
  --residual_anchor \
  > "$SMOKE_LOG" 2>&1
EPOCH_TIME=$(grep -oE 'epoch_time=[0-9.]+' "$SMOKE_LOG" | tail -1 | grep -oE '[0-9.]+' || echo "999")
echo "[bootstrap] smoke test per-epoch: $EPOCH_TIME s"
if [ -n "$EPOCH_TIME" ] && (( $(echo "$EPOCH_TIME > 50" | bc -l) )); then
  echo "[bootstrap] WARNING: per-epoch > 50s ($EPOCH_TIME). Continuing anyway (user override)."
fi

# 5. Launch the actual sweep.
START=${1:-0}; COUNT=${2:-50}; NGPU=${3:-8}
echo "[bootstrap] launching sweep: START=$START, COUNT=$COUNT, NGPU=$NGPU"
nohup bash scripts/_launch_caltech_offload.sh $START $COUNT $NGPU \
  > train_logs/offload/_master.log 2>&1 &
echo "[bootstrap] sweep launched, master PID $!"
echo "[bootstrap] tail -f train_logs/offload/_master.log to watch."
