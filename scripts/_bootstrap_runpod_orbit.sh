#!/bin/bash
# Pod 3-specific bootstrap for the B3 orbit-OOD experiment.
#
# Generates the orbit-shifted dataset (data_orbit_ood/m{1,2,4,8,16,32}/),
# runs a smoke test, and launches the 36-cell sweep cyclically across all
# 7 GPUs.  Output layout mirrors Caltech's orbit_ood.sbatch:
#   outputs/orbit_ood_runpod/raw/m${M}/dist_exp_rd_2d_orbit/clean/{model}/s{seed}/
#
# Usage: bash _bootstrap_runpod_orbit.sh
set -e

cd /workspace 2>/dev/null || { mkdir -p /workspace && cd /workspace; }
cd /workspace
if [ ! -d dde-fno/.git ]; then
  git clone https://github.com/ansschh/fno-dde.git dde-fno
fi
cd dde-fno
git fetch origin main
git reset --hard origin/main
git log --oneline -3

# Confirm v3+ HEAD.
if ! git merge-base --is-ancestor 1e32e83 HEAD 2>/dev/null; then
  echo "[orbit] WARNING: v3 commit 1e32e83 is not in HEAD ancestry."
fi

# Install deps (idempotent on RunPod's pre-baked images).
pip install -q numpy scipy matplotlib h5py pyyaml tqdm einops 2>&1 | tail -3
python3 -c "import torch; print(f'[orbit] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, ngpu: {torch.cuda.device_count()}')"

mkdir -p train_logs/orbit

# 1. Generate orbit-OOD dataset (CPU-only, ~30 min single-process; we use
#    parallel m-values to speed up — each m_value runs gen for that shift
#    count in a separate process).
need_gen=0
for m in 1 2 4 8 16 32; do
  if [ ! -f data_orbit_ood/m$m/dist_exp_rd_2d_orbit/test/shard_000.npz ]; then
    need_gen=1; break
  fi
done

if [ $need_gen -eq 1 ]; then
  echo "[orbit] generating data_orbit_ood/m{1,2,4,8,16,32}/ ..."
  # The audit-only step is a sanity check — fast (~10s).
  python3 scripts/gen_orbit_ood_data.py --audit_only \
    --tau 0.25 --tau_max 0.4 --dt 0.01 \
    --n_hist 64 --n_out 64 --n_grid 64 \
    --out_dir data_orbit_ood > train_logs/orbit/_audit.log 2>&1 || \
    { echo "[orbit] FATAL: audit failed:"; tail -20 train_logs/orbit/_audit.log; exit 2; }

  # Main gen: a single python invocation iterates all m-values internally.
  python3 scripts/gen_orbit_ood_data.py \
    --num_orbits 256 --num_val_orbits 32 \
    --m_values 1,2,4,8,16,32 \
    --n_hist 64 --n_out 64 --n_grid 64 \
    --dt 0.01 --tau 0.25 --tau_max 0.4 \
    --out_dir data_orbit_ood --seed 42 \
    > train_logs/orbit/_gen.log 2>&1 || \
    { echo "[orbit] FATAL: data gen failed:"; tail -20 train_logs/orbit/_gen.log; exit 3; }
  echo "[orbit] data gen complete:"
  ls -la data_orbit_ood/ | head -10
else
  echo "[orbit] data already present, skipping gen."
fi

# 2. Smoke test: train 5 epochs of one cell, verify save round-trip.
SMOKE_DIR=outputs/_orbit_smoke/dist_exp_rd_2d_orbit/clean/lemo_pc_nd/s42
rm -rf outputs/_orbit_smoke
echo "[orbit] running 5-epoch smoke test on GPU 0..."
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python3 -u scripts/train_apebench_smoke.py \
  --family dist_exp_rd_2d_orbit --model lemo_pc_nd --regime clean \
  --noise_std 0.0 --downsample_factor 1 \
  --epochs 5 --batch_size 4 \
  --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
  --seed 42 \
  --data_dir data_orbit_ood/m1 \
  --output_dir outputs/_orbit_smoke \
  --residual_anchor \
  > train_logs/orbit/_smoke.log 2>&1
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  echo "[orbit] FATAL: smoke trainer exited rc=$SMOKE_RC."
  tail -30 train_logs/orbit/_smoke.log
  exit 4
fi
for f in best_model.pt history.json test_results.json; do
  if [ ! -s "$SMOKE_DIR/$f" ]; then
    echo "[orbit] FATAL: smoke missing $SMOKE_DIR/$f"
    tail -30 train_logs/orbit/_smoke.log
    exit 5
  fi
done
python3 - <<'PYEOF' || { echo "[orbit] FATAL: best_model.pt corrupt"; exit 6; }
import torch
ckpt = torch.load("outputs/_orbit_smoke/dist_exp_rd_2d_orbit/clean/lemo_pc_nd/s42/best_model.pt", map_location="cpu", weights_only=False)
assert "model_state_dict" in ckpt and "config" in ckpt
print(f"[orbit-smoke] best_model.pt loads OK ({len(ckpt['model_state_dict'])} tensors)")
PYEOF
rm -rf outputs/_orbit_smoke
echo "[orbit] smoke PASSED."

# 3. Launch the 36-cell sweep cyclically across 7 GPUs, detached.
NGPU=7
MODELS=(lemo_pc_nd per_lag_mlp_nd)
M_VALUES=(1 2 4 8 16 32)
SEEDS=(42 123 456)

# Build the full 36-cell list as (model_idx, m_idx, seed_idx) tuples.
# Cell 0  = (0, 0, 0) = lemo_pc_nd, m=1, s42
# Cell 35 = (1, 5, 2) = per_lag_mlp_nd, m=32, s456
# Distribute cyclically across NGPU=7 GPUs.

mkdir -p train_logs/orbit
echo "[orbit] launching 36 cells across $NGPU GPUs..."

cat > train_logs/orbit/_master.sh <<'LAUNCHER'
#!/bin/bash
set -e
cd /workspace/dde-fno
NGPU=7
MODELS=(lemo_pc_nd per_lag_mlp_nd)
M_VALUES=(1 2 4 8 16 32)
SEEDS=(42 123 456)
N_SEEDS=${#SEEDS[@]}
N_M=${#M_VALUES[@]}

for gpu in $(seq 0 $((NGPU-1))); do
  (
    for cell_idx in $(seq 0 35); do
      if [ $((cell_idx % NGPU)) -ne $gpu ]; then continue; fi
      seed_idx=$((cell_idx % N_SEEDS))
      m_idx=$(( (cell_idx / N_SEEDS) % N_M ))
      model_idx=$(( cell_idx / (N_SEEDS * N_M) ))
      MODEL=${MODELS[$model_idx]}
      M=${M_VALUES[$m_idx]}
      SEED=${SEEDS[$seed_idx]}
      log=train_logs/orbit/cell_${cell_idx}_m${M}_${MODEL}_s${SEED}.log
      echo "[GPU $gpu] cell $cell_idx: $MODEL m=$M s$SEED -> $log"
      CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 \
        python3 -u scripts/train_apebench_smoke.py \
          --family dist_exp_rd_2d_orbit --model $MODEL --regime clean \
          --noise_std 0.0 --downsample_factor 1 \
          --epochs 200 --batch_size 4 \
          --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
          --seed $SEED \
          --data_dir data_orbit_ood/m${M} \
          --output_dir outputs/orbit_ood_runpod/raw/m${M} \
          --residual_anchor \
          > "$log" 2>&1 \
        || echo "[GPU $gpu] cell $cell_idx FAILED"
      echo "[GPU $gpu] done cell $cell_idx"
    done
    echo "[GPU $gpu] all cells done"
  ) &
done
wait
echo "[orbit-master] all 36 cells complete."
LAUNCHER
chmod +x train_logs/orbit/_master.sh

setsid bash train_logs/orbit/_master.sh \
  < /dev/null \
  > train_logs/orbit/_master.log 2>&1 &
MASTER_PID=$!
echo "[orbit] sweep launched, master PID $MASTER_PID (setsid'd, fully detached)"
sleep 3
if kill -0 $MASTER_PID 2>/dev/null; then
  echo "[orbit] master process confirmed running."
else
  echo "[orbit] WARNING: master PID $MASTER_PID not found 3s after launch — check master log."
  tail -20 train_logs/orbit/_master.log 2>/dev/null
fi
