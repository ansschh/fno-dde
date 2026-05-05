#!/bin/bash
# Post-sweep unified batch runner.
#
# Runs all 4 post-training evals on every sweep output dir on this pod,
# then bundles all results (artifacts + best_model.pt checkpoints) into a
# single tarball for SCP back to the user's laptop. Including best_model.pt
# makes the tarball larger (~34 MB / cell instead of ~15 MB / cell) but
# preserves reproducibility: any subsequent eval (new k grid, additional
# perturbation types, regenerated viz with different settings) can be run
# locally without going back to the pod.
#
# Pod-side launch (detached):
#   setsid nohup bash scripts/_post_sweep_batch.sh \
#     < /dev/null > train_logs/_post_sweep_batch.log 2>&1 &
#
# Idempotency: every eval script writes per-cell output files and skips
# cells that already have non-empty results, so re-running is safe.
#
# Output:
#   /workspace/post_sweep_artifacts_<hostname>.tar.gz
#
# After SCP to local:
#   tar -xzf post_sweep_artifacts_<hostname>.tar.gz \
#       -C "A:/dde research/dde-fno/extracted/<pod_label>/"
#   then re-run the figure scripts locally — they auto-pick up new cells.
#
# Artifacts the tarball preserves (per cell, ~34 MB / cell × ~50 cells / pod
# ≈ ~1.7 GB / pod):
#   - best_model.pt       (~19 MB, learned weights — kept for re-running evals)
#   - test_results.json   (final test rel-L2 + config)
#   - history.json        (per-epoch train/val/grad/opK trajectories)
#   - per_frame.json      (per-rollout-step rel-L2)
#   - viz_samples.npz     (input/target/pred fields for V01/V02/M-series)
#   - kernel_snapshot.npz (spectral weights + FiLM γ/β for F4/V05/V06/C22)
#   - residuals.npz       (per-sample rel-L2 + residual norms for A6/A7/T10)
#   - equivariance_dense.json (T1 cyclic-shift error per k for F08)
#   - adv_fgsm.json       (FGSM ε-sweep for F11 left)
#   - noise_sweep.json    (Gaussian σ-sweep for F11 right)

set -uo pipefail   # NOT -e: we want the script to continue on per-eval failures.
cd /workspace/dde-fno

HOST=$(hostname -s)
TS=$(date +%Y%m%d_%H%M)
LOG=train_logs/_post_sweep_batch_${HOST}_${TS}.log
TARBALL=/workspace/post_sweep_artifacts_${HOST}.tar.gz
mkdir -p train_logs
echo "=== post-sweep batch START $(date) host=${HOST} ===" > "$LOG"
echo "[batch] log file: $LOG" | tee -a "$LOG"

# ---------------------------------------------------------------------
# Step 0: discover sweep output dirs.
# ---------------------------------------------------------------------
SWEEP_DIRS=()
for d in outputs/*_runpod outputs/film_fix_full outputs/dist_kernel_v2_p1; do
    [ -d "$d/raw" ] && SWEEP_DIRS+=("$d")
done
if [ "${#SWEEP_DIRS[@]}" -eq 0 ]; then
    echo "[batch] FATAL: no sweep output dirs found under outputs/" | tee -a "$LOG"
    exit 1
fi
echo "[batch] sweep dirs to process (${#SWEEP_DIRS[@]}):" | tee -a "$LOG"
for d in "${SWEEP_DIRS[@]}"; do
    n_ckpts=$(find "$d/raw" -name best_model.pt 2>/dev/null | wc -l)
    echo "[batch]   $d  ($n_ckpts checkpoints)" | tee -a "$LOG"
done

# ---------------------------------------------------------------------
# Step 1: capture_paper_artifacts (per-cell viz, kernel, residuals, per-frame).
#         The script takes a SINGLE --layer_root, so loop per sweep dir.
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STEP 1: capture_paper_artifacts $(date) ===" | tee -a "$LOG"
for dir in "${SWEEP_DIRS[@]}"; do
    echo "[batch] capture: $dir" | tee -a "$LOG"
    python3 -u scripts/capture_paper_artifacts.py \
        --layer_root "$dir" \
        --data_dir data_dde_pde \
        --n_viz_samples 4 \
        >> "$LOG" 2>&1 \
        || echo "[batch]   capture FAILED on $dir (continuing)" | tee -a "$LOG"
done

# ---------------------------------------------------------------------
# Step 2: eval_equivariance_dense (multi-root, dense k grid, clean only).
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STEP 2: eval_equivariance_dense $(date) ===" | tee -a "$LOG"
python3 -u scripts/eval_equivariance_dense.py \
    --roots "${SWEEP_DIRS[@]}" \
    --data_dir data_dde_pde \
    --regimes clean \
    >> "$LOG" 2>&1 \
    || echo "[batch] equiv_dense FAILED (continuing)" | tee -a "$LOG"

# ---------------------------------------------------------------------
# Step 3: eval_adversarial_dense (FGSM ε-sweep).
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STEP 3: eval_adversarial_dense $(date) ===" | tee -a "$LOG"
python3 -u scripts/eval_adversarial_dense.py \
    --roots "${SWEEP_DIRS[@]}" \
    --data_dir data_dde_pde \
    >> "$LOG" 2>&1 \
    || echo "[batch] adv_dense FAILED (continuing)" | tee -a "$LOG"

# ---------------------------------------------------------------------
# Step 4: eval_noise_dense (Gaussian σ-sweep).
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STEP 4: eval_noise_dense $(date) ===" | tee -a "$LOG"
python3 -u scripts/eval_noise_dense.py \
    --roots "${SWEEP_DIRS[@]}" \
    --data_dir data_dde_pde \
    >> "$LOG" 2>&1 \
    || echo "[batch] noise_dense FAILED (continuing)" | tee -a "$LOG"

# ---------------------------------------------------------------------
# Step 5: tar everything except best_model.pt and __pycache__.
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STEP 5: tarball $(date) ===" | tee -a "$LOG"
TAR_INCLUDES=()
for d in "${SWEEP_DIRS[@]}"; do TAR_INCLUDES+=("$d"); done
TAR_INCLUDES+=("train_logs")

tar -czf "$TARBALL" \
    --exclude='__pycache__' \
    --exclude='*.tmp' \
    "${TAR_INCLUDES[@]}" \
    >> "$LOG" 2>&1 \
    || { echo "[batch] tar FAILED" | tee -a "$LOG"; exit 2; }

SIZE=$(du -h "$TARBALL" | cut -f1)
N_FILES=$(tar -tzf "$TARBALL" 2>/dev/null | wc -l)
echo "[batch] tarball: $SIZE, $N_FILES files at $TARBALL" | tee -a "$LOG"

# ---------------------------------------------------------------------
# Step 6: print SCP-back instructions.
# ---------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== POST-SWEEP BATCH DONE $(date) on $HOST ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Artifact summary written to $TARBALL ($SIZE)." | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Pull to local laptop:" | tee -a "$LOG"
echo "  scp -P <port> -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no \\" | tee -a "$LOG"
echo "      root@<ip>:$TARBALL \\" | tee -a "$LOG"
echo "      'A:/dde research/dde-fno/extracted/post_sweep_${HOST}.tar.gz'" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Then on local:" | tee -a "$LOG"
echo "  cd 'A:/dde research/dde-fno'" | tee -a "$LOG"
echo "  mkdir -p \"extracted/${HOST}\"" | tee -a "$LOG"
echo "  tar -xzf \"extracted/post_sweep_${HOST}.tar.gz\" -C \"extracted/${HOST}/\"" | tee -a "$LOG"
echo "  python scripts/make_phase2_figures.py    # re-renders all figures from the new artifacts" | tee -a "$LOG"
echo "  python scripts/make_paper_figures.py" | tee -a "$LOG"
