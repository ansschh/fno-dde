#!/usr/bin/env bash
# =============================================================================
# Follow-up sweep — 6h budget, 2 essentials only.
#
# Context: v2 dist-kernel sweep (270 cells, ~4h) ends with LEMO_PC beating UNet
# by 34% on clean+noisy.  Paper headline is locked.  This script closes the
# two remaining gaps that could lose us a reviewer:
#   A) Param-matched UNet (width=64) — proves win is architecture, not params.
#   B) Post-hoc OOD + E_orbit on v2 ckpts — adds the "M3 equivariance" figure
#      and the "tau OOD" figure for free.
#
# Skipped (out-of-budget): single-delay v2 reruns, v1-LEMO sanity, Causal LEMO.
# Causal LEMO code is in the repo (`causal_lemo_pc_nd`) for future runs if
# reviewers ask.
#
# Total: 45 training cells + post-hoc.  ~3h on parallel pods.
# =============================================================================

set -e

DIST_KERNEL_FAMILIES="dist_exp_rd_2d,dist_gaussian_rd_2d,dist_gamma_rd_2d,dist_uniform_rd_2d,dist_powerlaw_rd_2d"
REGIMES="clean,lowres,noisy"
SEEDS="42,123,456"

case "${1:-help}" in

  # -------------------------------------------------------------------------
  # Phase A — UNet width=64 (param-matched).  Run on pod 2 once dist_kernel_v2_p2
  # finishes (free GPU there first since FNO/Markov/Wind are fast).
  # 45 cells × ~50 min / 24 workers ≈ 2h.
  # -------------------------------------------------------------------------
  phase_a)
    echo "=== Phase A: UNet width=64 on dist-kernel ==="
    python3 scripts/run_apebench_sweep.py \
      --datasets ${DIST_KERNEL_FAMILIES} \
      --models unet_nd \
      --regimes ${REGIMES} \
      --seeds ${SEEDS} \
      --epochs 200 --batch_size 4 \
      --width 64 --n_layers 3 --lag_modes 24 --spatial_modes 12 \
      --noise_std 0.05 --downsample_factor 2 \
      --n_workers 24 --n_gpus 8 --residual_anchor \
      --data_dir data_dde_pde \
      --output_dir outputs/followup_a_unet_w64
    ;;

  # -------------------------------------------------------------------------
  # Phase E1 — OOD lag-shift eval (tau outside training range).
  # Reads ckpts from dist_kernel_v2_p{1,2}.  Single-GPU, ~30 min.
  # -------------------------------------------------------------------------
  phase_e_ood)
    echo "=== Phase E1: lag-shift OOD on v2 dist-kernel ckpts ==="
    # Generate OOD test sets if missing.  Skip families the generator does
    # not support (e.g. kernel-parameterized dist_* families: extending the
    # generator is a separate task).
    for fam in dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d; do
      if [ ! -d "data_dde_pde_ood/${fam}" ]; then
        echo "  attempting OOD set for ${fam}"
        python3 scripts/gen_dde_pde_ood.py --family ${fam} \
          --num_train 128 --num_val 64 --num_test 200 \
          2>/dev/null || echo "    skipped ${fam} (not supported by gen_dde_pde_ood.py)"
      fi
    done
    if [ -z "$(ls -A data_dde_pde_ood 2>/dev/null)" ]; then
      echo "  no OOD test sets generated; skipping phase_e_ood"
      exit 0
    fi
    for pod_dir in dist_kernel_v2_p1 dist_kernel_v2_p2 followup_a_unet_w64; do
      [ -d "outputs/${pod_dir}" ] || continue
      echo "  evaluating ${pod_dir}"
      python3 scripts/eval_lag_shift_ood.py \
        --layer5_root outputs/${pod_dir} \
        --ood_data_dir data_dde_pde_ood \
        --out_path outputs/followup_e_ood_${pod_dir}.json || true
    done
    ;;

  # -------------------------------------------------------------------------
  # Phase E2 — Cyclic-shift E_orbit (Metric M3, theorem T1 verification).
  # Reads same v2 ckpts.  Single-GPU, ~30 min.
  # -------------------------------------------------------------------------
  phase_e_eqorbit)
    echo "=== Phase E2: cyclic-shift E_orbit on v2 dist-kernel ckpts ==="
    for pod_dir in dist_kernel_v2_p1 dist_kernel_v2_p2 followup_a_unet_w64; do
      [ -d "outputs/${pod_dir}" ] || continue
      echo "  evaluating ${pod_dir}"
      python3 scripts/eval_equivariance.py \
        --layer5_root outputs/${pod_dir} \
        --data_dir data_dde_pde \
        --out_path outputs/followup_e_eqorbit_${pod_dir}.json \
        --shifts 1,4,16,32
    done
    ;;

  # -------------------------------------------------------------------------
  # Phase capture — post-hoc data capture for paper figures (per-frame relL2,
  # naive-copy baseline, viz samples, kernel snapshots, residuals).  Runs
  # against ALL sweep dirs found locally.  Idempotent.
  # -------------------------------------------------------------------------
  phase_capture)
    echo "=== Phase capture: post-hoc paper artifacts (parallel across 8 GPUs) ==="
    for sweep in outputs/dist_kernel_v2_p* \
                  outputs/layer5_final_sweep_p* \
                  outputs/layer4_audit \
                  outputs/followup_a_unet_w64; do
      [ -d "$sweep" ] || continue
      echo "  capturing on $sweep"
      bash scripts/capture_paper_artifacts_parallel.sh "$sweep" data_dde_pde || true
    done
    ;;

  # -------------------------------------------------------------------------
  # Auto-launcher: detect v2 sweep finish, then run the right phase.
  # Pass `pod1` or `pod2` to indicate which pod this is running on.
  # -------------------------------------------------------------------------
  auto_pod1)
    echo "=== auto_pod1: waiting for dist_kernel_v2_p1 to finish ==="
    # 90 cells expected.  Poll every 60s.
    target=90
    while true; do
      done_count=$(find outputs/dist_kernel_v2_p1 -name test_results.json 2>/dev/null | wc -l)
      running=$(ps -ef | grep train_apebench_smoke | grep -v grep | wc -l)
      echo "  [$(date +%H:%M:%S)] done=${done_count}/${target} running=${running}"
      if [ "$done_count" -ge "$target" ] && [ "$running" -eq 0 ]; then
        break
      fi
      sleep 60
    done
    echo "=== v2_p1 done.  Launching post-hoc evals ==="
    bash "$0" phase_e_ood
    bash "$0" phase_e_eqorbit
    bash "$0" phase_capture
    ;;

  auto_pod2)
    echo "=== auto_pod2: waiting for dist_kernel_v2_p2 to finish ==="
    # 180 cells expected.
    target=180
    while true; do
      done_count=$(find outputs/dist_kernel_v2_p2 -name test_results.json 2>/dev/null | wc -l)
      running=$(ps -ef | grep train_apebench_smoke | grep -v grep | wc -l)
      echo "  [$(date +%H:%M:%S)] done=${done_count}/${target} running=${running}"
      if [ "$done_count" -ge "$target" ] && [ "$running" -eq 0 ]; then
        break
      fi
      sleep 60
    done
    echo "=== v2_p2 done.  Launching Phase A (UNet width=64) ==="
    bash "$0" phase_a
    echo "=== Phase A done.  Launching post-hoc evals on pod 2 ckpts ==="
    bash "$0" phase_e_ood
    bash "$0" phase_e_eqorbit
    bash "$0" phase_capture
    ;;

  help|*)
    echo "Phases:"
    echo "  phase_a          UNet width=64 param-match on dist-kernel"
    echo "  phase_e_ood      OOD lag-shift eval on v2 ckpts"
    echo "  phase_e_eqorbit  E_orbit eval on v2 ckpts -- Metric M3"
    echo "  phase_capture    per-frame relL2 + naive baseline + viz + kernels + residuals"
    echo "  auto_pod1        wait for v2_p1, run E_ood + E_eqorbit + capture"
    echo "  auto_pod2        wait for v2_p2, run phase_a + E_ood + E_eqorbit + capture"
    ;;
esac
