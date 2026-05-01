#!/usr/bin/env bash
# =============================================================================
# Pull EVERYTHING from both RunPod nodes to local laptop.
#
# Run from the local repo: `A:/dde research/dde-fno/`.  Produces:
#   bundles/pod1_outputs.tar.gz   — pod 1 outputs/ + reports/
#   bundles/pod2_outputs.tar.gz   — pod 2 outputs/ + reports/
#   bundles/manifest.json         — what was bundled, sizes, timestamps
#
# Includes:
#   - All sweep outputs (history.json, test_results.json, per_frame.json,
#     viz_samples.npz, kernel_snapshot.npz, residuals.npz, best_model.pt)
#   - Manifests for data_dde_pde and data_apebench (raw shards excluded —
#     can be regenerated from `gen_dde_pde_data.py` / `gen_apebench_data.py`)
#   - reports/ if present
#
# Excluded by default:
#   - Raw NPZ shards (heavy, regenerable)
#   - `.tar.gz` artefacts already on the pod
#   - `__pycache__`
#
# Optional: `--no-ckpts` to drop best_model.pt files (saves ~70% size).
# =============================================================================

set -e

POD1_HOST="root@103.207.149.125"
POD1_PORT="17897"
POD2_HOST="root@103.207.149.137"
POD2_PORT="19573"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_REPO="/root/workspace/dde-fno"
LOCAL_BUNDLES="bundles"

NO_CKPTS=0
[ "${1:-}" = "--no-ckpts" ] && NO_CKPTS=1

mkdir -p "$LOCAL_BUNDLES"

bundle_pod () {
  local label=$1; local host=$2; local port=$3
  local out="$LOCAL_BUNDLES/${label}_outputs.tar.gz"
  echo ""
  echo "=== bundling ${label} (${host}:${port}) → ${out} ==="

  local exclude_args=(
    "--exclude=*.tar.gz"
    "--exclude=__pycache__"
    "--exclude=*/data/*/shard_*.npz"
    "--exclude=data_apebench/*/shard_*.npz"
    "--exclude=data_dde_pde/*/shard_*.npz"
    "--exclude=data_baseline_*/*/shard_*.npz"
    "--exclude=data_ood_*/*/shard_*.npz"
  )
  if [ "$NO_CKPTS" = "1" ]; then
    exclude_args+=("--exclude=best_model.pt")
  fi

  ssh -p "$port" -i "$SSH_KEY" "$host" "
    set -e
    cd ${REMOTE_REPO}
    echo '  remote disk usage:'
    du -sh outputs/ reports/ 2>/dev/null | sed 's/^/    /'
    echo '  collecting manifests + outputs...'
    tar czf - ${exclude_args[@]} \\
      outputs/ \\
      reports/ \\
      data_dde_pde/*/manifest.json \\
      data_apebench/*/manifest.json \\
      2>/dev/null
  " > "$out"

  local size=$(du -h "$out" | cut -f1)
  echo "  → ${size}"
}

# =============================================================================
# Run captures + bundle
# =============================================================================
echo "=== bundle_pods_for_download.sh ==="
echo "    --no-ckpts mode: ${NO_CKPTS}"
date

# Optional: trigger capture pipeline remotely if not yet run.  Idempotent.
trigger_capture () {
  local label=$1; local host=$2; local port=$3
  echo ""
  echo "=== triggering capture pipeline on ${label} ==="
  ssh -p "$port" -i "$SSH_KEY" "$host" "
    cd ${REMOTE_REPO}
    for sweep in outputs/dist_kernel_v2_p* outputs/layer5_final_sweep_p* outputs/layer4_audit outputs/followup_a_unet_w64; do
      [ -d \"\$sweep\" ] || continue
      echo \"  capturing on \$sweep\"
      python3 scripts/capture_paper_artifacts.py \\
        --layer_root \$sweep --data_dir data_dde_pde --device cuda \\
        2>&1 | tail -5
    done
  "
}

if [ "${SKIP_CAPTURE:-0}" != "1" ]; then
  trigger_capture pod1 "$POD1_HOST" "$POD1_PORT"
  trigger_capture pod2 "$POD2_HOST" "$POD2_PORT"
fi

bundle_pod pod1 "$POD1_HOST" "$POD1_PORT"
bundle_pod pod2 "$POD2_HOST" "$POD2_PORT"

# =============================================================================
# Manifest
# =============================================================================
cat > "$LOCAL_BUNDLES/manifest.json" <<EOF
{
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "pod1": {
    "host": "${POD1_HOST}",
    "port": "${POD1_PORT}",
    "tarball": "pod1_outputs.tar.gz",
    "size":   "$(du -h "$LOCAL_BUNDLES/pod1_outputs.tar.gz" | cut -f1)"
  },
  "pod2": {
    "host": "${POD2_HOST}",
    "port": "${POD2_PORT}",
    "tarball": "pod2_outputs.tar.gz",
    "size":   "$(du -h "$LOCAL_BUNDLES/pod2_outputs.tar.gz" | cut -f1)"
  },
  "no_ckpts": ${NO_CKPTS},
  "extracted_to_review": "tar -xzf bundles/pod{1,2}_outputs.tar.gz -C bundles/extracted/"
}
EOF

echo ""
echo "=== bundle complete ==="
cat "$LOCAL_BUNDLES/manifest.json"
echo ""
echo "Untar locally with:"
echo "  mkdir -p bundles/extracted_pod1 bundles/extracted_pod2"
echo "  tar -xzf bundles/pod1_outputs.tar.gz -C bundles/extracted_pod1/"
echo "  tar -xzf bundles/pod2_outputs.tar.gz -C bundles/extracted_pod2/"
