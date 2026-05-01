#!/usr/bin/env bash
# Run from your laptop AFTER you've SSH'd into Caltech HPC at least once
# (so Duo 2FA caches your session).
#
# This script bundles code + data and scp's it to your $HOME/dde-fno on
# Caltech HPC.  It does NOT submit any SLURM jobs — that's a separate step.
#
# Usage:
#   bash slurm/deploy_to_caltech.sh
#
# Prerequisites:
#   - You can `ssh caltech-hpc` without password prompt
#   - Your $HOME on Caltech HPC has at least 30 GB free
#   - Local repo at A:/dde research/dde-fno/

set -e
LOCAL_REPO="/a/dde research/dde-fno"
REMOTE=caltech-hpc
REMOTE_REPO="\$HOME/dde-fno"

cd "$LOCAL_REPO"

echo "=== STAGE 1: bundle code ==="
tar czf /tmp/lemo_code.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  --exclude='outputs' --exclude='bundles' --exclude='extracted' \
  --exclude='data_*' --exclude='reports' --exclude='paper/figures/*.pdf' \
  src/ scripts/ configs/ slurm/ requirements.txt
du -h /tmp/lemo_code.tar.gz

echo ""
echo "=== STAGE 2: bundle dist_kernel data ==="
# Only ship the 5 dist_kernel families (others not needed for σ-sweep)
tar czf /tmp/lemo_data.tar.gz \
  -C data_dde_pde \
  $(ls data_dde_pde | grep dist_ | grep -v dist_delay)
du -h /tmp/lemo_data.tar.gz

echo ""
echo "=== STAGE 3: scp to Caltech (~600 MB, 2-5 min) ==="
ssh ${REMOTE} "mkdir -p ${REMOTE_REPO} ${REMOTE_REPO}/slurm_logs ${REMOTE_REPO}/data_dde_pde"
scp /tmp/lemo_code.tar.gz ${REMOTE}:${REMOTE_REPO}/
scp /tmp/lemo_data.tar.gz ${REMOTE}:${REMOTE_REPO}/

echo ""
echo "=== STAGE 4: extract on Caltech ==="
ssh ${REMOTE} "cd ${REMOTE_REPO} && \
  tar xzf lemo_code.tar.gz --no-same-owner && \
  tar xzf lemo_data.tar.gz --no-same-owner -C data_dde_pde && \
  rm lemo_code.tar.gz lemo_data.tar.gz && \
  ls data_dde_pde && \
  echo 'extracted OK'"

echo ""
echo "=== DEPLOY DONE ==="
echo "Now SSH to Caltech and run:"
echo "    ssh caltech-hpc"
echo "    cd ~/dde-fno"
echo "    bash slurm/launch_caltech.sh"
