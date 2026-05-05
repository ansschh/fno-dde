#!/bin/bash
# Step 1: Kill stale processes and verify clean state.
echo '=== STEP 1: KILL STALE PROCESSES ==='
pkill -9 -f '_bootstrap_runpod_offload' 2>/dev/null
pkill -9 -f '_bootstrap_runpod_orbit' 2>/dev/null
pkill -9 -f '_launch_caltech_offload' 2>/dev/null
pkill -9 -f '_run_offload_cell' 2>/dev/null
pkill -9 -f 'gen_dde_pde_data' 2>/dev/null
pkill -9 -f 'gen_orbit_ood_data' 2>/dev/null
pkill -9 -f 'train_apebench_smoke' 2>/dev/null
pkill -9 -f '_smoke_models' 2>/dev/null
sleep 3
echo '--- after kill, remaining processes (should be empty): ---'
ps aux | grep -E '_bootstrap|gen_dde_pde|gen_orbit|train_apebench|_smoke_models|_launch_caltech|_run_offload' | grep -v grep
echo '--- end remaining ---'
echo ''
echo '=== STEP 2: CLEAN PRIOR FAILED RUN STATE ==='
if [ -d /workspace/dde-fno ]; then
  cd /workspace/dde-fno
  rm -rf data_dde_pde data_orbit_ood outputs/_smoke outputs/_orbit_smoke train_logs/offload train_logs/orbit data_gen_*.log
  echo 'cleaned data_dde_pde, data_orbit_ood, outputs/_smoke, outputs/_orbit_smoke, train_logs/offload, train_logs/orbit, data_gen_*.log'
else
  echo 'WARNING: /workspace/dde-fno does not exist yet'
fi
echo ''
echo '=== STEP 3: PULL v3+ CODE ==='
cd /workspace
if [ ! -d dde-fno/.git ]; then
  echo 'cloning fresh...'
  git clone https://github.com/ansschh/fno-dde.git dde-fno
fi
cd /workspace/dde-fno
git fetch origin main 2>&1 | tail -5
git reset --hard origin/main 2>&1 | tail -3
echo '--- HEAD log: ---'
git log --oneline -3
echo '--- end HEAD log ---'
echo ''
echo '=== STEP 4: CHECK BOOTSTRAP SCRIPT EXISTS ==='
ls -la scripts/_bootstrap_runpod_orbit.sh 2>&1
ls -la scripts/gen_orbit_ood_data.py 2>&1
ls -la scripts/train_apebench_smoke.py 2>&1
echo '=== END PRECHECK ==='
exit
