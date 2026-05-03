#!/bin/bash
# Restart data gen on a pod after pulling new gen code.
set -e
cd /workspace/dde-fno
pkill -f gen_dde_pde 2>/dev/null || true
pkill -f wait_and_train 2>/dev/null || true
pkill -f train_apebench_smoke 2>/dev/null || true
sleep 2
git pull
rm -rf data_dde_pde gen_logs
mkdir -p data_dde_pde gen_logs train_logs
for fam in dist_exp_rd_2d dist_gaussian_rd_2d dist_gamma_rd_2d dist_uniform_rd_2d dist_powerlaw_rd_2d; do
  nohup python3 -u scripts/gen_dde_pde_data.py --family $fam --dt 0.025 --T_total 8.0 --out_dir data_dde_pde > gen_logs/${fam}.log 2>&1 &
  echo "started $fam pid=$!"
done
sleep 3
echo "Active gens: $(ps aux | grep -c gen_dde_pde)"
