#!/bin/bash
echo '=== POD3 ORBIT-OOD PROGRESS CHECK ==='
cd /workspace/dde-fno
echo '--- bootstrap.log tail: ---'
tail -40 train_logs/orbit/_bootstrap.log 2>&1
echo ''
echo '--- gen.log tail (if exists): ---'
tail -20 train_logs/orbit/_gen.log 2>&1
echo ''
echo '--- audit.log tail (if exists): ---'
tail -10 train_logs/orbit/_audit.log 2>&1
echo ''
echo '--- live processes from this experiment: ---'
ps -ef | grep -E '_bootstrap_runpod_orbit|gen_orbit_ood|train_apebench_smoke|_master.sh' | grep -v grep
echo ''
echo '--- data_orbit_ood layout: ---'
ls -la data_orbit_ood/ 2>&1 | head -10
echo ''
echo '--- nvidia-smi short: ---'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 | head -8
echo '=== END PROGRESS CHECK ==='
exit
