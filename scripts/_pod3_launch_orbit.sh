#!/bin/bash
# Step 4: Launch the orbit-OOD bootstrap detached.
echo '=== LAUNCHING ORBIT-OOD BOOTSTRAP DETACHED ==='
cd /workspace/dde-fno
mkdir -p train_logs/orbit
rm -f train_logs/orbit/_bootstrap.log
nohup setsid bash scripts/_bootstrap_runpod_orbit.sh < /dev/null > train_logs/orbit/_bootstrap.log 2>&1 &
LAUNCH_PID=$!
disown $LAUNCH_PID 2>/dev/null
sleep 3
echo "BOOTSTRAP_LAUNCH_PID=$LAUNCH_PID"
echo '--- check process is alive: ---'
if kill -0 $LAUNCH_PID 2>/dev/null; then
  echo "BOOTSTRAP_PID_ALIVE=1"
  ps -p $LAUNCH_PID -o pid,ppid,sid,pgid,cmd 2>&1
else
  echo "BOOTSTRAP_PID_ALIVE=0 (already exited!)"
fi
echo '--- session check: ---'
ps -ef | grep -E '_bootstrap_runpod_orbit' | grep -v grep
echo '--- bootstrap.log first lines: ---'
sleep 2
head -30 train_logs/orbit/_bootstrap.log 2>&1
echo '=== END LAUNCH ==='
exit
