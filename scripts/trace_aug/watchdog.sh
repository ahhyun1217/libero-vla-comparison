#!/bin/bash
# Watchdog for the trace-aug experiment.
# ACTIONS: kill orphaned train_trace processes (PPID==1) that hoard /dev/shm and cause OOM cascades.
# EVENTS (stdout, deduped): ZOMBIE_KILLED / RAM_LOW / ARM_CRASH / CHAIN_DOWN / HEARTBEAT / EXPERIMENT_DONE.
# Exits 0 when the experiment finishes.
cd /home/user/amber/vla_wam
prev_ram=ok; prev_armcrash=0; prev_chain=up; n=0
while true; do
  # --- ACTION: kill orphan zombies (reparented to init) ---
  for p in $(pgrep -f "train_trace.py" 2>/dev/null); do
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    if [ "$ppid" = "1" ]; then
      sh=$(awk '/RssShmem/{print $2}' /proc/$p/status 2>/dev/null)
      kill -9 "$p" 2>/dev/null
      echo "ZOMBIE_KILLED pid=$p shmem=${sh}kB (freed RAM)"
    fi
  done

  # --- experiment finished? ---
  if grep -q "ALL DONE" logs/experiment.log 2>/dev/null; then
    echo "EXPERIMENT_DONE | $(grep RESULT logs/experiment.log | tr '\n' '|')"
    exit 0
  fi

  # --- RAM danger (edge-triggered) ---
  avail=$(free -m | awk '/Mem:/{print $7}')
  if [ "${avail:-9999}" -lt 1500 ]; then
    [ "$prev_ram" != "low" ] && echo "RAM_LOW avail=${avail}MB (OOM risk)"; prev_ram=low
  else prev_ram=ok; fi

  # --- an arm crashed: chain logged a missing checkpoint (edge-triggered) ---
  nc=$(grep -c "NO checkpoint" logs/experiment.log 2>/dev/null)
  if [ "${nc:-0}" != "$prev_armcrash" ]; then
    echo "ARM_CRASH | $(grep 'NO checkpoint' logs/experiment.log | tail -1)"; prev_armcrash=$nc
  fi

  # --- whole chain died before finishing (edge-triggered) ---
  if pgrep -f "run_experiment.sh" >/dev/null 2>&1; then prev_chain=up
  else
    [ "$prev_chain" != "down" ] && echo "CHAIN_DOWN before completion (needs relaunch)"; prev_chain=down
  fi

  # --- heartbeat every ~10 min so liveness is visible ---
  n=$((n+1))
  if [ $((n % 20)) -eq 1 ]; then
    st=$(grep -hoE "step [0-9]+/[0-9]+" logs/train_trace.log logs/train_ctrl.log 2>/dev/null | tail -1)
    echo "HEARTBEAT ${st:-starting} ram_avail=${avail}MB"
  fi
  sleep 30
done
