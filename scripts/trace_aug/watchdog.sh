#!/bin/bash
# trace-aug 실험 감시 스크립트.
#
# ACTION  고아가 된 train_trace 프로세스(PPID==1)를 죽임. 이것들이 /dev/shm 을 붙잡고 있어
#         RAM 이 연쇄적으로 말라 OOM 이 남
# EVENT   stdout 으로 중복 없이 출력됨 (edge-triggered)
#         ZOMBIE_KILLED / RAM_LOW / ARM_CRASH / CHAIN_DOWN / HEARTBEAT / EXPERIMENT_DONE
# 실험이 끝나면 0 으로 종료함.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

LOG="$LOG_DIR/experiment.log"
RAM_FLOOR_MB="${RAM_FLOOR_MB:-1500}"
POLL_SEC="${POLL_SEC:-30}"

prev_ram=ok; prev_armcrash=0; prev_chain=up; n=0

while true; do
  # ACTION: init 으로 재부모화된 고아 프로세스 정리
  for p in $(pgrep -f "train_trace.py" 2>/dev/null); do
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    if [ "$ppid" = "1" ]; then
      sh=$(awk '/RssShmem/{print $2}' "/proc/$p/status" 2>/dev/null)
      kill -9 "$p" 2>/dev/null
      echo "ZOMBIE_KILLED pid=$p shmem=${sh}kB (RAM 회수)"
    fi
  done

  # 실험 종료 확인
  if grep -q "ALL DONE" "$LOG" 2>/dev/null; then
    echo "EXPERIMENT_DONE | $(grep RESULT "$LOG" | tr '\n' '|')"
    exit 0
  fi

  # RAM 위험
  avail=$(free -m | awk '/Mem:/{print $7}')
  if [ "${avail:-9999}" -lt "$RAM_FLOOR_MB" ]; then
    [ "$prev_ram" != "low" ] && echo "RAM_LOW avail=${avail}MB (OOM 위험)"; prev_ram=low
  else
    prev_ram=ok
  fi

  # arm 하나가 죽어서 checkpoint 가 없는 경우
  nc=$(grep -c "checkpoint 없음" "$LOG" 2>/dev/null)
  if [ "${nc:-0}" != "$prev_armcrash" ]; then
    echo "ARM_CRASH | $(grep 'checkpoint 없음' "$LOG" | tail -1)"; prev_armcrash=$nc
  fi

  # 체인 전체가 완주 전에 죽은 경우.
  # pgrep -f 는 자기 자신의 command line 도 매칭하므로 watchdog 의 pid 를 반드시 제외해야 함
  # (제외하지 않으면 영원히 up 으로 보여 감시가 무력화됨)
  if pgrep -f "run_experiment.sh" 2>/dev/null | grep -qv "^$$\$"; then
    prev_chain=up
  else
    [ "$prev_chain" != "down" ] && echo "CHAIN_DOWN 완주 전 종료 (재실행 필요)"; prev_chain=down
  fi

  # 약 10 분마다 살아있음 표시
  n=$((n+1))
  if [ $((n % 20)) -eq 1 ]; then
    st=$(grep -hoE "step [0-9]+/[0-9]+" "$LOG_DIR"/train_*.log 2>/dev/null | tail -1)
    echo "HEARTBEAT ${st:-starting} ram_avail=${avail}MB"
  fi

  sleep "$POLL_SEC"
done
