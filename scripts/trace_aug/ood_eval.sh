#!/bin/bash
# OOD 예비 측정. init_states=false 로 초기 상태 분포를 이동시켜 조건당 100 episode 측정.
# 최종 수치는 tight_eval.sh (조건당 300 episode) 쪽을 봐야 함.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
source "$HERE/eval_lib.sh"
mkdir -p "$LOG_DIR"

RESULT="$LOG_DIR/ood.log"
echo "[ood] START init_states=false (분포이동) $(date)" > "$RESULT"

trace_eval ood_trace_aug "$TRACK_B/train_trace_out" 10 false "$RESULT"
trace_eval ood_control   "$TRACK_B/train_ctrl_out"  10 false "$RESULT"

echo "[ood] ALL DONE $(date)" >> "$RESULT"
