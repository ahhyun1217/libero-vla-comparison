#!/bin/bash
# 최종 측정. 조건당 300 episode (libero_spatial 10 task x 30) 로 표본을 늘려
# 표준오차를 ±2.9 %p 까지 줄임. n=100 예비 측정의 방향이 노이즈인지 확인하는 용도.
#
# in-dist(init_states=true) 와 OOD(init_states=false) 를 treatment/control 각각 측정함.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
source "$HERE/eval_lib.sh"
mkdir -p "$LOG_DIR"

RESULT="$LOG_DIR/tight.log"
echo "[tight] START 조건당 300 episode $(date)" > "$RESULT"

trace_eval tight_trace_id  "$TRACK_B/train_trace_out" 30 true  "$RESULT"
trace_eval tight_ctrl_id   "$TRACK_B/train_ctrl_out"  30 true  "$RESULT"
trace_eval tight_trace_ood "$TRACK_B/train_trace_out" 30 false "$RESULT"
trace_eval tight_ctrl_ood  "$TRACK_B/train_ctrl_out"  30 false "$RESULT"

echo "[tight] ALL DONE $(date)" >> "$RESULT"
