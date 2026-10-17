#!/bin/bash
# trace-aug 실험 본체.
#
# treatment(trace_weight=1.0) 와 control(trace_weight=0.0) 을 동일 설정으로 연달아 학습하고
# 각각 짧게 eval 함. 두 arm 의 차이는 trace_weight 하나뿐인 matched comparison 임.
# 최종 수치는 여기가 아니라 tight_eval.sh (조건당 300 episode) 쪽임.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
source "$HERE/eval_lib.sh"
mkdir -p "$LOG_DIR"

STEPS="${STEPS:-12000}"
LOG="$LOG_DIR/experiment.log"
echo "[exp] START $(date) steps=$STEPS" > "$LOG"

# tag / trace_weight / 출력 디렉터리
ARMS=(
  "trace_aug 1.0 $TRACK_B/train_trace_out"
  "ctrl      0.0 $TRACK_B/train_ctrl_out"
)

for arm in "${ARMS[@]}"; do
  read -r TAG WEIGHT OUT <<< "$arm"
  echo "[exp] === TRAIN $TAG (trace_weight=$WEIGHT) $(date) ===" >> "$LOG"
  rm -rf "$OUT"
  "$PY" "$HERE/train_trace.py" "$WEIGHT" "$OUT" "$STEPS" > "$LOG_DIR/train_$TAG.log" 2>&1
  trace_eval "$TAG" "$OUT" 10 true "$LOG"
done

echo "[exp] ALL DONE $(date)" >> "$LOG"
grep "RESULT" "$LOG" | sed 's/^/[exp] SUMMARY /' >> "$LOG"
