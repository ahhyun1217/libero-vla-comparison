#!/bin/bash
# 학습(PID 31296) 종료를 기다렸다가 ACT 100k 체크포인트를 평가
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

while kill -0 31296 2>/dev/null; do sleep 30; done
echo "[학습 종료 감지] $(date '+%H:%M:%S')"

CKPT=./outputs/train/act_libero_goal_task0_100k/checkpoints/100000/pretrained_model
[ -d "$CKPT" ] || CKPT=$(ls -d ./outputs/train/act_libero_goal_task0_100k/checkpoints/*/pretrained_model 2>/dev/null | tail -1)
echo "[평가 체크포인트] $CKPT"

SAMPLES=/tmp/vram3_act100k.txt; : > "$SAMPLES"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SAMPLES"; sleep 0.3; done ) &
SAMPLER=$!

T0=$(date +%s)
./.venv/bin/lerobot-eval --policy.path="$CKPT" --policy.device=cuda \
  --env.type=libero --env.task=libero_goal --env.task_ids='[0]' \
  --eval.batch_size=1 --eval.n_episodes=10 --seed=1000 > /tmp/eval3_act100k.log 2>&1
T1=$(date +%s)
kill $SAMPLER 2>/dev/null

PEAK=$(sort -n "$SAMPLES" | tail -1)
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/eval3_act100k.log | tail -1 | grep -oE "[0-9.]+$")
NE=$(grep -oE "'n_episodes': [0-9]+" /tmp/eval3_act100k.log | tail -1 | grep -oE "[0-9]+$")
EPS=$(grep -oE "'eval_ep_s': [0-9.]+" /tmp/eval3_act100k.log | tail -1 | grep -oE "[0-9.]+$")
printf "%-11s | 성공률 %6s%% | 에피소드 %3s | 에피당 %6.1fs | VRAM peak %6s MiB | 총 %ds\n" \
  "act-100k" "${SR:-ERR}" "${NE:-?}" "${EPS:-0}" "${PEAK:-?}" "$((T1-T0))" | tee -a /home/leap/ahhyun/libero-vla/env/compare3_results.txt
echo "[전체 완료] $(date '+%H:%M:%S')"
