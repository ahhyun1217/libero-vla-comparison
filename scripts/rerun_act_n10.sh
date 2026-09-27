#!/bin/bash
# ACT(20k, 100k)를 n_action_steps=10 으로 통일해 재평가 — 다른 모델과 완전 동일 조건
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/home/leap/ahhyun/libero-vla/env/compare_fair.txt
: > "$OUT"
echo "=== 완전 동일조건: libero_goal task0 / seed=1000 / 10ep / n_action_steps=10 ===" | tee -a "$OUT"

run_one () {
  NAME="$1"; CKPT="$2"
  S=/tmp/vramf_$NAME.txt; : > "$S"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$S"; sleep 0.3; done ) &
  SAMPLER=$!
  T0=$(date +%s)
  ./.venv/bin/lerobot-eval --policy.path="$CKPT" --policy.device=cuda \
    --policy.n_action_steps=10 \
    --env.type=libero --env.task=libero_goal --env.task_ids='[0]' \
    --eval.batch_size=1 --eval.n_episodes=10 --seed=1000 > /tmp/evalf_$NAME.log 2>&1
  T1=$(date +%s); kill $SAMPLER 2>/dev/null
  PEAK=$(sort -n "$S" | tail -1)
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/evalf_$NAME.log | tail -1 | grep -oE "[0-9.]+$")
  EPS=$(grep -oE "'eval_ep_s': [0-9.]+" /tmp/evalf_$NAME.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-11s | 성공률 %6s%% | 에피당 %6.1fs | VRAM peak %6s MiB | 총 %ds\n" \
    "$NAME" "${SR:-ERR}" "${EPS:-0}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
}

run_one "act-20k"  "./outputs/train/act_libero_goal_task0/checkpoints/020000/pretrained_model"
run_one "act-100k" "./outputs/train/act_libero_goal_task0_100k/checkpoints/100000/pretrained_model"
echo "=== 완료 ===" | tee -a "$OUT"
