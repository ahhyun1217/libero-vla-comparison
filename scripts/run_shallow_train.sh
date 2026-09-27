#!/bin/bash
# Shallow-pi 증류 3~4단계 재시도 (수집/잘라내기는 이미 완료)
#
# 1차 시도가 batch_size=8 에서 OOM (15.8GB 도달) 했다.
# X-VLA 는 SmolVLA 와 달리 전체를 학습하므로 훨씬 무겁다. 세 가지로 낮춘다:
#   - batch_size 8 -> 2
#   - VLM encoder 동결 (soft prompt + policy transformer 만 학습)
#   - expandable_segments 로 단편화 완화
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/home/leap/ahhyun/libero-vla/env/shallow_progress2.txt
: > "$LOG"

echo "[1/2] student 학습 시작 $(date '+%H:%M:%S')" | tee -a "$LOG"
rm -rf outputs/train/xvla_student_12
SF=/tmp/shallow2.vram; : > "$SF"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 1; done ) >/dev/null 2>&1 &
SP=$!
./.venv/bin/lerobot-train \
  --policy.path=./outputs/distill/xvla_student \
  --policy.device=cuda --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.freeze_language_encoder=true \
  --dataset.repo_id=xvla_rollouts \
  --dataset.root=./outputs/distill/xvla_rollouts \
  --dataset.video_backend=pyav \
  --steps=20000 --batch_size=2 --save_freq=20000 --log_freq=1000 \
  --output_dir=./outputs/train/xvla_student_12 --job_name=xvla_student_12 \
  > train_shallow.log 2>&1
RC=$?
kill $SP 2>/dev/null
echo "      종료코드=$RC | VRAM peak $(sort -n "$SF" | tail -1) MiB | $(date '+%H:%M:%S')" | tee -a "$LOG"
if [ "$RC" -ne 0 ]; then tail -6 train_shallow.log | tee -a "$LOG"; exit 1; fi

echo "[2/2] 평가 $(date '+%H:%M:%S')" | tee -a "$LOG"
OUT=/home/leap/ahhyun/libero-vla/env/shallow_results.txt
: > "$OUT"
echo "=== libero_10 task6 / seed=1000 / 50 episode / control_mode=absolute ===" | tee -a "$OUT"
./.venv/bin/lerobot-eval \
  --policy.path=./outputs/train/xvla_student_12/checkpoints/020000/pretrained_model \
  --policy.device=cuda --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_10 --env.task_ids='[6]' \
  --env.control_mode=absolute --env.episode_length=600 \
  --eval.batch_size=1 --eval.n_episodes=50 --seed=1000 > /tmp/shallow_eval.log 2>&1
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/shallow_eval.log | tail -1 | grep -oE "[0-9.]+$")
printf "%-22s | 성공률 %6s%%\n" "X-VLA 12층 (student)" "${SR:-ERR}" | tee -a "$OUT"
echo "=== 완료 $(date '+%H:%M:%S') ===" | tee -a "$OUT" | tee -a "$LOG"
