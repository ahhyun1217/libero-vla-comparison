#!/bin/bash
# Shallow-pi 방식 증류: X-VLA 24층(teacher) -> 12층(student)
#
# 우리 SmolVLA 실험이 58% 에서 막힌 원인 가설:
#   student 가 teacher 와 공유 가중치 0 인 상태로 백지에서 출발했다.
# 여기서는 student 를 teacher 의 앞 12층으로 초기화해 그 문제를 제거한다.
#
# 대기 루프를 넣지 말 것: pgrep -f 는 호출자 명령줄까지 잡아 자기 참조로 멈춘다.
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=/home/leap/ahhyun/libero-vla/env/shallow_progress.txt
: > "$LOG"

echo "[1/4] teacher(X-VLA 24층) rollout 수집 $(date '+%H:%M:%S')" | tee -a "$LOG"
rm -rf outputs/distill/_xtest outputs/distill/xvla_rollouts
./.venv/bin/python collect_xvla.py 6 36 ./outputs/distill/xvla_rollouts > collect_xvla.log 2>&1
grep -E "^완료:" collect_xvla.log | tee -a "$LOG"
if [ ! -d outputs/distill/xvla_rollouts/data ]; then
  echo "  수집 실패" | tee -a "$LOG"; tail -3 collect_xvla.log | tee -a "$LOG"; exit 1
fi

echo "[2/4] student(12층) 잘라내기 $(date '+%H:%M:%S')" | tee -a "$LOG"
./.venv/bin/python carve_student.py 12 ./outputs/distill/xvla_student > carve.log 2>&1
grep -E "^완료:" carve.log | tee -a "$LOG"

echo "[3/4] student 학습 $(date '+%H:%M:%S')" | tee -a "$LOG"
rm -rf outputs/train/xvla_student_12
SF=/tmp/shallow.vram; : > "$SF"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 1; done ) >/dev/null 2>&1 &
SP=$!
./.venv/bin/lerobot-train \
  --policy.path=./outputs/distill/xvla_student \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=xvla_rollouts \
  --dataset.root=./outputs/distill/xvla_rollouts \
  --dataset.video_backend=pyav \
  --steps=20000 --batch_size=8 --save_freq=20000 --log_freq=1000 \
  --output_dir=./outputs/train/xvla_student_12 --job_name=xvla_student_12 \
  > train_shallow.log 2>&1
RC=$?
kill $SP 2>/dev/null
echo "      종료코드=$RC | VRAM peak $(sort -n "$SF" | tail -1) MiB | $(date '+%H:%M:%S')" | tee -a "$LOG"
if [ "$RC" -ne 0 ]; then tail -5 train_shallow.log | tee -a "$LOG"; exit 1; fi

echo "[4/4] 평가 $(date '+%H:%M:%S')" | tee -a "$LOG"
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
