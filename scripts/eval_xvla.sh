#!/bin/bash
# X-VLA (0.9B) 평가.
# 주의: 여기에 대기 루프를 넣지 말 것. pgrep -f 는 호출자 명령줄까지 잡아
#       자기 자신을 감지해 영원히 멈춘다 (이 실험에서 두 번 겪음).
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=/home/leap/ahhyun/libero-vla/env/xvla_results.txt
: > "$OUT"
echo "=== libero_10 task6 / seed=1000 / 50 episode / n_action_steps=10 / control_mode=absolute ===" | tee -a "$OUT"

SF=/tmp/xvla.vram
: > "$SF"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
SP=$!

T0=$(date +%s)
./.venv/bin/lerobot-eval \
  --policy.path=lerobot/xvla-libero --policy.device=cuda --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_10 --env.task_ids='[6]' \
  --env.control_mode=absolute --env.episode_length=600 \
  --eval.batch_size=1 --eval.n_episodes=50 --seed=1000 > /tmp/xvla_eval.log 2>&1
T1=$(date +%s)
kill $SP 2>/dev/null

SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/xvla_eval.log | tail -1 | grep -oE "[0-9.]+$")
PEAK=$(sort -n "$SF" | tail -1)
printf "%-12s | 성공률 %6s%% | VRAM %6s MiB | %4ds\n" "X-VLA" "${SR:-ERR}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
echo "=== 완료 ===" | tee -a "$OUT"
