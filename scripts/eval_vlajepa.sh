#!/bin/bash
# VLA-JEPA (world model 을 학습 시에만 쓰는 VLA) 를 우리 비교 조건으로 평가
#   주의: VLA-JEPA 는 chunk_size=7 이라 n_action_steps 를 10 으로 맞출 수 없음.
#         네이티브 값 7 을 그대로 사용하고 결과에 명시한다.
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/home/leap/ahhyun/libero-vla/env/vlajepa_results.txt
: > "$OUT"
while pgrep -f "bin/lerobot-train|bin/lerobot-eval" > /dev/null; do sleep 20; done

echo "=== libero_10 task6 / seed=1000 / 50 episode / n_action_steps=7 (네이티브) ===" | tee -a "$OUT"
SF=/tmp/vlajepa.vram; : > "$SF"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
SP=$!
T0=$(date +%s)
./.venv/bin/lerobot-eval \
  --policy.path=lerobot/VLA-JEPA-LIBERO --policy.device=cuda \
  --env.type=libero --env.task=libero_10 --env.task_ids='[6]' --env.episode_length=600 \
  --eval.batch_size=1 --eval.n_episodes=50 --seed=1000 > /tmp/vlajepa_eval.log 2>&1
T1=$(date +%s); kill $SP 2>/dev/null
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/vlajepa_eval.log | tail -1 | grep -oE "[0-9.]+$")
PEAK=$(sort -n "$SF" | tail -1)
printf "%-18s | 성공률 %6s%% | VRAM %6s MiB | %4ds\n" "VLA-JEPA" "${SR:-ERR}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
echo "=== 완료 ===" | tee -a "$OUT"
