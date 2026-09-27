#!/bin/bash
# Pi0.5 / GR00T 를 HF 공식 경량(bf16 7.5GB) 체크포인트로 교체 실행
#   기존 로컬본 문제: pi05_libero(14GB)=RAM부족 / pi05_libero6k=타입미등록 / groot_lr_*(12GB)=RAM부족
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/home/leap/ahhyun/libero-vla/env/compare_fix2.txt
: > "$OUT"
echo "=== Pi0.5 / GR00T (HF 공식 bf16) / seed=1000 / n_action_steps=10 / 10과제x2ep ===" | tee -a "$OUT"

run () {
  LABEL="$1"; CKPT="$2"; SUITE="$3"; TAG="$4"
  SF=/tmp/f2_$TAG.vram; : > "$SF"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
  SP=$!
  T0=$(date +%s)
  ./.venv/bin/lerobot-eval --policy.path="$CKPT" --policy.device=cuda --policy.n_action_steps=10 \
    --env.type=libero --env.task="$SUITE" --env.episode_length=300 \
    --eval.batch_size=1 --eval.n_episodes=2 --seed=1000 > /tmp/f2_$TAG.log 2>&1
  T1=$(date +%s); kill $SP 2>/dev/null
  PEAK=$(sort -n "$SF" | tail -1)
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/f2_$TAG.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-9s | %-14s | 성공률 %6s%% | VRAM %6s MiB | %4ds\n" "$LABEL" "$SUITE" "${SR:-ERR}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
}

for S in goal spatial object; do
  run pi05  lerobot/pi05_libero_finetuned_v044 libero_$S pi05_$S
done
for S in goal spatial object; do
  run groot aractingi/lerobot-groot-ft-libero  libero_$S groot_$S
done
echo "=== 완료 ===" | tee -a "$OUT"
