#!/bin/bash
# 수정 재실행:
#  - SmolVLA libero_10 : num2words 재설치 완료
#  - GR00T            : --policy.base_model_path 로 로컬 베이스 지정
#  - Pi0.5            : RAM 15GB 한계로 14GB짜리 대신 pi05_libero6k(7GB, bf16) 사용
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/home/leap/ahhyun/libero-vla/env/compare_fix.txt
: > "$OUT"
echo "=== 수정 재실행 / seed=1000 / n_action_steps=10 / 10과제x2ep ===" | tee -a "$OUT"

M=/home/leap/ahhyun/vla-bench/models-nvme
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'

run () {
  LABEL="$1"; SUITE="$2"; EPLEN="$3"; TAG="$4"; shift 4
  SF=/tmp/fix_$TAG.vram; : > "$SF"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
  SP=$!
  T0=$(date +%s)
  ./.venv/bin/lerobot-eval "$@" \
    --env.type=libero --env.task="$SUITE" --env.episode_length="$EPLEN" \
    --eval.batch_size=1 --eval.n_episodes=2 --seed=1000 > /tmp/fix_$TAG.log 2>&1
  T1=$(date +%s); kill $SP 2>/dev/null
  PEAK=$(sort -n "$SF" | tail -1)
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/fix_$TAG.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-9s | %-14s | 성공률 %6s%% | VRAM %6s MiB | %4ds\n" "$LABEL" "$SUITE" "${SR:-ERR}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
}

# SmolVLA - libero_10 재시도
run smolvla libero_10 600 smol10 --policy.path=lerobot/smolvla_libero --policy.device=cuda --policy.n_action_steps=10 --rename_map="$RENAME"

# GR00T - 로컬 base 지정
for S in goal spatial object; do
  run groot libero_$S 300 groot_$S --policy.path=$M/groot_lr_$S --policy.device=cuda --policy.n_action_steps=10 --policy.base_model_path=$M/groot_base
done

# Pi0.5 - 작은 체크포인트(bf16 7GB)
for S in goal spatial object; do
  run pi05 libero_$S 300 pi05_$S --policy.path=$M/pi05_libero6k --policy.device=cuda --policy.n_action_steps=10
done
echo "=== 완료 ===" | tee -a "$OUT"
