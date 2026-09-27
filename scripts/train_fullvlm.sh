#!/bin/bash
# 가설 검증: SmolVLA 의 병목이 데이터가 아니라 "VLM 을 안 학습한다" 는 설계인가?
#
# 기본값 train_expert_only=True 는 VLM 전체를 requires_grad=False 로 얼린다
# (smolvlm_with_expert.py: set_requires_grad).
# 이를 False 로 바꾸면 VLM 언어층까지 학습된다. vision encoder 는 계속 얼려둔다.
#
# 데이터/스텝/시드는 DAgger 실행과 동일하게 두어 직접 비교 가능하게 한다.
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

SF=/tmp/fullvlm.vram
: > "$SF"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 1; done ) >/dev/null 2>&1 &
SP=$!

echo "[학습 시작] $(date '+%H:%M:%S')"
./.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --dataset.repo_id=dagger_aggregated \
  --dataset.root=./outputs/distill/dagger_aggregated \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --steps=20000 --batch_size=8 --save_freq=20000 --log_freq=1000 \
  --output_dir=./outputs/train/smolvla_fullvlm \
  --job_name=smolvla_fullvlm > train_fullvlm.log 2>&1
RC=$?
kill $SP 2>/dev/null
echo "[학습 종료] $(date '+%H:%M:%S') 코드=$RC"
echo "VRAM peak: $(sort -n "$SF" | tail -1) MiB"

if [ "$RC" -ne 0 ]; then
  echo "학습 실패. 로그 마지막:"
  tail -5 train_fullvlm.log
  exit 1
fi

echo "[평가 시작] $(date '+%H:%M:%S')"
OUT=/home/leap/ahhyun/libero-vla/env/fullvlm_results.txt
: > "$OUT"
echo "=== libero_10 task6 / seed=1000 / 50 episode / n_action_steps=10 ===" | tee -a "$OUT"
./.venv/bin/lerobot-eval \
  --policy.path=./outputs/train/smolvla_fullvlm/checkpoints/020000/pretrained_model \
  --policy.device=cuda --policy.n_action_steps=10 \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --env.type=libero --env.task=libero_10 --env.task_ids='[6]' --env.episode_length=600 \
  --eval.batch_size=1 --eval.n_episodes=50 --seed=1000 > /tmp/fullvlm_eval.log 2>&1
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/fullvlm_eval.log | tail -1 | grep -oE "[0-9.]+$")
printf "%-20s | 성공률 %6s%%\n" "smolvla_VLM학습" "${SR:-ERR}" | tee -a "$OUT"
echo "=== 완료 $(date '+%H:%M:%S') ===" | tee -a "$OUT"
