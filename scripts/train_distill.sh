#!/bin/bash
# SmolVLA 를 teacher(MolmoAct2) rollout 으로 학습 (증류)
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SF=/tmp/distill_train.vram; : > "$SF"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 1; done ) >/dev/null 2>&1 &
SP=$!
echo "[시작] $(date '+%H:%M:%S')"
./.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=teacher_rollouts \
  --dataset.root=./outputs/distill/teacher_rollouts \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --steps=20000 --batch_size=8 --save_freq=10000 --log_freq=500 \
  --output_dir=./outputs/train/smolvla_distilled \
  --job_name=smolvla_distilled
RC=$?
kill $SP 2>/dev/null
echo "[완료] $(date '+%H:%M:%S') 종료코드=$RC"
echo "VRAM peak: $(sort -n "$SF" | tail -1) MiB"
