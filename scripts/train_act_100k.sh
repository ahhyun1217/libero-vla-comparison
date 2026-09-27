#!/bin/bash
# ACT 재학습: libero_goal task0, 100k 스텝 (대기 루프 없음 - GPU는 이미 비어있음)
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EPS="399,405,410,421,437,443,463,469,473,490,496,497,516,519,525,541,547,552,561,566,577,585,599,611,631,633,659,664,672,678,687,693,703,722,725,734,754,765,772,779,780,792,802"
echo "[시작] $(date '+%H:%M:%S')"
./.venv/bin/lerobot-train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero --dataset.episodes="[$EPS]" \
  --steps=100000 --batch_size=8 --save_freq=20000 --log_freq=1000 \
  --output_dir=./outputs/train/act_libero_goal_task0_100k \
  --job_name=act_libero_goal_task0_100k
echo "[완료] $(date '+%H:%M:%S')"
