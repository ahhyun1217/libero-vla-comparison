#!/bin/bash
# ACT를 LIBERO 단일 과제로 학습
#   과제: libero_goal task 0 = "open the middle drawer of the cabinet"
#   데이터셋 task_index=19, 에피소드 43개
# 비교 대상(MolmoAct2 / SmolVLA)과 같은 과제로 평가하기 위한 학습.
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

EPS="399,405,410,421,437,443,463,469,473,490,496,497,516,519,525,541,547,552,561,566,577,585,599,611,631,633,659,664,672,678,687,693,703,722,725,734,754,765,772,779,780,792,802"

./.venv/bin/lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.episodes="[$EPS]" \
  --steps=20000 \
  --batch_size=8 \
  --save_freq=5000 \
  --log_freq=250 \
  --output_dir=./outputs/train/act_libero_goal_task0 \
  --job_name=act_libero_goal_task0
