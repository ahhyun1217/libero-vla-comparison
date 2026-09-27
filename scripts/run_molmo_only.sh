#!/bin/bash
# compare3.sh 의 molmoact2 구간만 재실행 (동일 조건) — 결과를 기존 파일에 이어붙임
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NEP=10
OUT=/home/leap/ahhyun/libero-vla/env/compare3_results.txt
SAMPLES=/tmp/vram3_molmoact2.txt; : > "$SAMPLES"

( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SAMPLES"; sleep 0.3; done ) &
SAMPLER=$!

T0=$(date +%s)
./.venv/bin/lerobot-eval \
  --policy.path=allenai/MolmoAct2-LIBERO-LeRobot \
  --policy.inference_action_mode=continuous \
  --policy.model_dtype=bfloat16 --policy.use_amp=true \
  --policy.enable_inference_cuda_graph=true \
  --policy.device=cuda --policy.n_action_steps=10 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
  --env.type=libero --env.task=libero_goal --env.task_ids='[0]' \
  --eval.batch_size=1 --eval.n_episodes=$NEP --seed=1000 \
  > /tmp/eval3_molmoact2.log 2>&1
T1=$(date +%s)
kill $SAMPLER 2>/dev/null

PEAK=$(sort -n "$SAMPLES" | tail -1)
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/eval3_molmoact2.log | tail -1 | grep -oE "[0-9.]+$")
NE=$(grep -oE "'n_episodes': [0-9]+" /tmp/eval3_molmoact2.log | tail -1 | grep -oE "[0-9]+$")
EPS=$(grep -oE "'eval_ep_s': [0-9.]+" /tmp/eval3_molmoact2.log | tail -1 | grep -oE "[0-9.]+$")
printf "%-11s | 성공률 %6s%% | 에피소드 %3s | 에피당 %6.1fs | VRAM peak %6s MiB | 총 %ds\n" \
  "molmoact2" "${SR:-ERR}" "${NE:-?}" "${EPS:-0}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
echo "=== 완료 ===" | tee -a "$OUT"
