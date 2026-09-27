#!/bin/bash
# 3개 모델 동일 과제 비교
#   과제: libero_goal task 0 = "open the middle drawer of the cabinet"
#   ACT가 이 과제만 학습했으므로 공정 비교를 위해 task 0 으로 통일
#   공통: seed=1000, n_action_steps=10, 10 에피소드
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

NEP=10
OUT=/home/leap/ahhyun/libero-vla/env/compare3_results.txt
: > "$OUT"

run_one () {
  local NAME="$1"; shift
  local SAMPLES="/tmp/vram3_$NAME.txt"; : > "$SAMPLES"

  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SAMPLES"
      sleep 0.3
    done ) &
  local SAMPLER=$!

  local T0=$(date +%s)
  ./.venv/bin/lerobot-eval "$@" \
    --env.type=libero --env.task=libero_goal --env.task_ids='[0]' \
    --eval.batch_size=1 --eval.n_episodes="$NEP" --seed=1000 \
    > "/tmp/eval3_$NAME.log" 2>&1
  local RC=$?
  local T1=$(date +%s)

  kill $SAMPLER 2>/dev/null
  local PEAK=$(sort -n "$SAMPLES" | tail -1)
  local SR=$(grep -oE "'pc_success': [0-9.]+" "/tmp/eval3_$NAME.log" | tail -1 | grep -oE "[0-9.]+$")
  local NE=$(grep -oE "'n_episodes': [0-9]+" "/tmp/eval3_$NAME.log" | tail -1 | grep -oE "[0-9]+$")
  local EPS=$(grep -oE "'eval_ep_s': [0-9.]+" "/tmp/eval3_$NAME.log" | tail -1 | grep -oE "[0-9.]+$")

  if [ -z "$SR" ]; then
    printf "%-11s | 실패(rc=%s) — 로그: /tmp/eval3_%s.log\n" "$NAME" "$RC" "$NAME" | tee -a "$OUT"
  else
    printf "%-11s | 성공률 %6s%% | 에피소드 %3s | 에피당 %6.1fs | VRAM peak %6s MiB | 총 %ds\n" \
      "$NAME" "$SR" "$NE" "${EPS:-0}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
  fi
}

echo "=== 과제: libero_goal task0 (open the middle drawer) / seed=1000 / ${NEP}ep ===" | tee -a "$OUT"

run_one "act" \
  --policy.path=./outputs/train/act_libero_goal_task0/checkpoints/020000/pretrained_model \
  --policy.device=cuda

run_one "smolvla" \
  --policy.path=lerobot/smolvla_libero \
  --policy.device=cuda --policy.n_action_steps=10 \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'

run_one "molmoact2" \
  --policy.path=allenai/MolmoAct2-LIBERO-LeRobot \
  --policy.inference_action_mode=continuous \
  --policy.model_dtype=bfloat16 --policy.use_amp=true \
  --policy.enable_inference_cuda_graph=true \
  --policy.device=cuda --policy.n_action_steps=10 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'

echo "=== 완료 ===" | tee -a "$OUT"
