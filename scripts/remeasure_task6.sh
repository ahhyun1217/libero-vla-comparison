#!/bin/bash
# README "전체 결과" 표의 수치 중 원본 기록이 없거나 조건이 달랐던 두 개를 재측정.
#   MolmoAct2  : 기존 task 6 기록은 libero_10 suite 평가 안의 2 episode 뿐이었음
#   X-VLA 대조군: 기존 기록은 10 episode 였음
# 조건은 표의 다른 task 6 평가(eval_xvla.sh, run_shallow_xvla.sh, train_fullvlm.sh)와 동일:
#   libero_10 task 6, episode_length 600, seed 1000, 50 episode, n_action_steps 10, batch 1
# RAM 15GB 라 두 평가를 순차로 돌림. 대기 루프는 넣지 않음 (pgrep -f 자기 매칭).
# 원본 기록인 /tmp/eval3_*.log, /tmp/ctrl24_eval.log 등을 덮어쓰지 않도록 로그 이름을 새로 씀.
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RES=/home/leap/ahhyun/libero-vla/repo/results
TASK6=(--env.type=libero --env.task=libero_10 --env.task_ids='[6]' --env.episode_length=600
       --eval.batch_size=1 --eval.n_episodes=50 --seed=1000)

run () {  # $1=라벨 $2=결과파일 $3=로그 나머지=정책/환경 인자
  local NAME="$1" OUT="$2" LOG="$3"; shift 3
  local SF="$LOG.vram"; : > "$SF"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
  local SP=$!
  local T0=$(date +%s)
  ./.venv/bin/lerobot-eval "$@" "${TASK6[@]}" > "$LOG" 2>&1
  local RC=$?
  local T1=$(date +%s)
  kill $SP 2>/dev/null
  local SR=$(grep -oE "'pc_success': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
  local NE=$(grep -oE "'n_episodes': [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+$")
  local PEAK=$(sort -n "$SF" | tail -1)
  if [ -z "$SR" ]; then
    printf "%-14s | 실패(rc=%s), 로그: %s\n" "$NAME" "$RC" "$LOG" | tee -a "$OUT"
  else
    printf "%-14s | 성공률 %6s%% | 에피소드 %3s | VRAM %6s MiB | %4ds\n" \
      "$NAME" "$SR" "$NE" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
  fi
  echo "=== 완료 ===" | tee -a "$OUT"
}

# ---- MolmoAct2: run_molmo_only.sh / run_rest.sh 와 같은 정책 설정, 과제와 episode 수만 변경 ----
OUT=$RES/molmoact2_task6_results.txt; : > "$OUT"
echo "=== libero_10 task6 / seed=1000 / 50 episode / n_action_steps=10 / bf16+AMP / continuous ===" | tee -a "$OUT"
run MolmoAct2 "$OUT" /tmp/remeasure_molmoact2_task6.log \
  --policy.path=allenai/MolmoAct2-LIBERO-LeRobot \
  --policy.inference_action_mode=continuous \
  --policy.model_dtype=bfloat16 --policy.use_amp=true \
  --policy.enable_inference_cuda_graph=true \
  --policy.device=cuda --policy.n_action_steps=10 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'

# ---- X-VLA 대조군: 24층을 carve_student.py 로 재저장한 checkpoint ----
OUT=$RES/xvla_control_results.txt; : > "$OUT"
echo "=== libero_10 task6 / seed=1000 / 50 episode / n_action_steps=10 / control_mode=absolute ===" | tee -a "$OUT"
run "X-VLA 대조군" "$OUT" /tmp/remeasure_xvla_ctrl24.log \
  --policy.path=outputs/distill/xvla_ctrl24 \
  --policy.device=cuda --policy.n_action_steps=10 \
  --env.control_mode=absolute
