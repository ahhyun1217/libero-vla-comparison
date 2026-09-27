#!/bin/bash
# 1) 앞서 실패한 eval 재실행  2) GR00T / Pi0.5 추가
# 설치는 절대 병행하지 않음 (bddl 삭제 사고 방지)
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/home/leap/ahhyun/libero-vla/env/compare_rest.txt
: > "$OUT"
echo "=== 재실행 + GR00T/Pi0.5 / seed=1000 / n_action_steps=10 ===" | tee -a "$OUT"

RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
CAMMAP='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
MOLMO_ARGS="--policy.inference_action_mode=continuous --policy.model_dtype=bfloat16 --policy.use_amp=true --policy.enable_inference_cuda_graph=true"

run () {  # $1=라벨 $2=suite $3=eplen $4=로그 ... 나머지는 eval 인자
  LABEL="$1"; SUITE="$2"; EPLEN="$3"; TAG="$4"; shift 4
  SF=/tmp/rest_$TAG.vram; : > "$SF"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
  SP=$!
  T0=$(date +%s)
  ./.venv/bin/lerobot-eval "$@" \
    --env.type=libero --env.task="$SUITE" --env.episode_length="$EPLEN" \
    --eval.batch_size=1 --eval.n_episodes=2 --seed=1000 > /tmp/rest_$TAG.log 2>&1
  T1=$(date +%s); kill $SP 2>/dev/null
  PEAK=$(sort -n "$SF" | tail -1)
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/rest_$TAG.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-11s | %-14s | 성공률 %6s%% | VRAM %6s MiB | %4ds\n" "$LABEL" "$SUITE" "${SR:-ERR}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
}

# ---- (1) 실패분 재실행 ----
run molmoact2 libero_object 300 molmo_obj --policy.path=allenai/MolmoAct2-LIBERO-LeRobot $MOLMO_ARGS --policy.device=cuda --policy.n_action_steps=10 --env.camera_name_mapping="$CAMMAP"
run smolvla   libero_10     600 smol_10   --policy.path=lerobot/smolvla_libero --policy.device=cuda --policy.n_action_steps=10 --rename_map="$RENAME"
run molmoact2 libero_10     600 molmo_10  --policy.path=allenai/MolmoAct2-LIBERO-LeRobot $MOLMO_ARGS --policy.device=cuda --policy.n_action_steps=10 --env.camera_name_mapping="$CAMMAP"

# ---- (2) GR00T (suite 전용 체크포인트) ----
M=/home/leap/ahhyun/vla-bench/models-nvme
run groot libero_goal    300 groot_goal    --policy.path=$M/groot_lr_goal    --policy.device=cuda --policy.n_action_steps=10
run groot libero_spatial 300 groot_spatial --policy.path=$M/groot_lr_spatial --policy.device=cuda --policy.n_action_steps=10
run groot libero_object  300 groot_object  --policy.path=$M/groot_lr_object  --policy.device=cuda --policy.n_action_steps=10

# ---- (3) Pi0.5 ----
run pi05 libero_goal    300 pi05_goal    --policy.path=$M/pi05_libero --policy.device=cuda --policy.n_action_steps=10
run pi05 libero_spatial 300 pi05_spatial --policy.path=$M/pi05_libero --policy.device=cuda --policy.n_action_steps=10
run pi05 libero_object  300 pi05_object  --policy.path=$M/pi05_libero --policy.device=cuda --policy.n_action_steps=10

# ---- (4) ACT 제로샷: task0만 학습한 모델을 task1 에 ----
SF=/tmp/rest_act0.vram; : > "$SF"
./.venv/bin/lerobot-eval \
  --policy.path=./outputs/train/act_libero_goal_task0_100k/checkpoints/100000/pretrained_model \
  --policy.device=cuda --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_goal --env.task_ids='[1]' --env.episode_length=300 \
  --eval.batch_size=1 --eval.n_episodes=10 --seed=1000 > /tmp/rest_act0.log 2>&1
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/rest_act0.log | tail -1 | grep -oE "[0-9.]+$")
printf "%-11s | %-14s | 성공률 %6s%% | (학습 안 한 과제=제로샷)\n" act-100k "goal task1" "${SR:-ERR}" | tee -a "$OUT"
echo "=== 완료 ===" | tee -a "$OUT"
