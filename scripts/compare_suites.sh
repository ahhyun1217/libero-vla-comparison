#!/bin/bash
# SmolVLA vs MolmoAct2 : LIBERO 나머지 suite 비교 (함수/명령치환 없이 단순 구조)
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/home/leap/ahhyun/libero-vla/env/compare_suites.txt
: > "$OUT"
echo "=== suite 비교 / seed=1000 / n_action_steps=10 / 과제10종x2ep ===" | tee -a "$OUT"

RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
CAMMAP='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'

for SUITE in libero_spatial libero_object libero_10; do
  if [ "$SUITE" = "libero_10" ]; then EPLEN=600; else EPLEN=300; fi

  # ---------- SmolVLA ----------
  SF=/tmp/vs_smolvla_$SUITE.vram; : > "$SF"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
  SP=$!
  T0=$(date +%s)
  ./.venv/bin/lerobot-eval --policy.path=lerobot/smolvla_libero --policy.device=cuda \
    --policy.n_action_steps=10 --rename_map="$RENAME" \
    --env.type=libero --env.task="$SUITE" --env.episode_length="$EPLEN" \
    --eval.batch_size=1 --eval.n_episodes=2 --seed=1000 > /tmp/vs_smolvla_$SUITE.log 2>&1
  T1=$(date +%s)
  kill $SP 2>/dev/null
  PEAK=$(sort -n "$SF" | tail -1)
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/vs_smolvla_$SUITE.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-10s | %-14s | 성공률 %6s%% | VRAM %6s MiB | %4ds\n" smolvla "$SUITE" "${SR:-ERR}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"

  # ---------- MolmoAct2 ----------
  SF=/tmp/vs_molmo_$SUITE.vram; : > "$SF"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SF"; sleep 0.5; done ) >/dev/null 2>&1 &
  SP=$!
  T0=$(date +%s)
  ./.venv/bin/lerobot-eval --policy.path=allenai/MolmoAct2-LIBERO-LeRobot \
    --policy.inference_action_mode=continuous --policy.model_dtype=bfloat16 \
    --policy.use_amp=true --policy.enable_inference_cuda_graph=true \
    --policy.device=cuda --policy.n_action_steps=10 --env.camera_name_mapping="$CAMMAP" \
    --env.type=libero --env.task="$SUITE" --env.episode_length="$EPLEN" \
    --eval.batch_size=1 --eval.n_episodes=2 --seed=1000 > /tmp/vs_molmo_$SUITE.log 2>&1
  T1=$(date +%s)
  kill $SP 2>/dev/null
  PEAK=$(sort -n "$SF" | tail -1)
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/vs_molmo_$SUITE.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-10s | %-14s | 성공률 %6s%% | VRAM %6s MiB | %4ds\n" molmoact2 "$SUITE" "${SR:-ERR}" "${PEAK:-?}" "$((T1-T0))" | tee -a "$OUT"
done

# 덤: ACT(task0만 학습)를 학습 안 한 task1 에 투입
./.venv/bin/lerobot-eval \
  --policy.path=./outputs/train/act_libero_goal_task0_100k/checkpoints/100000/pretrained_model \
  --policy.device=cuda --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_goal --env.task_ids='[1]' --env.episode_length=300 \
  --eval.batch_size=1 --eval.n_episodes=10 --seed=1000 > /tmp/vs_act_unseen.log 2>&1
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/vs_act_unseen.log | tail -1 | grep -oE "[0-9.]+$")
printf "%-10s | %-14s | 성공률 %6s%% | (학습 안 한 과제)\n" act-100k "goal task1" "${SR:-ERR}" | tee -a "$OUT"
echo "=== 완료 ===" | tee -a "$OUT"
