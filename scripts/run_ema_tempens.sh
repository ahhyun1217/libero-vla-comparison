#!/bin/bash
# (A) SmolVLA + EMA : 같은 DAgger 데이터로 EMA 켜고 재학습, 일반 vs EMA 가중치 비교
# (B) ACT + temporal ensemble : 추가 학습 없이 추론 옵션만 켜서 비교
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=/home/leap/ahhyun/libero-vla/env/ema_progress.txt
OUT=/home/leap/ahhyun/libero-vla/env/ema_results.txt
: > "$LOG"; : > "$OUT"
while pgrep -f "bin/lerobot-eval|bin/lerobot-train" > /dev/null; do sleep 20; done

RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'

# ---------- (A) SmolVLA + EMA ----------
echo "[1/3] SmolVLA + EMA 학습 $(date '+%H:%M:%S')" | tee -a "$LOG"
rm -rf outputs/train/smolvla_dagger_ema
./.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=dagger_aggregated \
  --dataset.root=./outputs/distill/dagger_aggregated \
  --dataset.video_backend=pyav --rename_map="$RENAME" \
  --ema.enable=true \
  --steps=20000 --batch_size=8 --save_freq=20000 --log_freq=1000 \
  --output_dir=./outputs/train/smolvla_dagger_ema --job_name=smolvla_dagger_ema \
  > train_ema.log 2>&1
echo "      종료코드=$? $(date '+%H:%M:%S')" | tee -a "$LOG"

echo "[2/3] SmolVLA 평가 (일반 vs EMA) $(date '+%H:%M:%S')" | tee -a "$LOG"
echo "=== SmolVLA / libero_10 task6 / seed=1000 / 50 episode ===" | tee -a "$OUT"
for V in "일반" "EMA"; do
  if [ "$V" = "EMA" ]; then P=./outputs/train/smolvla_dagger_ema/checkpoints/020000/pretrained_model_ema
  else P=./outputs/train/smolvla_dagger_ema/checkpoints/020000/pretrained_model; fi
  ./.venv/bin/lerobot-eval --policy.path="$P" --policy.device=cuda --policy.n_action_steps=10 \
    --rename_map="$RENAME" \
    --env.type=libero --env.task=libero_10 --env.task_ids='[6]' --env.episode_length=600 \
    --eval.batch_size=1 --eval.n_episodes=50 --seed=1000 > /tmp/ema_$V.log 2>&1
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/ema_$V.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-22s | 성공률 %6s%%\n" "DAgger 가중치=$V" "${SR:-ERR}" | tee -a "$OUT"
done

# ---------- (B) ACT + temporal ensemble ----------
echo "[3/3] ACT temporal ensemble 비교 $(date '+%H:%M:%S')" | tee -a "$LOG"
echo "=== ACT 100k / libero_goal task0 / seed=1000 / 50 episode ===" | tee -a "$OUT"
ACT=./outputs/train/act_libero_goal_task0_100k/checkpoints/100000/pretrained_model
for C in "off" "0.01"; do
  if [ "$C" = "off" ]; then EXTRA="--policy.n_action_steps=10"
  else EXTRA="--policy.temporal_ensemble_coeff=$C --policy.n_action_steps=1"; fi
  ./.venv/bin/lerobot-eval --policy.path="$ACT" --policy.device=cuda $EXTRA \
    --env.type=libero --env.task=libero_goal --env.task_ids='[0]' --env.episode_length=300 \
    --eval.batch_size=1 --eval.n_episodes=50 --seed=1000 > /tmp/act_te_$C.log 2>&1
  SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/act_te_$C.log | tail -1 | grep -oE "[0-9.]+$")
  EPS=$(grep -oE "'eval_ep_s': [0-9.]+" /tmp/act_te_$C.log | tail -1 | grep -oE "[0-9.]+$")
  printf "%-22s | 성공률 %6s%% | 에피당 %6.1fs\n" "temporal_ens=$C" "${SR:-ERR}" "${EPS:-0}" | tee -a "$OUT"
done
echo "=== 완료 $(date '+%H:%M:%S') ===" | tee -a "$OUT" | tee -a "$LOG"
