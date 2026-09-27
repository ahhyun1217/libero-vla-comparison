#!/bin/bash
# DAgger (Ross et al. 2011) - Dataset Aggregation 포함 버전
#   1단계(학생 상태 수집)는 이미 완료되어 outputs/distill/student_states 에 있음
#   2) teacher 가 학생 상태에 라벨 부여
#   3) D <- teacher 궤적 u 학생 교정데이터  (논문의 Aggregation)
#   4) 합친 데이터로 재학습
#   5) 평가
cd /home/leap/ahhyun/libero-vla/env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=/home/leap/ahhyun/libero-vla/env/dagger2_progress.txt
: > "$LOG"
while pgrep -f "bin/lerobot-eval|bin/lerobot-train" > /dev/null; do sleep 20; done

echo "[1/4] teacher 라벨링 $(date '+%H:%M:%S')" | tee -a "$LOG"
rm -rf outputs/distill/student_relabeled
./.venv/bin/python relabel_student.py ./outputs/distill/student_states \
  ./outputs/distill/student_relabeled > relabel_student.log 2>&1
grep -E "^\[5/5\]" relabel_student.log | tee -a "$LOG" || { echo "  라벨링 실패" | tee -a "$LOG"; tail -3 relabel_student.log | tee -a "$LOG"; exit 1; }

echo "[2/4] Dataset Aggregation $(date '+%H:%M:%S')" | tee -a "$LOG"
rm -rf outputs/distill/dagger_aggregated
./.venv/bin/python merge_datasets.py ./outputs/distill/dagger_aggregated \
  ./outputs/distill/teacher_rollouts ./outputs/distill/student_relabeled \
  > merge.log 2>&1
grep -E "^완료:" merge.log | tee -a "$LOG" || { echo "  병합 실패" | tee -a "$LOG"; tail -3 merge.log | tee -a "$LOG"; exit 1; }

echo "[3/4] 재학습 $(date '+%H:%M:%S')" | tee -a "$LOG"
rm -rf outputs/train/smolvla_dagger
./.venv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=dagger_aggregated \
  --dataset.root=./outputs/distill/dagger_aggregated \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --steps=20000 --batch_size=8 --save_freq=10000 --log_freq=500 \
  --output_dir=./outputs/train/smolvla_dagger --job_name=smolvla_dagger \
  > train_dagger.log 2>&1
echo "      종료코드=$? $(date '+%H:%M:%S')" | tee -a "$LOG"

echo "[4/4] 평가 50 episode $(date '+%H:%M:%S')" | tee -a "$LOG"
OUT=/home/leap/ahhyun/libero-vla/env/dagger_results.txt
: > "$OUT"
echo "=== libero_10 task6 / seed=1000 / 50 episode ===" | tee -a "$OUT"
./.venv/bin/lerobot-eval \
  --policy.path=./outputs/train/smolvla_dagger/checkpoints/020000/pretrained_model \
  --policy.device=cuda --policy.n_action_steps=10 \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --env.type=libero --env.task=libero_10 --env.task_ids='[6]' --env.episode_length=600 \
  --eval.batch_size=1 --eval.n_episodes=50 --seed=1000 > /tmp/dagger_eval.log 2>&1
SR=$(grep -oE "'pc_success': [0-9.]+" /tmp/dagger_eval.log | tail -1 | grep -oE "[0-9.]+$")
printf "%-18s | 성공률 %6s%%\n" "smolvla_DAgger" "${SR:-ERR}" | tee -a "$OUT"
echo "=== 완료 $(date '+%H:%M:%S') ===" | tee -a "$OUT" | tee -a "$LOG"
