# lerobot-eval 공통 러너.
#
# run_experiment.sh / tight_eval.sh / ood_eval.sh 가 각자 갖고 있던 동일한 eval 호출을
# 여기 하나로 모음. 세 스크립트는 이제 표본 크기와 init_states 만 다른 thin caller 임.
#
# trace_eval <tag> <ckpt_parent> <n_episodes_per_task> <init_states> <result_log>
#   tag                  결과 로그와 출력 디렉터리 이름. 호출부끼리 겹치지 않게 줄 것
#   ckpt_parent          train 출력 디렉터리 (안의 checkpoints/last 를 씀)
#   n_episodes_per_task  libero_spatial 은 task 10 개라 30 이면 조건당 300 episode
#   init_states          true = in-distribution, false = OOD (초기 상태 분포이동)
trace_eval () {
  local TAG="$1" PARENT="$2" NEP="$3" INIT="$4" RESULT="$5"
  local CK="$PARENT/checkpoints/last/pretrained_model"
  local OUT="$LOG_DIR/eval_$TAG"

  if [ ! -f "$CK/model.safetensors" ]; then
    echo "[eval] !! $TAG: checkpoint 없음 ($CK) -> skip. 학습이 죽었을 가능성" >> "$RESULT"
    return 1
  fi

  echo "[eval] $TAG n_per_task=$NEP init_states=$INIT  $(date)" >> "$RESULT"
  rm -rf "$OUT"
  "$PY" -m lerobot.scripts.lerobot_eval \
    --policy.path="$CK" --policy.device=cuda \
    --env.type=libero --env.task=libero_spatial --env.init_states="$INIT" \
    --eval.n_episodes="$NEP" --eval.batch_size=1 \
    --output_dir="$OUT" > "$LOG_DIR/eval_$TAG.log" 2>&1

  "$PY" -c "
import json
d = json.load(open('$OUT/eval_info.json'))['overall']
print('[eval] RESULT $TAG:', d['pc_success'], 'n=', d['n_episodes'])
" >> "$RESULT" 2>&1
}
