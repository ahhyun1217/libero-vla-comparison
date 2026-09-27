#!/bin/bash
source /home/user/amber/vla_wam/common/env.sh
cd /home/user/amber/vla_wam
echo "[ood] START init_states=false (분포이동) $(date)" > logs/ood.log
run() { # tag, parent
  CK=$2/checkpoints/last/pretrained_model
  rm -rf logs/ood_$1
  env PYTORCH_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH \
    $PY -m lerobot.scripts.lerobot_eval --policy.path=$CK --policy.device=cuda \
    --env.type=libero --env.task=libero_spatial --env.init_states=false \
    --eval.n_episodes=10 --eval.batch_size=1 --output_dir=logs/ood_$1 > logs/ood_$1.log 2>&1
  python3 -c "import json;d=json.load(open('logs/ood_$1/eval_info.json'));print('[ood] RESULT $1 (init_states=false):',d['overall']['pc_success'],'n=',d['overall']['n_episodes'])" >> logs/ood.log 2>&1
}
run trace_aug trackB_pointtrace/train_trace_out
run control   trackB_pointtrace/train_ctrl_out
echo "[ood] ALL DONE $(date)" >> logs/ood.log
