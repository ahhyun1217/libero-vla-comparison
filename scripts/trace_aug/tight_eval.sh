#!/bin/bash
source /home/user/amber/vla_wam/common/env.sh
cd /home/user/amber/vla_wam
echo "[tight] START n=30/task (300ep/조건) $(date)" > logs/tight.log
ev() { # tag, parent, init_states
  CK=$2/checkpoints/last/pretrained_model
  rm -rf logs/tight_$1
  env PYTORCH_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH \
    $PY -m lerobot.scripts.lerobot_eval --policy.path=$CK --policy.device=cuda \
    --env.type=libero --env.task=libero_spatial --env.init_states=$3 \
    --eval.n_episodes=30 --eval.batch_size=1 --output_dir=logs/tight_$1 > logs/tight_$1.log 2>&1
  python3 -c "import json;d=json.load(open('logs/tight_$1/eval_info.json'));print('[tight] RESULT $1:',d['overall']['pc_success'],'n=',d['overall']['n_episodes'])" >> logs/tight.log 2>&1
}
ev trace_id  trackB_pointtrace/train_trace_out true
ev ctrl_id   trackB_pointtrace/train_ctrl_out  true
ev trace_ood trackB_pointtrace/train_trace_out false
ev ctrl_ood  trackB_pointtrace/train_ctrl_out  false
echo "[tight] ALL DONE $(date)" >> logs/tight.log
