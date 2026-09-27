#!/bin/bash
source /home/user/amber/vla_wam/common/env.sh
cd /home/user/amber/vla_wam
LOG=logs/experiment.log
STEPS=12000
TR=/home/user/amber/vla_wam/trackB_pointtrace/train_trace_out
CT=/home/user/amber/vla_wam/trackB_pointtrace/train_ctrl_out
echo "[exp] START $(date) steps=$STEPS" > $LOG

run_eval () { # ckpt_parent, evaldir, tag
  CK=$1/checkpoints/last/pretrained_model
  if [ ! -f "$CK/model.safetensors" ]; then
     echo "[exp] !! $3: NO checkpoint at $CK -> skip eval (training likely crashed)" >> $LOG
     return 1
  fi
  echo "[exp] EVAL $3: $CK -> $2  $(date)" >> $LOG
  rm -rf $2
  env PYTORCH_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH \
    $PY -m lerobot.scripts.lerobot_eval --policy.path=$CK --policy.device=cuda \
    --env.type=libero --env.task=libero_spatial --eval.n_episodes=10 --eval.batch_size=1 \
    --output_dir=$2 >> logs/eval_$3.log 2>&1
  $PY -c "import json;d=json.load(open('$2/eval_info.json'));print('[exp] RESULT $3 pc_success:',d['overall']['pc_success'],'n=',d['overall']['n_episodes'])" >> $LOG 2>&1
}

echo "[exp] === TRAIN treatment (trace_weight=1.0) $(date) ===" >> $LOG
rm -rf $TR
$PY trackB_pointtrace/train_trace.py 1.0 $TR $STEPS >> logs/train_trace.log 2>&1
run_eval $TR logs/eval_trace_aug trace_aug

echo "[exp] === TRAIN control (trace_weight=0.0) $(date) ===" >> $LOG
rm -rf $CT
$PY trackB_pointtrace/train_trace.py 0.0 $CT $STEPS >> logs/train_ctrl.log 2>&1
run_eval $CT logs/eval_ctrl ctrl

echo "[exp] ALL DONE $(date)" >> $LOG
echo "[exp] ===== SUMMARY =====" >> $LOG
grep "RESULT" $LOG >> $LOG
