# trace-aug 실험 공통 환경 변수.
#
# 경로는 전부 override 가능함. 다른 머신에서 돌릴 때는 VLA_WAM 만 export 하면 됨.
#   VLA_WAM=/my/path bash run_experiment.sh
# 기본값은 실제로 이 실험을 돌렸던 머신(rtx4070ti-super)의 경로임.

export VLA_WAM="${VLA_WAM:-/home/user/amber/vla_wam}"
export PY="${PY:-$VLA_WAM/.venv/bin/python}"
export TRACK_B="${TRACK_B:-$VLA_WAM/trackB_pointtrace}"
export LOG_DIR="${LOG_DIR:-$VLA_WAM/logs}"

export LIBERO_CONFIG_PATH="$VLA_WAM/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_ALLOC_CONF=expandable_segments:True
