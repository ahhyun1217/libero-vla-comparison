"""Teacher(MolmoAct2)가 직접 로봇을 굴리며 (관측, action) 을 수집.

사람 시연과 달리 teacher 자신의 상태 분포(삐끗했을 때 복구하는 구간 포함)가 담김.
성공한 episode 만 저장함.

사용: python collect_teacher.py <task_id> <n_episodes> <출력경로>
"""

import sys
from pathlib import Path

from vla_collect import CollectConfig, collect

collect(CollectConfig(
    ckpt="allenai/MolmoAct2-LIBERO-LeRobot",
    task_id=int(sys.argv[1]) if len(sys.argv) > 1 else 6,
    n_episodes=int(sys.argv[2]) if len(sys.argv) > 2 else 36,
    out=Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./outputs/distill/teacher_rollouts"),
    wrist_key="wrist_image",
    policy_overrides={
        "inference_action_mode": "continuous",
        "model_dtype": "bfloat16",
    },
))
