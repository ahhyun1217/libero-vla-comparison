"""Teacher(X-VLA)가 로봇을 굴리며 (관측, action) 을 수집.

X-VLA 는 절대좌표 제어를 전제로 학습됐으므로 control_mode=absolute 가 필수임.
관측 키도 image / image2 를 쓴다.

사용: python collect_xvla.py <task_id> <n_episodes> <출력경로>
"""

import sys
from pathlib import Path

from vla_collect import CollectConfig, collect

collect(CollectConfig(
    ckpt="lerobot/xvla-libero",
    task_id=int(sys.argv[1]) if len(sys.argv) > 1 else 6,
    n_episodes=int(sys.argv[2]) if len(sys.argv) > 2 else 36,
    out=Path(sys.argv[3]) if len(sys.argv) > 3 else Path("./outputs/distill/xvla_rollouts"),
    control_mode="absolute",
    wrist_key="image2",
))
