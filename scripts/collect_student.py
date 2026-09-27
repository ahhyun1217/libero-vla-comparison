"""DAgger 1단계: 학생(SmolVLA)이 로봇을 굴리며 방문한 상태를 수집.

teacher 수집과 결정적으로 다른 점은 성공/실패를 가리지 않고 전부 저장한다는 것.
학생이 실패하는 상황이야말로 teacher 라벨이 필요한 지점이기 때문임.
여기 저장된 action 은 이후 relabel 단계에서 teacher 것으로 교체됨.

사용: python collect_student.py <task_id> <n_episodes> <학생체크포인트> <출력경로>
"""

import sys
from pathlib import Path

from vla_collect import CollectConfig, collect

DEFAULT_STUDENT = "./outputs/train/smolvla_distilled/checkpoints/020000/pretrained_model"

collect(CollectConfig(
    ckpt=sys.argv[3] if len(sys.argv) > 3 else DEFAULT_STUDENT,
    task_id=int(sys.argv[1]) if len(sys.argv) > 1 else 6,
    n_episodes=int(sys.argv[2]) if len(sys.argv) > 2 else 30,
    out=Path(sys.argv[4]) if len(sys.argv) > 4 else Path("./outputs/distill/student_states"),
    wrist_key="wrist_image",
    keep_failures=True,   # DAgger 의 핵심: 실패 구간이 교정 대상임
    # SmolVLA 는 camera1 / camera2 이름을 기대하므로 env 출력 이름을 맞춰줌
    rename_map={
        "observation.images.image": "observation.images.camera1",
        "observation.images.wrist_image": "observation.images.camera2",
    },
))
