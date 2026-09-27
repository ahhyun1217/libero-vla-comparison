"""DAgger 의 Dataset Aggregation: 여러 LeRobotDataset 을 하나로 합친다.

Ross et al. 2011 의 D <- D u D_i 단계에 해당한다.
teacher 성공 궤적(과제를 푸는 법)과 학생 상태 교정 데이터(실수 수습법)를
모두 유지해야 성능이 무너지지 않는다.

사용: python merge_datasets.py <출력root> <입력root1> <입력root2> ...
"""

import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT = Path(sys.argv[1])
SRCS = [Path(p) for p in sys.argv[2:]]
assert SRCS, "입력 dataset 을 하나 이상 지정하세요"


def to_hwc_uint8(a):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    if a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return a


print(f"[1/3] 입력 dataset {len(SRCS)}개 로드", flush=True)
datasets = []
for p in SRCS:
    ds = LeRobotDataset(p.name, root=str(p), video_backend="pyav")
    print(f"      {p.name}: episode {ds.meta.total_episodes} / frame {len(ds)}", flush=True)
    datasets.append(ds)

base = datasets[0]
print(f"[2/3] 병합 dataset 생성 -> {OUT}", flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
dst = LeRobotDataset.create(
    repo_id=OUT.name,
    fps=int(base.meta.fps),
    features=base.meta.features,
    root=str(OUT),
    use_videos=True,
)

print("[3/3] frame 복사", flush=True)
total = 0
for ds, p in zip(datasets, SRCS):
    ep_index = np.array(ds.hf_dataset["episode_index"])
    cur = None
    for i in range(len(ds)):
        item = ds[i]
        ep = int(ep_index[i])
        if ep != cur:
            if cur is not None:
                dst.save_episode()
            cur = ep
        frame = {}
        for k, spec in base.meta.features.items():
            if k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                continue
            if k not in item:
                continue
            v = item[k]
            if spec.get("dtype") == "video":
                v = to_hwc_uint8(v)
            else:
                v = v.numpy() if isinstance(v, torch.Tensor) else v
            frame[k] = v
        frame["task"] = item.get("task", "")
        dst.add_frame(frame)
        total += 1
    if cur is not None:
        dst.save_episode()
    print(f"      {p.name} 완료 (누적 {total} frame)", flush=True)

dst.finalize()
print(f"완료: {total} frame, {dst.meta.total_episodes} episode -> {OUT}", flush=True)
