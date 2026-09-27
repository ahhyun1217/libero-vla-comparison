"""DAgger 2단계: 학생이 방문한 상태에 teacher(MolmoAct2) action 을 라벨링.

학생이 실패하며 빠진 상황까지 포함된 상태 분포 위에 teacher 의 정답을 붙인다.
teacher 성공 궤적만으로는 얻을 수 없는, "학생이 틀리는 지점의 교정" 데이터가 된다.

사용: python relabel_student.py <학생상태_root> <출력_root>
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import lerobot.policies  # 정책 타입 등록 트리거
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

SRC_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./outputs/distill/student_states")
OUT_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./outputs/distill/student_relabeled")
CKPT = "allenai/MolmoAct2-LIBERO-LeRobot"

print(f"[1/5] 학생 상태 dataset 로드 ({SRC_ROOT})", flush=True)
src = LeRobotDataset("student_states", root=str(SRC_ROOT), video_backend="pyav")
print(f"      episode {src.meta.total_episodes}개 / frame {len(src)}개", flush=True)

print("[2/5] teacher 정책 로드 (MolmoAct2 5B, bf16)", flush=True)
pcfg = PreTrainedConfig.from_pretrained(CKPT)
pcfg.pretrained_path = CKPT
pcfg.device = "cuda"
pcfg.inference_action_mode = "continuous"
pcfg.model_dtype = "bfloat16"
pcfg.n_action_steps = 10

# 데이터셋은 image/image2, teacher 는 image/wrist_image 를 기대하므로 이름을 맞춘다
RENAME = {"observation.images.image2": "observation.images.wrist_image"}
policy = make_policy(cfg=pcfg, ds_meta=src.meta, rename_map=RENAME)
policy.eval()
pre, post = make_pre_post_processors(
    policy_cfg=pcfg,
    pretrained_path=CKPT,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "rename_observations_processor": {"rename_map": RENAME},
    },
)

print("[3/5] 출력 dataset 생성", flush=True)
OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
dst = LeRobotDataset.create(
    repo_id="student_relabeled",
    fps=int(src.meta.fps),
    features=src.meta.features,
    root=str(OUT_ROOT),
    use_videos=True,
)

print("[4/5] teacher 라벨링 시작", flush=True)
ep_index = np.array(src.hf_dataset["episode_index"])
cur_ep = None
n_frames = 0

with torch.no_grad(), torch.autocast(device_type="cuda"):
    for i in tqdm(range(len(src)), ncols=80):
        item = src[i]
        ep = int(ep_index[i])
        if ep != cur_ep:
            if cur_ep is not None:
                dst.save_episode()
            policy.reset()
            cur_ep = ep

        # teacher 는 예측에 관측만 필요하다. action 을 같이 넘기면 전처리기가
        # 학생 action 의 gripper 범위를 검증하다 실패하므로 관측만 전달한다.
        batch = {}
        for k, v in item.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not k.startswith("observation."):
                continue
            batch[k] = v.unsqueeze(0).cuda()
        batch["task"] = [item["task"]] if isinstance(item.get("task"), str) else item.get("task")

        processed = pre(batch)
        action = policy.select_action(processed)
        action = post(action)
        teacher_action = action.squeeze(0).float().cpu().numpy()

        # timestamp/index 류는 dataset 이 자동 관리하므로 넣지 않는다
        SKIP = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        frame = {}
        for k, spec in src.meta.features.items():
            if k in SKIP:
                continue
            if k == ACTION:
                frame[k] = teacher_action.astype(np.float32)
                continue
            if k not in item:
                continue
            v = item[k]
            v = v.numpy() if isinstance(v, torch.Tensor) else v
            if spec.get("dtype") == "video":
                # dataset 은 CHW float[0,1] 로 주지만 writer 는 HWC uint8 을 기대한다
                v = (np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
            frame[k] = v
        frame["task"] = item["task"]
        dst.add_frame(frame)
        n_frames += 1

if cur_ep is not None:
    dst.save_episode()

dst.finalize()  # meta/episodes flush (없으면 이후 로드 실패)
print(f"[5/5] 완료: {n_frames} frame, {len(episodes)} episode -> {OUT_ROOT}", flush=True)
