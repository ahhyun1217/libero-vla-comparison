"""LIBERO rollout 수집 공통 모듈.

collect_teacher / collect_student / collect_xvla 가 공유하던 코드를 모았음.
세 스크립트는 이 모듈을 설정만 바꿔 호출하는 얇은 래퍼가 됨.

이 모듈에 담긴 것은 전부 실험 중 실제로 부딪혀 해결한 지점임:
  - make_env 반환이 {suite: {task_id: VectorEnv}} 2단 중첩임
  - observation.state 는 env preprocessor 통과 후에야 생성되고 policy 마다 차원이 다름
  - dataset writer 는 HWC uint8 을 받는데 dataset 은 CHW float[0,1] 을 줌
  - LeRobotDataset.create(fps=...) 에 float 를 넣으면 비디오 인코딩이 실패함
  - 모든 episode 를 쓴 뒤 finalize() 를 호출해야 meta/episodes 가 flush 됨
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import lerobot.policies  # noqa: F401  (정책 타입 등록 트리거)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

DEFAULT_MAX_STEPS = 600
DEFAULT_FPS = 10
IMG_SIZE = 256
STATE_DIM = 8
ACTION_DIM = 7


@dataclass
class CollectConfig:
    """수집 1회의 설정."""

    ckpt: str
    out: Path
    task_id: int = 6
    n_episodes: int = 36
    suite: str = "libero_10"
    control_mode: str = "relative"
    wrist_key: str = "wrist_image"          # 정책이 기대하는 손목 카메라 이름
    n_action_steps: int = 10
    keep_failures: bool = False             # True 면 실패 episode 도 저장 (DAgger 용)
    max_steps: int = DEFAULT_MAX_STEPS
    seed_base: int = 2000
    policy_overrides: dict = field(default_factory=dict)   # pcfg 에 덮어쓸 값
    rename_map: dict = field(default_factory=dict)         # 관측 키 이름 변환


def build_env(cfg: CollectConfig):
    """LIBERO 환경 생성. make_env 의 2단 중첩 dict 를 풀어서 돌려줌."""
    env_cfg = LiberoEnv(
        task=cfg.suite,
        task_ids=[cfg.task_id],
        episode_length=cfg.max_steps,
        control_mode=cfg.control_mode,
        observation_height=IMG_SIZE,
        observation_width=IMG_SIZE,
        camera_name_mapping={
            "agentview_image": "image",
            "robot0_eye_in_hand_image": cfg.wrist_key,
        },
    )
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)
    while isinstance(env, dict):
        env = list(env.values())[0]
    return env_cfg, env


def build_policy(cfg: CollectConfig, env_cfg):
    """정책과 전처리기들을 준비."""
    pcfg = PreTrainedConfig.from_pretrained(cfg.ckpt)
    pcfg.pretrained_path = cfg.ckpt
    pcfg.device = "cuda"
    pcfg.n_action_steps = cfg.n_action_steps
    for k, v in cfg.policy_overrides.items():
        setattr(pcfg, k, v)

    policy = make_policy(cfg=pcfg, env_cfg=env_cfg, rename_map=cfg.rename_map or None)
    policy.eval()

    overrides = {"device_processor": {"device": "cuda"}}
    if cfg.rename_map:
        overrides["rename_observations_processor"] = {"rename_map": cfg.rename_map}
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=cfg.ckpt, preprocessor_overrides=overrides
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
    return policy, pre, post, env_pre, env_post


def build_dataset(out: Path, fps: int = DEFAULT_FPS) -> LeRobotDataset:
    """저장용 dataset 생성. fps 는 반드시 int 여야 비디오 인코딩이 통과함."""
    features = {
        "observation.images.image": {
            "dtype": "video", "shape": (IMG_SIZE, IMG_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "video", "shape": (IMG_SIZE, IMG_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=out.name, fps=int(fps), features=features, root=str(out), use_videos=True
    )


def to_hwc_uint8(t) -> np.ndarray:
    """(C,H,W) float[0,1] 또는 배치 차원이 붙은 텐서를 (H,W,C) uint8 로."""
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.ndim == 4:
        a = a[0]
    if a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return a


def run_episode(env, policy, pre, post, env_pre, env_post, cfg: CollectConfig, seed: int):
    """episode 1회 실행. (프레임 목록, 성공여부, 스텝수) 를 돌려줌."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    frames, success, step = [], False, 0

    while step < cfg.max_steps:
        observation = preprocess_observation(obs)
        raw = deepcopy(observation)
        try:
            observation["task"] = list(env.call("task_description"))
        except Exception:
            observation["task"] = [""] * env.num_envs

        observation = env_pre(observation)
        env_obs = observation          # observation.state 가 여기서 생성됨
        observation = pre(observation)

        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            action = policy.select_action(observation)
        action = post(action)
        action = env_post({ACTION: action})[ACTION]   # 내부 패딩을 환경용 차원으로 풂
        action_np = action.to("cpu").numpy()

        wrist = env_obs.get(f"observation.images.{cfg.wrist_key}")
        if wrist is None:
            wrist = raw[f"observation.images.{cfg.wrist_key}"]
        frames.append({
            "observation.images.image": to_hwc_uint8(env_obs["observation.images.image"]),
            "observation.images.image2": to_hwc_uint8(wrist),
            # policy 마다 state 를 다른 차원으로 패딩하므로 실제 차원만 남김
            "observation.state": env_obs["observation.state"][0, :STATE_DIM]
                                 .detach().cpu().numpy().astype(np.float32),
            "action": action_np[0].astype(np.float32),
            "task": observation["task"][0] if observation.get("task") else "",
        })

        obs, _, terminated, truncated, info = env.step(action_np)
        step += 1
        if info.get("is_success") is not None and np.any(info["is_success"]):
            success = True
        final = info.get("final_info")
        if final is not None:
            for e in (final if isinstance(final, (list, tuple)) else [final]):
                if isinstance(e, dict) and e.get("is_success"):
                    success = True
        if np.any(terminated) or np.any(truncated):
            break

    return frames, success, step


def collect(cfg: CollectConfig) -> None:
    """설정대로 rollout 을 모아 dataset 으로 저장."""
    print(f"[1/4] 환경 생성 ({cfg.suite} task {cfg.task_id}, control={cfg.control_mode})", flush=True)
    env_cfg, env = build_env(cfg)

    print(f"[2/4] 정책 로드 ({cfg.ckpt})", flush=True)
    policy, pre, post, env_pre, env_post = build_policy(cfg, env_cfg)

    print(f"[3/4] 출력 dataset 생성 -> {cfg.out}", flush=True)
    dst = build_dataset(cfg.out)

    mode = "성공/실패 모두 저장" if cfg.keep_failures else "성공분만 저장"
    print(f"[4/4] rollout 수집 시작 (목표 {cfg.n_episodes} episode, {mode})", flush=True)

    saved = attempts = n_fail = 0
    limit = cfg.n_episodes if cfg.keep_failures else cfg.n_episodes * 3
    while saved < cfg.n_episodes and attempts < limit:
        attempts += 1
        frames, success, step = run_episode(
            env, policy, pre, post, env_pre, env_post, cfg, cfg.seed_base + attempts
        )
        if success or cfg.keep_failures:
            for fr in frames:
                dst.add_frame(fr)
            dst.save_episode()
            saved += 1
            n_fail += 0 if success else 1
            state = "성공" if success else "실패(보존)"
        else:
            state = "실패 폐기"
        print(f"  시도 {attempts}: {state} ({step} step) | 저장 {saved}/{cfg.n_episodes}", flush=True)

    dst.finalize()   # 이걸 빠뜨리면 meta/episodes 가 flush 되지 않아 이후 로드가 실패함
    tail = f" (실패 {n_fail}개 포함)" if cfg.keep_failures else ""
    print(f"완료: {saved} episode 저장{tail} -> {cfg.out}", flush=True)
