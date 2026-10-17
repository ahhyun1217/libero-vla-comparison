#!/usr/bin/env python
"""
Track B: trace-augmented SmolVLA training (lerobot 인프라를 재사용한 커스텀 루프).

미래 2D point-trace (ATM/MolmoAct 계열) 를 예측하는 보조 헤드를 붙여, trace loss 가
action expert 의 표현을 함께 다듬도록 함.

같은 스크립트로 두 arm 을 돌려 matched comparison 을 만듦:
  trace_weight=1.0  treatment (action loss + trace loss)
  trace_weight=0.0  control   (동일 설정, trace 항만 0)

설정 주의:
  train_expert_only=true 를 유지함. false 로 두면 사전학습 VLM 이 망가져 eval 이 0 % 가 됨.
  그래서 trace 헤드는 얼어 있는 VLM 이 아니라 학습되는 expert 특징에 붙임
  (얼린 쪽에 붙이면 gradient 가 흐르지 않는 무효 probe 가 됨).

경로는 VLA_WAM 환경변수로 override 가능함. 기본값은 이 실험을 돌렸던 머신 경로임.

사용: python train_trace.py <trace_weight> <output_dir> [steps]
"""
import sys, os, glob, json
import numpy as np
import torch
import torch.nn as nn


def cycle(iterable):
    """Non-caching infinite iterator. NOTE: itertools.cycle() caches every yielded
    batch (images included) -> ~15MB/step RAM leak -> OOM. This restarts instead."""
    it = iter(iterable)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(iterable)

ROOT     = os.environ.get("VLA_WAM", "/home/user/amber/vla_wam")
TRACK_B  = os.environ.get("TRACK_B", f"{ROOT}/trackB_pointtrace")
os.environ.setdefault("LIBERO_CONFIG_PATH", f"{ROOT}/.libero")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

TRACE_WEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUTPUT_DIR   = sys.argv[2] if len(sys.argv) > 2 else f"{TRACK_B}/train_trace_out"
STEPS        = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
H_HORIZON    = 16          # future frames to predict
IMG_SIZE     = 256.0       # trace pixel normalization
SAVE_FREQ    = 1000       # frequent so an OOM-kill never loses everything
LOG_FREQ     = 100
BATCH        = 8

import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.utils.train_utils import save_checkpoint, update_last_checkpoint, get_step_checkpoint_dir

# lerobot/libero 전체에서 libero_spatial 에 해당하는 episode index 목록 (432 개).
# 이 저장소에는 포함돼 있지 않음. 없으면 어떤 파일인지 모른 채 죽으므로 먼저 안내함.
EPISODES_JSON = os.environ.get("EPISODES_JSON", f"{TRACK_B}/spatial_episodes.json")
if not os.path.exists(EPISODES_JSON):
    raise SystemExit(
        f"episode 목록이 없음: {EPISODES_JSON}\n"
        "lerobot/libero 에서 libero_spatial episode index 만 골라 담은 JSON 배열이 필요함.\n"
        "EPISODES_JSON 으로 다른 경로를 지정할 수 있음."
    )
eps = json.load(open(EPISODES_JSON))
eps_arg = "[" + ",".join(str(e) for e in sorted(eps)) + "]"

cli = [
    "--policy.type=smolvla",
    "--policy.pretrained_path=lerobot/smolvla_base",
    "--policy.push_to_hub=false",
    "--policy.device=cuda",
    "--policy.train_expert_only=true",      # STABLE (base got 58% this way). false wrecked the
                                            # pretrained VLM -> 0%. Trace instead shapes the (trainable) expert.
    "--policy.freeze_vision_encoder=true",
    "--dataset.repo_id=lerobot/libero",
    f"--dataset.episodes={eps_arg}",
    f"--batch_size={BATCH}",
    f"--steps={STEPS}",
    f"--save_freq={SAVE_FREQ}",
    f"--output_dir={OUTPUT_DIR}",
]
cfg = draccus.parse(TrainPipelineConfig, args=cli)
cfg.validate()   # populates optimizer/scheduler from policy preset, sets up output dir
print(f"[train_trace] trace_weight={TRACE_WEIGHT} steps={STEPS} out={OUTPUT_DIR}", flush=True)

dataset = make_dataset(cfg)
device = torch.device("cuda")
policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
policy.train().to(device)

processor_overrides = {
    "device_processor": {"device": device.type},
    "normalizer_processor": {
        "stats": dataset.meta.stats,
        "features": {**policy.config.input_features, **policy.config.output_features},
        "norm_map": policy.config.normalization_mapping,
    },
    "rename_observations_processor": {"rename_map": cfg.rename_map},
}
# BUG FIX: also override the postprocessor's unnormalizer with THIS dataset's action
# stats. Without this it kept smolvla_base's SO-100 stats -> eval unnormalized actions
# at the wrong scale -> 0% everywhere (training was fine; only eval was broken).
postprocessor_overrides = {
    "unnormalizer_processor": {
        "stats": dataset.meta.stats,
        "features": policy.config.output_features,
        "norm_map": policy.config.normalization_mapping,
    },
}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
    dataset_stats=dataset.meta.stats, preprocessor_overrides=processor_overrides,
    postprocessor_overrides=postprocessor_overrides,
)
optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

# ---- trace targets: per-episode tracks (use frame_index for local t) ----
TRACES = {}
for f in glob.glob(f"{ROOT}/data_cache/traces/ep_*.npz"):
    d = np.load(f); TRACES[int(d["episode_index"])] = d["tracks"].astype(np.float32)  # (T,N,2)
N_POINTS = next(iter(TRACES.values())).shape[1]
print(f"[train_trace] loaded {len(TRACES)} trace files, N={N_POINTS} points, H={H_HORIZON}", flush=True)

def build_trace_target(raw_batch):
    """Use frame_index (local t) + episode_index -> (B,H,N,2) target, (B,H) mask."""
    fidx = raw_batch["frame_index"].tolist()
    epix = raw_batch["episode_index"].tolist()
    B = len(fidx)
    tgt = np.zeros((B, H_HORIZON, N_POINTS, 2), np.float32)
    msk = np.zeros((B, H_HORIZON), np.float32)
    for i in range(B):
        ep = int(epix[i]); tr = TRACES.get(ep)
        if tr is None: continue
        t = int(fidx[i]); T = tr.shape[0]
        for h in range(H_HORIZON):
            tt = t + h + 1
            if tt < T:
                tgt[i, h] = tr[tt]; msk[i, h] = 1.0
    return (torch.from_numpy(tgt).to(device) / IMG_SIZE, torch.from_numpy(msk).to(device))

# Capture the trainable EXPERT features (input to action_out_proj) via a pre-hook,
# so the trace head shapes the expert — which IS trained under expert-only. Reading
# the frozen VLM features instead would make the trace loss a no-op probe.
_feat = {}
policy.model.action_out_proj.register_forward_pre_hook(lambda m, inp: _feat.__setitem__("x", inp[0]))

trace_head = None
# 학습 루프를 직접 돌리므로 cfg.num_workers 는 쓰이지 않음. 실제로 적용되는 건 여기임.
# worker 프로세스가 video frame tensor 로 /dev/shm 을 새게 해서 예전 실행이 8-21GB 까지
# 올라간 뒤 OOM 으로 죽었음. 2 로 낮추고 pin_memory 를 끄면 안정적이었음.
# 그래도 새면 DL_WORKERS=0 으로 공유메모리 경로를 아예 없앨 수 있음 (느려짐).
DL_WORKERS = int(os.environ.get("DL_WORKERS", "2"))
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True,
                                     num_workers=DL_WORKERS,
                                     prefetch_factor=2 if DL_WORKERS else None,
                                     pin_memory=False, drop_last=True)

step = 0
for raw_batch in cycle(loader):
    tgt, msk = build_trace_target(raw_batch)                 # before preprocessing (needs index)
    batch = preprocessor(raw_batch)                          # to device, normalize, tokenize
    action_loss, ld = policy.forward(batch)

    # trainable expert features captured during policy.forward (input to action_out_proj)
    pooled = _feat["x"].mean(dim=1)                              # (B, D_expert)
    if trace_head is None:
        D = pooled.shape[-1]
        trace_head = nn.Sequential(nn.Linear(D, 512), nn.GELU(),
                                   nn.Linear(512, H_HORIZON * N_POINTS * 2)).to(device)
        # add to existing param group (not a new group) so the LR scheduler stays consistent
        optimizer.param_groups[0]["params"].extend(list(trace_head.parameters()))
        print(f"[train_trace] trace_head created D={D} -> {H_HORIZON*N_POINTS*2}", flush=True)
    pred = trace_head(pooled).view(-1, H_HORIZON, N_POINTS, 2)
    per = ((pred - tgt) ** 2).mean(dim=3).mean(dim=2)            # (B,H)
    trace_loss = (per * msk).sum() / msk.sum().clamp(min=1.0)

    loss = action_loss + TRACE_WEIGHT * trace_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
    optimizer.step()
    lr_scheduler.step()

    if step % LOG_FREQ == 0:
        print(f"step {step}/{STEPS}  action={action_loss.item():.4f}  trace={trace_loss.item():.4f}  "
              f"total={loss.item():.4f}", flush=True)
    step += 1
    if step % SAVE_FREQ == 0 or step >= STEPS:
        cdir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
        save_checkpoint(checkpoint_dir=cdir, step=step, cfg=cfg, policy=policy,
                        optimizer=optimizer, scheduler=lr_scheduler,
                        preprocessor=preprocessor, postprocessor=postprocessor)
        update_last_checkpoint(cdir)
        print(f"[train_trace] saved checkpoint @ step {step}", flush=True)
    if step >= STEPS:
        break

print("[train_trace] DONE", flush=True)
