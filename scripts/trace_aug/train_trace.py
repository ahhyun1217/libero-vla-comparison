#!/usr/bin/env python
"""
Track B — trace-augmented SmolVLA training (custom loop reusing lerobot infra).

Adds an auxiliary head that predicts the future 2D point-trace (ATM/MolmoAct-style)
from the shared VLM image features, so the trace loss shapes the (trainable) scene
representation the action expert consumes.

Same script runs both arms for a matched comparison:
  trace_weight=1.0  -> treatment (action loss + trace loss)
  trace_weight=0.0  -> control   (pure baseline, identical config)

Config note: train_expert_only=false so the connector is trainable (otherwise the
trace head would only probe frozen features and change nothing).

Usage: python train_trace.py <trace_weight> <output_dir> [steps]
Everything stays under amber/vla_wam.
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

ROOT = "/home/user/amber/vla_wam"
os.environ.setdefault("LIBERO_CONFIG_PATH", f"{ROOT}/.libero")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

TRACE_WEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUTPUT_DIR   = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/trackB_pointtrace/train_trace_out"
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

eps = json.load(open(f"{ROOT}/trackB_pointtrace/spatial_episodes.json"))
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
    "--num_workers=4",
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
# num_workers=0: the worker processes leak /dev/shm via video-frame tensors
# (prior runs climbed to 8-21GB shmem -> OOM-killed). Loading in the main process
# eliminates the shared-memory path entirely. Slightly slower, but leak-proof.
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True,
                                     num_workers=2, prefetch_factor=2,
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
