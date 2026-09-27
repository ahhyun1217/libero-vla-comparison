#!/usr/bin/env python
"""
Track B — trace label generation (ATM-style point tracking) for LIBERO.

Uses CoTracker3 *online* model (sliding window) so memory is bounded regardless
of episode length (the offline model OOMs on long clips).

For each episode, tracks a grid of points over the agentview video and saves the
2D trajectories — the "trace" target SmolVLA will learn to predict.

Output per episode: data_cache/traces/ep_{idx:06d}.npz
  tracks: float32 (T, N, 2)  pixel coords (x,y)
  vis:    bool    (T, N)
  + episode_index, task_index, W, H, grid_size

GPU inference ~2-4GB, NO EGL/sim (safe alongside other GPU work if VRAM available).
Everything stays under amber/vla_wam.
"""
import argparse, os
import numpy as np
import torch


def get_episode_ranges(ds):
    edi = getattr(ds, "episode_data_index", None)
    ranges = []
    if edi is not None and "from" in edi:
        froms = edi["from"].tolist(); tos = edi["to"].tolist()
        for i, (a, b) in enumerate(zip(froms, tos)):
            ranges.append((i, int(a), int(b)))
    else:
        ep_col = np.asarray(ds.hf_dataset["episode_index"])
        for i in np.unique(ep_col):
            idx = np.where(ep_col == i)[0]
            ranges.append((int(i), int(idx.min()), int(idx.max()) + 1))
    return ranges


def load_agentview_video(ds, a, b, cam_key):
    frames = []
    for i in range(a, b):
        img = ds[i][cam_key]                       # (3,H,W) float [0,1]
        img = (img.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        frames.append(img)
    return np.stack(frames, 0)                     # (T,H,W,3) uint8


@torch.no_grad()
def run_cotracker_online(tracker, video_thw, grid_size, device):
    """video_thw: (T,H,W,3) uint8 → returns tracks (T,N,2), vis (T,N)."""
    T = video_thw.shape[0]
    step = tracker.step                            # window stride (usually 8)
    frames = [torch.from_numpy(video_thw[i]).to(device).float() for i in range(T)]

    def _chunk(buf):
        v = torch.stack(buf[-step * 2:], 0).permute(0, 3, 1, 2)[None]  # (1,W,3,H,W)
        return v

    is_first = True
    buf = []
    pred_tracks = pred_vis = None
    for i in range(T):
        buf.append(frames[i])
        if i % step == 0 and i != 0:
            pred_tracks, pred_vis = tracker(_chunk(buf), is_first_step=is_first, grid_size=grid_size)
            is_first = False
    # flush last partial window
    pred_tracks, pred_vis = tracker(_chunk(buf), is_first_step=is_first, grid_size=grid_size)
    return pred_tracks[0].cpu().numpy().astype(np.float32), pred_vis[0].cpu().numpy().astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="lerobot/libero")
    ap.add_argument("--cam", default="observation.images.image")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data_cache", "traces"))
    ap.add_argument("--episodes", default="0:5", help="'a:b' slice, comma list, or 'all'")
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="cotracker3_online")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo)
    ranges = get_episode_ranges(ds)
    if args.episodes == "all":
        sel = ranges
    elif ":" in args.episodes:
        a, b = args.episodes.split(":"); sel = ranges[int(a):int(b)]
    else:
        want = {int(x) for x in args.episodes.split(",")}; sel = [r for r in ranges if r[0] in want]
    print(f"[gen_traces] repo={args.repo} episodes={len(sel)} grid={args.grid_size} "
          f"model={args.model} device={args.device}", flush=True)

    tracker = torch.hub.load("facebookresearch/co-tracker", args.model).to(args.device).eval()
    print(f"[gen_traces] tracker loaded (step={tracker.step})", flush=True)

    for (ep, a, b) in sel:
        vid = load_agentview_video(ds, a, b, args.cam)          # (T,H,W,3)
        T, H, W, _ = vid.shape
        tracks, vis = run_cotracker_online(tracker, vid, args.grid_size, args.device)
        task_idx = int(np.asarray(ds[a]["task_index"]))
        outp = os.path.join(args.out, f"ep_{ep:06d}.npz")
        np.savez_compressed(outp, tracks=tracks, vis=vis, episode_index=ep,
                            task_index=task_idx, W=W, H=H, grid_size=args.grid_size)
        print(f"  ep {ep:4d} | T={T:3d} N={tracks.shape[1]:3d} tracks{tracks.shape} | {os.path.basename(outp)}",
              flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print("[gen_traces] done ->", os.path.abspath(args.out), flush=True)


if __name__ == "__main__":
    main()
