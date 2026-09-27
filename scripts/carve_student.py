"""Shallow-pi 방식: teacher 체크포인트에서 얕은 student 를 잘라낸다.

X-VLA 의 policy transformer 는 blocks.0 ~ blocks.23 (24층) 이다.
student 는 앞쪽 N 층만 물려받고 나머지 가중치(임베딩, soft prompt, 헤드 등)는 그대로 복사한다.

이렇게 하면 student 가 백지에서 시작하지 않는다. 우리 SmolVLA 실험이 실패에 가까웠던
가장 큰 원인(teacher 와 공유 가중치가 0)을 제거하는 것이 목적이다.

사용: python carve_student.py <남길_층수> <출력경로>
"""

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

KEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./outputs/distill/xvla_student")
TEACHER = "lerobot/xvla-libero"

print(f"[1/4] teacher 체크포인트 받기: {TEACHER}", flush=True)
src = Path(snapshot_download(TEACHER))
print(f"      {src}", flush=True)

print(f"[2/4] 출력 폴더 준비: {OUT}", flush=True)
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for f in src.iterdir():
    if f.is_file() and f.name != "model.safetensors":
        shutil.copy2(f, OUT / f.name)

print(f"[3/4] 가중치 슬라이싱 (24층 -> {KEEP}층)", flush=True)
sd = load_file(str(src / "model.safetensors"))
new_sd = {}
dropped = 0
for k, v in sd.items():
    m = re.search(r"blocks\.(\d+)\.", k)
    if m is None:
        new_sd[k] = v          # 블록이 아닌 가중치는 전부 유지
        continue
    idx = int(m.group(1))
    if idx < KEEP:
        new_sd[k] = v          # 앞쪽 N 층만 물려받음
    else:
        dropped += 1
print(f"      유지 {len(new_sd)} 텐서 / 제거 {dropped} 텐서", flush=True)
save_file(new_sd, str(OUT / "model.safetensors"), metadata={"format": "pt"})

print(f"[4/4] config 의 depth 를 {KEEP} 으로 수정", flush=True)
cfg_path = OUT / "config.json"
cfg = json.loads(cfg_path.read_text())
before = cfg.get("depth")
cfg["depth"] = KEEP
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"      depth: {before} -> {cfg['depth']}", flush=True)

size = sum(f.stat().st_size for f in OUT.iterdir() if f.is_file()) / 1e9
print(f"완료: {OUT} ({size:.2f} GB)", flush=True)
