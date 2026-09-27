"""MolmoAct2 롤아웃 영상을 :99 가상화면에 재생 (VNC로 실시간 시청용).

eval이 생성한 mp4를 순서대로 크게 띄운다. --follow 를 주면 새 영상이
생길 때마다 자동으로 이어서 재생하므로, eval을 동시에 돌리면
"실시간에 가깝게" 로봇이 움직이는 걸 볼 수 있다.

사용:
  DISPLAY=:99 python watch_live.py               # 최신 eval 결과 재생
  DISPLAY=:99 python watch_live.py --follow      # 새 영상 계속 감시
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISPLAY", ":99")

import cv2

ROOT = Path("/home/leap/ahhyun/libero-vla/env/outputs/eval")
WINDOW = "MolmoAct2 - LIBERO"
FOLLOW = "--follow" in sys.argv


def newest_run() -> Path | None:
    """정책 종류와 무관하게 가장 최근 eval 실행 폴더를 고른다."""
    runs = sorted(ROOT.glob("*/*_libero_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def play(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    # 실행 폴더명 끝의 정책 이름(molmoact2 / smolvla / act ...)을 함께 표시
    run_name = path.parents[2].name
    policy = run_name.split("_libero_")[-1] if "_libero_" in run_name else run_name
    label = f"[{policy}] {path.parent.name}/{path.name}"
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        h, w = frame.shape[:2]
        scale = 900 / max(h, 1)
        big = cv2.resize(frame, (int(w * scale), 900), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(big, (0, 0), (big.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(big, f"{label}  frame {n}", (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
        cv2.imshow(WINDOW, big)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            cap.release()
            raise KeyboardInterrupt
    cap.release()
    print(f"  재생 완료: {label} ({n} frames)", flush=True)


cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1200, 900)

seen: set[Path] = set()
print("VNC(100.71.184.57:5900) 화면을 보세요. 종료는 창에서 q.", flush=True)

try:
    while True:
        run = newest_run()
        vids = sorted(run.rglob("*.mp4")) if run else []
        fresh = [v for v in vids if v not in seen]

        for v in fresh:
            seen.add(v)
            print(f"재생: {v.relative_to(ROOT)}", flush=True)
            play(v)

        if not FOLLOW:
            if not vids:
                print("영상이 없습니다. 먼저 eval을 실행하세요.", flush=True)
            break

        if not fresh:
            time.sleep(2)
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    print("플레이어 종료", flush=True)
