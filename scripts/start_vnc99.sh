#!/bin/bash
# 물리 모니터 없이 가상 화면(:99)을 만들고 VNC로 중계
# 사용: bash start_vnc99.sh [비밀번호]
PW="${1:-molmoact}"
DISP=":99"
GEOM="1920x1080x24"

# 기존 정리
pkill -9 -f "Xvfb $DISP" 2>/dev/null
pkill -9 -f "x11vnc" 2>/dev/null
pkill -9 -f "fluxbox" 2>/dev/null
sleep 1

# 1) 가상 모니터 생성
Xvfb $DISP -screen 0 $GEOM -ac +extension GLX +render -noreset \
  > "$HOME/.vnc/xvfb.log" 2>&1 &
sleep 2

# 2) 창 관리자 (가벼움)
DISPLAY=$DISP fluxbox > "$HOME/.vnc/fluxbox.log" 2>&1 &
sleep 1

# 3) VNC 비밀번호
mkdir -p "$HOME/.vnc"
x11vnc -storepasswd "$PW" "$HOME/.vnc/passwd" >/dev/null 2>&1

# 4) :99 를 VNC로 중계
x11vnc -display $DISP \
  -rfbauth "$HOME/.vnc/passwd" \
  -rfbport 5900 \
  -forever -shared -noxdamage -repeat \
  -bg -o "$HOME/.vnc/x11vnc.log"
sleep 2

echo "===== 가상 화면 VNC 실행됨 ====="
echo "해상도    : $GEOM"
echo "접속 주소 : 100.71.184.57:5900"
echo "비밀번호  : $PW"
echo
echo "[프로세스]"
pgrep -f "Xvfb $DISP" >/dev/null && echo "  Xvfb    OK" || echo "  Xvfb    실패"
pgrep -f fluxbox      >/dev/null && echo "  fluxbox OK" || echo "  fluxbox 실패"
pgrep -f x11vnc       >/dev/null && echo "  x11vnc  OK" || echo "  x11vnc  실패"
echo "[포트]"
ss -tln 2>/dev/null | grep 5900 || echo "  5900 안 열림"
