#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Neuro-CRISPR-KAN — one-command launcher
#   ./start.sh            run both backend (8000) + frontend (3000)
#   ./start.sh stop       kill anything we left running
#   ./start.sh logs       tail the server log
# ──────────────────────────────────────────────────────────────
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$ROOT/web"
LOG_DIR="$ROOT/logs"
BACKEND_PORT=8000
FRONTEND_PORT=3000
mkdir -p "$LOG_DIR"

# colors
C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'

cmd="${1:-up}"

free_port() {
  local p=$1
  if lsof -ti :"$p" >/dev/null 2>&1; then
    echo -e "${Y}port $p is taken — killing existing process${N}"
    lsof -ti :"$p" | xargs -r kill -9 2>/dev/null || true
    sleep 1
  fi
}

case "$cmd" in
  stop)
    echo -e "${Y}stopping Neuro-CRISPR-KAN…${N}"
    pkill -9 -f "uvicorn backend.server" 2>/dev/null || true
    pkill -9 -f "http.server $FRONTEND_PORT" 2>/dev/null || true
    free_port "$BACKEND_PORT"
    free_port "$FRONTEND_PORT"
    echo -e "${G}stopped.${N}"
    exit 0
    ;;
  logs)
    tail -f "$LOG_DIR/server.log" "$LOG_DIR/uvicorn.out" "$LOG_DIR/frontend.out" 2>/dev/null
    exit 0
    ;;
  up|"")
    ;;
  *)
    echo "usage: $0 [up|stop|logs]"; exit 1
    ;;
esac

echo -e "${C}Neuro-CRISPR-KAN launcher${N}"
echo "  root:     $ROOT"
echo "  backend:  http://localhost:$BACKEND_PORT"
echo "  frontend: http://localhost:$FRONTEND_PORT"
echo "  logs:     $LOG_DIR/"
echo

free_port "$BACKEND_PORT"
free_port "$FRONTEND_PORT"

# ── Backend (FastAPI / uvicorn) ───────────────────────────────
echo -e "${C}▶ starting backend (model warm-up takes ~5s on A100)…${N}"
cd "$ROOT"
nohup uvicorn backend.server:app \
  --host 0.0.0.0 --port "$BACKEND_PORT" \
  > "$LOG_DIR/uvicorn.out" 2>&1 &
BACKEND_PID=$!
echo "  pid=$BACKEND_PID  log=$LOG_DIR/uvicorn.out"

# Wait for /  to return 200 (model loaded)
echo -n "  waiting for backend"
for i in $(seq 1 60); do
  if curl -sf -o /dev/null "http://localhost:$BACKEND_PORT/"; then
    echo -e " ${G}ready${N}"; break
  fi
  echo -n "."; sleep 1
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo -e " ${R}backend crashed — see $LOG_DIR/uvicorn.out${N}"
    tail -20 "$LOG_DIR/uvicorn.out"
    exit 1
  fi
  if [ "$i" -eq 60 ]; then
    echo -e " ${R}timed out${N}"; tail -20 "$LOG_DIR/uvicorn.out"; exit 1
  fi
done

# ── Frontend (static http.server) ─────────────────────────────
echo -e "${C}▶ starting frontend…${N}"
cd "$WEB_DIR"
nohup python3 -m http.server "$FRONTEND_PORT" \
  > "$LOG_DIR/frontend.out" 2>&1 &
FRONTEND_PID=$!
echo "  pid=$FRONTEND_PID  log=$LOG_DIR/frontend.out"
sleep 1
if ! curl -sf -o /dev/null "http://localhost:$FRONTEND_PORT/"; then
  echo -e "${R}frontend failed${N}"; tail -20 "$LOG_DIR/frontend.out"; exit 1
fi

cat <<EOF

${G}✓ Neuro-CRISPR-KAN is up${N}

   Open:    http://localhost:$FRONTEND_PORT
   Health:  http://localhost:$BACKEND_PORT/
   Logs:    http://localhost:$BACKEND_PORT/api/logs?kind=predictions&n=20

   Tail live:   ./start.sh logs
   Shut down:   ./start.sh stop      (or Ctrl+C here)

EOF

# Trap Ctrl+C — gracefully kill both children
cleanup() {
  echo -e "\n${Y}stopping…${N}"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  free_port "$BACKEND_PORT"
  free_port "$FRONTEND_PORT"
  echo -e "${G}stopped.${N}"
  exit 0
}
trap cleanup INT TERM

# Foreground tail so the script stays alive and prints log lines
tail -F "$LOG_DIR/server.log" "$LOG_DIR/uvicorn.out" 2>/dev/null
