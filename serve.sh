#!/usr/bin/env bash
# serve.sh — host ASR Studio: one backend process serving the built frontend and the API on one
# port, optionally behind an ngrok tunnel with basic auth. Settings come from ./.env (see .env.example).
#
#   ./serve.sh start      # (default) build the frontend, (re)start the server and tunnel, print the URL
#   ./serve.sh stop
#   ./serve.sh restart
#   ./serve.sh status
#   ./serve.sh logs [n]
#
# Dev mode (hot reload, two ports, no tunnel) is still ./start.sh.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

[[ -f "$ROOT/.env" ]] || { echo "error: no .env (copy .env.example to .env)" >&2; exit 1; }
set -a; source "$ROOT/.env"; set +a
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8767}"
TUNNEL="${TUNNEL:-1}"; DOMAIN="${DOMAIN:-}"
BASIC_AUTH_USER="${BASIC_AUTH_USER:-}"; BASIC_AUTH_PASS="${BASIC_AUTH_PASS:-}"
NGROK_WEB="127.0.0.1:4041"   # the agent's local API; 4040 is usually taken by another project's agent

RUN="$ROOT/.run"; mkdir -p "$RUN"
BE_PID="$RUN/backend.pid"; BE_LOG="$RUN/backend.log"
NG_PID="$RUN/ngrok.pid";   NG_LOG="$RUN/ngrok.log"; NG_CFG="$RUN/ngrok.yml"

NGROK_BIN="${NGROK_BIN:-}"
if [[ -z "$NGROK_BIN" ]]; then
  for c in "$HOME/.local/bin/ngrok" "$(command -v ngrok 2>/dev/null || true)"; do
    [[ -n "$c" && -x "$c" ]] && { NGROK_BIN="$c"; break; }
  done
fi

kill_pidfile() {  # <pidfile> <name>
  local f="$1" name="$2"
  if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
    local pid; pid="$(cat "$f")"
    echo ">> stopping $name (pid $pid)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.3; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
}

stop_all() {
  kill_pidfile "$NG_PID" ngrok
  kill_pidfile "$BE_PID" backend
  # a backend started by hand on the same port would block ours
  pkill -f "uvicorn app.main:app --host $HOST --port $PORT" 2>/dev/null || true
}

public_url() {
  curl -fsS "http://$NGROK_WEB/api/tunnels" 2>/dev/null | grep -o '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4
}

write_ngrok_config() {
  {
    echo 'version: "2"'
    echo "web_addr: $NGROK_WEB"
    echo 'tunnels:'
    echo '  asr-studio:'
    echo '    proto: http'
    echo "    addr: $HOST:$PORT"
    [[ -n "$DOMAIN" ]] && echo "    domain: $DOMAIN"
    if [[ -n "$BASIC_AUTH_USER" || -n "$BASIC_AUTH_PASS" ]]; then
      echo '    basic_auth:'
      echo "      - \"$BASIC_AUTH_USER:$BASIC_AUTH_PASS\""
    fi
  } > "$NG_CFG"
  chmod 600 "$NG_CFG"
}

start_all() {
  [[ -d backend/.venv && -d frontend/node_modules ]] || { echo "Run ./setup.sh first." >&2; exit 1; }
  if [[ "$TUNNEL" == "1" ]]; then
    [[ -n "$NGROK_BIN" ]] || { echo "error: ngrok not found (set NGROK_BIN or TUNNEL=0)" >&2; exit 1; }
    if [[ -n "$BASIC_AUTH_PASS" && ${#BASIC_AUTH_PASS} -lt 8 ]]; then
      echo "error: ngrok needs BASIC_AUTH_PASS of at least 8 characters" >&2; exit 1
    fi
  fi

  echo ">> building frontend"
  ( cd frontend && npm run build >"$RUN/build.log" 2>&1 ) || { echo "frontend build failed — see $RUN/build.log" >&2; tail -n 20 "$RUN/build.log" >&2; exit 1; }

  stop_all
  echo ">> starting backend on $HOST:$PORT"
  cd backend
  nohup .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" >>"$BE_LOG" 2>&1 &
  echo $! > "$BE_PID"
  cd "$ROOT"
  echo -n ">> waiting for backend"
  for _ in $(seq 1 60); do
    curl -fsS -o /dev/null "http://$HOST:$PORT/api/health" 2>/dev/null && { echo " — up"; break; }
    kill -0 "$(cat "$BE_PID")" 2>/dev/null || { echo; echo "backend exited — see $BE_LOG" >&2; tail -n 15 "$BE_LOG" >&2; exit 1; }
    echo -n "."; sleep 1
  done

  if [[ "$TUNNEL" == "1" ]]; then
    write_ngrok_config
    local cfgs=()
    for c in "$HOME/.config/ngrok/ngrok.yml" "$HOME/snap/ngrok/current/.config/ngrok/ngrok.yml"; do
      [[ -f "$c" ]] && { cfgs=(--config "$c"); break; }
    done
    echo ">> starting ngrok tunnel${DOMAIN:+ on $DOMAIN}"
    nohup "$NGROK_BIN" start asr-studio --log "$NG_LOG" "${cfgs[@]}" --config "$NG_CFG" >/dev/null 2>&1 &
    echo $! > "$NG_PID"
    echo -n ">> waiting for ngrok"
    for _ in $(seq 1 20); do
      [[ -n "$(public_url)" ]] && { echo " — up"; break; }
      kill -0 "$(cat "$NG_PID")" 2>/dev/null || { echo; echo "ngrok exited — see $NG_LOG" >&2; tail -n 10 "$NG_LOG" >&2; exit 1; }
      echo -n "."; sleep 1
    done
  fi

  echo
  echo "==================================================================="
  [[ "$TUNNEL" == "1" ]] && echo "  public : $(public_url || echo '(no tunnel — see .run/ngrok.log)')${BASIC_AUTH_USER:+   (user: $BASIC_AUTH_USER, password in .env)}"
  echo "  local  : http://$HOST:$PORT"
  echo "  logs   : ./serve.sh logs      stop: ./serve.sh stop"
  echo "==================================================================="
}

status_all() {
  for pair in "backend:$BE_PID" "ngrok:$NG_PID"; do
    local name="${pair%%:*}" f="${pair##*:}"
    if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then echo "$name: running (pid $(cat "$f"))"; else echo "$name: stopped"; fi
  done
  curl -fsS "http://$HOST:$PORT/api/health" 2>/dev/null && echo || echo "backend: not answering on $HOST:$PORT"
  [[ "$TUNNEL" == "1" ]] && echo "public: $(public_url || echo '(none)')"
}

case "${1:-start}" in
  start|restart) start_all ;;
  stop)    stop_all; echo ">> stopped" ;;
  status)  status_all ;;
  logs)    tail -n "${2:-40}" -f "$BE_LOG" "$NG_LOG" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
