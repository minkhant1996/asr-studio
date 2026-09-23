#!/usr/bin/env bash
# Starts backend (port 8767) and frontend (port 5175); Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
[ -d backend/.venv ] && [ -d frontend/node_modules ] || { echo "Run ./setup.sh first."; exit 1; }
trap 'kill 0' EXIT INT TERM
( cd backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8767 --reload ) &
( cd frontend && npm run dev -- --port 5175 --strictPort ) &
echo "Frontend: http://localhost:5175   Backend API: http://127.0.0.1:8767/docs"
wait
