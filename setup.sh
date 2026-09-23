#!/usr/bin/env bash
# One-shot setup for backend + frontend. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1 — $2"; exit 1; }; }
need python3 "install Python 3.10+ from https://python.org"
need node "install Node 18+ from https://nodejs.org"
command -v ffmpeg >/dev/null 2>&1 || echo "Note: ffmpeg not found. Most audio decodes without it, but install it for full format coverage (apt install ffmpeg / brew install ffmpeg)."

echo "==> Backend: virtual environment + dependencies (torch is large, this takes a few minutes)"
cd backend
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r requirements.txt
fi
[ -f .env.example ] && [ ! -f .env ] && cp .env.example .env && echo "    created backend/.env"
mkdir -p data
cd ..

echo "==> Frontend: npm dependencies"
cd frontend && npm install && cd ..

echo
echo "Setup complete. Start everything with:  ./start.sh"
echo "Model weights download on first use (Whisper tiny ~150 MB, small ~1 GB, large-v3-turbo ~1.6 GB)."
