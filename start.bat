@echo off
REM Starts backend (port 8767) and frontend (port 5175) in two windows.
cd /d "%~dp0"
if not exist backend\.venv (echo Run setup.bat first. & exit /b 1)
if not exist frontend\node_modules (echo Run setup.bat first. & exit /b 1)
start "asr-backend"  cmd /k "cd backend && .venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8767 --reload"
start "asr-frontend" cmd /k "cd frontend && npm run dev -- --port 5175 --strictPort"
echo Frontend: http://localhost:5175   Backend API: http://127.0.0.1:8767/docs
