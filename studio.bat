@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m keyloom_mania studio
) else (
  python -m keyloom_mania studio
)
if errorlevel 1 pause
