@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\python.exe" (
  echo Python environment missing. Read README.md.
  pause
  exit /b 1
)
".venv\python.exe" -m litter_detector.ui
if errorlevel 1 pause
