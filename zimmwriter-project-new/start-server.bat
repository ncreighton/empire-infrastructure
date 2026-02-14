@echo off
echo Starting ZimmWriter Controller API on port 8765...
echo Docs: http://localhost:8765/docs
echo.
cd /d "%~dp0"
python -m uvicorn src.api:app --host 0.0.0.0 --port 8765 --reload
pause
