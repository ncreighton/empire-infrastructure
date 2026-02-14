@echo off
echo =============================================
echo  ZimmWriter Controller - Setup
echo =============================================
echo.

python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Verifying...
python -c "import pywinauto; import fastapi; print('All OK')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Install failed. Try: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo =============================================
echo  Setup complete!
echo.
echo  Next steps:
echo    1. Open ZimmWriter
echo    2. Run: python scripts/discover_controls.py
echo    3. Run: start-server.bat
echo.
echo  API docs: http://localhost:8765/docs
echo =============================================
pause
