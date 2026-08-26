@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python trayApp.py
echo.
echo ==========================================
echo App exited. Error message is shown above.
echo ==========================================
pause