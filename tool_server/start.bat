@echo off
title 净源哨兵 6.0 - Tool Server
echo ============================================
echo   净源哨兵 6.0 — 工具服务器
echo   端口: 5050
echo   健康检查: http://127.0.0.1:5050/health
echo ============================================
echo.
cd /d "%~dp0"
pip install -q -r requirements.txt 2>nul
echo [启动中...]
python server.py
pause
