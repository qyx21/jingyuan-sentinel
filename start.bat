@echo off
chcp 65001 >nul
title 净源哨兵 — 环境启动

echo ========================================
echo   净源哨兵 — 环境启动检查
echo ========================================
echo.

REM ─── 1. 检查 Ollama ──────────────────────────
echo [1/5] 检查 Ollama Embedding 服务...
curl -s http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 (
    echo   ✓ Ollama 已在运行
) else (
    echo   ! Ollama 未运行，正在启动...
    start "" "ollama app" 2>nul
    REM 后台启动 ollama serve（前台启动会卡住本脚本）
    start "" /min ollama serve >nul 2>&1
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if not errorlevel 1 (
        echo   ✓ Ollama 启动成功
    ) else (
        echo   ✗ Ollama 启动失败！请手动启动 Ollama
    )
)

REM 检查 bge-m3 模型
curl -s http://localhost:11434/api/tags | findstr "bge-m3" >nul 2>&1
if not errorlevel 1 (
    echo   ✓ bge-m3 模型已就绪
) else (
    echo   ! bge-m3 模型未安装，正在拉取...
    ollama pull bge-m3
    echo   ✓ bge-m3 安装完成
)
echo.

REM ─── 2. 检查 Dify Docker ──────────────────────
echo [2/5] 检查 Dify Docker 服务...
docker ps --format "{{.Names}}" | findstr "docker-api" >nul 2>&1
if not errorlevel 1 (
    echo   ✓ Dify 容器已运行
) else (
    echo   ! Dify 容器未运行，请进入 dify\docker 目录执行:
    echo     docker compose up -d
    echo.
    echo   然后重新运行本脚本。
    pause
    exit /b 1
)
echo.

REM ─── 3. 检查 Dify API ─────────────────────────
echo [3/5] 检查 Dify API 连通性...
curl -s http://localhost/v1/chat-messages -X POST -H "Content-Type: application/json" -d "{}" 2>&1 | findstr "unauthorized" >nul 2>&1
if not errorlevel 1 (
    echo   ✓ Dify API 正常响应（需鉴权，符合预期）
) else (
    echo   ? Dify API 状态未知，登录 http://localhost 确认
)
echo.

REM ─── 4. 检查工具服务器 ────────────────────────
echo [4/5] 检查工具服务器（端口 5050）...
curl -s --max-time 3 http://127.0.0.1:5050/health >nul 2>&1
if not errorlevel 1 (
    echo   ✓ 工具服务器已运行
) else (
    echo   ! 工具服务器未运行，正在启动...
    start "净源哨兵-工具服务器" cmd /k "cd /d %~dp0tool_server && python server.py"
    timeout /t 6 /nobreak >nul
    curl -s --max-time 3 http://127.0.0.1:5050/health >nul 2>&1
    if not errorlevel 1 (
        echo   ✓ 工具服务器启动成功
    ) else (
        echo   ✗ 工具服务器启动失败！请手动执行: cd tool_server ^&^& python server.py
    )
)
echo.

REM ─── 5. 检查知识库 ────────────────────────────
echo [5/5] 检查知识库...
echo   - 知识库文件: knowledge_base\ 目录下 8 个文件（105 条目）
echo   - 知识库由 kb-search 工具本地读取，无需 Dify 索引
echo.

REM ─── 完成 ──────────────────────────────────────
echo ========================================
echo   全部检查完成！
echo.
echo   访问净源哨兵 Agent:
echo   http://localhost/agent/q0b0miVr6N3EdvaK
echo   前端页面: frontend.html（浏览器直接打开）
echo.
echo   演示前检查清单：
echo   [ ] Ollama 运行正常 + bge-m3 已加载
echo   [ ] Dify Docker 所有容器 healthy
echo   [ ] 工具服务器已运行（http://127.0.0.1:5050/health）
echo   [ ] Dify Agent 可正常对话
echo   [ ] 备选模型已配置（硅基流动 DeepSeek）
echo   [ ] 测试关键用例 P11 + A1 通过
echo ========================================
pause
