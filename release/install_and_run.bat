@echo off
chcp 65001 >nul
title PaperFree Installer and Launcher

echo ==========================================
echo PaperFree 一键安装与启动
echo ==========================================
echo.

cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 conda。
    echo 请先安装 Anaconda 或 Miniconda，然后重新运行本脚本。
    pause
    exit /b 1
)

call conda info --envs >nul 2>nul
if errorlevel 1 (
    echo [错误] conda 无法正常工作。
    echo 请先手动打开 Anaconda Prompt 测试 conda 是否可用。
    pause
    exit /b 1
)

echo [1/5] 检查环境 paperfree 是否存在...
call conda env list | findstr /R /C:"^paperfree " >nul
if errorlevel 1 (
    echo 未找到环境，正在创建 conda 环境 paperfree...
    call conda create -n paperfree python=3.11 -y
    if errorlevel 1 (
        echo [错误] conda 环境创建失败。
        pause
        exit /b 1
    )
) else (
    echo 已找到环境 paperfree
)

echo.
echo [2/5] 激活环境...
call conda activate paperfree
if errorlevel 1 (
    echo 尝试使用 conda.bat 激活环境...
    call "%UserProfile%\miniconda3\condabin\conda.bat" activate paperfree
)
if errorlevel 1 (
    echo [错误] 无法激活环境 paperfree。
    pause
    exit /b 1
)

echo.
echo [3/5] 安装依赖...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败。
        pause
        exit /b 1
    )
) else (
    echo 未找到 requirements.txt，改为安装基础依赖...
    pip install streamlit pandas requests tqdm
    if errorlevel 1 (
        echo [错误] 基础依赖安装失败。
        pause
        exit /b 1
    )
)

echo.
echo [4/5] 检查必要目录...
if not exist data mkdir data
if not exist output mkdir output

echo.
echo [5/5] 启动网页应用...
if exist integrated_paper_pipeline.py (
    streamlit run integrated_paper_pipeline.py
) else (
    echo [错误] 未找到 integrated_paper_pipeline.py
    pause
    exit /b 1
)

pause