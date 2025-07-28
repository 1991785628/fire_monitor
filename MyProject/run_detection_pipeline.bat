@echo off
chcp 65001 >nul

REM 直接使用绝对路径执行Python脚本
set "PYTHON_SCRIPT1=d:\MyProject\fire_monitor\populate_database.py"
set "PYTHON_SCRIPT2=d:\MyProject\fire_monitor\process_database_images.py"

cls
echo === 火灾检测处理流程 ===

echo 1. 运行数据库填充...
python "%PYTHON_SCRIPT1%"
if %errorlevel% neq 0 (
    echo 错误: 数据库填充失败
    pause
    exit /b 1
)

echo 2. 运行图像预测...
python "%PYTHON_SCRIPT2%"
if %errorlevel% neq 0 (
    echo 错误: 图像预测失败
    pause
    exit /b 1
)

echo === 处理完成 ===
pause