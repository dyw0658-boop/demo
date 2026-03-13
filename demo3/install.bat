@echo off
echo ========================================
echo  ACGAN + PPO 项目安装脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未检测到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo 检测到Python环境
echo.

REM 安装依赖
echo 正在安装依赖包...
pip install -r requirements.txt

if errorlevel 1 (
    echo 警告: 依赖安装可能有问题，请手动检查
    echo.
)

REM 创建输出目录
echo.
echo 创建输出目录...
if not exist "outputs" mkdir outputs
if not exist "outputs\checkpoints" mkdir outputs\checkpoints
if not exist "outputs\images" mkdir outputs\images
if not exist "outputs\logs" mkdir outputs\logs

REM 验证安装
echo.
echo 验证安装...
python simple_test.py

if errorlevel 1 (
    echo 警告: 项目验证可能有问题
) else (
    echo 项目验证通过！
)

echo.
echo ========================================
echo 安装完成！
echo.
echo 下一步操作：
echo 1. 预训练: scripts\\train_pretrain.sh
echo 2. PPO微调: scripts\\train_ppo.sh
echo 3. 评估: scripts\\eval.sh
echo ========================================
echo.
pause