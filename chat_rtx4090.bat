@echo off


chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion
title CPULiteLM RTX 4090 Chat

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHONPATH=%CD%;%PYTHONPATH%"
set "CUDA_VISIBLE_DEVICES=0"

set "MODEL=rtx4090_max_ckpt"
set "TOKENIZER=tokenizer_rtx4090_32k"
set "CONFIG=configs\colab_medium.json"
set "DEVICE=cpu"
set "AMP_DTYPE=off"
set "MAX_NEW_TOKENS=256"
set "TEMPERATURE=0.7"
set "TOP_K=50"
set "SYSTEM_PROMPT="

python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if "%ERRORLEVEL%"=="0" (
    set "DEVICE=cuda"
    set "AMP_DTYPE=fp16"
)

if not exist "%MODEL%\pytorch_model.bin" (
    if exist "artifacts\rtx4090_max_ckpt\pytorch_model.bin" (
        set "MODEL=artifacts\rtx4090_max_ckpt"
    )
)

if not exist "%TOKENIZER%\tokenizer.json" (
    if exist "artifacts\tokenizer_rtx4090_32k\tokenizer.json" (
        set "TOKENIZER=artifacts\tokenizer_rtx4090_32k"
    )
)

if not exist "%MODEL%\pytorch_model.bin" (
    echo Missing model checkpoint: "%MODEL%\pytorch_model.bin"
    echo Train first, or edit MODEL in chat_rtx4090.bat.
    pause
    exit /b 1
)

if not exist "%TOKENIZER%\tokenizer.json" (
    echo Missing tokenizer: "%TOKENIZER%\tokenizer.json"
    echo Train tokenizer first, or edit TOKENIZER in chat_rtx4090.bat.
    pause
    exit /b 1
)

:loop
cls
echo ======================================================
echo   CPULiteLM RTX 4090 Chat
echo ======================================================
echo Model:     %MODEL%
echo Tokenizer: %TOKENIZER%
echo Device:    %DEVICE%  AMP: %AMP_DTYPE%
echo.

set "USER_INPUT="
set /p "USER_INPUT=질문을 입력하세요 (종료: exit): "

if /i "%USER_INPUT%"=="exit" goto end
if /i "%USER_INPUT%"=="quit" goto end
if /i "%USER_INPUT%"=="q" goto end
if "%USER_INPUT%"=="" goto loop

echo.
echo [AI 답변]
echo ------------------------------------------------------

python -u -m cpu_lite_lm.generate --model "%MODEL%" --tokenizer "%TOKENIZER%" --config "%CONFIG%" --device "%DEVICE%" --amp-dtype "%AMP_DTYPE%" --system "%SYSTEM_PROMPT%" --user "%USER_INPUT%" --max-new-tokens "%MAX_NEW_TOKENS%" --temperature "%TEMPERATURE%" --top-k "%TOP_K%"

echo.
echo ------------------------------------------------------
pause
goto loop

:end
echo 프로그램을 종료합니다.
pause
