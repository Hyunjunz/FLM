@echo off
setlocal
title CPULiteLM Chat - CPU Optimized Mode (스트리밍 지원)

:: PYTHONPATH 설정
set PYTHONPATH=.

:loop
cls
echo ======================================================
echo   CPULiteLM 저사양 CPU 최적화 추론 모드 (스트리밍 지원)
echo   (Static KV Cache + High Stability Mode)
echo ======================================================
echo.

:: 시스템 프롬프트 설정
set SYSTEM_PROMPT=당신은 친절한 AI 조수입니다.

:: 사용자 입력 받기
set /p USER_INPUT="질문을 입력하세요 (종료하려면 exit): "

if "%USER_INPUT%"=="exit" goto end

echo.
echo [AI 답변 생성 중...]
echo ------------------------------------------------------

:: 추론 실행
:: 품질 안정을 위해 아직 학습이 부족한 모델에서는 --speculative를 일시적으로 끕니다.
:: --top-k 50을 추가하여 답변의 다양성과 품질을 보정합니다.
python -m cpu_lite_lm.generate ^
    --model keural_sft_ckpt ^
    --tokenizer tokenizer_keural_32k ^
    --config configs/colab_medium.json ^
    --device cpu ^
    --system "%SYSTEM_PROMPT%" ^
    --user "%USER_INPUT%" ^
    --max-new-tokens 128 ^
    --temperature 0.5 ^
    --top-k 40

echo.
echo ------------------------------------------------------
pause
goto loop

:end
echo 프로그램을 종료합니다.
pause
