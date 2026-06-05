#!/bin/bash

# CPULiteLM MoE 대용량 데이터 학습 스크립트
# 1. 토크나이저 학습 (이미 되어있다면 건너뜀)
# 2. CARP 추론 토큰 추가
# 3. 스트리밍 모드로 전체 데이터 학습 (Overfitting 방지)

# --- 설정 ---
VOCAB_SIZE=32000
REASONING_TOKENS=128
DATA_PATH="data/hf_cache/HAERAE-HUB___korean-webtext"
TOKENIZER_DIR="artifacts/tokenizer_moe"
MODEL_DIR="artifacts/moe_cpu_large_ckpt"
CONFIG="configs/moe_cpu_large.json"

# 에러 발생 시 중단
set -e

# --- STEP 1 & 2: 토크나이저 준비 ---
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
    echo "--- 토크나이저가 없으므로 새로 학습합니다 ---"
    python -m cpu_lite_lm.tokenizer_train \
        --data "$DATA_PATH" \
        --output-dir "$TOKENIZER_DIR" \
        --vocab-size "$VOCAB_SIZE" \
        --max-docs 200000 \
        --text-column "text"

    python scripts/add_carp_tokens.py \
        --tokenizer "$TOKENIZER_DIR" \
        --output-dir "$TOKENIZER_DIR" \
        --count "$REASONING_TOKENS"
fi

echo "--- STEP 3: MoE Full-Scale Streaming Training 시작 ---"
# --max-docs -1, --max-chars -1 로 설정하여 데이터 제한을 완전히 풉니다.
# --streaming 을 사용하여 수십 GB 데이터를 메모리 부담 없이 학습합니다.
# --block-size 를 256으로 키워 긴 문맥과 추론 능력을 확보합니다.

python -m cpu_lite_lm.train \
    --config "$CONFIG" \
    --tokenizer "$TOKENIZER_DIR" \
    --data "$DATA_PATH" \
    --output-dir "$MODEL_DIR" \
    --vocab-size $((VOCAB_SIZE + REASONING_TOKENS + 10)) \
    --streaming \
    --shuffle-buffer 5000 \
    --block-size 512 \
    --max-docs -1 \
    --max-chars -1 \
    --batch-size 8 \
    --grad-accum-steps 8 \
    --learning-rate 2e-4 \
    --weight-decay 0.1 \
    --log-every 10 \
    --save-every 1000 \
    --amp-dtype off \
    --foreach-optimizer

echo "--- 학습이 정상적으로 완료되었습니다 ---"
echo "최종 모델 위치: $MODEL_DIR"
