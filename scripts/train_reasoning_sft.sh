#!/bin/bash
set -e

if [ ! -f data/reasoning_sft.jsonl ]; then
  python scripts/prepare_reasoning_artifacts.py \
    --train data/train.jsonl \
    --eval data/eval.jsonl \
    --reasoning-sft data/reasoning_sft.jsonl \
    --reasoning-eval data/reasoning_eval.jsonl \
    --verifier-train data/verifier_train.jsonl \
    --verifier-eval data/verifier_eval.jsonl
fi

python -m cpu_lite_lm.train_reasoning_sft \
  --model artifacts/base_ckpt \
  --tokenizer artifacts/tokenizer \
  --data data/reasoning_sft.jsonl \
  --eval-data data/reasoning_eval.jsonl \
  --output-dir artifacts/reasoning_sft_ckpt \
  --block-size 1024 \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --learning-rate 5e-5 \
  --epochs 2
