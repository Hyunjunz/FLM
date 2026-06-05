#!/bin/bash
set -e

if [ ! -f data/verifier_train.jsonl ]; then
  python scripts/prepare_reasoning_artifacts.py \
    --train data/train.jsonl \
    --eval data/eval.jsonl \
    --reasoning-sft data/reasoning_sft.jsonl \
    --reasoning-eval data/reasoning_eval.jsonl \
    --verifier-train data/verifier_train.jsonl \
    --verifier-eval data/verifier_eval.jsonl
fi

python -m cpu_lite_lm.train_verifier \
  --model artifacts/reasoning_sft_ckpt \
  --tokenizer artifacts/tokenizer \
  --data data/verifier_train.jsonl \
  --eval-data data/verifier_eval.jsonl \
  --output-dir artifacts/verifier_ckpt \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --learning-rate 1e-4 \
  --verifier-loss-weight 1.0
