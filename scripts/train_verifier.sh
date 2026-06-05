#!/bin/bash
set -e

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
