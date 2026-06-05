#!/bin/bash
set -e

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
