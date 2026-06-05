#!/bin/bash
set -e

# 1. Prepare the mixed reasoning-oriented dataset.
python scripts/prepare_700m_dataset.py \
  --output data/train.jsonl \
  --total-count 100000 \
  --seed 42 \
  --max-text-chars 4096

# 2. Base continued pretraining.
python -m cpu_lite_lm.train \
  --config configs/carp_700m.json \
  --tokenizer artifacts/tokenizer \
  --data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --output-dir artifacts/base_ckpt \
  --block-size 512 \
  --batch-size 8 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --eval-every 500 \
  --eval-max-batches 20

# 3. Reasoning SFT.
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

# 4. Verifier head training.
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

# 5. Full-depth reasoning evaluation.
python -m cpu_lite_lm.eval_reasoning \
  --model artifacts/reasoning_sft_ckpt \
  --tokenizer artifacts/tokenizer \
  --data data/reasoning_eval.jsonl \
  --route hard \
  --full-depth \
  --temperature 0.0 \
  --max-new-tokens 256

# 6. CPU inference benchmark.
python -m cpu_lite_lm.benchmark \
  --model artifacts/reasoning_sft_ckpt \
  --config configs/carp_700m.json \
  --prompt-tokens 64 \
  --generated-tokens 64 \
  --use-cache \
  --moe-top-k 1
