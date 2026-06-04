#!/usr/bin/env bash
set -euo pipefail

# CPU-target CARP large checkpoint training preset for Vast/RTX 4090.
# Goal: train a larger but still CPU-runnable checkpoint.
# Expected CPU inference depends heavily on CPU memory bandwidth, threads, and quantization.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_ROOT="$LAUNCH_ROOT"

if [[ ! -d "$CODE_ROOT/cpu_lite_lm" ]]; then
  for candidate in "$LAUNCH_ROOT/cpu_llm_lab" "$LAUNCH_ROOT/CPU_LLM_LAB" "$LAUNCH_ROOT/flm" "$PWD"; do
    if [[ -d "$candidate/cpu_lite_lm" ]]; then
      CODE_ROOT="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ ! -d "$CODE_ROOT/cpu_lite_lm" ]]; then
  echo "Cannot find cpu_lite_lm package." >&2
  exit 1
fi

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="${DATASET:-tau/commonsense_qa}"
SPLIT="${SPLIT:-train}"
DATA="${DATA:-data/carp_commonsenseqa_train.jsonl}"
TOKENIZER="${TOKENIZER:-artifacts/tokenizer_rtx4090_32k}"
CONFIG="${CONFIG:-configs/carp_cpu_large.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/carp_cpu_large_ckpt}"
BASE_MODEL="${BASE_MODEL:-}"

MAX_EXAMPLES="${MAX_EXAMPLES:-9741}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
REASONING_TOKENS="${REASONING_TOKENS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
ROUTER_LOSS_WEIGHT="${ROUTER_LOSS_WEIGHT:-0.05}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
CPU_THREADS="${CPU_THREADS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"

echo "CARP CPU-large Vast training"
echo "  dataset=$DATASET split=$SPLIT max_examples=$MAX_EXAMPLES"
echo "  config=$CONFIG"
echo "  tokenizer=$TOKENIZER"
echo "  base_model=${BASE_MODEL:-<none>}"
echo "  output=$OUTPUT_DIR"
echo "  block_size=$BLOCK_SIZE batch_size=$BATCH_SIZE max_steps=$MAX_STEPS"
echo "  device=$DEVICE amp=$AMP_DTYPE"
echo

python -u scripts/download_carp_language.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --output "$DATA" \
  --max-examples "$MAX_EXAMPLES"

if [[ ! -e "$TOKENIZER/tokenizer.json" && ! -e "$TOKENIZER" ]]; then
  echo "Missing tokenizer: $TOKENIZER"
  echo "Training tokenizer from $DATA ..."
  python -u scripts/train_tokenizer.py \
    --data "$DATA" \
    --output-dir "$TOKENIZER" \
    --vocab-size "$VOCAB_SIZE" \
    --max-docs 0
fi

args=(
  --data "$DATA"
  --config "$CONFIG"
  --tokenizer "$TOKENIZER"
  --output-dir "$OUTPUT_DIR"
  --reasoning-tokens "$REASONING_TOKENS"
  --block-size "$BLOCK_SIZE"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --router-loss-weight "$ROUTER_LOSS_WEIGHT"
  --device "$DEVICE"
  --amp-dtype "$AMP_DTYPE"
  --cpu-threads "$CPU_THREADS"
)

if [[ -n "$BASE_MODEL" ]]; then
  args+=(--base-model "$BASE_MODEL")
fi

python -u scripts/train_carp_sft.py "${args[@]}"

python -u scripts/eval_carp_router.py \
  --model "$OUTPUT_DIR" \
  --data "$DATA" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --block-size "$BLOCK_SIZE" \
  --device "$DEVICE"

python -u scripts/eval_carp_language_answer.py \
  --model "$OUTPUT_DIR" \
  --dataset "$DATASET" \
  --split validation \
  --max-examples 200 \
  --device "$DEVICE" \
  --amp-dtype "$AMP_DTYPE"

echo
echo "Saved CPU-target large CARP checkpoint to $OUTPUT_DIR"
echo "CPU smoke inference:"
echo "  python -u scripts/carp_language_infer.py --model $OUTPUT_DIR --question 'Where would you keep a pillow when you sleep?' --choices 'A. garage\nB. bed\nC. oven\nD. road\nE. shower' --device cpu --amp-dtype off --show-carp"
