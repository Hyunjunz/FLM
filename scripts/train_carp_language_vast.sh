#!/usr/bin/env bash
set -euo pipefail

# CARP language-reasoning run for Vast/Linux CUDA.
# Default dataset: tau/commonsense_qa.
# Override example:
#   MAX_EXAMPLES=8000 MAX_STEPS=2000 BATCH_SIZE=64 bash scripts/train_carp_language_vast.sh

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

DATASET="${DATASET:-mix_language}"
EVAL_DATASET="${EVAL_DATASET:-tau/commonsense_qa}"
SPLIT="${SPLIT:-train}"
DATA="${DATA:-data/carp_commonsenseqa_train.jsonl}"
TOKENIZER="${TOKENIZER:-artifacts/tokenizer_carp_language}"
CONFIG="${CONFIG:-configs/carp_micro.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/carp_language_ckpt}"

MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
VOCAB_SIZE="${VOCAB_SIZE:-4096}"
REASONING_TOKENS="${REASONING_TOKENS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-192}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-0}"
SAVE_STEP_DIRS="${SAVE_STEP_DIRS:-0}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
ROUTER_LOSS_WEIGHT="${ROUTER_LOSS_WEIGHT:-0.2}"
RANKING_LOSS_WEIGHT="${RANKING_LOSS_WEIGHT:-0.5}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
CPU_THREADS="${CPU_THREADS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

echo "CARP language Vast run"
echo "  dataset=$DATASET split=$SPLIT max_examples=$MAX_EXAMPLES"
echo "  data=$DATA"
echo "  tokenizer=$TOKENIZER vocab_size=$VOCAB_SIZE"
echo "  output=$OUTPUT_DIR"
echo "  device=$DEVICE amp=$AMP_DTYPE batch_size=$BATCH_SIZE max_steps=$MAX_STEPS"
echo

python -u scripts/download_carp_language.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --output "$DATA" \
  --max-examples "$MAX_EXAMPLES"

if [[ ! -e "$TOKENIZER/tokenizer.json" && ! -e "$TOKENIZER" ]]; then
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
  --ranking-loss-weight "$RANKING_LOSS_WEIGHT"
  --device "$DEVICE"
  --amp-dtype "$AMP_DTYPE"
  --cpu-threads "$CPU_THREADS"
  --save-every "$SAVE_EVERY"
)

if [[ "$SAVE_STEP_DIRS" == "1" ]]; then
  args+=(--save-step-dirs)
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
  --dataset "$EVAL_DATASET" \
  --split validation \
  --max-examples 200 \
  --device "$DEVICE" \
  --amp-dtype "$AMP_DTYPE"

echo
echo "Done. This trains on mixed language data and evaluates answer scoring on $EVAL_DATASET."
