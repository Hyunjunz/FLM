#!/usr/bin/env bash
set -euo pipefail

# 24GB VRAM preset for the CPU-target 1B+ top-1 MoE model.
# Override values at launch, for example:
#   BLOCK_SIZE=768 GRAD_ACCUM_STEPS=48 bash scripts/train_cpu_1b_moe_24gb.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$CODE_ROOT/cpu_lite_lm" ]]; then
  echo "Cannot find cpu_lite_lm package." >&2
  exit 1
fi

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA="${DATA:-data/hf_cache/HAERAE-HUB___korean-webtext}"
DATASET_NAME="${DATASET_NAME:-HAERAE-HUB/KOREAN-WEBTEXT}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
CACHE_DIR="${CACHE_DIR:-./hf_cache}"
TOKENIZER="${TOKENIZER:-tokenizer_rtx4090_32k}"
CONFIG="${CONFIG:-configs/cpu_1b_moe_fast.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/cpu_1b_moe_24gb_ckpt}"
RESUME_FROM="${RESUME_FROM:-}"

VOCAB_SIZE="${VOCAB_SIZE:-32000}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
MAX_STEPS="${MAX_STEPS:-200000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-8192}"
TOKENIZER_MAX_DOCS="${TOKENIZER_MAX_DOCS:-500000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_DOCS="${EVAL_DOCS:-512}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-20}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
COMPILE="${COMPILE:-0}"
QUALITY_FILTER="${QUALITY_FILTER:-1}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"

if [[ ! -e "$DATA" ]]; then
  if [[ "$DOWNLOAD_IF_MISSING" == "1" ]]; then
    echo "Missing dataset: $DATA"
    echo "Downloading $DATASET_NAME split=$DATASET_SPLIT ..."
    python -u scripts/download_korean_webtext.py \
      --dataset-name "$DATASET_NAME" \
      --split "$DATASET_SPLIT" \
      --output-dir "$DATA" \
      --cache-dir "$CACHE_DIR"
  else
    echo "Missing dataset: $DATA" >&2
    echo "Set DATA=/path/to/text-or-arrow-cache, or set DOWNLOAD_IF_MISSING=1." >&2
    exit 1
  fi
fi

args=(
  --data "$DATA"
  --tokenizer "$TOKENIZER"
  --output-dir "$OUTPUT_DIR"
  --config "$CONFIG"
  --vocab-size "$VOCAB_SIZE"
  --block-size "$BLOCK_SIZE"
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --beta1 0.9
  --beta2 0.95
  --text-column text
  --max-docs none
  --min-chars 100
  --skip-docs 512
  --max-chars 0
  --tokenizer-max-docs "$TOKENIZER_MAX_DOCS"
  --tokenizer-log-every 1000
  --streaming
  --shuffle-buffer "$SHUFFLE_BUFFER"
  --seed 1234
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --device cuda
  --amp-dtype "$AMP_DTYPE"
  --tf32
  --foreach-optimizer
  --num-workers "$NUM_WORKERS"
  --eval-every "$EVAL_EVERY"
  --eval-docs "$EVAL_DOCS"
  --eval-skip-docs 0
  --eval-max-chars 1000000
  --eval-max-batches "$EVAL_MAX_BATCHES"
)

if [[ "$QUALITY_FILTER" == "1" ]]; then
  args+=(--quality-filter)
fi

if [[ "$COMPILE" == "1" ]]; then
  args+=(--compile)
fi

if [[ -n "$RESUME_FROM" ]]; then
  args+=(--resume-from "$RESUME_FROM")
fi

echo "Starting 1B+ MoE training for 24GB VRAM"
echo "  data=$DATA"
echo "  config=$CONFIG"
echo "  tokenizer=$TOKENIZER"
echo "  output=$OUTPUT_DIR"
echo "  batch_size=$BATCH_SIZE block_size=$BLOCK_SIZE grad_accum_steps=$GRAD_ACCUM_STEPS"
echo "  effective_tokens_per_step=$((BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS))"
echo "  max_steps=$MAX_STEPS lr=$LEARNING_RATE amp=$AMP_DTYPE compile=$COMPILE"
echo

python -u -m cpu_lite_lm.train "${args[@]}"

echo
echo "Saved checkpoint to $OUTPUT_DIR"
echo "CPU benchmark after training:"
echo "  python -m cpu_lite_lm.benchmark --model $OUTPUT_DIR --config $CONFIG --generated-tokens 64 --threads 16 --dynamic-int8"
