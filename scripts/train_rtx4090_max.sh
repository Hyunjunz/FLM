#!/usr/bin/env bash
set -euo pipefail

# RTX 4090 24GB long-run preset.
# Override any value at launch, for example:
#   BATCH_SIZE=24 GRAD_ACCUM_STEPS=3 MAX_STEPS=200000 bash scripts/train_rtx4090_max.sh

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
  echo "Current script root: $LAUNCH_ROOT" >&2
  echo "Put this script in the repository root, or run it from the directory that contains cpu_lite_lm/." >&2
  exit 1
fi

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_DATA="data/hf_cache/HAERAE-HUB___korean-webtext"
if [[ "$CODE_ROOT" != "$LAUNCH_ROOT" ]]; then
  DEFAULT_DATA="$LAUNCH_ROOT/data/hf_cache/HAERAE-HUB___korean-webtext"
fi

DATA="${DATA:-$DEFAULT_DATA}"
DATASET_NAME="${DATASET_NAME:-HAERAE-HUB/KOREAN-WEBTEXT}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
CACHE_DIR="${CACHE_DIR:-./hf_cache}"
TOKENIZER="${TOKENIZER:-artifacts/tokenizer_rtx4090_32k}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/rtx4090_max_ckpt}"
CONFIG="${CONFIG:-configs/colab_medium.json}"

VOCAB_SIZE="${VOCAB_SIZE:-32000}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_STEPS="${MAX_STEPS:-1000000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-32768}"
TOKENIZER_MAX_DOCS="${TOKENIZER_MAX_DOCS:-500000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_DOCS="${EVAL_DOCS:-1024}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
COMPILE="${COMPILE:-1}"
QUALITY_FILTER="${QUALITY_FILTER:-1}"
RESUME_FROM="${RESUME_FROM:-}"
DOWNLOAD_IF_MISSING="${DOWNLOAD_IF_MISSING:-1}"

if [[ ! -e "$DATA" ]]; then
  if [[ "$DOWNLOAD_IF_MISSING" == "1" ]]; then
    echo "Missing dataset: $DATA"
    echo "Downloading $DATASET_NAME split=$DATASET_SPLIT ..."
    if [[ -f scripts/download_korean_webtext.py ]]; then
      python -u scripts/download_korean_webtext.py \
        --dataset-name "$DATASET_NAME" \
        --split "$DATASET_SPLIT" \
        --output-dir "$DATA" \
        --cache-dir "$CACHE_DIR"
    else
      DATASET_NAME="$DATASET_NAME" DATASET_SPLIT="$DATASET_SPLIT" DATA="$DATA" CACHE_DIR="$CACHE_DIR" python -u - <<'PY'
import os
from pathlib import Path

from datasets import load_dataset

dataset_name = os.environ["DATASET_NAME"]
split = os.environ["DATASET_SPLIT"]
output_dir = Path(os.environ["DATA"])
cache_dir = os.environ["CACHE_DIR"]

print(f"Downloading {dataset_name} split={split} cache_dir={cache_dir}", flush=True)
ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
output_dir.parent.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(output_dir))
print(f"Saved Korean webtext dataset to {output_dir}", flush=True)
PY
    fi
  else
    echo "Missing dataset: $DATA" >&2
    echo "Set DATA=/path/to/text-or-arrow-cache, or set DOWNLOAD_IF_MISSING=1." >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  --eval-max-chars 2000000
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

echo "Starting RTX 4090 long training run"
echo "  data=$DATA"
echo "  output=$OUTPUT_DIR"
echo "  batch_size=$BATCH_SIZE block_size=$BLOCK_SIZE grad_accum_steps=$GRAD_ACCUM_STEPS"
echo "  effective_tokens_per_step=$((BATCH_SIZE * BLOCK_SIZE * GRAD_ACCUM_STEPS))"
echo "  max_steps=$MAX_STEPS amp=$AMP_DTYPE compile=$COMPILE"

python -u -m cpu_lite_lm.train "${args[@]}"
