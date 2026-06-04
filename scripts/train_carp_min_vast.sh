#!/usr/bin/env bash
set -euo pipefail

# Minimal CARP proof run for Vast/Linux CUDA instances.
# Override at launch, for example:
#   EXAMPLES=5000 MAX_STEPS=1000 BATCH_SIZE=64 bash scripts/train_carp_min_vast.sh

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
  echo "Run this from the repository root, or place it next to cpu_lite_lm/." >&2
  exit 1
fi

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA="${DATA:-data/carp_synthetic_min.jsonl}"
TOKENIZER="${TOKENIZER:-artifacts/tokenizer}"
CONFIG="${CONFIG:-configs/carp_micro.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/carp_sft_min_ckpt}"

EXAMPLES="${EXAMPLES:-1000}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
REASONING_TOKENS="${REASONING_TOKENS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
ROUTER_LOSS_WEIGHT="${ROUTER_LOSS_WEIGHT:-0.2}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
CPU_THREADS="${CPU_THREADS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

echo "CARP minimal Vast run"
echo "  code_root=$CODE_ROOT"
echo "  data=$DATA examples=$EXAMPLES"
echo "  tokenizer=$TOKENIZER"
echo "  config=$CONFIG"
echo "  output=$OUTPUT_DIR"
echo "  device=$DEVICE amp=$AMP_DTYPE batch_size=$BATCH_SIZE max_steps=$MAX_STEPS"
echo

if [[ -f scripts/make_carp_synthetic.py ]]; then
  python -u scripts/make_carp_synthetic.py \
    --output "$DATA" \
    --examples "$EXAMPLES"
else
  echo "Missing scripts/make_carp_synthetic.py; generating synthetic CARP data inline."
  DATA="$DATA" EXAMPLES="$EXAMPLES" python - <<'PY'
import json
import os
import random
from pathlib import Path

out = Path(os.environ["DATA"])
examples = int(os.environ["EXAMPLES"])
rng = random.Random(1234)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    for idx in range(examples):
        kind = idx % 4
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        c = rng.randint(1, 20)
        if kind == 0:
            row = {
                "question": f"What is {a} + {b}?",
                "answer": str(a + b),
                "reasoning_tokens": ["<R0>"],
                "difficulty": "medium",
            }
        elif kind == 1:
            row = {
                "question": f"What is {a} - {b}?",
                "answer": str(a - b),
                "reasoning_tokens": ["<R1>"],
                "difficulty": "medium",
            }
        elif kind == 2:
            row = {
                "question": f"If x + {a} = {b}, what is x?",
                "answer": str(b - a),
                "reasoning_tokens": ["<R0>", "<R2>"],
                "difficulty": "hard",
            }
        else:
            row = {
                "question": f"Compute ({a} + {b}) * {c}.",
                "answer": str((a + b) * c),
                "reasoning_tokens": ["<R0>", "<R3>"],
                "difficulty": "hard",
            }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Wrote {examples} synthetic CARP traces to {out}", flush=True)
PY
fi

if [[ ! -e "$TOKENIZER/tokenizer.json" && ! -e "$TOKENIZER" ]]; then
  echo "Missing tokenizer: $TOKENIZER"
  echo "Training a minimal tokenizer from $DATA ..."
  python -u scripts/train_tokenizer.py \
    --data "$DATA" \
    --output-dir "$TOKENIZER" \
    --vocab-size "$VOCAB_SIZE" \
    --max-docs 0
fi

python -u scripts/train_carp_sft.py \
  --data "$DATA" \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT_DIR" \
  --reasoning-tokens "$REASONING_TOKENS" \
  --block-size "$BLOCK_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --router-loss-weight "$ROUTER_LOSS_WEIGHT" \
  --device "$DEVICE" \
  --amp-dtype "$AMP_DTYPE" \
  --cpu-threads "$CPU_THREADS"

python -u scripts/eval_carp_router.py \
  --model "$OUTPUT_DIR" \
  --data "$DATA" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --block-size "$BLOCK_SIZE" \
  --device "$DEVICE"

echo
echo "Done. Check router_accuracy above. A useful first target is >= 0.90 on this synthetic set."
