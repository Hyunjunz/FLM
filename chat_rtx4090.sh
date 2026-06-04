#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"

if [[ ! -d "$CODE_ROOT/cpu_lite_lm" ]]; then
  for candidate in "$SCRIPT_DIR/cpu_llm_lab" "$SCRIPT_DIR/CPU_LLM_LAB" "$SCRIPT_DIR/flm" "$PWD"; do
    if [[ -d "$candidate/cpu_lite_lm" ]]; then
      CODE_ROOT="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ ! -d "$CODE_ROOT/cpu_lite_lm" ]]; then
  echo "Cannot find cpu_lite_lm package." >&2
  echo "Run this from the repository root, or place it next to the cpu_lite_lm/ directory." >&2
  exit 1
fi

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DEFAULT_MODEL="rtx4090_max_ckpt"
DEFAULT_TOKENIZER="tokenizer_rtx4090_32k"
if [[ ! -e "$DEFAULT_MODEL/pytorch_model.bin" && -e "artifacts/rtx4090_max_ckpt/pytorch_model.bin" ]]; then
  DEFAULT_MODEL="artifacts/rtx4090_max_ckpt"
fi
if [[ ! -e "$DEFAULT_TOKENIZER/tokenizer.json" && -e "artifacts/tokenizer_rtx4090_32k/tokenizer.json" ]]; then
  DEFAULT_TOKENIZER="artifacts/tokenizer_rtx4090_32k"
fi

MODEL="${MODEL:-$DEFAULT_MODEL}"
TOKENIZER="${TOKENIZER:-$DEFAULT_TOKENIZER}"
CONFIG="${CONFIG:-configs/colab_medium.json}"
DEVICE="${DEVICE:-cuda}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-50}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-당신은 친절하고 정확한 AI 조수입니다. 한국어로 답변하세요.}"

if [[ ! -e "$MODEL/pytorch_model.bin" ]]; then
  echo "Missing model checkpoint: $MODEL/pytorch_model.bin" >&2
  echo "Train first, or set MODEL=/path/to/checkpoint." >&2
  exit 1
fi

if [[ ! -e "$TOKENIZER/tokenizer.json" && ! -e "$TOKENIZER" ]]; then
  echo "Missing tokenizer: $TOKENIZER" >&2
  echo "Set TOKENIZER=/path/to/tokenizer_rtx4090_32k or checkpoint tokenizer." >&2
  exit 1
fi

echo "RTX 4090 chat"
echo "  model=$MODEL"
echo "  tokenizer=$TOKENIZER"
echo "  device=$DEVICE amp=$AMP_DTYPE"
echo "Type exit or quit to stop."
echo

while true; do
  read -r -p "You> " USER_INPUT || break
  case "${USER_INPUT,,}" in
    exit|quit|q)
      break
      ;;
  esac

  if [[ -z "${USER_INPUT// }" ]]; then
    continue
  fi

  echo -n "AI> "
  python -u -m cpu_lite_lm.generate \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --amp-dtype "$AMP_DTYPE" \
    --system "$SYSTEM_PROMPT" \
    --user "$USER_INPUT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K"
  echo
done

echo "bye"
