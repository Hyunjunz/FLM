# CARP Implementation Status

This repository now has a runnable first CARP layer on top of CPULiteLM.

## Implemented

- `cpu_lite_lm.carp`
  - `HeuristicDifficultyRouter`
  - `ReasoningCompressor`
  - `<R0>` style reasoning token helpers
  - `TinyVerifier` based on candidate mean log probability plus simple penalties
  - `CARPGenerator` end-to-end adaptive generation pipeline
- `scripts/add_carp_tokens.py`
  - Adds `<R0>` through `<R127>` or another count to an existing tokenizer.
- `scripts/carp_generate.py`
  - Runs adaptive CARP generation from an existing CPULiteLM checkpoint.
- `cpu_lite_lm.carp_data`
  - Parses teacher traces with `question`, `answer`, `reasoning_tokens`, and
    `difficulty`.
  - Builds CARP SFT text in the same `### Question` / `### Answer` style as the
    current SFT pipeline.
  - Produces router labels for later supervised router-head training.
- `scripts/prepare_carp_sft.py`
  - Converts JSONL teacher traces into preformatted SFT JSONL.
- `scripts/make_carp_synthetic.py`
  - Creates deterministic arithmetic CARP traces for smoke-proof experiments.
- `scripts/train_carp_sft.py`
  - Trains LM loss plus router-head loss from CARP JSONL traces.
- `scripts/eval_carp_router.py`
  - Reports router-head accuracy and a 4x4 confusion matrix.
- `scripts/download_carp_language.py`
  - Downloads language reasoning datasets and converts them to CARP traces.
  - Supports `tau/commonsense_qa` and `google/boolq`.
- `scripts/train_carp_language_vast.sh`
  - Vast/Linux CUDA one-command run for real language reasoning data.
- `configs/carp_cpu_large.json`
  - 12-layer, hidden-768 CPU-target large CARP config.
- `scripts/train_carp_cpu_large_vast.sh`
  - Vast training preset for the larger CPU-target checkpoint.
- `scripts/carp_language_infer.py`
  - Runs a single CommonsenseQA-style inference prompt.
- `scripts/eval_carp_language_answer.py`
  - Measures generated answer exact-match accuracy on CommonsenseQA or BoolQ.
- `cpu_lite_lm.carp_train`
  - Adds `carp_sft_loss()` for combined LM/router training.
- `CPULiteConfig`
  - Added CARP metadata fields:
    - `carp_num_reasoning_tokens`
    - `carp_router_labels`
    - `carp_verifier_labels`
- `CPULiteForCausalLM`
  - Optional router and verifier heads are created when `carp_router_labels` or
    `carp_verifier_labels` are greater than zero.
  - `carp_heads(input_ids)` returns pooled router/verifier logits for future
    supervised training.
  - `resize_token_embeddings(new_size)` can expand a checkpoint after tokenizer
    growth while preserving existing token rows.
- Tests
  - Router classification
  - Reasoning token creation
  - Minimal CARP generation smoke test

## How To Try

Add CARP tokens to a tokenizer:

```bash
python scripts/add_carp_tokens.py --tokenizer artifacts/tokenizer --output-dir artifacts/tokenizer_carp --count 128
```

Run CARP generation:

```bash
python scripts/carp_generate.py \
  --model artifacts/micro_ckpt \
  --config configs/micro.json \
  --tokenizer artifacts/tokenizer_carp \
  --user "Analyze why this code can deadlock." \
  --max-new-tokens 120 \
  --show-carp
```

Prepare CARP SFT JSONL:

```bash
python scripts/prepare_carp_sft.py \
  --input data/carp_traces.jsonl \
  --output data/carp_sft.jsonl \
  --max-reasoning-tokens 128
```

Run a small proof-style synthetic experiment:

```bash
python scripts/make_carp_synthetic.py --output data/carp_synthetic.jsonl --examples 1000
python scripts/train_carp_sft.py \
  --data data/carp_synthetic.jsonl \
  --config configs/carp_micro.json \
  --tokenizer artifacts/tokenizer \
  --output-dir artifacts/carp_sft_ckpt \
  --max-steps 200
python scripts/eval_carp_router.py \
  --model artifacts/carp_sft_ckpt \
  --data data/carp_synthetic.jsonl
```

Run a real language-reasoning dataset experiment on Vast:

```bash
bash scripts/train_carp_language_vast.sh
```

The default dataset is `tau/commonsense_qa`. Override it with:

```bash
DATASET=google/boolq MAX_EXAMPLES=5000 bash scripts/train_carp_language_vast.sh
```

Train the larger CPU-target checkpoint:

```bash
bash scripts/train_carp_cpu_large_vast.sh
```

Run one trained inference:

```bash
python scripts/carp_language_infer.py \
  --model artifacts/carp_language_ckpt \
  --question "Where would you keep a pillow when you sleep?" \
  --choices "A. garage\nB. bed\nC. oven\nD. road\nE. shower" \
  --show-carp
```

Evaluate generated answers:

```bash
python scripts/eval_carp_language_answer.py \
  --model artifacts/carp_language_ckpt \
  --dataset tau/commonsense_qa \
  --split validation \
  --max-examples 200
```

If the model checkpoint was trained before adding reasoning tokens, expand the
embedding matrix with `resize_token_embeddings()` before saving a CARP checkpoint,
or pass `--reasoning-tokens 0` to use routing, candidate scoring, and speculative
decoding without internal CARP tokens.

## What This Is

This is a practical PoC runtime implementation. It does not claim that the
current tiny checkpoint has learned compressed reasoning. The runtime path is in
place so training and evaluation can be added incrementally.

## Remaining Work

- Train a tokenizer with CARP tokens from the start.
- Build `question -> reasoning_tokens -> answer` SFT data from teacher traces.
- Improve router supervision beyond difficulty classification, including
  reasoning budget and verifier-required prediction.
- Add a trainable verifier head and pairwise ranking loss.
- Distill a separate draft model or improve the current self-speculative mode.
- Track speculative accept rate and mode-level latency.
- Add benchmark scripts for easy/medium/hard/critical routing.
- Add GGUF metadata mapping for CARP token ranges.
- Evaluate on GSM8K, HumanEval/MBPP, Korean instruction data, and local latency
  benchmarks.

## Practical Next Step

The best next engineering step is to make a CARP tokenizer and train a small
checkpoint with `carp_num_reasoning_tokens=128` in the config. After that, use
the existing `CARPGenerator` to measure routing distribution, candidate quality,
and latency before implementing learned router/verifier heads.
