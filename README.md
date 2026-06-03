# cpu_llm_lab

`cpu_llm_lab` is a minimal, runnable CPU-only language model project. The model,
`CPULiteLM`, is a small decoder-only Transformer with RMSNorm, RoPE, grouped
query attention, SwiGLU, tied token embeddings, causal masking, and KV-cache
generation.

The goal is not model quality. The goal is an end-to-end baseline that can train
for a few steps, generate text, benchmark CPU inference, and test dynamic int8
quantization without downloading a large model.

## Install

```bash
pip install -r requirements.txt
```

`transformers` is optional at runtime. If it is unavailable, the project uses a
small local fallback for config/model save and load.

## Quick Start

Run from this directory:

```bash
python scripts/train_micro.py
python scripts/generate_micro.py --prompt "안녕하세요, 저는" --max_new_tokens 20
python scripts/benchmark_cpu.py
python scripts/quantize_dynamic.py
pytest -q
```

## Tokenizer Training

The tokenizer uses Hugging Face `tokenizers` with byte-level BPE. Byte-level
pre-tokenization gives robust handling for Korean, English, code, whitespace,
and rare characters.

```bash
python scripts/train_tokenizer.py --data data/sample_corpus.txt --output-dir artifacts/tokenizer --vocab-size 1024
```

## Micro Training

```bash
python scripts/train_micro.py --max-steps 5
```

The script creates `artifacts/tokenizer` if missing, trains the micro model on
`data/hf_cache/HAERAE-HUB___korean-webtext` when that local HF cache exists.
If the cache is missing, it falls back to `data/sample_corpus.txt`. It saves
`artifacts/micro_ckpt`.

## Colab GPU Training

On Colab, enable `Runtime > Change runtime type > GPU`, then run:

```bash
cd /content
# Upload or clone this repo, then enter it.
cd cpu_llm_lab
pip install -r requirements.txt
```

If your HF cache is copied into `data/hf_cache/HAERAE-HUB___korean-webtext`,
start the GPU preset:

```bash
python scripts/train_colab_gpu.py
```

The preset uses:

- `configs/colab_small.json`
- 16K tokenizer
- `block_size=512`
- `batch_size=12`
- `grad_accum_steps=4`
- FP16 autocast
- TF32 matmul on supported NVIDIA GPUs
- streaming Arrow dataset
- shuffle buffer
- save every 5000 optimizer steps

For a quick Colab sanity check:

```bash
python scripts/train_colab_gpu.py --max-steps 100 --save-every 50
```

For better quality, run longer:

```bash
python scripts/train_colab_gpu.py --max-steps 50000 --save-every 5000
```

If Colab shows no output for a long time, use unbuffered Python and the L4 fast
preset:

```bash
python -u scripts/train_colab_l4_fast.py
```

This prints immediately, trains the tokenizer from 50K documents, then streams
the full dataset for model training. To force visible tokenizer progress:

```bash
python -u scripts/train_colab_l4_fast.py --tokenizer-log-every 100
```

For a better L4 run with validation loss and noisy-webtext filtering:

```bash
python -u scripts/train_colab_l4_quality.py
```

This preset uses `configs/colab_medium.json`, SDPA attention when available,
quality filtering, a held-out validation prefix, and logs validation perplexity
every 1000 optimizer steps. If it OOMs, reduce `--batch-size 16` to `8` and
increase `--grad-accum-steps 2` to `4`.

Useful diagnostics:

```bash
python scripts/inspect_korean_webtext.py --quality-filter --max-docs 1000
```

If you have a stronger GPU and enough time, try the larger config:

```bash
python scripts/train_colab_gpu.py \
  --config configs/colab_medium.json \
  --tokenizer artifacts/tokenizer_colab_32k \
  --output-dir artifacts/colab_medium_ckpt \
  --vocab-size 32000 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --max-steps 100000
```

Generate from the Colab checkpoint:

```bash
python scripts/generate_colab.py --prompt "대한민국의 수도는" --max_new_tokens 120
```

## Keural SFT

To download `mkd-chanwoo/keural-SFT` and immediately run supervised fine-tuning:

```bash
python -u scripts/train_keural_sft.py \
  --config configs/colab_medium.json \
  --output-dir artifacts/keural_sft_ckpt \
  --max-steps 3000
```

`scripts/train_keural_sft.py` downloads automatically when
`datasets/keural-SFT` is missing. If the tokenizer is missing, it trains one
from the SFT `text` column. If `--base-model` is missing or `none`, it
initializes from config. The dataset cache goes to `./hf_cache`.

To SFT an existing pretraining checkpoint, pass it explicitly:

```bash
python -u scripts/train_keural_sft.py \
  --base-model artifacts/l4_quality_ckpt \
  --tokenizer artifacts/l4_quality_ckpt \
  --output-dir artifacts/keural_sft_ckpt \
  --max-steps 3000
```

Download only:

```bash
python scripts/download_keural_sft.py
```

Generate from the SFT checkpoint:

```bash
python scripts/generate_colab.py \
  --model artifacts/keural_sft_ckpt \
  --tokenizer artifacts/keural_sft_ckpt \
  --config configs/colab_medium.json \
  --prompt "### 질문:\n대한민국의 수도는?\n\n### 답변:\n" \
  --max_new_tokens 80
```

The HF cache reader streams the Arrow shards and uses the `text` column by
default. To keep CPU smoke runs fast, only a bounded prefix is tokenized:

```bash
python scripts/train_micro.py --data data/hf_cache/HAERAE-HUB___korean-webtext --text-column text --max-docs 2000 --max-chars 200000
```

For a full streaming pass over the local HAERAE Korean webtext cache:

```bash
python scripts/train_korean_webtext.py
```

This uses all Arrow shards with `--streaming`, `--max-docs None`,
`--max-chars 0`, and a chunk shuffle buffer. It is intended for long CPU runs.
For a quick check:

```bash
python scripts/train_korean_webtext.py --max-steps 10
```

## Text Generation

```bash
python scripts/generate_micro.py --prompt "안녕하세요, 저는" --max_new_tokens 20
```

Generation supports temperature, top-k sampling, and KV cache:

```bash
python scripts/generate_micro.py --temperature 0.8 --top-k 20
python scripts/generate_micro.py --no-cache
```

## CPU Benchmark

```bash
python scripts/benchmark_cpu.py --threads 4 --prompt-tokens 64 --generated-tokens 64
```

The benchmark reports prompt prefill tok/s, decode tok/s, RSS memory, thread
count, and cache mode.

## Dynamic Quantization

```bash
python scripts/quantize_dynamic.py
```

This applies PyTorch dynamic int8 quantization to `nn.Linear` layers and compares
serialized state sizes. Dynamic quantization support depends on the local PyTorch
CPU backend.

## Architecture

- Decoder-only Transformer
- Pre-RMSNorm residual blocks
- RoPE on query/key
- Grouped Query Attention / Multi Query Attention via `num_key_value_heads`
- SwiGLU MLP
- Tied token embeddings
- Causal LM loss
- KV cache for autoregressive decoding

Configs:

- `configs/micro.json`: fast tests and smoke training
- `configs/tiny.json`: larger CPU experiment template

## Why CPU-Friendly

- GQA/MQA reduces KV cache size compared with full MHA.
- Small hidden sizes keep batch-1 GEMV costs low.
- Bias-free linear layers and RMSNorm keep kernels simple.
- KV cache generation avoids recomputing old keys and values.
- Dynamic int8 quantization can shrink linear weights for CPU experiments.
- The implementation avoids CUDA, Triton, FlashAttention, and custom extensions.

## Current Limits

- This is still a Python/PyTorch implementation, but it now uses PyTorch SDPA
  attention when available.
- The toy corpus is too small for meaningful language quality.
- Korean webtext is noisy; use `--quality-filter` for long training runs.
- Train loss alone is misleading; use `--eval-every` or the L4 quality preset.
- ONNX export is a skeleton and does not yet export cache-aware generation.
- GGUF conversion is documented as a future mapping task.

## Next Steps

- Add proper train/validation split and perplexity reporting.
- Add longer mixed Korean/English/code corpora.
- Add LoRA or QLoRA-style fine-tuning hooks.
- Add calibrated quantization comparisons.
- Add llama.cpp tensor-name mapping and GGUF metadata export.

## GGUF / llama.cpp Plan

The architecture intentionally resembles LLaMA-style blocks:

- `embed_tokens`
- per-layer `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`
- RMSNorm weights
- tied `lm_head`

To support GGUF, add a converter that writes config metadata, tokenizer metadata,
and maps PyTorch tensor names to llama.cpp-compatible names. The first practical
target should be a GGUF file that llama.cpp can inspect, followed by a custom
architecture entry if exact tensor names are not accepted by existing loaders.
