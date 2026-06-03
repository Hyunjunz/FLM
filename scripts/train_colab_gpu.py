"""Colab GPU training preset for CPULiteLM.

This script assumes the HAERAE Korean webtext cache is already available under
`data/hf_cache/HAERAE-HUB___korean-webtext` or that you pass `--data`.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.train import build_parser, train


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        data="data/hf_cache/HAERAE-HUB___korean-webtext",
        tokenizer="artifacts/tokenizer_colab_16k",
        output_dir="artifacts/colab_gpu_ckpt",
        config="configs/colab_small.json",
        vocab_size=16000,
        block_size=512,
        batch_size=12,
        grad_accum_steps=4,
        max_steps=50000,
        learning_rate=3e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        text_column="text",
        max_docs=None,
        min_chars=100,
        max_chars=0,
        tokenizer_max_docs=200000,
        streaming=True,
        shuffle_buffer=8192,
        seed=1234,
        log_every=20,
        save_every=5000,
        device="auto",
        amp_dtype="fp16",
        tf32=True,
        num_workers=0,
    )
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Missing dataset cache: {args.data}. Upload/copy it first, or pass --data."
        )
    train(args)


if __name__ == "__main__":
    main()

