"""L4 Colab quality preset.

Compared with the fast preset, this enables:
- quality filtering for noisy webtext
- validation loss logging
- larger microbatch with lower accumulation when possible
- frequent checkpointing
"""

from pathlib import Path
import sys

print("Booting train_colab_l4_quality.py ...", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.train import build_parser, train


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        data="data/hf_cache/HAERAE-HUB___korean-webtext",
        tokenizer="artifacts/tokenizer_colab_32k",
        output_dir="artifacts/l4_quality_ckpt",
        config="configs/colab_medium.json",
        vocab_size=32000,
        block_size=768,
        batch_size=16,
        grad_accum_steps=2,
        max_steps=100000,
        learning_rate=2e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        text_column="text",
        max_docs=None,
        min_chars=100,
        skip_docs=512,
        quality_filter=True,
        max_chars=0,
        tokenizer_max_docs=200000,
        tokenizer_log_every=1000,
        streaming=True,
        shuffle_buffer=16384,
        seed=1234,
        log_every=20,
        save_every=5000,
        device="auto",
        amp_dtype="fp16",
        tf32=True,
        num_workers=0,
        eval_every=1000,
        eval_docs=512,
        eval_skip_docs=0,
        eval_max_chars=1_000_000,
        eval_max_batches=40,
    )
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Missing dataset cache: {args.data}. Upload/copy it first, or pass --data."
        )
    train(args)


if __name__ == "__main__":
    main()

