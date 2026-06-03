"""L4 Colab preset with faster startup and frequent logs.

Use this when the larger Colab preset spends too long in tokenizer training
without visible progress. It trains a 16K tokenizer from a bounded subset, then
uses the full streaming dataset for model training.
"""

from pathlib import Path
import sys

print("Booting train_colab_l4_fast.py ...", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.train import build_parser, train


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        data="data/hf_cache/HAERAE-HUB___korean-webtext",
        tokenizer="artifacts/tokenizer_l4_16k",
        output_dir="artifacts/l4_fast_ckpt",
        config="configs/colab_small.json",
        vocab_size=16000,
        block_size=768,
        batch_size=10,
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
        tokenizer_max_docs=50000,
        tokenizer_log_every=500,
        streaming=True,
        shuffle_buffer=8192,
        seed=1234,
        log_every=10,
        save_every=2500,
        device="auto",
        amp_dtype="fp16",
        tf32=True,
        num_workers=0,
    )
    args = parser.parse_args()
    print(f"Args parsed. Data path: {args.data}", flush=True)
    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Missing dataset cache: {args.data}. Upload/copy it first, or pass --data."
        )
    train(args)


if __name__ == "__main__":
    main()

