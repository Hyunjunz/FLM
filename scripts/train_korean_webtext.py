"""Train CPULiteLM on the full local HAERAE Korean webtext HF cache.

Default behavior streams every Arrow shard for one full epoch. This can take a
long time on CPU. For a short smoke run, pass `--max-steps 10`.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.train import build_parser, train


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        data="data/hf_cache/HAERAE-HUB___korean-webtext",
        tokenizer="artifacts/tokenizer_korean_webtext",
        output_dir="artifacts/korean_webtext_ckpt",
        config="configs/micro.json",
        vocab_size=1024,
        block_size=128,
        batch_size=2,
        max_steps=0,
        learning_rate=1e-3,
        text_column="text",
        max_docs=None,
        min_chars=0,
        max_chars=0,
        tokenizer_max_docs=None,
        streaming=True,
        shuffle_buffer=2048,
        seed=1234,
        log_every=10,
        save_every=500,
    )
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Expected local HF cache at {args.data}. "
            "Put the HAERAE-HUB korean-webtext cache there or pass --data."
        )
    train(args)


if __name__ == "__main__":
    main()
