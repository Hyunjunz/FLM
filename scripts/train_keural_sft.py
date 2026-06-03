from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.train_sft import build_parser, train_sft


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(
        download_if_missing=True,
        train_tokenizer_if_missing=True,
        tokenizer="artifacts/tokenizer_keural_32k",
        base_model="none",
    )
    train_sft(parser.parse_args())
