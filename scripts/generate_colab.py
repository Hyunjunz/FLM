from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.generate import build_parser, generate


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        model="artifacts/colab_gpu_ckpt",
        tokenizer="artifacts/tokenizer_colab_16k",
        config="configs/colab_small.json",
        device="auto",
        amp_dtype="fp16",
        max_new_tokens=120,
        temperature=0.8,
        top_k=50,
    )
    generate(parser.parse_args())


if __name__ == "__main__":
    main()

