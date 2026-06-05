from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.helix_data import prepare_helix_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/helix_train.jsonl")
    parser.add_argument(
        "--preset",
        choices=["big_reasoning", "balanced_reasoning", "reasoning_mix", "small_reasoning", "synthetic"],
        default="big_reasoning",
    )
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--synthetic-examples", type=int, default=1000)
    parser.add_argument("--no-balance-data", action="store_true")
    parser.add_argument("--no-skip-download-errors", action="store_true")
    parser.add_argument("--max-per-difficulty", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    prepare_helix_dataset(
        args.output,
        preset=args.preset,
        max_examples_per_dataset=args.max_examples,
        cache_dir=args.cache_dir,
        synthetic_examples=args.synthetic_examples,
        seed=args.seed,
        balance=not args.no_balance_data,
        max_per_difficulty=args.max_per_difficulty,
        skip_errors=not args.no_skip_download_errors,
    )


if __name__ == "__main__":
    main()
