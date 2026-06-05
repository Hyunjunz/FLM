from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.helix_data import prepare_helix_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/helix_train.jsonl")
    parser.add_argument("--preset", choices=["reasoning_mix", "small_reasoning", "synthetic"], default="reasoning_mix")
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--synthetic-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    prepare_helix_dataset(
        args.output,
        preset=args.preset,
        max_examples_per_dataset=args.max_examples,
        cache_dir=args.cache_dir,
        synthetic_examples=args.synthetic_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
