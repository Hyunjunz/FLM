from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.helix_data import convert_jsonl_to_helix


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_jsonl_to_helix(args.input, args.output)


if __name__ == "__main__":
    main()
