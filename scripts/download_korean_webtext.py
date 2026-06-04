"""Download HAERAE Korean webtext into the local Arrow dataset path."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="HAERAE-HUB/KOREAN-WEBTEXT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/hf_cache/HAERAE-HUB___korean-webtext")
    parser.add_argument("--cache-dir", default="./hf_cache")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"Dataset already exists: {output_dir}", flush=True)
        return

    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Downloading Korean webtext requires `pip install datasets`.") from exc

    print(
        f"Downloading {args.dataset_name} split={args.split} "
        f"cache_dir={args.cache_dir}",
        flush=True,
    )
    ds = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"Saved Korean webtext dataset to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
