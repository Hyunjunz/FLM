from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from cpu_lite_lm.train_sft import load_jsonl_shards


DATASET_NAME = "mkd-chanwoo/keural-SFT"
CACHE_DIR = "./hf_cache"
SAVE_DIR = "./datasets/keural-SFT"


def main() -> None:
    print(f"Loading {DATASET_NAME} ...", flush=True)
    try:
        ds = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)
    except Exception as exc:
        print(
            "Direct load_dataset failed; loading shard_*.jsonl only. "
            f"Original error: {exc}",
            flush=True,
        )
        ds = load_jsonl_shards(DATASET_NAME, CACHE_DIR, "train")
    print(ds[0], flush=True)
    Path(SAVE_DIR).parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(SAVE_DIR)
    print(f"saved to {SAVE_DIR}", flush=True)


if __name__ == "__main__":
    main()
