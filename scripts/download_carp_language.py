"""Download language reasoning datasets and convert them to CARP traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def commonsense_qa_to_trace(row: Dict[str, Any]) -> Dict[str, Any]:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    answer_key = str(row.get("answerKey", "")).strip()
    answer_text = ""
    for label, text in zip(labels, texts):
        if str(label) == answer_key:
            answer_text = str(text)
            break
    choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    question = (
        "Choose the best commonsense answer.\n\n"
        f"Question: {row['question']}\n\n"
        f"Choices:\n{choices}"
    )
    answer = f"{answer_key}. {answer_text}".strip()
    return {
        "question": question,
        "answer": answer,
        "reasoning_tokens": ["<R32>", "<R33>"],
        "difficulty": "hard",
        "verifier_required": True,
        "candidate_count": 3,
        "source": "tau/commonsense_qa",
    }


def boolq_to_trace(row: Dict[str, Any]) -> Dict[str, Any]:
    question = (
        "Answer the yes/no question using only the passage.\n\n"
        f"Passage: {row['passage']}\n\n"
        f"Question: {row['question']}"
    )
    answer = "yes" if bool(row["answer"]) else "no"
    return {
        "question": question,
        "answer": answer,
        "reasoning_tokens": ["<R48>", "<R49>"],
        "difficulty": "hard",
        "verifier_required": True,
        "candidate_count": 3,
        "source": "google/boolq",
    }


def iter_dataset(name: str, split: str, max_examples: int | None) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install datasets to download CARP language data: pip install datasets") from exc

    ds = load_dataset(name, split=split)
    limit = len(ds) if max_examples is None or max_examples <= 0 else min(max_examples, len(ds))
    for idx in range(limit):
        yield dict(ds[idx])


def convert(name: str, split: str, output: str | Path, max_examples: int | None) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if name in {"tau/commonsense_qa", "commonsense_qa"}:
        dataset_name = "tau/commonsense_qa"
        formatter = commonsense_qa_to_trace
    elif name in {"google/boolq", "boolq"}:
        dataset_name = "google/boolq"
        formatter = boolq_to_trace
    else:
        raise ValueError(
            "Unsupported dataset. Use tau/commonsense_qa or google/boolq."
        )

    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_dataset(dataset_name, split, max_examples):
            trace = formatter(row)
            if trace["question"] and trace["answer"]:
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                written += 1
    print(f"Wrote {written} CARP language traces from {dataset_name}/{split} to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tau/commonsense_qa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/carp_commonsenseqa_train.jsonl")
    parser.add_argument("--max-examples", type=int, default=5000)
    args = parser.parse_args()
    convert(args.dataset, args.split, args.output, args.max_examples)


if __name__ == "__main__":
    main()
