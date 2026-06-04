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
    candidates = [f"{label}. {text}" for label, text in zip(labels, texts)]
    return {
        "question": question,
        "answer": answer,
        "candidates": candidates,
        "gold_label": answer_key,
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
        "candidates": ["yes", "no"],
        "gold_label": answer,
        "reasoning_tokens": ["<R48>", "<R49>"],
        "difficulty": "hard",
        "verifier_required": True,
        "candidate_count": 3,
        "source": "google/boolq",
    }


def hellaswag_to_trace(row: Dict[str, Any]) -> Dict[str, Any]:
    endings = list(row["endings"])
    labels = ["A", "B", "C", "D"]
    gold_idx = int(row["label"])
    gold_label = labels[gold_idx]
    candidates = [f"{label}. {ending}" for label, ending in zip(labels, endings)]
    context = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
    question = (
        "Choose the most plausible continuation.\n\n"
        f"Context: {context}\n\n"
        "Choices:\n" + "\n".join(candidates)
    )
    return {
        "question": question,
        "answer": candidates[gold_idx],
        "candidates": candidates,
        "gold_label": gold_label,
        "reasoning_tokens": ["<R32>", "<R40>"],
        "difficulty": "hard",
        "verifier_required": True,
        "candidate_count": 4,
        "source": "Rowan/hellaswag",
    }


def arc_to_trace(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    candidates = [f"{label}. {text}" for label, text in zip(labels, texts)]
    answer_key = str(row["answerKey"]).strip()
    answer = next((candidate for candidate in candidates if candidate.startswith(f"{answer_key}.")), "")
    question = (
        "Choose the best answer.\n\n"
        f"Question: {row['question']}\n\n"
        "Choices:\n" + "\n".join(candidates)
    )
    return {
        "question": question,
        "answer": answer or answer_key,
        "candidates": candidates,
        "gold_label": answer_key,
        "reasoning_tokens": ["<R32>", "<R41>"],
        "difficulty": "hard",
        "verifier_required": True,
        "candidate_count": len(candidates),
        "source": source,
    }


def openbookqa_to_trace(row: Dict[str, Any]) -> Dict[str, Any]:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    candidates = [f"{label}. {text}" for label, text in zip(labels, texts)]
    answer_key = str(row["answerKey"]).strip()
    answer = next((candidate for candidate in candidates if candidate.startswith(f"{answer_key}.")), "")
    question_text = row.get("question_stem", row.get("question", ""))
    fact = str(row.get("fact1", "") or "").strip()
    question = (
        "Choose the best answer.\n\n"
        + (f"Fact: {fact}\n\n" if fact else "")
        + f"Question: {question_text}\n\n"
        + "Choices:\n"
        + "\n".join(candidates)
    )
    return {
        "question": question,
        "answer": answer or answer_key,
        "candidates": candidates,
        "gold_label": answer_key,
        "reasoning_tokens": ["<R32>", "<R42>"],
        "difficulty": "hard",
        "verifier_required": True,
        "candidate_count": len(candidates),
        "source": "allenai/openbookqa",
    }


def iter_dataset(name: str, split: str, max_examples: int | None) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install datasets to download CARP language data: pip install datasets") from exc

    if "::" in name:
        dataset_name, config_name = name.split("::", 1)
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(name, split=split)
    limit = len(ds) if max_examples is None or max_examples <= 0 else min(max_examples, len(ds))
    for idx in range(limit):
        yield dict(ds[idx])


def convert_one(name: str, split: str, max_examples: int | None) -> List[Dict[str, Any]]:
    if name in {"tau/commonsense_qa", "commonsense_qa"}:
        dataset_name = "tau/commonsense_qa"
        formatter = commonsense_qa_to_trace
    elif name in {"google/boolq", "boolq"}:
        dataset_name = "google/boolq"
        formatter = boolq_to_trace
    elif name in {"Rowan/hellaswag", "hellaswag"}:
        dataset_name = "Rowan/hellaswag"
        formatter = hellaswag_to_trace
    elif name in {"allenai/ai2_arc:ARC-Challenge", "ai2_arc_challenge", "arc_challenge"}:
        dataset_name = "allenai/ai2_arc::ARC-Challenge"
        formatter = lambda row: arc_to_trace(row, "allenai/ai2_arc/ARC-Challenge")
    elif name in {"allenai/ai2_arc:ARC-Easy", "ai2_arc_easy", "arc_easy"}:
        dataset_name = "allenai/ai2_arc::ARC-Easy"
        formatter = lambda row: arc_to_trace(row, "allenai/ai2_arc/ARC-Easy")
    elif name in {"allenai/openbookqa", "openbookqa"}:
        dataset_name = "allenai/openbookqa"
        formatter = openbookqa_to_trace
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    rows = []
    for row in iter_dataset(dataset_name, split, max_examples):
        trace = formatter(row)
        if trace["question"] and trace["answer"]:
            rows.append(trace)
    print(f"Loaded {len(rows)} traces from {dataset_name}/{split}", flush=True)
    return rows


def convert(name: str, split: str, output: str | Path, max_examples: int | None) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if name in {"mix_language", "mixed_language", "mix"}:
        specs = [
            ("tau/commonsense_qa", "train", max_examples),
            ("google/boolq", "train", max_examples),
            ("Rowan/hellaswag", "train", max_examples),
            ("allenai/ai2_arc:ARC-Challenge", "train", max_examples),
            ("allenai/ai2_arc:ARC-Easy", "train", max_examples),
            ("allenai/openbookqa", "train", max_examples),
        ]
        traces = []
        for dataset_name, dataset_split, limit in specs:
            traces.extend(convert_one(dataset_name, dataset_split, limit))
    else:
        traces = convert_one(name, split, max_examples)
    with output.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    print(f"Wrote {len(traces)} CARP language traces to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mix_language")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/carp_commonsenseqa_train.jsonl")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()
    convert(args.dataset, args.split, args.output, args.max_examples)


if __name__ == "__main__":
    main()
