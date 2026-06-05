"""Dataset builders for HelixMind router/verifier training."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


HelixRow = Dict[str, Any]


def normalize_helix_row(item: Dict[str, Any]) -> HelixRow | None:
    prompt = _extract_prompt(item)
    if not prompt:
        return None
    difficulty = item.get("difficulty", item.get("router_label"))
    if isinstance(difficulty, int):
        difficulty = ("easy", "medium", "hard")[max(0, min(2, difficulty))]
    elif isinstance(difficulty, str):
        lowered = difficulty.lower()
        if lowered in {"0", "easy"}:
            difficulty = "easy"
        elif lowered in {"1", "medium"}:
            difficulty = "medium"
        elif lowered in {"2", "hard", "critical"}:
            difficulty = "hard"
        else:
            difficulty = None

    verifier = item.get("verifier_label")
    if verifier is None:
        verifier = item.get("accepted", item.get("correct", item.get("is_correct", None)))
    accepted = True if verifier is None else bool(verifier)

    row: HelixRow = {
        "prompt": prompt,
        "accepted": accepted,
        "source": item.get("source", "converted"),
    }
    if difficulty is not None:
        row["difficulty"] = difficulty
    if "weight" in item:
        row["weight"] = float(item["weight"])
    return row


def _extract_prompt(item: Dict[str, Any]) -> str:
    for key in ("prompt", "question", "user", "input", "instruction"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            if key == "instruction" and isinstance(item.get("output"), str):
                return f"Instruction:\n{value.strip()}\n\nExpected answer:\n{item['output'].strip()}"
            return value.strip()

    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    messages = item.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "user"))
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(f"{role}: {content.strip()}")
        if parts:
            return "\n".join(parts)

    if isinstance(item.get("answer"), str):
        answer = item["answer"].strip()
        for key in ("context", "passage"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return f"{value.strip()}\n\nAnswer: {answer}"
    return ""


def write_jsonl(rows: Iterable[HelixRow], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with tmp_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp_output, output)
    print(f"Wrote {count} Helix rows to {output}", flush=True)
    return output


def iter_jsonl_objects(path: str | Path, skip_bad: bool = True):
    path = Path(path)
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if not skip_bad:
                    raise
                skipped += 1
                print(f"Skipped malformed JSONL record at {path}:{line_no}: {exc}", flush=True)
    if skipped:
        print(f"Skipped {skipped} malformed JSONL records from {path}", flush=True)


def summarize_helix_rows(rows: Iterable[HelixRow]) -> Dict[str, Dict[str, int]]:
    summary = {
        "difficulty": {"easy": 0, "medium": 0, "hard": 0, "missing": 0},
        "accepted": {"true": 0, "false": 0},
    }
    for row in rows:
        difficulty = str(row.get("difficulty", "missing")).lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "missing"
        summary["difficulty"][difficulty] += 1
        summary["accepted"]["true" if bool(row.get("accepted", True)) else "false"] += 1
    return summary


def print_helix_summary(rows: List[HelixRow], title: str = "Helix data") -> None:
    summary = summarize_helix_rows(rows)
    print(
        f"{title}: rows={len(rows)} "
        f"difficulty={summary['difficulty']} accepted={summary['accepted']}",
        flush=True,
    )


def balance_helix_rows(
    rows: List[HelixRow],
    max_per_difficulty: int = 12000,
    verifier_false_ratio: float = 0.5,
    seed: int = 1234,
) -> List[HelixRow]:
    rng = random.Random(seed)
    buckets: Dict[str, List[HelixRow]] = {"easy": [], "medium": [], "hard": []}
    missing: List[HelixRow] = []
    for row in rows:
        difficulty = str(row.get("difficulty", "")).lower()
        if difficulty in buckets:
            buckets[difficulty].append(row)
        else:
            missing.append(row)
    balanced: List[HelixRow] = []
    for difficulty, bucket in buckets.items():
        rng.shuffle(bucket)
        balanced.extend(bucket[:max_per_difficulty])
    balanced.extend(missing[: max(0, max_per_difficulty // 4)])

    true_rows = [row for row in balanced if bool(row.get("accepted", True))]
    false_rows = [row for row in balanced if not bool(row.get("accepted", True))]
    if true_rows and false_rows:
        target_false = int(len(true_rows) * verifier_false_ratio / max(1e-9, 1.0 - verifier_false_ratio))
        if len(false_rows) > target_false:
            rng.shuffle(false_rows)
            false_rows = false_rows[:target_false]
        elif len(false_rows) < target_false:
            false_rows = false_rows + _build_extra_negative_rows(true_rows, target_false - len(false_rows), rng)
        balanced = true_rows + false_rows
    rng.shuffle(balanced)
    return balanced


def _build_extra_negative_rows(rows: List[HelixRow], count: int, rng: random.Random) -> List[HelixRow]:
    bad_claims = [
        "The draft gives an unsupported exact fact. Mark whether it is correct.",
        "The draft contradicts the question. Mark whether it is correct.",
        "The draft skips the required condition and reaches the wrong conclusion.",
        "The draft uses a wrong arithmetic step. Mark whether it is correct.",
    ]
    out = []
    for _ in range(count):
        src = rng.choice(rows)
        out.append(
            {
                "prompt": f"{src['prompt']}\n\nDraft answer: {rng.choice(bad_claims)}\nIs the draft correct?",
                "difficulty": src.get("difficulty", "medium"),
                "accepted": False,
                "source": f"{src.get('source', 'converted')}_balanced_negative",
            }
        )
    return out


def convert_jsonl_to_helix(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    converted: List[HelixRow] = []
    skipped = 0
    for item in iter_jsonl_objects(input_path, skip_bad=True):
        row = normalize_helix_row(item)
        if row is None:
            skipped += 1
            continue
        converted.append(row)
    output = write_jsonl(converted, output_path)
    if skipped:
        print(f"Skipped {skipped} rows without usable text fields", flush=True)
    return output


def build_synthetic_helix_rows(examples: int = 2000, seed: int = 1234) -> List[HelixRow]:
    rng = random.Random(seed)
    rows: List[HelixRow] = []
    templates = [
        ("What is {a} + {b}?", "easy", True),
        ("Summarize this sentence in one short phrase: the user wants a fast answer.", "easy", True),
        ("Debug this Python function and explain the likely bug: def f(x): return x[1]", "medium", True),
        ("Compare these two implementation options and choose the safer CPU path.", "medium", True),
        ("Prove the loop invariant for n={a}, then find a counterexample if it fails.", "hard", True),
        ("Solve step by step: if x + {a} = {b}, what is x and why?", "hard", True),
    ]
    bad_drafts = [
        "Draft answer: 2 + 2 = 5. Decide if this draft is correct.",
        "Draft answer: The code is safe because it never uses memory. Decide if this draft is correct.",
        "Draft answer: I know the exact source without evidence. Decide if this draft is correct.",
    ]
    for idx in range(examples):
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        text, difficulty, accepted = templates[idx % len(templates)]
        rows.append(
            {
                "prompt": text.format(a=a, b=b),
                "difficulty": difficulty,
                "accepted": accepted,
                "source": "helix_synthetic",
            }
        )
        if idx % 5 == 0:
            rows.append(
                {
                    "prompt": bad_drafts[idx % len(bad_drafts)],
                    "difficulty": "easy",
                    "accepted": False,
                    "source": "helix_synthetic_negative",
                }
            )
    return rows


HF_DATASET_ALIASES = {
    "gsm8k": "openai/gsm8k",
    "piqa": "ybisk/piqa",
    "winogrande": "allenai/winogrande",
    "race": "ehovy/race",
    "sciq": "allenai/sciq",
}


def _load_hf_dataset(name: str, config: str | None, split: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Auto download requires `pip install datasets`.") from exc
    name = HF_DATASET_ALIASES.get(name, name)
    if config:
        return load_dataset(name, config, split=split, cache_dir=cache_dir)
    return load_dataset(name, split=split, cache_dir=cache_dir)


def _choice_prompt(question: str, choices: List[str], instruction: str = "Choose the best answer.") -> str:
    joined = "\n".join(choices)
    return f"{instruction}\n\nQuestion: {question}\n\nChoices:\n{joined}"


def _commonsense_qa(row: Dict[str, Any]) -> List[HelixRow]:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    choices = [f"{label}. {text}" for label, text in zip(labels, texts)]
    answer = str(row.get("answerKey", "")).strip()
    prompt = _choice_prompt(str(row["question"]), choices, "Choose the best commonsense answer.")
    rows = [{"prompt": prompt, "difficulty": "medium", "accepted": True, "source": "tau/commonsense_qa"}]
    wrong = next((choice for choice in choices if not choice.startswith(f"{answer}.")), "")
    if wrong:
        rows.append(
            {
                "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
                "difficulty": "medium",
                "accepted": False,
                "source": "tau/commonsense_qa_negative",
            }
        )
    return rows


def _boolq(row: Dict[str, Any]) -> List[HelixRow]:
    prompt = (
        "Answer the yes/no question using only the passage.\n\n"
        f"Passage: {row['passage']}\n\nQuestion: {row['question']}"
    )
    gold = "yes" if bool(row["answer"]) else "no"
    wrong = "no" if gold == "yes" else "yes"
    difficulty = "hard" if len(str(row["passage"]).split()) > 140 else "medium"
    return [
        {"prompt": prompt, "difficulty": difficulty, "accepted": True, "source": "google/boolq"},
        {
            "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
            "difficulty": difficulty,
            "accepted": False,
            "source": "google/boolq_negative",
        },
    ]


def _hellaswag(row: Dict[str, Any]) -> List[HelixRow]:
    endings = [str(value) for value in row["endings"]]
    labels = ["A", "B", "C", "D"]
    choices = [f"{label}. {ending}" for label, ending in zip(labels, endings)]
    gold_idx = int(row["label"])
    context = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
    prompt = _choice_prompt(context, choices, "Choose the most plausible continuation.")
    wrong = choices[(gold_idx + 1) % len(choices)]
    return [
        {"prompt": prompt, "difficulty": "medium", "accepted": True, "source": "Rowan/hellaswag"},
        {
            "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
            "difficulty": "medium",
            "accepted": False,
            "source": "Rowan/hellaswag_negative",
        },
    ]


def _arc(row: Dict[str, Any], source: str) -> List[HelixRow]:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    choices = [f"{label}. {text}" for label, text in zip(labels, texts)]
    answer = str(row["answerKey"]).strip()
    prompt = _choice_prompt(str(row["question"]), choices)
    wrong = next((choice for choice in choices if not choice.startswith(f"{answer}.")), "")
    difficulty = "hard" if "Challenge" in source else "medium"
    rows = [{"prompt": prompt, "difficulty": difficulty, "accepted": True, "source": source}]
    if wrong:
        rows.append(
            {
                "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
                "difficulty": difficulty,
                "accepted": False,
                "source": f"{source}_negative",
            }
        )
    return rows


def _openbookqa(row: Dict[str, Any]) -> List[HelixRow]:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    choices = [f"{label}. {text}" for label, text in zip(labels, texts)]
    fact = str(row.get("fact1", "") or "").strip()
    question = str(row.get("question_stem", row.get("question", "")))
    prompt = _choice_prompt((f"Fact: {fact}\n\n" if fact else "") + question, choices)
    answer = str(row["answerKey"]).strip()
    wrong = next((choice for choice in choices if not choice.startswith(f"{answer}.")), "")
    rows = [{"prompt": prompt, "difficulty": "medium", "accepted": True, "source": "allenai/openbookqa"}]
    if wrong:
        rows.append(
            {
                "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
                "difficulty": "medium",
                "accepted": False,
                "source": "allenai/openbookqa_negative",
            }
        )
    return rows


def _gsm8k(row: Dict[str, Any]) -> List[HelixRow]:
    question = str(row["question"])
    answer = str(row["answer"])
    prompt = f"Solve the grade-school math problem step by step.\n\nQuestion: {question}"
    wrong = "The answer is 0 because no calculation is needed."
    return [
        {"prompt": f"{prompt}\n\nReference answer:\n{answer}", "difficulty": "hard", "accepted": True, "source": "gsm8k"},
        {
            "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
            "difficulty": "hard",
            "accepted": False,
            "source": "gsm8k_negative",
        },
    ]


def _piqa(row: Dict[str, Any]) -> List[HelixRow]:
    goal = str(row["goal"])
    choices = [f"A. {row['sol1']}", f"B. {row['sol2']}"]
    gold = int(row["label"])
    wrong = choices[1 - gold]
    prompt = _choice_prompt(goal, choices, "Choose the physically plausible solution.")
    return [
        {"prompt": prompt, "difficulty": "medium", "accepted": True, "source": "piqa"},
        {
            "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
            "difficulty": "medium",
            "accepted": False,
            "source": "piqa_negative",
        },
    ]


def _winogrande(row: Dict[str, Any]) -> List[HelixRow]:
    sentence = str(row["sentence"])
    choices = [f"A. {row['option1']}", f"B. {row['option2']}"]
    answer = str(row["answer"]).strip()
    wrong = choices[1 if answer == "1" else 0]
    prompt = _choice_prompt(sentence, choices, "Choose the option that correctly fills the blank.")
    return [
        {"prompt": prompt, "difficulty": "medium", "accepted": True, "source": "winogrande"},
        {
            "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
            "difficulty": "medium",
            "accepted": False,
            "source": "winogrande_negative",
        },
    ]


def _race(row: Dict[str, Any]) -> List[HelixRow]:
    options = [str(option) for option in row["options"]]
    labels = ["A", "B", "C", "D"]
    choices = [f"{label}. {option}" for label, option in zip(labels, options)]
    answer = str(row["answer"]).strip()
    wrong = next((choice for choice in choices if not choice.startswith(f"{answer}.")), choices[0])
    prompt = (
        "Answer the reading comprehension question using the article.\n\n"
        f"Article: {row['article']}\n\nQuestion: {row['question']}\n\nChoices:\n"
        + "\n".join(choices)
    )
    return [
        {"prompt": prompt, "difficulty": "hard", "accepted": True, "source": "race"},
        {
            "prompt": f"{prompt}\n\nDraft answer: {wrong}\nIs the draft correct?",
            "difficulty": "hard",
            "accepted": False,
            "source": "race_negative",
        },
    ]


def _sciq(row: Dict[str, Any]) -> List[HelixRow]:
    choices = [
        f"A. {row['correct_answer']}",
        f"B. {row['distractor1']}",
        f"C. {row['distractor2']}",
        f"D. {row['distractor3']}",
    ]
    prompt = (
        "Choose the correct science answer.\n\n"
        f"Support: {row.get('support', '')}\n\nQuestion: {row['question']}\n\nChoices:\n"
        + "\n".join(choices)
    )
    return [
        {"prompt": prompt, "difficulty": "medium", "accepted": True, "source": "sciq"},
        {
            "prompt": f"{prompt}\n\nDraft answer: {choices[1]}\nIs the draft correct?",
            "difficulty": "medium",
            "accepted": False,
            "source": "sciq_negative",
        },
    ]


HF_PRESETS: Dict[str, List[Tuple[str, str | None, str, Callable[[Dict[str, Any]], List[HelixRow]]]]] = {
    "reasoning_mix": [
        ("tau/commonsense_qa", None, "train", _commonsense_qa),
        ("google/boolq", None, "train", _boolq),
        ("Rowan/hellaswag", None, "train", _hellaswag),
        ("allenai/ai2_arc", "ARC-Challenge", "train", lambda row: _arc(row, "allenai/ai2_arc/ARC-Challenge")),
        ("allenai/ai2_arc", "ARC-Easy", "train", lambda row: _arc(row, "allenai/ai2_arc/ARC-Easy")),
        ("allenai/openbookqa", "main", "train", _openbookqa),
    ],
    "small_reasoning": [
        ("tau/commonsense_qa", None, "train", _commonsense_qa),
        ("google/boolq", None, "train", _boolq),
        ("allenai/ai2_arc", "ARC-Easy", "train", lambda row: _arc(row, "allenai/ai2_arc/ARC-Easy")),
    ],
    "balanced_reasoning": [
        ("tau/commonsense_qa", None, "train", _commonsense_qa),
        ("google/boolq", None, "train", _boolq),
        ("Rowan/hellaswag", None, "train", _hellaswag),
        ("allenai/ai2_arc", "ARC-Challenge", "train", lambda row: _arc(row, "allenai/ai2_arc/ARC-Challenge")),
        ("allenai/ai2_arc", "ARC-Easy", "train", lambda row: _arc(row, "allenai/ai2_arc/ARC-Easy")),
        ("allenai/openbookqa", "main", "train", _openbookqa),
    ],
    "big_reasoning": [
        ("tau/commonsense_qa", None, "train", _commonsense_qa),
        ("google/boolq", None, "train", _boolq),
        ("Rowan/hellaswag", None, "train", _hellaswag),
        ("allenai/ai2_arc", "ARC-Challenge", "train", lambda row: _arc(row, "allenai/ai2_arc/ARC-Challenge")),
        ("allenai/ai2_arc", "ARC-Easy", "train", lambda row: _arc(row, "allenai/ai2_arc/ARC-Easy")),
        ("allenai/openbookqa", "main", "train", _openbookqa),
        ("gsm8k", "main", "train", _gsm8k),
        ("piqa", None, "train", _piqa),
        ("winogrande", "winogrande_xl", "train", _winogrande),
        ("race", "all", "train", _race),
        ("sciq", None, "train", _sciq),
    ],
}


def build_hf_helix_rows(
    preset: str = "reasoning_mix",
    max_examples_per_dataset: int = 2000,
    cache_dir: str | None = None,
    skip_errors: bool = True,
) -> List[HelixRow]:
    if preset not in HF_PRESETS:
        raise ValueError(f"unknown Helix dataset preset: {preset}")
    rows: List[HelixRow] = []
    for name, config, split, formatter in HF_PRESETS[preset]:
        try:
            ds = _load_hf_dataset(name, config, split, cache_dir)
        except Exception as exc:
            if not skip_errors:
                raise
            print(f"Skipped HF dataset {name}/{config or 'default'}/{split}: {exc}", flush=True)
            continue
        limit = len(ds) if max_examples_per_dataset <= 0 else min(max_examples_per_dataset, len(ds))
        before = len(rows)
        for idx in range(limit):
            rows.extend(formatter(dict(ds[idx])))
        print(f"Loaded {len(rows) - before} Helix rows from {name}/{config or 'default'}/{split}", flush=True)
    return rows


def prepare_helix_dataset(
    output: str | Path,
    preset: str = "reasoning_mix",
    max_examples_per_dataset: int = 2000,
    cache_dir: str | None = None,
    synthetic_examples: int = 0,
    seed: int = 1234,
    balance: bool = True,
    max_per_difficulty: int = 12000,
    skip_errors: bool = True,
) -> Path:
    if preset == "synthetic":
        rows = build_synthetic_helix_rows(max_examples_per_dataset, seed)
    else:
        rows = build_hf_helix_rows(preset, max_examples_per_dataset, cache_dir, skip_errors=skip_errors)
        if synthetic_examples > 0:
            rows.extend(build_synthetic_helix_rows(synthetic_examples, seed))
    print_helix_summary(rows, "Helix data before balance")
    if balance:
        rows = balance_helix_rows(rows, max_per_difficulty=max_per_difficulty, seed=seed)
        print_helix_summary(rows, "Helix data after balance")
    return write_jsonl(rows, output)
