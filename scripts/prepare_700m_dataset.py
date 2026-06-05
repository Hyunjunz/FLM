"""Prepare a safer 700M CARP dataset mix with augmentations.

Key fixes over the Gemini-cli draft:
- normalizes every multiple-choice dataset to A/B/C/D/E labels
- shuffles choices by original index, not answer text
- avoids random CARP reasoning tokens; uses source/task-based deterministic tokens
- increases default dataset size for 700M smoke training
- keeps text-only LM/instruction rows rank/router-safe by omitting choice metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    print("Please install datasets: pip install datasets")
    raise SystemExit(1)

# Sample-level ratios. Token-level ratios will differ because general text rows are longer.
RATIO_GENERAL = 0.35
RATIO_INSTRUCTION = 0.20
RATIO_CHOICE = 0.25
RATIO_CARP = 0.10
RATIO_MATH_CODE = 0.10

LETTERS = [chr(65 + i) for i in range(26)]
CHOICE_RE = re.compile(r"^\s*([^.)\s]+)\s*[.)]\s*(.*)\s*$")

# Keep this aligned with cpu_lite_lm.carp.ReasoningCompressor.category_slots.
CATEGORY_SLOTS = {
    "math": 0,
    "code": 32,
    "logic": 64,
    "physical": 80,
    "retrieval": 96,
    "coreference": 112,
    "writing": 128,
    "social": 144,
    "safety": 160,
    "uncertainty": 192,
    "korean": 224,
}

SOURCE_CATEGORIES = {
    "tau/commonsense_qa": ["logic", "physical"],
    "google/boolq": ["retrieval", "logic"],
    "Rowan/hellaswag": ["logic", "physical"],
    "allenai/ai2_arc": ["retrieval", "logic"],
    "allenai/openbookqa": ["retrieval", "logic"],
    "piqa": ["physical", "logic"],
    "social_i_qa": ["social", "logic"],
    "winogrande": ["coreference", "logic"],
    "openai/gsm8k": ["math", "logic"],
}


def stable_int(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def parse_choice(choice: str) -> Tuple[str, str]:
    match = CHOICE_RE.match(choice)
    if not match:
        return "", choice.strip()
    return match.group(1).strip(), match.group(2).strip()


def normalize_choices(labels: Sequence[Any], texts: Sequence[Any], answer_key: Any) -> Tuple[List[str], str]:
    """Convert arbitrary dataset labels to A/B/C/D/E and remap gold label.

    This handles ARC-style numeric labels, CSQA letter labels, and mixed label formats.
    """
    raw_labels = [str(label).strip() for label in labels]
    raw_texts = [str(text).strip() for text in texts]
    answer = str(answer_key).strip()

    if len(raw_labels) != len(raw_texts):
        raise ValueError(f"labels/texts length mismatch: {len(raw_labels)} vs {len(raw_texts)}")
    if not raw_labels:
        raise ValueError("empty choices")

    old_to_new = {old: LETTERS[i] for i, old in enumerate(raw_labels)}
    if answer not in old_to_new:
        # Some datasets use 0/1 indices or numeric strings that are not in labels.
        try:
            idx = int(answer)
            if 0 <= idx < len(raw_labels):
                answer = raw_labels[idx]
            elif 1 <= idx <= len(raw_labels):
                answer = raw_labels[idx - 1]
        except Exception:
            pass

    if answer not in old_to_new:
        raise ValueError(f"answer_key={answer!r} not found in labels={raw_labels!r}")

    candidates = [f"{LETTERS[i]}. {text}" for i, text in enumerate(raw_texts)]
    return candidates, old_to_new[answer]


def rebuild_question_with_choices(question_prefix: str, candidates: Sequence[str]) -> str:
    prefix = question_prefix.split("Choices:", 1)[0].rstrip()
    return f"{prefix}\n\nChoices:\n" + "\n".join(candidates)


def shuffle_choices(question: str, choices: List[str], gold_label: str) -> Tuple[str, List[str], str]:
    """Shuffle normalized A/B/C/D/E choices by index and remap the gold label."""
    parsed = [parse_choice(choice) for choice in choices]
    labels = [label for label, _ in parsed]
    texts = [text for _, text in parsed]

    if gold_label not in labels:
        raise ValueError(f"gold_label={gold_label!r} not found in choices={choices!r}")
    gold_index = labels.index(gold_label)

    indices = list(range(len(texts)))
    random.shuffle(indices)
    shuffled_texts = [texts[i] for i in indices]
    new_candidates = [f"{LETTERS[i]}. {text}" for i, text in enumerate(shuffled_texts)]
    new_gold_index = indices.index(gold_index)
    new_gold_label = LETTERS[new_gold_index]
    new_question = rebuild_question_with_choices(question, new_candidates)
    return new_question, new_candidates, new_gold_label


def choice_text(candidates: Sequence[str], gold_label: str) -> str:
    for choice in candidates:
        label, text = parse_choice(choice)
        if label == gold_label:
            return text
    return ""


def format_answer(gold_label: str, gold_text: str, format_type: str) -> str:
    if format_type == "parsed":
        return f"parsed={gold_label}"
    if format_type == "label_only":
        return gold_label
    if format_type == "label_text":
        return f"{gold_label}. {gold_text}"
    if format_type == "natural":
        return f"The answer is {gold_label}."
    if format_type == "korean_natural":
        return f"정답은 {gold_label}입니다."
    return f"{gold_label}. {gold_text}"


def load_streaming_dataset(name: str, config: Optional[str] = None, split: str = "train"):
    if config:
        return load_dataset(name, config, split=split, streaming=True)
    return load_dataset(name, split=split, streaming=True)


def take_from_stream(ds: Iterable[Dict[str, Any]], count: int) -> Iterable[Dict[str, Any]]:
    iterator = iter(ds)
    for _ in range(max(0, count)):
        try:
            yield next(iterator)
        except StopIteration:
            break


def get_general_lm(count: int, max_chars: int) -> List[Dict[str, Any]]:
    print(f"Loading {count} General LM samples...")
    rows: List[Dict[str, Any]] = []

    en_count = int(count * 0.6)
    ko_count = count - en_count

    try:
        ds_en = load_streaming_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")
        for row in take_from_stream(ds_en, en_count):
            text = clip_text(row.get("text", ""), max_chars)
            if text:
                rows.append({"text": text, "source": "fineweb-edu", "type": "general_en"})
    except Exception as exc:
        print(f"Error loading FineWeb-Edu: {exc}")

    try:
        ds_ko = load_streaming_dataset("wikimedia/wikipedia", "20231101.ko")
        for row in take_from_stream(ds_ko, ko_count):
            text = clip_text(row.get("text", ""), max_chars)
            if text:
                rows.append({"text": text, "source": "wikipedia-ko", "type": "general_ko"})
    except Exception as exc:
        print(f"Error loading Wikipedia-ko: {exc}")

    return rows


def get_instruction(count: int, max_chars: int) -> List[Dict[str, Any]]:
    print(f"Loading {count} Instruction samples...")
    rows: List[Dict[str, Any]] = []
    try:
        ds = load_streaming_dataset("yahma/alpaca-cleaned")
        for row in take_from_stream(ds, count):
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            output = clip_text(str(row.get("output", "")), max_chars)
            if not instruction or not output:
                continue
            prompt = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                prompt += f"### Input:\n{input_text}\n\n"
            prompt += "### Response:\n"
            rows.append({
                "text": prompt + output,
                "prompt": prompt,
                "answer": output,
                "source": "alpaca",
                "type": "instruction",
            })
    except Exception as exc:
        print(f"Error loading Alpaca: {exc}")
    return rows


def normalize_to_trace(ds_name: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if "commonsense_qa" in ds_name:
            candidates, gold_label = normalize_choices(row["choices"]["label"], row["choices"]["text"], row["answerKey"])
            question = f"Question: {row['question']}\n\nChoices:\n" + "\n".join(candidates)
            return {"question": question, "candidates": candidates, "gold_label": gold_label, "task_family": "commonsense"}

        if "boolq" in ds_name:
            candidates = ["A. yes", "B. no"]
            gold_label = "A" if bool(row["answer"]) else "B"
            question = f"Passage: {row['passage']}\n\nQuestion: {row['question']}\n\nChoices:\n" + "\n".join(candidates)
            return {"question": question, "candidates": candidates, "gold_label": gold_label, "task_family": "retrieval"}

        if "hellaswag" in ds_name:
            labels = ["A", "B", "C", "D"]
            candidates, gold_label = normalize_choices(labels, row["endings"], labels[int(row["label"])])
            question = f"Context: {row.get('ctx_a', '')} {row.get('ctx_b', '')}\n\nChoices:\n" + "\n".join(candidates)
            return {"question": question, "candidates": candidates, "gold_label": gold_label, "task_family": "continuation"}

        if "ai2_arc" in ds_name or "openbookqa" in ds_name:
            candidates, gold_label = normalize_choices(row["choices"]["label"], row["choices"]["text"], row["answerKey"])
            question_text = row.get("question") or row.get("question_stem") or ""
            question = f"Question: {question_text}\n\nChoices:\n" + "\n".join(candidates)
            return {"question": question, "candidates": candidates, "gold_label": gold_label, "task_family": "science"}

        if "piqa" in ds_name:
            candidates, gold_label = normalize_choices(["A", "B"], [row["sol1"], row["sol2"]], int(row["label"]))
            question = f"Goal: {row['goal']}\n\nChoices:\n" + "\n".join(candidates)
            return {"question": question, "candidates": candidates, "gold_label": gold_label, "task_family": "physical"}

        if "social_i_qa" in ds_name:
            candidates, gold_label = normalize_choices(
                ["A", "B", "C"],
                [row["answerA"], row["answerB"], row["answerC"]],
                int(row["label"]) - 1,
            )
            question = f"Context: {row['context']}\n\nQuestion: {row['question']}\n\nChoices:\n" + "\n".join(candidates)
            return {"question": question, "candidates": candidates, "gold_label": gold_label, "task_family": "social"}

        if "winogrande" in ds_name:
            candidates, gold_label = normalize_choices(["A", "B"], [row["option1"], row["option2"]], int(row["answer"]) - 1)
            question = f"Sentence: {row['sentence']}\n\nChoices:\n" + "\n".join(candidates)
            return {"question": question, "candidates": candidates, "gold_label": gold_label, "task_family": "coreference"}
    except Exception as exc:
        print(f"Skipping malformed {ds_name} row: {exc}")
        return None
    return None


def get_choice_reasoning(count: int) -> List[Dict[str, Any]]:
    print(f"Loading {count} Choice Reasoning samples...")
    rows: List[Dict[str, Any]] = []
    datasets_to_load: List[Tuple[str, Optional[str]]] = [
        ("tau/commonsense_qa", None),
        ("google/boolq", None),
        ("Rowan/hellaswag", None),
        ("allenai/ai2_arc", "ARC-Challenge"),
        ("allenai/ai2_arc", "ARC-Easy"),
        ("allenai/openbookqa", "main"),
        ("piqa", "plain_text"),
        ("social_i_qa", "plain_text"),
        ("winogrande", "winogrande_xl"),
    ]
    samples_per_ds = max(1, count // len(datasets_to_load)) if datasets_to_load else count

    for ds_name, config in datasets_to_load:
        source_id = ds_name if config is None else f"{ds_name}/{config}"
        try:
            ds = load_streaming_dataset(ds_name, config)
            loaded = 0
            for row in take_from_stream(ds, samples_per_ds * 3):  # allow skips
                if loaded >= samples_per_ds:
                    break
                trace = normalize_to_trace(ds_name, row)
                if not trace:
                    continue

                if random.random() < 0.5 and trace.get("candidates"):
                    try:
                        q, cand, gold = shuffle_choices(trace["question"], trace["candidates"], trace["gold_label"])
                        trace["question"] = q
                        trace["candidates"] = cand
                        trace["gold_label"] = gold
                        trace["shuffled"] = True
                    except Exception as exc:
                        print(f"Choice shuffle failed for {source_id}: {exc}")
                        trace["shuffled"] = False
                else:
                    trace["shuffled"] = False

                gold_text = choice_text(trace["candidates"], trace["gold_label"])
                fmt = random.choice(["parsed", "label_only", "label_text", "natural", "korean_natural"])
                answer = format_answer(trace["gold_label"], gold_text, fmt)
                prompt = f"### Question:\n{trace['question']}\n\n### Answer:\n"
                rows.append({
                    "text": prompt + answer,
                    "prompt": prompt,
                    "answer": answer,
                    "router_label": {
                        "difficulty": 2,
                        "verifier_required": True,
                        "candidate_count": len(trace["candidates"]),
                        "reasoning_budget": 0,
                    },
                    "candidates": trace["candidates"],
                    "gold_label": trace["gold_label"],
                    "gold_text": gold_text,
                    "source": source_id,
                    "type": "choice",
                    "task_family": trace.get("task_family", "choice"),
                    "shuffled": trace.get("shuffled", False),
                })
                loaded += 1
        except Exception as exc:
            print(f"Error loading {source_id}: {exc}")
    return rows[:count]


def reasoning_tokens_for(row: Dict[str, Any], num_tokens: int = 256) -> List[str]:
    source = str(row.get("source", ""))
    family = str(row.get("task_family", ""))
    text = str(row.get("prompt") or row.get("text") or "")

    categories: List[str] = []
    for key, value in SOURCE_CATEGORIES.items():
        if key in source:
            categories.extend(value)
            break
    if family == "physical":
        categories.extend(["physical", "logic"])
    elif family == "social":
        categories.extend(["social", "logic"])
    elif family == "coreference":
        categories.extend(["coreference", "logic"])
    elif family == "science":
        categories.extend(["retrieval", "logic"])
    elif family == "retrieval":
        categories.extend(["retrieval", "logic"])

    if any("\uac00" <= ch <= "\ud7a3" for ch in text):
        categories.append("korean")
    if not categories:
        categories = ["logic"]

    # Deduplicate while preserving order.
    deduped = list(dict.fromkeys(categories))
    seed = stable_int(source + "\n" + family + "\n" + text[:512])
    budget = 2 + (seed % 3)  # 2~4 tokens, not always fixed length.
    tokens: List[str] = []
    for i in range(budget):
        category = deduped[i % len(deduped)]
        slot = CATEGORY_SLOTS.get(category, CATEGORY_SLOTS["logic"])
        # Keep 32 ids per broad category to match runtime compressor slots.
        token_id = (slot + ((seed >> (i * 8)) + i) % 32) % num_tokens
        tokens.append(f"<R{token_id}>")
    return tokens


def get_carp_traces(count: int, choice_samples: List[Dict[str, Any]], reasoning_token_count: int) -> List[Dict[str, Any]]:
    print(f"Generating {count} CARP traces from choice samples...")
    rows: List[Dict[str, Any]] = []
    if not choice_samples:
        return rows

    for _ in range(count):
        base = random.choice(choice_samples).copy()
        r_tokens = reasoning_tokens_for(base, reasoning_token_count)
        prompt = base["prompt"].replace(
            "### Answer:",
            "### Reasoning Tokens:\n" + " ".join(r_tokens) + "\n\n### Answer:",
        )
        base["text"] = prompt + base["answer"]
        base["prompt"] = prompt
        base["router_label"] = dict(base.get("router_label", {}))
        base["router_label"]["reasoning_budget"] = len(r_tokens)
        base["reasoning_tokens"] = r_tokens
        base["type"] = "carp"
        rows.append(base)
    return rows


def get_math_code(count: int, max_chars: int) -> List[Dict[str, Any]]:
    print(f"Loading {count} Math/Code samples...")
    rows: List[Dict[str, Any]] = []
    try:
        ds = load_streaming_dataset("openai/gsm8k", "main")
        for row in take_from_stream(ds, count):
            prompt = f"### Question:\n{row['question']}\n\n### Answer:\n"
            answer = clip_text(row.get("answer", ""), max_chars)
            rows.append({
                "text": prompt + answer,
                "prompt": prompt,
                "answer": answer,
                "source": "openai/gsm8k",
                "type": "math",
                "task_family": "math",
            })
    except Exception as exc:
        print(f"Error loading GSM8K: {exc}")
    if len(rows) < count:
        rows.extend(get_synthetic_reasoning(count - len(rows)))
    return rows[:count]


def get_synthetic_reasoning(count: int) -> List[Dict[str, Any]]:
    print(f"Generating {count} synthetic reasoning samples...")
    rows: List[Dict[str, Any]] = []
    templates = [
        lambda a, b: {
            "question": f"What is {a} * {b}?",
            "plan": "Multiply the two integers and keep the final number separate.",
            "solution": f"{a} * {b} = {a * b}.",
            "answer": str(a * b),
            "type": "math_synthetic",
        },
        lambda a, b: {
            "question": f"x + {a} = {b}. Solve for x.",
            "plan": "Subtract the same value from both sides.",
            "solution": f"x = {b} - {a} = {b - a}.",
            "answer": str(b - a),
            "type": "math_synthetic",
        },
        lambda a, b: {
            "question": "다음 Python 코드의 버그를 찾아라: for i in range(len(xs)+1): print(xs[i])",
            "plan": "반복 범위와 인덱스 접근을 확인한다.",
            "solution": "range(len(xs)+1)는 마지막에 xs[len(xs)]를 접근하므로 범위를 벗어난다.\n수정 코드: for i in range(len(xs)): print(xs[i])",
            "answer": "인덱스 범위 오류",
            "type": "code_debug_ko_synthetic",
        },
        lambda a, b: {
            "question": f"철수는 사과 {a}개를 사고 {b}개를 더 샀다. 몇 개인가?",
            "plan": "처음 개수와 추가 개수를 더한다.",
            "solution": f"{a} + {b} = {a + b}.",
            "answer": str(a + b),
            "type": "math_ko_synthetic",
        },
    ]
    for idx in range(count):
        a = random.randint(2, 20)
        b = random.randint(5, 40)
        row = templates[idx % len(templates)](a, b)
        prompt = (
            f"### Question:\n{row['question']}\n\n"
            f"### Plan:\n{row['plan']}\n\n"
            f"### Solution:\n{row['solution']}\n\n"
            "### Answer:\n"
        )
        rows.append({
            "text": prompt + row["answer"],
            "prompt": prompt,
            "answer": row["answer"],
            "source": "synthetic_reasoning",
            "type": row["type"],
            "task_family": "math_code",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/carp_700m_mix.jsonl")
    parser.add_argument("--total-count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-chars", type=int, default=4096)
    parser.add_argument("--reasoning-tokens", type=int, default=256)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total = args.total_count
    general = get_general_lm(int(total * RATIO_GENERAL), args.max_text_chars)
    instr = get_instruction(int(total * RATIO_INSTRUCTION), args.max_text_chars)
    choice = get_choice_reasoning(int(total * RATIO_CHOICE))
    carp = get_carp_traces(int(total * RATIO_CARP), choice, args.reasoning_tokens)
    math_code = get_math_code(int(total * RATIO_MATH_CODE), args.max_text_chars)

    all_rows = general + instr + choice + carp + math_code
    random.shuffle(all_rows)

    with open(args.output, "w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Finished! Wrote {len(all_rows)} samples to {args.output}")
    print(
        f"General: {len(general)}, Instr: {len(instr)}, Choice: {len(choice)}, "
        f"CARP: {len(carp)}, Math: {len(math_code)}"
    )


if __name__ == "__main__":
    main()
