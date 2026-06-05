"""Build reasoning SFT/eval and verifier JSONL files from a mixed training set."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


REASONING_TYPES = ("math", "code", "debug", "reason", "logic", "synthetic")
HARD_HINTS = re.compile(
    r"(### Plan:|### Solution:|계산|단계별|풀이|코드|버그|debug|solve|calculate|reason)",
    re.IGNORECASE,
)


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def is_reasoning_row(row: Dict[str, Any]) -> bool:
    marker = " ".join(str(row.get(key, "")) for key in ("type", "task_family", "source", "text", "prompt"))
    return any(token in marker.lower() for token in REASONING_TYPES) or HARD_HINTS.search(marker) is not None


def extract_question_answer(row: Dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or row.get("prompt") or "").strip()
    answer = str(row.get("answer") or row.get("final_answer") or row.get("output") or "").strip()
    text = str(row.get("text") or "")
    if not question and "### Question:" in text:
        question = text.split("### Question:", 1)[1].split("### Plan:", 1)[0].split("### Solution:", 1)[0].split("### Answer:", 1)[0].strip()
    if not answer and "### Answer:" in text:
        answer = text.split("### Answer:", 1)[1].strip()
    return question, answer


def to_reasoning_sft(row: Dict[str, Any]) -> Dict[str, Any] | None:
    question, answer = extract_question_answer(row)
    if not question or not answer:
        return None
    text = str(row.get("text") or "")
    plan = str(row.get("plan") or "").strip()
    solution = str(row.get("solution") or row.get("rationale") or "").strip()
    if not plan:
        plan = "Identify the required operation and keep the final answer separate."
    if not solution:
        if "### Solution:" in text:
            solution = text.split("### Solution:", 1)[1].split("### Answer:", 1)[0].strip()
        else:
            solution = "Use the information in the question to derive the answer concisely."
    return {
        "question": question,
        "plan": plan,
        "solution": solution,
        "answer": answer,
        "category": row.get("task_family", row.get("type", "reasoning")),
    }


def make_negative(answer: str) -> str:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if numbers:
        target = numbers[-1]
        replacement = str(float(target) + 1.0) if "." in target else str(int(target) + 1)
        idx = answer.rfind(target)
        return answer[:idx] + replacement + answer[idx + len(target) :]
    if "인덱스" in answer or "index" in answer.lower():
        return "환경 설정 문제이며 코드 수정은 필요 없다."
    return "The candidate answer is plausible but misses a required condition."


def build_verifier_rows(reasoning_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in reasoning_rows:
        question = str(row["question"])
        answer = str(row["answer"])
        rows.append({"question": question, "candidate_answer": answer, "verifier_label": 1})
        rows.append({"question": question, "candidate_answer": make_negative(answer), "verifier_label": 0})
    return rows


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--eval", default="data/eval.jsonl")
    parser.add_argument("--reasoning-sft", default="data/reasoning_sft.jsonl")
    parser.add_argument("--reasoning-eval", default="data/reasoning_eval.jsonl")
    parser.add_argument("--verifier-train", default="data/verifier_train.jsonl")
    parser.add_argument("--verifier-eval", default="data/verifier_eval.jsonl")
    parser.add_argument("--max-sft", type=int, default=20000)
    parser.add_argument("--max-eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = [to_reasoning_sft(row) for row in iter_jsonl(args.train) if is_reasoning_row(row)]
    train_reasoning = [row for row in train_rows if row is not None]
    eval_rows = [to_reasoning_sft(row) for row in iter_jsonl(args.eval) if is_reasoning_row(row)]
    eval_reasoning = [row for row in eval_rows if row is not None]
    rng.shuffle(train_reasoning)
    rng.shuffle(eval_reasoning)
    train_reasoning = train_reasoning[: args.max_sft]
    eval_reasoning = eval_reasoning[: args.max_eval]
    if not eval_reasoning:
        eval_reasoning = train_reasoning[: min(args.max_eval, len(train_reasoning))]
    if not train_reasoning:
        raise ValueError("No reasoning rows found. Check train/eval input files.")

    write_jsonl(args.reasoning_sft, train_reasoning)
    write_jsonl(args.reasoning_eval, eval_reasoning)
    write_jsonl(args.verifier_train, build_verifier_rows(train_reasoning))
    write_jsonl(args.verifier_eval, build_verifier_rows(eval_reasoning))


if __name__ == "__main__":
    main()
