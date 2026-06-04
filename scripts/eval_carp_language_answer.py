"""Evaluate generated answer accuracy on language reasoning datasets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.carp import CARPGenerator
from cpu_lite_lm.carp import ReasoningCompressor
from cpu_lite_lm.generate import _amp_dtype, _resolve_device
from cpu_lite_lm.modeling_cpu_lite import CPULiteForCausalLM
from cpu_lite_lm.tokenizer_train import load_tokenizer


def parse_choice(text: str) -> str:
    match = re.search(r"\b([A-E])\b\s*[\.\)]?", text.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def load_commonsense_qa(split: str, max_examples: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install datasets: pip install datasets") from exc

    ds = load_dataset("tau/commonsense_qa", split=split)
    limit = len(ds) if max_examples <= 0 else min(max_examples, len(ds))
    rows: List[Dict[str, Any]] = []
    for idx in range(limit):
        row = dict(ds[idx])
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(row["choices"]["label"], row["choices"]["text"])
        )
        prompt = (
            "Choose the best commonsense answer.\n\n"
            f"Question: {row['question']}\n\n"
            f"Choices:\n{choices}\n\n"
            "Answer with the option letter and text."
        )
        candidates = [
            f"{label}. {text}"
            for label, text in zip(row["choices"]["label"], row["choices"]["text"])
        ]
        rows.append({"prompt": prompt, "gold": str(row["answerKey"]).upper(), "candidates": candidates})
    return rows


def load_boolq(split: str, max_examples: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install datasets: pip install datasets") from exc

    ds = load_dataset("google/boolq", split=split)
    limit = len(ds) if max_examples <= 0 else min(max_examples, len(ds))
    rows: List[Dict[str, Any]] = []
    for idx in range(limit):
        row = dict(ds[idx])
        prompt = (
            "Answer the yes/no question using only the passage.\n\n"
            f"Passage: {row['passage']}\n\n"
            f"Question: {row['question']}\n\n"
            "Answer yes or no."
        )
        rows.append({"prompt": prompt, "gold": "yes" if bool(row["answer"]) else "no"})
    return rows


def parse_boolq(text: str) -> str:
    lowered = text.lower()
    yes_pos = lowered.find("yes")
    no_pos = lowered.find("no")
    if yes_pos < 0 and no_pos < 0:
        return ""
    if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
        return "yes"
    return "no"


@torch.no_grad()
def mean_logprob_score(model, tokenizer, prompt_text: str, answer_text: str, device: torch.device) -> float:
    prompt_ids = tokenizer.encode(prompt_text).ids
    answer_ids = tokenizer.encode(answer_text).ids
    if not answer_ids:
        return -1e9
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    out = model(ids, use_cache=False)
    logits = out.logits[:, max(0, len(prompt_ids) - 1) : -1, :]
    labels = ids[:, len(prompt_ids) :]
    if logits.numel() == 0 or labels.numel() == 0:
        return -1e9
    logprobs = torch.log_softmax(logits, dim=-1)
    chosen = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return float(chosen.mean().item())


def build_scoring_prompt(generator: CARPGenerator, prompt: str) -> str:
    route = generator.router.route(prompt)
    reasoning_tokens = generator.compressor.select(prompt, route)
    return generator._build_prompt(prompt, reasoning_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/carp_language_ckpt")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--dataset", choices=["tau/commonsense_qa", "commonsense_qa", "google/boolq", "boolq"], default="tau/commonsense_qa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--baseline", action="store_true", help="Use direct model generation instead of CARP routing.")
    parser.add_argument("--generate", action="store_true", help="Use free-form generation instead of multiple-choice scoring.")
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer or args.model)
    model = CPULiteForCausalLM.from_pretrained(args.model).to(device).eval()
    generator = CARPGenerator(model, tokenizer)
    amp_dtype = _amp_dtype(args.amp_dtype)
    if args.dataset in {"tau/commonsense_qa", "commonsense_qa"}:
        rows = load_commonsense_qa(args.split, args.max_examples)
        parser_fn = parse_choice
    else:
        rows = load_boolq(args.split, args.max_examples)
        parser_fn = parse_boolq

    correct = 0
    parsed = 0
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
    ):
        for idx, row in enumerate(rows, start=1):
            if args.baseline:
                prompt_text = f"### Question:\n{row['prompt']}\n\n### Answer:\n"
                if not args.generate and "candidates" in row:
                    scored = [
                        (mean_logprob_score(model, tokenizer, prompt_text, candidate, device), candidate)
                        for candidate in row["candidates"]
                    ]
                    answer = max(scored, key=lambda item: item[0])[1]
                else:
                    ids = torch.tensor([tokenizer.encode(prompt_text).ids], dtype=torch.long, device=device)
                    out = model.generate_simple(
                        ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        eos_token_id=None,
                    )
                    new_ids = out[0, ids.size(1) :].tolist()
                    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            else:
                if not args.generate and "candidates" in row:
                    prompt_text = build_scoring_prompt(generator, row["prompt"])
                    scored = [
                        (mean_logprob_score(model, tokenizer, prompt_text, candidate, device), candidate)
                        for candidate in row["candidates"]
                    ]
                    answer = max(scored, key=lambda item: item[0])[1]
                else:
                    result = generator.generate(
                        row["prompt"],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        use_speculative=not args.no_speculative,
                        eos_token_id=None,
                    )
                    answer = result.answer
            pred = parser_fn(answer)
            if pred:
                parsed += 1
            if pred == row["gold"]:
                correct += 1
            if args.print_every > 0 and (idx == 1 or idx % args.print_every == 0):
                acc = correct / idx
                parse_rate = parsed / idx
                print(
                    f"eval {idx}/{len(rows)} acc={acc:.4f} parse_rate={parse_rate:.4f} "
                    f"gold={row['gold']} pred={pred or '-'} answer={answer[:80]!r}",
                    flush=True,
                )

    total = len(rows)
    print(f"answer_accuracy={correct / max(total, 1):.4f} correct={correct} total={total}")
    print(f"parse_rate={parsed / max(total, 1):.4f} parsed={parsed} total={total}")


if __name__ == "__main__":
    main()
