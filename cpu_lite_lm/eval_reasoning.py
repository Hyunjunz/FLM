"""Small fixed-set reasoning evaluation for CPULiteLM."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from .generate import load_model
from .helix_runtime import HelixMindRuntime, HelixRuntimeState
from .tokenizer_train import load_tokenizer
from .train import resolve_device


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")


def is_correct(generated: str, answer: str) -> bool:
    gen = normalize_answer(generated)
    gold = normalize_answer(answer)
    if not gold:
        return False
    return gold in gen or gen.endswith(gold)


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = resolve_device(args.device)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model, args.config).to(device).eval()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    eos_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<|endoftext|>") or tokenizer.token_to_id("<eos>")

    by_cat = defaultdict(lambda: [0, 0])
    lengths: List[int] = []
    total_tokens = 0
    t0 = time.perf_counter()
    verifier_hits = 0
    verifier_total = 0

    runtime = None
    if not args.full_depth:
        runtime = HelixMindRuntime(
            model,
            tokenizer,
            HelixRuntimeState(hard_full_depth=args.route == "hard", verify_before_accept=args.verify_before_accept),
        )

    for row in rows:
        question = str(row["question"])
        answer = str(row.get("answer", ""))
        category = str(row.get("category", "unknown"))
        prompt = f"### Question:\n{question}\n\n### Answer:\n"
        with torch.inference_mode():
            if runtime is not None:
                generated = runtime.infer(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    eos_token_id=eos_id,
                )
                verifier_score = runtime.state.stats.get("last_verifier_score")
                if verifier_score is not None:
                    verifier_total += 1
                    verifier_hits += int((verifier_score >= 0.5) == is_correct(generated, answer))
            else:
                ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
                out = model.generate_simple(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    use_cache=True,
                    eos_token_id=eos_id,
                )
                new_ids = out[0, ids.size(1) :].tolist()
                generated = tokenizer.decode(new_ids, skip_special_tokens=True)
        ok = is_correct(generated, answer)
        by_cat[category][0] += int(ok)
        by_cat[category][1] += 1
        gen_tokens = len(tokenizer.encode(generated).ids)
        lengths.append(gen_tokens)
        total_tokens += gen_tokens

    elapsed = max(time.perf_counter() - t0, 1e-9)
    correct = sum(v[0] for v in by_cat.values())
    total = sum(v[1] for v in by_cat.values())
    for category, (cat_ok, cat_total) in sorted(by_cat.items()):
        print(f"{category}: accuracy {cat_ok / max(cat_total, 1):.3f} ({cat_ok}/{cat_total})")
    print(f"overall_accuracy: {correct / max(total, 1):.3f} ({correct}/{total})")
    print(f"average_generation_length: {sum(lengths) / max(len(lengths), 1):.2f}")
    print(f"tokens_per_sec: {total_tokens / elapsed:.2f}")
    print(f"decode_mode: {'full-depth' if args.full_depth else 'router/early-exit'}")
    if verifier_total:
        print(f"verifier_accuracy: {verifier_hits / verifier_total:.3f} ({verifier_hits}/{verifier_total})")
    return {"accuracy": correct / max(total, 1), "tokens_per_sec": total_tokens / elapsed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/reasoning_sft_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--data", default="data/reasoning_eval.jsonl")
    parser.add_argument("--route", choices=["easy", "medium", "hard", "auto"], default="hard")
    parser.add_argument("--full-depth", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verify-before-accept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
