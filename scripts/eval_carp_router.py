"""Evaluate CARP router-head accuracy on CARP JSONL traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_lite_lm.carp_data import CARPJsonlSFTDataset, collate_carp_sft
from cpu_lite_lm.modeling_cpu_lite import CPULiteForCausalLM
from cpu_lite_lm.tokenizer_train import load_tokenizer
from cpu_lite_lm.train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/carp_sft_ckpt")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--data", default="data/carp_synthetic.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer_path = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_path)
    model = CPULiteForCausalLM.from_pretrained(args.model).to(device).eval()
    if model.router_head is None:
        raise RuntimeError("Checkpoint has no router_head. Train with carp_router_labels > 0.")
    ds = CARPJsonlSFTDataset(
        args.data,
        tokenizer,
        block_size=args.block_size,
        max_reasoning_tokens=getattr(model.config, "carp_num_reasoning_tokens", 128),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_carp_sft(batch, model.config.pad_token_id),
    )
    total = 0
    correct = 0
    confusion = torch.zeros(4, 4, dtype=torch.long)
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            heads = model.carp_heads(batch["input_ids"], attention_mask=batch["attention_mask"])
            pred = torch.argmax(heads.router_logits, dim=-1)
            gold = batch["router_difficulty"]
            correct += int((pred == gold).sum().item())
            total += int(gold.numel())
            for g, p in zip(gold.cpu().tolist(), pred.cpu().tolist()):
                if 0 <= g < 4 and 0 <= p < 4:
                    confusion[g, p] += 1
    acc = correct / max(total, 1)
    print(f"router_accuracy={acc:.4f} correct={correct} total={total}")
    print("confusion_rows_gold_cols_pred=")
    for row in confusion.tolist():
        print(row)


if __name__ == "__main__":
    main()
