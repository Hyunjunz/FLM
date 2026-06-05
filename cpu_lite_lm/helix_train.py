"""Train HelixMind router and verifier heads with the base model frozen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .configuration_cpu_lite import CPULiteConfig
from .generate import load_model
from .helix_data import prepare_helix_dataset
from .helix_runtime import HelixDifficultyRouter
from .modeling_cpu_lite import CPULiteForCausalLM
from .tokenizer_train import load_tokenizer
from .train import autocast_dtype, resolve_device


DIFFICULTY_TO_ID = {"easy": 0, "medium": 1, "hard": 2}


class HelixJsonlDataset(Dataset):
    """JSONL dataset for frozen-base Helix head training.

    Supported fields:
      prompt/question/user/input: text prompt
      difficulty/router_label: easy|medium|hard or 0|1|2
      verifier_label/accepted/correct/is_correct: verifier target, 0|1

    Missing difficulty labels are filled by the deterministic Helix router.
    Missing verifier labels default to 1, which is useful for route-only data.
    """

    def __init__(self, path: str | Path, tokenizer, block_size: int = 256) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.router = HelixDifficultyRouter()
        self.rows = [self._parse(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"no rows found in {self.path}")

    def _parse(self, line: str) -> Dict[str, Any]:
        item = json.loads(line)
        prompt = item.get("prompt") or item.get("question") or item.get("user") or item.get("input")
        if not prompt:
            raise ValueError("each row needs prompt/question/user/input")
        ids = self.tokenizer.encode(str(prompt)).ids[: self.block_size]
        if not ids:
            ids = [1]
        difficulty = item.get("difficulty", item.get("router_label"))
        if difficulty is None:
            difficulty = self.router.classify(str(prompt), len(ids))
        if isinstance(difficulty, str):
            difficulty_id = DIFFICULTY_TO_ID[difficulty.lower()]
        else:
            difficulty_id = max(0, min(2, int(difficulty)))

        verifier = item.get("verifier_label")
        if verifier is None:
            verifier = item.get("accepted", item.get("correct", item.get("is_correct", 1)))
        verifier_id = 1 if bool(verifier) else 0
        weight = float(item.get("weight", 1.0))
        return {
            "input_ids": ids,
            "router_label": difficulty_id,
            "verifier_label": verifier_id,
            "weight": weight,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def collate_helix(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    router_labels = torch.empty((len(batch),), dtype=torch.long)
    verifier_labels = torch.empty((len(batch),), dtype=torch.long)
    weights = torch.empty((len(batch),), dtype=torch.float)
    for i, row in enumerate(batch):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[i, : ids.numel()] = ids
        attention_mask[i, : ids.numel()] = 1
        router_labels[i] = int(row["router_label"])
        verifier_labels[i] = int(row["verifier_label"])
        weights[i] = float(row["weight"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "router_labels": router_labels,
        "verifier_labels": verifier_labels,
        "weights": weights,
    }


def ensure_helix_heads(model: CPULiteForCausalLM) -> None:
    hidden = model.config.hidden_size
    if model.router_head is None or getattr(model.config, "carp_router_labels", 0) != 3:
        model.config.carp_router_labels = 3
        model.router_head = nn.Linear(hidden, 3)
    if model.verifier_head is None or getattr(model.config, "carp_verifier_labels", 0) != 2:
        model.config.carp_verifier_labels = 2
        model.verifier_head = nn.Linear(hidden, 2)


def freeze_base_train_heads(model: CPULiteForCausalLM) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for head in (model.router_head, model.verifier_head):
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad = True


def helix_head_loss(model: CPULiteForCausalLM, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    heads = model.carp_heads(batch["input_ids"], attention_mask=batch["attention_mask"])
    if heads.router_logits is None or heads.verifier_logits is None:
        raise RuntimeError("Helix router/verifier heads are required")
    router_loss = F.cross_entropy(heads.router_logits, batch["router_labels"], reduction="none")
    verifier_loss = F.cross_entropy(heads.verifier_logits, batch["verifier_labels"], reduction="none")
    weights = batch["weights"].to(router_loss.dtype)
    loss = ((router_loss + verifier_loss) * weights).sum() / weights.sum().clamp_min(1.0)
    router_acc = (torch.argmax(heads.router_logits, dim=-1) == batch["router_labels"]).float().mean()
    verifier_acc = (torch.argmax(heads.verifier_logits, dim=-1) == batch["verifier_labels"]).float().mean()
    return {
        "loss": loss,
        "router_loss": router_loss.mean().detach(),
        "verifier_loss": verifier_loss.mean().detach(),
        "router_acc": router_acc.detach(),
        "verifier_acc": verifier_acc.detach(),
    }


def build_lr_lambda(max_steps: int, warmup_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/helix_train.jsonl")
    parser.add_argument("--auto-download", action="store_true", help="Download/prepare Helix data when --data is missing")
    parser.add_argument(
        "--download-preset",
        choices=["reasoning_mix", "small_reasoning", "synthetic"],
        default="reasoning_mix",
    )
    parser.add_argument("--download-max-examples", type=int, default=2000, help="Examples per HF dataset")
    parser.add_argument("--download-cache-dir", default="data/hf_cache")
    parser.add_argument("--synthetic-examples", type=int, default=1000)
    parser.add_argument("--model", default="artifacts/micro_ckpt")
    parser.add_argument("--config", default="configs/micro.json")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--output-dir", default="artifacts/helix_ckpt")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["off", "fp16", "bf16"], default="off")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))

    tokenizer_path = args.tokenizer or str(Path(args.model) / "tokenizer.json")
    data_path = Path(args.data)
    if args.auto_download and not data_path.exists():
        prepare_helix_dataset(
            data_path,
            preset=args.download_preset,
            max_examples_per_dataset=args.download_max_examples,
            cache_dir=args.download_cache_dir,
            synthetic_examples=args.synthetic_examples,
            seed=args.seed,
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Helix training data not found: {data_path}. "
            "Pass --auto-download to download/prepare it automatically."
        )
    tokenizer = load_tokenizer(tokenizer_path)
    device = resolve_device(args.device)
    model = load_model(args.model, args.config)
    ensure_helix_heads(model)
    model.resize_token_embeddings(tokenizer.get_vocab_size())
    freeze_base_train_heads(model)
    model.to(device).train()

    dataset = HelixJsonlDataset(data_path, tokenizer, block_size=args.block_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_helix(batch, model.config.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(args.max_steps, args.warmup_steps, args.min_lr_ratio),
    )
    amp_dtype = autocast_dtype(args.amp_dtype)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Helix trainable params: {trainable:,}", flush=True)

    step = 0
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = helix_head_loss(model, batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                print(
                    f"step {step}/{args.max_steps} loss {float(out['loss'].detach()):.4f} "
                    f"router {float(out['router_loss']):.4f} verifier {float(out['verifier_loss']):.4f} "
                    f"router_acc {float(out['router_acc']):.3f} verifier_acc {float(out['verifier_acc']):.3f} "
                    f"lr {scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    save_checkpoint(model, tokenizer, args, Path(args.output_dir), step)


def save_checkpoint(model, tokenizer, args: argparse.Namespace, output: Path, step: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save(str(output / "tokenizer.json"))
    payload = vars(args).copy()
    payload["last_step"] = step
    payload["trained"] = "router_head, verifier_head; base weights frozen"
    (output / "helix_train_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Helix checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()
