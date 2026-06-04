from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from cpu_lite_lm.modeling_cpu_lite import CPULiteForCausalLM
from cpu_lite_lm.tokenizer_train import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="./carp_700m_ckpt")
    p.add_argument("--question", default="What is a pillow used for?")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = load_tokenizer(str(Path(args.model) / "tokenizer.json"))
    model = CPULiteForCausalLM.from_pretrained(args.model).to(device).eval()

    prompt = f"### Question:\n{args.question}\n\n### Answer:\n"
    input_ids = tokenizer.encode(prompt).ids

    print("PROMPT:")
    print(repr(prompt))
    print("input_len:", len(input_ids))
    print("special ids:")
    print("pad", getattr(model.config, "pad_token_id", None))
    print("eos", getattr(model.config, "eos_token_id", None))
    print("bos", getattr(model.config, "bos_token_id", None))

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    chunks = []
    with torch.no_grad():
        for i, tok in enumerate(
            model.generate_streaming(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_cache=True,
                eos_token_id=None,  # 일단 자동 eos 중단 막지 말고 None
            )
        ):
            print("STEP", i, "shape", tuple(tok.shape), "tok", tok.detach().cpu().tolist())

            # generate_streaming이 [1,1]이든 [1,total]이든 마지막 토큰만 사용
            if tok.ndim == 2:
                tid = int(tok[0, -1].item())
            else:
                tid = int(tok[-1].item())

            chunks.append(tid)

            print("  token id:", tid)
            print("  raw token:", repr(tokenizer.decode([tid], skip_special_tokens=False)))
            print("  clean token:", repr(tokenizer.decode([tid], skip_special_tokens=True)))

    print("\nCHUNKS:", chunks)
    print("RAW DECODE:")
    print(repr(tokenizer.decode(chunks, skip_special_tokens=False)))
    print("CLEAN DECODE:")
    print(repr(tokenizer.decode(chunks, skip_special_tokens=True)))


if __name__ == "__main__":
    main()