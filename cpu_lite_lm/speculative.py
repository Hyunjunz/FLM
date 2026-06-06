"""Self-Speculative Decoding for CPULiteLM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .modeling_cpu_lite import CPULiteForCausalLM

class SelfSpeculativeGenerator:
    def __init__(
        self, 
        model: CPULiteForCausalLM, 
        draft_layer: int = 1, 
        lookahead: int = 3
    ) -> None:
        self.model = model
        self.draft_layer = draft_layer
        self.lookahead = lookahead

    @torch.no_grad()
    def generate_streaming(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
    ):
        self.model.eval()
        device = input_ids.device
        bsz = input_ids.size(0)
        eos = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id
        
        generated = input_ids.clone()
        
        max_len = generated.size(1) + max_new_tokens + self.lookahead + 1
        kv_cache = self.model.allocate_kv_cache(bsz, max_len, device=device)
            
        cur_pos = generated.size(1)
        target_len = input_ids.size(1) + max_new_tokens
        # Initial prefill
        self.model(
            generated, 
            use_cache=True, 
            past_key_values=kv_cache, 
            cache_position=torch.arange(0, cur_pos, device=device)
        )
        
        while generated.size(1) < target_len:
            # 1. Draft phase: predict 'lookahead' tokens using early exit
            draft_ids = generated[:, -1:]
            draft_tokens = []
            
            # Re-score the final committed token at its real position to draft
            # the next token from the same context as full-depth decoding.
            temp_pos = cur_pos - 1
            for _ in range(self.lookahead):
                out = self.model(
                    draft_ids, 
                    use_cache=True, 
                    past_key_values=kv_cache, 
                    cache_position=torch.tensor([temp_pos], device=device),
                    output_layer=self.draft_layer
                )
                logits = out.logits[:, -1, :]
                if temperature <= 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k and top_k > 0:
                        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(next_token)
                draft_ids = next_token
                temp_pos += 1
                if eos is not None and (next_token == eos).any():
                    break
            
            if not draft_tokens:
                break
                
            # 2. Verify phase
            all_draft_tokens = torch.cat(draft_tokens, dim=1)
            verify_input = torch.cat([generated[:, -1:], all_draft_tokens], dim=1)
            verify_pos = torch.arange(cur_pos - 1, cur_pos + all_draft_tokens.size(1), device=device)
            
            out = self.model(
                verify_input,
                use_cache=True,
                past_key_values=kv_cache,
                cache_position=verify_pos
            )
            
            # 3. Acceptance phase
            verify_logits = out.logits[:, :-1, :]
            
            accepted_count = 0
            for i in range(all_draft_tokens.size(1)):
                best_token = torch.argmax(verify_logits[:, i, :], dim=-1, keepdim=True)
                if (best_token == all_draft_tokens[:, i:i+1]).all():
                    accepted_count += 1
                else:
                    break
            
            # Append accepted tokens + the first correction token
            if accepted_count < all_draft_tokens.size(1):
                next_token = torch.argmax(verify_logits[:, accepted_count, :], dim=-1, keepdim=True)
                accepted_slice = all_draft_tokens[:, :accepted_count]
                chunks = [accepted_slice, next_token]
            else:
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                chunks = [all_draft_tokens, next_token]

            for chunk in chunks:
                remaining = target_len - generated.size(1)
                if remaining <= 0:
                    break
                chunk = chunk[:, :remaining]
                if chunk.size(1) == 0:
                    continue
                generated = torch.cat([generated, chunk], dim=1)
                yield chunk
            
            # Update position
            cur_pos = generated.size(1)
            # print(f"Accepted {accepted_count}, total size {cur_pos}")
            
            if eos is not None and (generated[:, -1] == eos).any():
                break
