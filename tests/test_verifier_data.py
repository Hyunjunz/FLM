import json

import torch

from cpu_lite_lm import CPULiteConfig, CPULiteForCausalLM
from cpu_lite_lm.helix_train import HelixJsonlDataset, collate_helix, ensure_helix_heads, helix_head_loss
from cpu_lite_lm.reasoning_data import VERIFIER_IGNORE_INDEX, format_verifier_prompt, make_hard_negative


class TinyTokenizer:
    class Encoded:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        return self.Encoded([min(63, max(1, ord(ch) % 64)) for ch in text[:64]] or [1])


def test_missing_verifier_label_is_ignored(tmp_path):
    data = tmp_path / "verifier.jsonl"
    data.write_text(json.dumps({"question": "12*13?", "candidate_answer": "156"}) + "\n", encoding="utf-8")
    ds = HelixJsonlDataset(data, TinyTokenizer(), block_size=64)
    assert ds[0]["verifier_label"] == VERIFIER_IGNORE_INDEX

    cfg = CPULiteConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=1)
    model = CPULiteForCausalLM(cfg)
    ensure_helix_heads(model)
    batch = collate_helix([ds[0]], pad_token_id=0)
    out = helix_head_loss(model, batch)
    assert torch.isfinite(out["loss"])


def test_verifier_prompt_and_hard_negative():
    prompt = format_verifier_prompt("12 * 13은?", "156", include_label=True, label=1)
    assert "### Candidate Answer:" in prompt
    assert prompt.rstrip().endswith("1")
    assert make_hard_negative("12*13?", "156") != "156"
