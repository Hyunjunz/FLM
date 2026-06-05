import torch

from cpu_lite_lm import CPULiteConfig, CPULiteForCausalLM, HelixMindRuntime, HelixRuntimeState
from cpu_lite_lm.helix_data import build_synthetic_helix_rows, convert_jsonl_to_helix, prepare_helix_dataset
from cpu_lite_lm.helix_train import (
    collate_helix,
    ensure_helix_heads,
    freeze_base_train_heads,
    helix_head_loss,
    materialize_tokenizer_corpus,
)


class TinyTokenizer:
    class Encoded:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        ids = [min(63, max(1, ord(ch) % 64)) for ch in text[:16]]
        return self.Encoded(ids or [1])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def test_helix_runtime_generates_without_retraining():
    cfg = CPULiteConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        eos_token_id=None,
    )
    model = CPULiteForCausalLM(cfg).eval()
    runtime = HelixMindRuntime(model, TinyTokenizer(), HelixRuntimeState())

    with torch.inference_mode():
        text = runtime.infer("debug this proof with reasoning", max_new_tokens=3, temperature=0.0, eos_token_id=None)

    assert isinstance(text, str)
    assert runtime.state.stats["last_regens"] >= 0


def test_helix_quant_policy_has_layer_entries():
    cfg = CPULiteConfig(vocab_size=64, num_hidden_layers=2)
    model = CPULiteForCausalLM(cfg)
    from cpu_lite_lm.helix_runtime import estimate_quant_policy

    policy = estimate_quant_policy(model)
    assert "layers.0.self_attn.qkv" in policy
    assert "lm_head" in policy


def test_helix_head_training_loss_backprops_only_heads():
    cfg = CPULiteConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=32,
        eos_token_id=None,
    )
    model = CPULiteForCausalLM(cfg)
    ensure_helix_heads(model)
    freeze_base_train_heads(model)
    batch = collate_helix(
        [
            {"input_ids": [1, 2, 3], "router_label": 0, "verifier_label": 1, "weight": 1.0},
            {"input_ids": [4, 5], "router_label": 2, "verifier_label": 0, "weight": 1.0},
        ],
        pad_token_id=0,
    )

    out = helix_head_loss(model, batch)
    out["loss"].backward()

    assert model.router_head.weight.grad is not None
    assert model.verifier_head.weight.grad is not None
    assert not model.model.embed_tokens.weight.requires_grad


def test_prepare_synthetic_helix_dataset(tmp_path):
    output = tmp_path / "helix.jsonl"
    prepare_helix_dataset(output, preset="synthetic", max_examples_per_dataset=8, synthetic_examples=0)
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) >= 8
    built = build_synthetic_helix_rows(8)
    assert any(row["accepted"] is False for row in built)


def test_materialize_tokenizer_corpus_from_helix_jsonl(tmp_path):
    data = tmp_path / "helix.jsonl"
    data.write_text('{"prompt":"hello tokenizer","difficulty":"easy","accepted":true}\n', encoding="utf-8")
    corpus = materialize_tokenizer_corpus(data, tmp_path / "corpus.txt")
    assert corpus.read_text(encoding="utf-8").strip() == "hello tokenizer"


def test_convert_mixed_jsonl_to_helix(tmp_path):
    src = tmp_path / "mixed.jsonl"
    src.write_text(
        "\n".join(
            [
                '{"text":"raw sft text","router_label":1}',
                '{"instruction":"solve this","output":"answer","difficulty":"hard"}',
                '{"question":"what is this?","answer":"that","accepted":true}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "helix.jsonl"
    convert_jsonl_to_helix(src, out)
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert "prompt" in rows[0]
