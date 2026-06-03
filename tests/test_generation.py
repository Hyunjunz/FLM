import torch

from cpu_lite_lm import CPULiteConfig, CPULiteForCausalLM


def test_generation_adds_tokens():
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
    x = torch.randint(0, 64, (1, 4))
    out = model.generate_simple(x, max_new_tokens=5, temperature=0.0, use_cache=True, eos_token_id=None)
    assert out.shape == (1, 9)

