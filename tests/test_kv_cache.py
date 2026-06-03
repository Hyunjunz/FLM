import torch

from cpu_lite_lm import CPULiteConfig, CPULiteForCausalLM


def test_kv_cache_shapes_and_decode_step():
    cfg = CPULiteConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    model = CPULiteForCausalLM(cfg)
    x = torch.randint(0, 64, (1, 6))
    out = model(x, use_cache=True)
    assert len(out.past_key_values) == cfg.num_hidden_layers
    k, v = out.past_key_values[0]
    assert k.shape == (1, cfg.num_key_value_heads, 6, cfg.head_dim)
    assert v.shape == (1, cfg.num_key_value_heads, 6, cfg.head_dim)
    nxt = torch.randint(0, 64, (1, 1))
    out2 = model(nxt, past_key_values=out.past_key_values, use_cache=True)
    k2, v2 = out2.past_key_values[0]
    assert out2.logits.shape == (1, 1, 64)
    assert k2.shape[2] == 7
    assert v2.shape[2] == 7

