import json
from pathlib import Path

import pytest

from cpu_lite_lm import CPULiteConfig


def test_micro_config_loads():
    cfg = CPULiteConfig.from_json_file(Path("configs/micro.json"))
    assert cfg.hidden_size == 128
    assert cfg.head_dim == 32
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0


def test_invalid_hidden_size():
    with pytest.raises(ValueError, match="hidden_size"):
        CPULiteConfig(hidden_size=130, num_attention_heads=4)


def test_invalid_kv_heads():
    with pytest.raises(ValueError, match="num_key_value_heads"):
        CPULiteConfig(num_attention_heads=8, num_key_value_heads=3)


def test_invalid_moe_top_k():
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        CPULiteConfig(num_experts=4, num_experts_per_tok=0)


def test_cpu_1b_moe_config_static_param_count():
    cfg = CPULiteConfig.from_json_file(Path("configs/cpu_1b_moe_fast.json"))
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    embed = cfg.vocab_size * cfg.hidden_size
    attn = (
        cfg.hidden_size * (cfg.num_attention_heads * head_dim)
        + 2 * cfg.hidden_size * (cfg.num_key_value_heads * head_dim)
        + cfg.hidden_size * cfg.hidden_size
    )
    expert = 3 * cfg.hidden_size * cfg.intermediate_size
    layer = attn + cfg.num_experts * expert + cfg.hidden_size * cfg.num_experts + 2 * cfg.hidden_size
    total = embed + cfg.num_hidden_layers * layer + cfg.hidden_size
    assert total >= 1_000_000_000
    assert cfg.num_experts_per_tok == 1


def test_save_load_config(tmp_path):
    cfg = CPULiteConfig(vocab_size=1234)
    cfg.save_pretrained(tmp_path)
    loaded = CPULiteConfig.from_pretrained(tmp_path)
    assert loaded.vocab_size == 1234
