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


def test_save_load_config(tmp_path):
    cfg = CPULiteConfig(vocab_size=1234)
    cfg.save_pretrained(tmp_path)
    loaded = CPULiteConfig.from_pretrained(tmp_path)
    assert loaded.vocab_size == 1234

