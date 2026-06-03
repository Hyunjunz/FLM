import torch

from cpu_lite_lm import CPULiteConfig, CPULiteForCausalLM


def micro_cfg():
    return CPULiteConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )


def test_forward_shape():
    model = CPULiteForCausalLM(micro_cfg())
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out.logits.shape == (2, 8, 128)
    assert out.loss is None


def test_forward_with_labels_has_loss():
    model = CPULiteForCausalLM(micro_cfg())
    x = torch.randint(0, 128, (2, 8))
    out = model(x, labels=x)
    assert out.loss is not None
    assert out.loss.ndim == 0

