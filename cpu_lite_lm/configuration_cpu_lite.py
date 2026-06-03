"""Configuration for CPULiteLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    from transformers import PretrainedConfig
except Exception:  # pragma: no cover - used when transformers is unavailable
    class PretrainedConfig:
        model_type = ""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def from_pretrained(cls, path: str | Path, **kwargs: Any):
            config_path = Path(path)
            if config_path.is_dir():
                config_path = config_path / "config.json"
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data.update(kwargs)
            return cls(**data)

        def to_dict(self) -> Dict[str, Any]:
            return dict(self.__dict__)

        def save_pretrained(self, save_directory: str | Path) -> None:
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            (save_path / "config.json").write_text(
                json.dumps(self.to_dict(), indent=2), encoding="utf-8"
            )


class CPULiteConfig(PretrainedConfig):
    model_type = "cpu_lite_lm"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 384,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        use_sdpa: bool = True,
        initializer_range: float = 0.02,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sdpa = use_sdpa
        self.initializer_range = initializer_range
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"{self.hidden_size} vs {self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_key_value_heads must divide num_attention_heads: "
                f"{self.num_key_value_heads} vs {self.num_attention_heads}"
            )
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "CPULiteConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["model_type"] = self.model_type
        return data
