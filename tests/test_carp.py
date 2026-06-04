import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from cpu_lite_lm import CPULiteConfig, CPULiteForCausalLM
from cpu_lite_lm.carp import (
    CARPGenerator,
    DifficultyLevel,
    HeuristicDifficultyRouter,
    add_reasoning_tokens,
    reasoning_token_strings,
)
from cpu_lite_lm.carp_data import build_carp_instruction_text, build_router_label, parse_carp_trace
from cpu_lite_lm.carp_data import CARPJsonlSFTDataset, collate_carp_sft
from cpu_lite_lm.carp_train import carp_sft_loss


def build_tokenizer():
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "###": 4,
        "Question:": 5,
        "Answer:": 6,
        "Reasoning": 7,
        "Tokens:": 8,
        "안녕": 9,
        "코드": 10,
        "deadlock": 11,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return tok


def test_reasoning_token_strings():
    assert reasoning_token_strings(3) == ["<R0>", "<R1>", "<R2>"]


def test_add_reasoning_tokens():
    tok = build_tokenizer()
    added = add_reasoning_tokens(tok, 4)
    assert added == 4
    assert tok.token_to_id("<R3>") is not None


def test_router_levels():
    router = HeuristicDifficultyRouter()
    assert router.route("안녕").difficulty_level == DifficultyLevel.EASY
    assert router.route("이 코드 deadlock 원인을 분석해줘").difficulty_level == DifficultyLevel.HARD
    assert router.route("보안 취약점 분석").difficulty_level == DifficultyLevel.CRITICAL


def test_carp_generator_runs():
    tok = build_tokenizer()
    add_reasoning_tokens(tok, 16)
    cfg = CPULiteConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        eos_token_id=None,
        carp_num_reasoning_tokens=16,
    )
    model = CPULiteForCausalLM(cfg)
    gen = CARPGenerator(model, tok, num_reasoning_tokens=16, lookahead=1)
    result = gen.generate("안녕", max_new_tokens=3, temperature=0.0, use_speculative=False, eos_token_id=None)
    assert result.route.difficulty_level == DifficultyLevel.EASY
    assert isinstance(result.answer, str)
    assert len(result.candidates) == 1


def test_carp_heads_and_resize_embeddings():
    cfg = CPULiteConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=32,
        carp_router_labels=4,
        carp_verifier_labels=6,
    )
    model = CPULiteForCausalLM(cfg)
    x = torch.randint(0, 16, (2, 4))
    heads = model.carp_heads(x)
    assert heads.router_logits.shape == (2, 4)
    assert heads.verifier_logits.shape == (2, 6)
    model.resize_token_embeddings(20)
    assert model.config.vocab_size == 20
    assert model.lm_head.weight.shape[0] == 20


def test_carp_trace_formatting():
    trace = parse_carp_trace(
        {
            "question": "2+2?",
            "final_answer": "4",
            "reasoning_tokens": ["<R0>", "<R999>", "<R1>"],
            "difficulty": "hard",
        },
        max_reasoning_tokens=4,
    )
    prompt, answer = build_carp_instruction_text(trace)
    assert "<R0> <R1>" in prompt
    assert answer == "4"
    label = build_router_label(trace)
    assert label["difficulty"] == 2
    assert label["reasoning_budget"] == 2


def test_carp_sft_loss(tmp_path):
    tok = build_tokenizer()
    add_reasoning_tokens(tok, 8)
    data = tmp_path / "carp.jsonl"
    data.write_text(
        (
            '{"question":"What is 1 + 2?","answer":"A. 3",'
            '"candidates":["A. 3","B. 4"],"gold_label":"A",'
            '"reasoning_tokens":["<R0>"],"difficulty":"medium"}\n'
        ),
        encoding="utf-8",
    )
    ds = CARPJsonlSFTDataset(data, tok, block_size=32, max_reasoning_tokens=8)
    batch = collate_carp_sft([ds[0]], pad_token_id=0)
    cfg = CPULiteConfig(
        vocab_size=tok.get_vocab_size(),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        carp_num_reasoning_tokens=8,
        carp_router_labels=4,
    )
    model = CPULiteForCausalLM(cfg)
    out = carp_sft_loss(model, batch)
    assert out.loss.ndim == 0
    assert out.router_loss.ndim == 0
    assert out.ranking_loss.ndim == 0
