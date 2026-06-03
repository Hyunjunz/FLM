from cpu_lite_lm.sft_data import (
    _is_preformatted_text_example,
    build_instruction_text,
    format_sft_example,
)


def test_format_instruction_output():
    prompt, answer = format_sft_example(
        {"instruction": "Answer the capital.", "input": "Korea", "output": "Seoul."}
    )
    assert "capital" in prompt
    assert "Korea" in prompt
    assert answer == "Seoul."


def test_format_messages():
    prompt, answer = format_sft_example(
        {
            "messages": [
                {"role": "user", "content": "Hello?"},
                {"role": "assistant", "content": "Hi."},
            ]
        }
    )
    assert "Hello" in prompt
    assert answer == "Hi."


def test_instruction_template():
    prompt, answer = build_instruction_text("question", "answer")
    assert "### Question:" in prompt
    assert "### Answer:" in prompt
    assert answer == "answer"


def test_preformatted_text_schema():
    assert _is_preformatted_text_example(
        {"text": "### Question:\nA\n\n### Answer:\nB", "source_name": "x", "n_tokens": 10}
    )
    assert not _is_preformatted_text_example({"text": "x", "output": "y"})

