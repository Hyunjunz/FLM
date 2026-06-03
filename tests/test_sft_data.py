from cpu_lite_lm.sft_data import build_instruction_text, format_sft_example


def test_format_instruction_output():
    prompt, answer = format_sft_example(
        {"instruction": "수도를 답하세요.", "input": "대한민국", "output": "서울입니다."}
    )
    assert "수도" in prompt
    assert "대한민국" in prompt
    assert answer == "서울입니다."


def test_format_messages():
    prompt, answer = format_sft_example(
        {
            "messages": [
                {"role": "user", "content": "안녕?"},
                {"role": "assistant", "content": "안녕하세요."},
            ]
        }
    )
    assert "안녕" in prompt
    assert answer == "안녕하세요."


def test_instruction_template():
    prompt, answer = build_instruction_text("질문", "답")
    assert "### 질문:" in prompt
    assert "### 답변:" in prompt
    assert answer == "답"

