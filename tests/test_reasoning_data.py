import json

from cpu_lite_lm.reasoning_data import ReasoningSFTJsonlDataset, format_reasoning_example


class TinyTokenizer:
    class Encoded:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        return self.Encoded([max(1, ord(ch) % 63) for ch in text] or [1])

    def token_to_id(self, token):
        return 2 if token == "<eos>" else None


def test_reasoning_sft_formatting_has_answer_boundary():
    prompt, answer = format_reasoning_example(
        {"question": "2+3?", "plan": "Add.", "solution": "2+3=5.", "answer": "5"}
    )
    assert "### Plan:" in prompt
    assert "### Solution:" in prompt
    assert prompt.endswith("### Answer:\n")
    assert answer == "5"


def test_reasoning_sft_dataset_jsonl(tmp_path):
    data = tmp_path / "reasoning.jsonl"
    data.write_text(json.dumps({"question": "1+1?", "solution": "1+1=2.", "answer": "2"}) + "\n")
    ds = ReasoningSFTJsonlDataset(data, TinyTokenizer(), block_size=64)
    item = ds[0]
    assert item["input_ids"].numel() >= 2
    assert item["labels"].shape == item["input_ids"].shape
