from cpu_lite_lm.helix_runtime import HelixDifficultyRouter


def test_router_classifies_math_code_as_hard():
    router = HelixDifficultyRouter()
    assert router.classify("단계별로 계산하고 풀이를 보여줘: 12 * 13은?", 12) == "hard"
    assert router.classify("debug this Python bug and explain the code fix", 10) == "hard"


def test_router_classifies_short_factual_as_easy():
    router = HelixDifficultyRouter()
    assert router.classify("Capital of France?", 4) == "easy"
