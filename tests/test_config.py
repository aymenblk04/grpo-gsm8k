import pytest

from grpo_math.config import apply_overrides, load_config


def test_overrides_types():
    cfg = {"grpo": {"beta": 0.04}}
    apply_overrides(cfg, ["grpo.beta=0.0", "grpo.bf16=false", "new.key=[a, b]"])
    assert cfg["grpo"]["beta"] == 0.0
    assert cfg["grpo"]["bf16"] is False
    assert cfg["new"]["key"] == ["a", "b"]


def test_bad_override():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equal_sign"])


def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg["grpo"]["num_generations"] == 8
    assert set(cfg["rewards"]) <= {"correctness", "lenient_correctness", "soft_format", "strict_format", "numeric"}
    assert cfg["rewards"]["numeric"] == 0  # disabled after the v1 reward hack
