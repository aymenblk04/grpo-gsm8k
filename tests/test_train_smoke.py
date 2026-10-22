"""End-to-end: a few GRPO steps on a tiny model must run and log rewards."""

import pytest
from datasets import Dataset

from grpo_math.config import load_config
from grpo_math.data import prepare
from grpo_math.train import build_reward_funcs, train


def test_build_reward_funcs_skips_zero_weight():
    funcs, weights = build_reward_funcs({"correctness": 1.0, "numeric": 0})
    assert [f.__name__ for f in funcs] == ["correctness_reward"]
    assert weights == [1.0]


def test_build_reward_funcs_rejects_unknown():
    with pytest.raises(KeyError):
        build_reward_funcs({"nope": 1.0})


@pytest.mark.parametrize("lora", [False, True])
def test_grpo_smoke(lora, tiny_model_path, tiny_tokenizer, tmp_path):
    raw = Dataset.from_dict(
        {"question": ["What is 2+2?"] * 8, "answer": ["2+2=4\n#### 4"] * 8}
    )
    cfg = load_config(
        "configs/base.yaml",
        [
            f"output_dir={tmp_path}",
            f"lora.enabled={str(lora).lower()}",
            "grpo.num_generations=2",
            "grpo.per_device_train_batch_size=2",
            "grpo.gradient_accumulation_steps=1",
            "grpo.max_completion_length=8",
            "grpo.max_steps=2",
            "grpo.logging_steps=1",
            "grpo.gradient_checkpointing=false",
            "grpo.save_steps=1000",
        ],
    )
    trainer = train(cfg, model=tiny_model_path, tokenizer=tiny_tokenizer, train_dataset=prepare(raw))
    logs = [h for h in trainer.state.log_history if "reward" in h]
    assert logs, "trainer should log the mean reward"
    assert trainer.state.global_step == 2
