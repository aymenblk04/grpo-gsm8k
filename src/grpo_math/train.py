"""Build and run a GRPOTrainer from a config dict."""

from __future__ import annotations

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from grpo_math.data import load_gsm8k
from grpo_math.rewards import REWARD_REGISTRY


def build_reward_funcs(reward_cfg: dict) -> tuple[list, list[float]]:
    """Turn {"name": weight} into (functions, weights); weight 0 disables a reward."""
    funcs, weights = [], []
    for name, weight in reward_cfg.items():
        if name not in REWARD_REGISTRY:
            raise KeyError(f"unknown reward {name!r}, choose from {sorted(REWARD_REGISTRY)}")
        if weight:
            funcs.append(REWARD_REGISTRY[name])
            weights.append(float(weight))
    if not funcs:
        raise ValueError("at least one reward must have a non-zero weight")
    return funcs, weights


def build_peft_config(lora_cfg: dict | None) -> LoraConfig | None:
    if not lora_cfg or not lora_cfg.get("enabled"):
        return None
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )


def build_grpo_args(cfg: dict, reward_weights: list[float]) -> GRPOConfig:
    grpo = dict(cfg["grpo"])
    if not torch.cuda.is_available():
        # lets the smoke test (and laptops) run on CPU
        grpo["bf16"] = False
        grpo["use_cpu"] = True
    return GRPOConfig(
        output_dir=cfg["output_dir"],
        seed=cfg.get("seed", 42),
        reward_weights=reward_weights,
        **grpo,
    )


def train(cfg: dict, model=None, tokenizer=None, train_dataset=None) -> GRPOTrainer:
    """``model``/``tokenizer``/``train_dataset`` can be injected (used in tests)."""
    reward_funcs, weights = build_reward_funcs(cfg["rewards"])
    if train_dataset is None:
        train_dataset = load_gsm8k(
            "train", num_samples=cfg["data"].get("train_samples"), seed=cfg.get("seed", 0)
        )
    trainer = GRPOTrainer(
        model=model if model is not None else cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=build_grpo_args(cfg, weights),
        train_dataset=train_dataset,
        peft_config=build_peft_config(cfg.get("lora")),
    )
    trainer.train()
    return trainer
