"""GSM8K loading and prompt formatting."""

from __future__ import annotations

from datasets import Dataset, load_dataset

from grpo_math.rewards import extract_gsm8k_gold

SYSTEM_PROMPT = (
    "You are a careful math assistant. Reason step by step inside "
    "<think> </think> tags, then give only the final number inside "
    "<answer> </answer> tags."
)


def to_grpo_example(example: dict) -> dict:
    """Map a raw GSM8K row to the conversational format GRPOTrainer expects."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": extract_gsm8k_gold(example["answer"]),
    }


def prepare(dataset: Dataset, num_samples: int | None = None, seed: int = 0) -> Dataset:
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    return dataset.map(to_grpo_example, remove_columns=dataset.column_names)


def load_gsm8k(split: str = "train", num_samples: int | None = None, seed: int = 0) -> Dataset:
    raw = load_dataset("openai/gsm8k", "main", split=split)
    return prepare(raw, num_samples=num_samples, seed=seed)
