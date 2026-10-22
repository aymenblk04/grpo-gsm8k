"""GSM8K loading and prompt formatting."""

from __future__ import annotations

from datasets import Dataset, load_dataset

from grpo_math.rewards import extract_gsm8k_gold

SYSTEM_PROMPT = (
    "You are a careful math assistant. Reason step by step inside "
    "<think> </think> tags, then give only the final number inside "
    "<answer> </answer> tags."
)


# One worked example shown as a previous turn. In v2 the model never produced
# the tags on its own, so the format rewards were never triggered; a single
# demonstration makes the format part of what it samples.
FEW_SHOT = [
    {
        "role": "user",
        "content": "A baker makes 24 cookies and puts them into bags of 4. "
        "He sells each bag for $3. How much money does he make?",
    },
    {
        "role": "assistant",
        "content": "<think>\n24 cookies in bags of 4 gives 24 / 4 = 6 bags.\n"
        "Each bag sells for $3, so he makes 6 * 3 = 18 dollars.\n</think>\n"
        "<answer>18</answer>",
    },
]


def make_prompt(question: str, few_shot: bool = False) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        messages += FEW_SHOT
    return messages + [{"role": "user", "content": question}]


def to_grpo_example(example: dict, few_shot: bool = False) -> dict:
    """Map a raw GSM8K row to the conversational format GRPOTrainer expects."""
    return {
        "prompt": make_prompt(example["question"], few_shot),
        "answer": extract_gsm8k_gold(example["answer"]),
    }


def prepare(
    dataset: Dataset, num_samples: int | None = None, seed: int = 0, few_shot: bool = False
) -> Dataset:
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    return dataset.map(
        to_grpo_example, fn_kwargs={"few_shot": few_shot}, remove_columns=dataset.column_names
    )


def load_gsm8k(
    split: str = "train", num_samples: int | None = None, seed: int = 0, few_shot: bool = False
) -> Dataset:
    raw = load_dataset("openai/gsm8k", "main", split=split)
    return prepare(raw, num_samples=num_samples, seed=seed, few_shot=few_shot)
