"""Rule-based reward functions for GRPO.

Every function follows the TRL reward signature:
    fn(prompts, completions, **dataset_columns) -> list[float]

Completions can be plain strings or conversational messages
(``[{"role": "assistant", "content": ...}]``), so we normalise first.
"""

from __future__ import annotations

import re
from fractions import Fraction

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
STRICT_FORMAT_RE = re.compile(
    r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL
)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    # conversational format: list of messages, we score the last one
    return completion[-1]["content"]


def extract_answer(text: str) -> str | None:
    """Return the content of the last <answer> block, if any."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_gsm8k_gold(solution: str) -> str:
    """GSM8K reference solutions end with '#### <number>'."""
    return solution.split("####")[-1].strip()


def parse_number(text: str | None) -> Fraction | None:
    """Parse the last number in ``text`` ("1,200", "-3.5", "7/2", "$18")."""
    if text is None:
        return None
    found = NUMBER_RE.findall(text)
    if not found:
        return None
    raw = found[-1].replace(",", "")
    try:
        return Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(prediction: str | None, gold: str) -> bool:
    pred, ref = parse_number(prediction), parse_number(gold)
    return pred is not None and ref is not None and pred == ref


# --- reward functions -------------------------------------------------------


def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """1.0 if the number inside <answer> matches the gold answer, else 0.0."""
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_answer(completion_text(completion))
        rewards.append(1.0 if is_correct(pred, gold) else 0.0)
    return rewards


def strict_format_reward(prompts, completions, **kwargs) -> list[float]:
    """Whole completion must be exactly <think>...</think><answer>...</answer>."""
    return [
        0.5 if STRICT_FORMAT_RE.match(completion_text(c)) else 0.0
        for c in completions
    ]


def soft_format_reward(prompts, completions, **kwargs) -> list[float]:
    """Partial credit for each tag, so the model gets signal early in training.

    Each tag present exactly once gives 0.125; text after </answer> is
    lightly penalised to discourage rambling.
    """
    rewards = []
    for c in completions:
        text = completion_text(c)
        score = 0.0
        for tag in ("<think>", "</think>", "<answer>", "</answer>"):
            if text.count(tag) == 1:
                score += 0.125
        if "</answer>" in text:
            trailing = text.split("</answer>")[-1].strip()
            score -= min(len(trailing) * 0.001, 0.125)
        rewards.append(score)
    return rewards


def numeric_answer_reward(prompts, completions, **kwargs) -> list[float]:
    """Small bonus when the answer block is a bare number (easier to verify)."""
    rewards = []
    for c in completions:
        pred = extract_answer(completion_text(c))
        ok = pred is not None and re.fullmatch(r"-?[\d,]+(?:\.\d+)?", pred)
        rewards.append(0.25 if ok else 0.0)
    return rewards


REWARD_REGISTRY = {
    "correctness": correctness_reward,
    "strict_format": strict_format_reward,
    "soft_format": soft_format_reward,
    "numeric": numeric_answer_reward,
}
