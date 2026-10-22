from fractions import Fraction

from grpo_math.rewards import (
    correctness_reward,
    extract_answer,
    extract_gsm8k_gold,
    lenient_correctness_reward,
    numeric_answer_reward,
    parse_number,
    soft_format_reward,
    strict_format_reward,
)

GOOD = "<think>3 apples + 4 apples = 7</think>\n<answer>7</answer>"


def test_extract_answer_takes_last_block():
    text = "<answer>1</answer> wait <answer>2</answer>"
    assert extract_answer(text) == "2"
    assert extract_answer("no tags") is None


def test_gsm8k_gold():
    assert extract_gsm8k_gold("blah\n#### 1,234") == "1,234"


def test_parse_number_variants():
    assert parse_number("$1,200") == Fraction(1200)
    assert parse_number("-3.5") == Fraction(-7, 2)
    assert parse_number("7/2") == Fraction(7, 2)
    assert parse_number("answer is 4 then 5") == Fraction(5)
    assert parse_number("none") is None


def test_correctness_reward_plain_and_conversational():
    comps = [GOOD, "<answer>8</answer>", [{"role": "assistant", "content": GOOD}]]
    assert correctness_reward(None, comps, answer=["7", "7", "7.0"]) == [1.0, 0.0, 1.0]


def test_correctness_requires_answer_tag():
    assert correctness_reward(None, ["the answer is 7"], answer=["7"]) == [0.0]


def test_strict_format():
    assert strict_format_reward(None, [GOOD, "<answer>7</answer>"]) == [0.5, 0.0]


def test_soft_format_partial_credit():
    full, partial, trailing = soft_format_reward(
        None, [GOOD, "<think>3 + 4 makes 7 in total</think> 7", GOOD + " and more text"]
    )
    assert full == 0.5
    assert partial == 0.25
    assert trailing < full


def test_v1_reward_hack_gets_no_format_reward():
    # what the v1 policy converged to: no reasoning, just the answer tag
    hacked = ["<answer>15</answer>", "<think></think><answer>15</answer>", "<think>ok</think><answer>15</answer>"]
    assert soft_format_reward(None, hacked) == [0.0, 0.0, 0.0]
    assert strict_format_reward(None, hacked) == [0.0, 0.0, 0.0]


def test_lenient_correctness():
    comps = [GOOD, "so the total is 7 apples", "<answer>8</answer> but really 7", "no idea"]
    assert lenient_correctness_reward(None, comps, answer=["7"] * 4) == [1.0, 0.5, 0.0, 0.0]


def test_numeric_answer_reward():
    out = numeric_answer_reward(None, [GOOD, "<answer>seven</answer>", "x"])
    assert out == [0.25, 0.0, 0.0]
