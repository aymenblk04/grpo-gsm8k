from datasets import Dataset

from grpo_math.data import SYSTEM_PROMPT, prepare


def _raw():
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?", "What is 10/2?", "What is 3*3?"],
            "answer": ["2+2=4\n#### 4", "10/2=5\n#### 5", "3*3=9\n#### 9"],
        }
    )


def test_prepare_formats_prompt_and_gold():
    ds = prepare(_raw())
    assert ds.column_names == ["prompt", "answer"]
    row = ds[0]
    assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert row["prompt"][1]["content"] == "What is 2+2?"
    assert row["answer"] == "4"


def test_prepare_subsamples():
    assert len(prepare(_raw(), num_samples=2)) == 2
