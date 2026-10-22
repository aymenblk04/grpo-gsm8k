import json

from datasets import Dataset

from grpo_math.data import prepare
from grpo_math.evaluate import evaluate, load_model, score


def test_score_metrics():
    metrics, records = score(
        ["<think>a</think><answer>4</answer>", "<answer>5</answer>", "no idea, maybe 3"],
        ["4", "4", "4"],
    )
    assert metrics["accuracy"] == 1 / 3
    assert metrics["format_rate"] == 1 / 3
    assert metrics["answer_tag_rate"] == 2 / 3
    assert metrics["accuracy_lenient"] == 1 / 3
    assert records[1]["prediction"] == "5"


def test_evaluate_end_to_end(tiny_model_path, tmp_path):
    raw = Dataset.from_dict({"question": ["2+2?"] * 3, "answer": ["#### 4"] * 3})
    model, tok = load_model(tiny_model_path)
    metrics = evaluate(model, tok, prepare(raw), max_new_tokens=4, batch_size=2, out_dir=tmp_path)
    assert metrics["n"] == 3
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == metrics
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 3
