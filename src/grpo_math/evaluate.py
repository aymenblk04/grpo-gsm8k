"""Greedy evaluation on GSM8K: accuracy and format compliance."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_math.rewards import STRICT_FORMAT_RE, extract_answer, is_correct


def pick_dtype() -> torch.dtype:
    """bf16 on Ampere+, fp16 on older GPUs such as the Colab T4, fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def load_model(model_path: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompts: list[list[dict]], max_new_tokens: int, batch_size: int):
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts[i : i + batch_size]
        ]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


def score(completions: list[str], golds: list[str]) -> tuple[dict, list[dict]]:
    records = []
    for text, gold in zip(completions, golds):
        pred = extract_answer(text)
        records.append(
            {
                "completion": text,
                "prediction": pred,
                "gold": gold,
                "correct": is_correct(pred, gold),
                # last number anywhere in the text: fair to models that ignore the tags
                "correct_lenient": is_correct(pred if pred is not None else text, gold),
                "format_ok": bool(STRICT_FORMAT_RE.match(text)),
                "num_chars": len(text),
            }
        )
    n = max(len(records), 1)
    metrics = {
        "n": len(records),
        "accuracy": sum(r["correct"] for r in records) / n,
        "accuracy_lenient": sum(r["correct_lenient"] for r in records) / n,
        "format_rate": sum(r["format_ok"] for r in records) / n,
        "answer_tag_rate": sum(r["prediction"] is not None for r in records) / n,
        "mean_chars": sum(r["num_chars"] for r in records) / n,
    }
    return metrics, records


def evaluate(model, tokenizer, dataset, max_new_tokens=400, batch_size=16, out_dir=None):
    completions = generate(model, tokenizer, dataset["prompt"], max_new_tokens, batch_size)
    metrics, records = score(completions, dataset["answer"])
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        with open(out / "samples.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return metrics
