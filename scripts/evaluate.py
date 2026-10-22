"""Evaluate a base model or a GRPO checkpoint on the GSM8K test split.

    python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --out results/base
    python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/base --out results/grpo
"""

import argparse
import json

from grpo_math.data import load_gsm8k
from grpo_math.evaluate import evaluate, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="LoRA adapter directory")
    p.add_argument("--num-samples", type=int, default=None, help="default: full test set (1319)")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)
    dataset = load_gsm8k("test", num_samples=args.num_samples)
    metrics = evaluate(model, tokenizer, dataset, args.max_new_tokens, args.batch_size, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
