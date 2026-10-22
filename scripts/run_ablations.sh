#!/usr/bin/env bash
# Ablations on top of configs/base.yaml. Each run takes ~2-3h on one A100
# with the default 300 steps; lower grpo.max_steps for a quicker sweep.
set -euo pipefail

BASE=configs/base.yaml
MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$BASE'))['model_name'])")

run() {
  local name=$1; shift
  echo "=== $name ==="
  python scripts/train_grpo.py --config "$BASE" output_dir="outputs/$name" "$@"
  python scripts/evaluate.py --model "$MODEL" --adapter "outputs/$name" --out "results/$name"
}

# baseline model, no training
python scripts/evaluate.py --model "$MODEL" --out results/zero_shot

run base

# 1. KL penalty: does the reference model matter at this scale?
run beta_0      grpo.beta=0.0
run beta_0.1    grpo.beta=0.1

# 2. group size G (batch size is kept divisible by G)
run g4          grpo.num_generations=4
run g16         grpo.num_generations=16 grpo.per_device_train_batch_size=16 grpo.gradient_accumulation_steps=2

# 3. reward shaping: correctness only vs. full shaped reward
run correct_only rewards.soft_format=0 rewards.strict_format=0 rewards.numeric=0
