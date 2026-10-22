# GRPO on GSM8K with verifiable rewards

![tests](https://github.com/<your-username>/grpo-gsm8k/actions/workflows/tests.yml/badge.svg)

Reinforcement-learning post-training of a small instruction model
(Qwen2.5-0.5B-Instruct, swappable for Falcon3-1B or similar) on grade-school
math with **GRPO** and **rule-based, verifiable rewards**: no reward model,
only checks of the final number and the output format.

The goal is to study, at small scale, the effect of three design choices on
accuracy and training dynamics:

1. the **KL penalty** β to the reference model,
2. the **group size** G used to compute advantages,
3. **reward shaping** (correctness only vs. correctness + format rewards).

## Method

For each prompt *q*, GRPO samples a group of G completions
*o₁ … o_G* from the current policy and scores them with the reward function.
The advantage of each completion is its reward normalised within the group:

$$\hat A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$

so no value network (critic) is needed, unlike PPO. The policy is updated with a
clipped PPO-style objective on these advantages, plus a KL term
β·KL(π_θ ‖ π_ref) that keeps it close to the starting model.

### Rewards (`src/grpo_math/rewards.py`)

The model is asked to answer as `<think>…</think><answer>N</answer>`.

| reward | value | purpose |
|---|---|---|
| `correctness` | 1.0 if the number in `<answer>` equals the gold answer | the actual task |
| `strict_format` | 0.5 if the whole output matches the template | parseable outputs |
| `soft_format` | 0.125 per tag, minus a small penalty for text after `</answer>` | dense signal early in training |
| `numeric` | 0.25 if the answer is a bare number | easier verification |

Answers are compared as exact fractions, so `7`, `7.0`, `14/2` and `$7`
all match. Rewards are weighted and summed (weights in the config).

## Project layout

```
configs/base.yaml        hyper-parameters of the baseline run
src/grpo_math/
  rewards.py             verifiable reward functions
  data.py                GSM8K loading + chat prompt
  train.py               GRPOTrainer setup (TRL, optional LoRA)
  evaluate.py            greedy evaluation, accuracy / format metrics
  config.py              YAML + dotted CLI overrides
scripts/
  train_grpo.py          training CLI
  evaluate.py            evaluation CLI
  run_ablations.sh       full ablation sweep
  plot.py                training curves + results table
notebooks/colab_quickrun.ipynb   short run on a free T4
tests/                   unit tests + offline CPU smoke test
```

## Usage

```bash
pip install -r requirements.txt
pip install -e .

# zero-shot baseline
python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --out results/zero_shot

# GRPO training (LoRA), then evaluation of the adapter
python scripts/train_grpo.py --config configs/base.yaml
python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/base --out results/base

# any config value can be overridden from the command line
python scripts/train_grpo.py --config configs/base.yaml grpo.beta=0.0 output_dir=outputs/beta_0

# full sweep + figures
bash scripts/run_ablations.sh
python scripts/plot.py --runs base beta_0 beta_0.1 g4 g16 correct_only
```

No GPU? `notebooks/colab_quickrun.ipynb` runs a shortened version on a free
Colab T4.

### Tests

```bash
pytest -q
```

The test suite runs offline on CPU. It includes an end-to-end smoke test that
trains a tiny randomly initialised Llama for two GRPO steps (with and without
LoRA), so the full training path is checked in CI without downloading a model.

## Results

> Not run yet. This section will be filled with the numbers produced by
> `scripts/plot.py` (`figures/results.md` and `figures/training_curves.png`).

| run | accuracy | format rate | mean chars |
|---|---|---|---|
| zero_shot | – | – | – |
| base (β=0.04, G=8, all rewards) | – | – | – |
| β = 0 | – | – | – |
| β = 0.1 | – | – | – |
| G = 4 | – | – | – |
| G = 16 | – | – | – |
| correctness only | – | – | – |

## Notes

- `warmup_ratio` was removed from `TrainingArguments` in transformers 5, so the
  config uses `warmup_steps`.
- With β > 0 and no LoRA, TRL rebuilds the reference model from the policy's
  config path; with LoRA the reference is obtained by disabling the adapter, so
  no second model is kept in memory.
- Evaluation uses left padding: with right padding, batched generation for
  decoder-only models appends tokens after the pad tokens.

## References

- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO).
- DeepSeek-AI, 2025. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.*
- Cobbe et al., 2021. *Training Verifiers to Solve Math Word Problems* (GSM8K).
- Hugging Face TRL, `GRPOTrainer`.
