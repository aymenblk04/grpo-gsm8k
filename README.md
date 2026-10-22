# GRPO on GSM8K with verifiable rewards

![tests](https://github.com/aymenblk04/grpo-gsm8k/actions/workflows/tests.yml/badge.svg)

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

### v1: a reward-hacking run (Colab T4)

First short run: 100 GRPO steps, G = 4, 500 training questions, LoRA, fp16,
`max_completion_length=200`, evaluated greedily on 200 test questions
(`configs/v1.yaml` + the Colab overrides). Raw outputs are in
[`docs/results/v1`](docs/results/v1).

| model | accuracy (tagged) | accuracy (lenient) | format rate | mean length (chars) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, zero-shot | 0.0% | **32.0%** | 0.0% | 798 |
| + GRPO v1 (100 steps) | 7.5% | **7.5%** | 0.0% | 20 |

*Tagged* only counts the number inside `<answer>`; *lenient* takes the last
number of the output.

![v1 training curves](docs/results/v1/training_curves.png)

**What happened.** The total reward rises from 0.03 to about 0.55 in 30 steps
while the completion length collapses from ~180 to ~10 tokens. By the end, the
policy answers every question with a bare `<answer>N</answer>`:

```
gold 400  →  "<answer> 185 </answer>"
gold 25   →  "<answer>15</answer>"
```

The reward was **hacked**. The two answer tags (0.25 from `soft_format`) and a
bare number (0.25 from `numeric`) were worth 0.5 *without any reasoning*,
which is far easier to collect than the correctness reward: the base model
never used the tags, so correctness was almost always 0 at the start. The
strict format reward stayed at 0 for the whole run because no sample ever
contained a `<think>` block. Lenient accuracy shows the cost: the model went
from 32% to 7.5% by giving up step-by-step reasoning.

A second issue was evaluation: with 256 new tokens, ~80% of the zero-shot
answers were cut off, and the tagged metric scored the base model at 0%.
The baseline was therefore underestimated.

### v1b: the bare-number bonus alone is enough

An intermediate run (details in
[`docs/results/v1b/NOTES.md`](docs/results/v1b/NOTES.md)) used format rewards
that already required reasoning, but kept the v1 config: strict correctness
and the 0.25 bare-number bonus.

| model | accuracy (tagged) | accuracy (lenient) | format rate | mean length (chars) |
|---|---|---|---|---|
| zero-shot (400 new tokens) | 0.0% | **42.5%** | 0.0% | 911 |
| + GRPO v1b (100 steps) | 7.0% | **7.0%** | 0.0% | 20 |

The same collapse happened. Both format rewards stayed at 0 for the whole
run, so the bare-number bonus (~0.23 of the ~0.30 mean reward) **drove the
hack on its own**. A single cheap reward term is enough. The run also shows
how much evaluation settings matter for the baseline: allowing 400 instead of
256 new tokens moves the zero-shot lenient accuracy from 32% to 42.5%, since
fewer answers are cut off. Both evaluations use the same 200 questions
(seeded shuffle), so the only difference is the token budget.

### v2: no more hacking, but almost no learning

Changes from v1:
- format rewards give **nothing unless `<think>` contains real reasoning**
  (≥ 20 characters), and the exact v1 outputs are a regression test
  (`tests/test_rewards.py`);
- the bare-number bonus is off;
- new `lenient_correctness` reward: 1.0 for a correct tagged answer, 0.5 for a
  correct untagged one;
- 300 completion tokens in training, 400 at evaluation.

Same Colab budget as before (100 steps, G = 4, 500 questions, ~38 min on a T4),
evaluated on the same 200 test questions. Raw outputs are in
[`docs/results/v2`](docs/results/v2).

| model | accuracy (tagged) | accuracy (lenient) | format rate | mean length (chars) |
|---|---|---|---|---|
| zero-shot | 0.0% | 42.5% | 0.0% | 911 |
| + GRPO v2 (100 steps) | 0.0% | **45.5%** | 0.0% | 916 |

![v2 training curves](docs/results/v2/training_curves.png)

**The collapse is gone.** Completion length stays around 220-240 tokens for the
whole run, against ~10 tokens in v1/v1b, and the model keeps reasoning step by
step.

**But the policy barely moved.**
- The KL to the reference stays at ~0.001 (v1 reached 0.4).
- 62 of the 200 evaluation outputs are identical to the zero-shot ones.
- The +3 points are not significant. On the same questions, 13 go from wrong to
  right and 7 from right to wrong, and an exact McNemar test gives p ≈ 0.26.

**Why.** Two problems with the learning signal:
1. **The format is never discovered.** The model never writes `<think>` or
   `<answer>` on its own (0 of 200 eval outputs), so both format rewards are 0
   on every sample of the run, and the reward reduces to the 0.5 lenient
   credit. GRPO can only reinforce behaviour that the model already samples
   sometimes; the format reward is too sparse to be found by exploration.
2. **Many groups carry no signal.** With G = 4, 40-60% of the groups have
   identical rewards (all right or all wrong), so their advantages are zero.

Together with the low learning rate (5e-6 with LoRA), this gives very small
updates. The v1 hack moved the policy only because a trivial behaviour was
paid well.

### Next steps
- **Bootstrap the format** before RL: a few-shot example in the prompt, or a
  short SFT warm-up on formatted solutions (the "cold start" used for
  DeepSeek-R1, see [sft-dpo-math](https://github.com/aymenblk04/sft-dpo-math)).
- **Larger groups** (G = 8-16) to reduce the share of zero-advantage groups, and
  a higher LoRA learning rate (1e-5 to 5e-5).
- Longer runs: 100 steps × 4 prompts only covers 400 questions.

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
