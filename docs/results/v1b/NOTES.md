# v1b: gated format rewards, bare-number bonus still on

Accidental intermediate run: the reward code already required reasoning in
`<think>` for the format rewards, but the config was still v1:

| reward | weight |
|---|---|
| correctness (strict, tagged only) | 1.0 |
| soft_format (now gated on reasoning) | 1.0 |
| strict_format (now gated on reasoning) | 1.0 |
| numeric (0.25 for a bare number in `<answer>`) | 1.0 |

Colab settings as in v1 (100 steps, G = 4, 500 questions,
max_completion_length = 200); evaluation with 400 new tokens.

Result: the policy collapsed to `<answer>N</answer>` again (~10 tokens from
step 35). Both format rewards stayed at exactly 0 for the whole run, so the
hack is driven by the `numeric` bonus alone, about 0.23 of the ~0.30 mean
reward. A single cheap reward term is enough to trigger the collapse.
