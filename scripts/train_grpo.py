"""Train with GRPO on GSM8K.

    python scripts/train_grpo.py --config configs/base.yaml
    python scripts/train_grpo.py --config configs/base.yaml grpo.beta=0.0 output_dir=outputs/no_kl
"""

import argparse

from grpo_math.config import load_config
from grpo_math.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    trainer = train(cfg)
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
