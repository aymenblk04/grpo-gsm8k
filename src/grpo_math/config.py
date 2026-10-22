"""YAML config loading with dotted command-line overrides."""

from __future__ import annotations

from pathlib import Path

import yaml


def _coerce(value: str):
    # let YAML decide: "0.1" -> float, "true" -> bool, "[a, b]" -> list
    return yaml.safe_load(value)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.sub=value`` overrides in place, e.g. ``grpo.beta=0.0``."""
    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _coerce(value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides or [])
