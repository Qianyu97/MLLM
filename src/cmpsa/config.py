"""Configuration loading: YAML defaults + optional override + env interpolation.

Usage::

    from cmpsa.config import load_config
    cfg = load_config()                      # configs/default.yaml
    cfg = load_config("configs/exp_a.yaml")  # merged on top of default.yaml
    print(cfg.cmota.lambda_ot)               # attribute access
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from . import paths

DEFAULT_CONFIG = paths.PROJECT_ROOT / "configs" / "default.yaml"
MODELS_CONFIG = paths.PROJECT_ROOT / "configs" / "models.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_ns(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for dotted access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(override: str | Path | None = None, as_namespace: bool = True):
    """Load default.yaml (+ optional override) and the model registry.

    The resolved ``data_root``/``models_root``/``results_root`` always come from
    :mod:`cmpsa.paths` (which honours the CMPSA_* env vars), so they stay correct
    after moving the folder to a server.
    """
    cfg = load_yaml(DEFAULT_CONFIG)
    if override is not None:
        cfg = _deep_merge(cfg, load_yaml(override))
    cfg["models"] = load_yaml(MODELS_CONFIG)
    cfg["data_root"] = str(paths.DATA_ROOT)
    cfg["models_root"] = str(paths.MODELS_ROOT)
    cfg["results_root"] = str(paths.RESULTS_ROOT)
    return _to_ns(cfg) if as_namespace else cfg
