"""CMPSA evaluation package.

Each ``eval_<bench>.py`` exposes a functional entry point::

    run(model, method, limit, cfg) -> dict   # returns the standard metrics dict

and is also runnable as ``python -m cmpsa.eval.eval_<bench> [--model ...] [--method ...]``.

All predictions / metrics are written through :mod:`cmpsa.eval.common` so they obey
the standard prediction-jsonl row schema and the standard metrics-json schema.
"""
from __future__ import annotations

__all__ = [
    "common",
    "eval_pope",
    "eval_chair",
    "eval_amber",
    "eval_hallusionbench",
    "eval_mmhal",
    "eval_mme",
    "eval_vg_rel",
    "run_all",
]
