"""POPE evaluation (object-existence hallucination, yes/no).

Reads the three POPE subsets (``paths.POPE_SUBSETS`` = random / popular / adversarial),
each referencing COCO val2014 images. For every question we call
``method.answer_yes_no(image, question)`` and score Accuracy / Precision / Recall /
F1 / Yes-Ratio overall *and* per subset.

POPE json is read robustly: it may be a JSON list or one JSON object per line.

Run::

    python -m cmpsa.eval.eval_pope --model llava-1.5-7b --method vanilla --limit 8
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed

from cmpsa.eval.common import (
    add_common_eval_args,
    apply_limit,
    build_method_for,
    compute_and_save_metrics,
    load_image,
    save_predictions,
)

_LOG = get_logger("cmpsa.eval.pope")
BENCH = "pope"


def _read_pope_json(path) -> List[Dict[str, Any]]:
    """Read a POPE subset file as a list of dicts (robust to list vs jsonl)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    # Fall back to one JSON object per line.
    items: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _load_pope_items(limit: Optional[int]) -> List[Dict[str, Any]]:
    """Load all subsets, tagging each item with its subset name and image path."""
    out: List[Dict[str, Any]] = []
    for subset, path in paths.POPE_SUBSETS.items():
        if not path.exists():
            _LOG.warning("POPE subset '%s' missing: %s (skipping)", subset, path)
            continue
        raw = _read_pope_json(path)
        raw = apply_limit(raw, limit)  # per-subset limit for a balanced smoke test
        for it in raw:
            img_name = it.get("image")
            out.append(
                {
                    "id": f"{subset}_{it.get('question_id')}",
                    "subset": subset,
                    "image_name": img_name,
                    "image_path": str(paths.coco_val2014_image(img_name)) if img_name else "",
                    "question": it.get("text") or it.get("question"),
                    "label": str(it.get("label", "")).strip().lower(),
                }
            )
    return out


def run(model: str, method: str, limit: Optional[int], cfg: Any) -> Dict[str, Any]:
    """Functional entry point used by ``run_all`` and ``__main__``."""
    set_seed(getattr(cfg, "seed", 42))
    items = _load_pope_items(limit)
    if not items:
        raise RuntimeError(
            f"No POPE questions found. Expected subsets under {paths.POPE_OUTPUT_COCO}."
        )

    m = build_method_for(model, method, cfg)

    rows: List[Dict[str, Any]] = []
    for it in items:
        image = load_image(it["image_path"])
        pred, _conf = m.answer_yes_no(image, it["question"])
        label = it["label"]
        rows.append(
            {
                "id": it["id"],
                "image": it["image_name"],
                "question": it["question"],
                "gt": label if label in ("yes", "no") else None,
                "pred": pred,
                "label": 1 if label == "yes" else (0 if label == "no" else None),
                "type": "object",        # POPE probes object existence
                "subset": it["subset"],
            }
        )

    save_predictions(rows, BENCH, model, method)

    # Per-subset F1 / Acc / Yes-Ratio as extra metrics (the by_type aggregation in
    # common is by type; POPE splits are by subset, so compute them explicitly).
    extra = _per_subset_metrics(rows)
    return compute_and_save_metrics(BENCH, model, method, rows, extra_metrics=extra)


def _per_subset_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    from cmpsa.eval.common import _binary_metrics  # reuse the same definition

    extra: Dict[str, float] = {}
    subsets = sorted({r["subset"] for r in rows if r.get("subset")})
    for s in subsets:
        sub = [r for r in rows if r.get("subset") == s]
        mt = _binary_metrics(sub)
        if mt is None:
            continue
        extra[f"{s}_f1"] = mt["f1"]
        extra[f"{s}_accuracy"] = mt["accuracy"]
        extra[f"{s}_yes_ratio"] = mt["yes_ratio"]
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description="POPE yes/no hallucination eval")
    cfg = load_config()
    add_common_eval_args(parser, cfg.mllm.key)
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    run(args.model, args.method, args.limit, cfg)


if __name__ == "__main__":
    main()
