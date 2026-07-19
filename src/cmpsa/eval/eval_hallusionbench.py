"""HallusionBench evaluation (yes/no, figure / question / all accuracy).

Source of items:
  * Preferred: ``paths.HALLUSION_META`` (jsonl produced by ``parquet_to_images``),
    each row carrying the extracted image path + question + gt_answer.
  * Fallback: read ``paths.HALLUSION_PARQUET`` directly (pyarrow), decoding the
    embedded image bytes in-memory. This keeps the eval runnable even if the
    images have not been materialized yet.

Each question's gt_answer is "0"/"1" (1 == the statement/answer is correct -> "yes").
We score three HallusionBench accuracies:

  aAcc : per-question (every individual item)                     -> "all"
  qAcc : per-question-group: a question is correct only if ALL its
         visual variants are answered correctly (grouped by set_id+question_id)
  fAcc : per-figure: a figure is correct only if ALL its questions
         are answered correctly (grouped by set_id+figure_id)

Run::

    python -m cmpsa.eval.eval_hallusionbench --model llava-1.5-7b --limit 8
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, read_jsonl, set_seed

from cmpsa.eval.common import (
    add_common_eval_args,
    apply_limit,
    build_method_for,
    compute_and_save_metrics,
    load_image,
    save_predictions,
)

_LOG = get_logger("cmpsa.eval.hallusionbench")
BENCH = "hallusionbench"


def _gt_to_yesno(value: Any) -> Optional[str]:
    s = str(value).strip()
    if s in ("1", "yes", "Yes", "YES"):
        return "yes"
    if s in ("0", "no", "No", "NO"):
        return "no"
    return None


def _load_from_meta(limit: Optional[int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for r in read_jsonl(paths.HALLUSION_META):
        filename = r.get("filename")
        image_path = r.get("image") or r.get("image_path")
        if image_path is None and filename:
            image_path = str(paths.HALLUSION_IMAGES / str(filename))
        items.append(
            {
                "id": str(r.get("question_id", r.get("id", len(items)))),
                "set_id": r.get("set_id"),
                "figure_id": r.get("figure_id"),
                "question_id": r.get("question_id"),
                "question": r.get("question"),
                "gt": _gt_to_yesno(r.get("gt_answer")),
                "image_path": image_path,
                "image_bytes": None,
                "image_name": filename or (str(image_path or "").split("/")[-1]),
            }
        )
    return apply_limit(items, limit)


def _load_from_parquet(limit: Optional[int]) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq  # lazy; data dep only

    table = pq.read_table(paths.HALLUSION_PARQUET)
    cols = table.column_names
    n = table.num_rows
    take = n if (limit is None or limit <= 0) else min(n, limit)

    def col(name):
        return table.column(name).to_pylist() if name in cols else [None] * n

    set_ids = col("set_id")
    figure_ids = col("figure_id")
    question_ids = col("question_id")
    questions = col("question")
    gts = col("gt_answer")
    filenames = col("filename")
    images = col("image")

    items: List[Dict[str, Any]] = []
    for i in range(take):
        img = images[i]
        img_bytes = None
        if isinstance(img, dict):
            img_bytes = img.get("bytes")
        items.append(
            {
                "id": str(question_ids[i] if question_ids[i] is not None else i),
                "set_id": set_ids[i],
                "figure_id": figure_ids[i],
                "question_id": question_ids[i],
                "question": questions[i],
                "gt": _gt_to_yesno(gts[i]),
                "image_path": None,
                "image_bytes": img_bytes,
                "image_name": filenames[i] or f"hallusion_{i}.png",
            }
        )
    return items


def _load_items(limit: Optional[int]) -> List[Dict[str, Any]]:
    if paths.HALLUSION_META.exists():
        _LOG.info("loading HallusionBench from meta: %s", paths.HALLUSION_META)
        return _load_from_meta(limit)
    if paths.HALLUSION_PARQUET.exists():
        _LOG.info("meta absent; reading parquet directly: %s", paths.HALLUSION_PARQUET)
        return _load_from_parquet(limit)
    raise RuntimeError(
        f"No HallusionBench data found. Expected {paths.HALLUSION_META} "
        f"or {paths.HALLUSION_PARQUET}."
    )


def _group_accuracy(rows: List[Dict[str, Any]], key_fn) -> float:
    """Accuracy where a group counts as correct only if ALL its items are correct."""
    groups: Dict[Any, List[bool]] = defaultdict(list)
    for r in rows:
        if r["gt"] is None:
            continue
        from cmpsa.eval.common import _norm_yesno

        correct = _norm_yesno(r["pred"]) == r["gt"]
        groups[key_fn(r)].append(bool(correct))
    if not groups:
        return 0.0
    n_correct = sum(1 for v in groups.values() if all(v))
    return n_correct / len(groups)


def run(model: str, method: str, limit: Optional[int], cfg: Any) -> Dict[str, Any]:
    """Functional entry point used by ``run_all`` and ``__main__``."""
    set_seed(getattr(cfg, "seed", 42))
    items = _load_items(limit)
    if not items:
        raise RuntimeError("HallusionBench produced no items.")

    m = build_method_for(model, method, cfg)

    rows: List[Dict[str, Any]] = []
    for it in items:
        src = it["image_bytes"] if it["image_bytes"] is not None else it["image_path"]
        if src is None:
            _LOG.warning("no image for item %s (skipping)", it["id"])
            continue
        image = load_image(src)
        pred, _conf = m.answer_yes_no(image, it["question"])
        rows.append(
            {
                "id": it["id"],
                "image": it["image_name"],
                "question": it["question"],
                "gt": it["gt"],
                "pred": pred,
                "label": (1 if it["gt"] == "yes" else 0) if it["gt"] in ("yes", "no") else None,
                "type": "overall",
                "subset": None,
                # keep grouping keys around on the row for the group accuracies
                "_set_id": it["set_id"],
                "_figure_id": it["figure_id"],
                "_question_id": it["question_id"],
            }
        )

    a_acc_rows = rows  # aAcc == overall accuracy from common
    q_acc = _group_accuracy(
        a_acc_rows, key_fn=lambda r: (r["_set_id"], r["_question_id"])
    )
    f_acc = _group_accuracy(
        a_acc_rows, key_fn=lambda r: (r["_set_id"], r["_figure_id"])
    )

    # Strip the private grouping keys before persisting the standard rows.
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    save_predictions(clean_rows, BENCH, model, method)
    extra = {"qAcc": float(q_acc), "fAcc": float(f_acc)}
    out = compute_and_save_metrics(BENCH, model, method, clean_rows, extra_metrics=extra)
    # rename overall accuracy to aAcc as well (keep both for viz convenience)
    if "accuracy" in out["metrics"]:
        out["metrics"]["aAcc"] = out["metrics"]["accuracy"]
        from cmpsa.utils import save_json

        save_json(out, paths.metrics_path(BENCH, model, method))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="HallusionBench yes/no eval")
    cfg = load_config()
    add_common_eval_args(parser, cfg.mllm.key)
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    run(args.model, args.method, args.limit, cfg)


if __name__ == "__main__":
    main()
