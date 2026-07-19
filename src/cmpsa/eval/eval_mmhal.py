"""MMHal-Bench evaluation (96 items, GPT-4 judged).

MMHal-Bench scores open-ended answers with a GPT-4 judge; there is no closed-form
metric. This script therefore:

  1. Loads the 96 MMHal questions + images (``paths.MMHAL_IMAGES`` / MMHAL_DIR).
  2. Generates an answer with ``method.answer_yes_no`` is NOT appropriate here;
     MMHal questions are open-ended, so we use ``method.caption(image, question)``
     (the Method caption() path accepts an arbitrary prompt).
  3. Writes standard prediction rows that already contain everything a GPT-4 judge
     needs (question, gt answer, generated answer, the gt-objects / question-type).
  4. Writes a placeholder ``score`` of 0.0 and ``hallucination_rate`` of 0.0, with a
     clear note that an *external* GPT-4 judge must be run to fill these in
     (see ``response_template`` written next to the predictions).

The MMHal json (``response_template.json`` in the official release) is a list of
records with keys like ``image_src``, ``question``, ``gt_answer``,
``question_type``, ``gt_objects``. We read it robustly and tolerate a few field-name
variants.

Run::

    python -m cmpsa.eval.eval_mmhal --model llava-1.5-7b --limit 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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

_LOG = get_logger("cmpsa.eval.mmhal")
BENCH = "mmhal"


def _find_mmhal_json() -> Optional[Path]:
    """Locate the MMHal question json under MMHAL_DIR (several known names)."""
    candidates = [
        paths.MMHAL_DIR / "response_template.json",
        paths.MMHAL_DIR / "mmhal-bench_answer_template.json",
        paths.MMHAL_DIR / "mmhal_bench.json",
        paths.MMHAL_DIR / "data" / "response_template.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Otherwise any json directly under MMHAL_DIR.
    for p in sorted(paths.MMHAL_DIR.glob("*.json")):
        return p
    return None


def _resolve_image(record: Dict[str, Any], idx: int) -> Optional[Path]:
    """Resolve an MMHal image path from common field names."""
    for key in ("image", "image_path", "image_src", "img_path", "filename"):
        v = record.get(key)
        if not v:
            continue
        name = str(v)
        # image_src is sometimes a URL; use only the basename against MMHAL_IMAGES
        base = name.split("/")[-1].split("?")[0]
        p = paths.MMHAL_IMAGES / base
        if p.exists():
            return p
        p2 = Path(name)
        if p2.exists():
            return p2
    # Fallback: positional image file.
    imgs = sorted(paths.MMHAL_IMAGES.glob("*"))
    if idx < len(imgs):
        return imgs[idx]
    return None


def _load_items(limit: Optional[int]) -> List[Dict[str, Any]]:
    qjson = _find_mmhal_json()
    if qjson is None:
        raise RuntimeError(
            f"No MMHal question json found under {paths.MMHAL_DIR}. "
            "Expected response_template.json (or similar)."
        )
    with open(qjson, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data.get("questions", [data]))
    items = apply_limit(list(data), limit)
    _LOG.info("loaded %d MMHal records from %s", len(items), qjson)
    return items


def run(model: str, method: str, limit: Optional[int], cfg: Any) -> Dict[str, Any]:
    """Functional entry point used by ``run_all`` and ``__main__``.

    NOTE: the returned metrics contain placeholder ``score``/``hallucination_rate``
    (0.0) that MUST be overwritten by an external GPT-4 judge. The predictions file
    holds everything the judge needs.
    """
    set_seed(getattr(cfg, "seed", 42))
    items = _load_items(limit)
    m = build_method_for(model, method, cfg)

    rows: List[Dict[str, Any]] = []
    judge_records: List[Dict[str, Any]] = []
    for i, rec in enumerate(items):
        img_path = _resolve_image(rec, i)
        if img_path is None:
            _LOG.warning("no image for MMHal record %d (skipping)", i)
            continue
        question = rec.get("question") or rec.get("query") or ""
        gt_answer = rec.get("gt_answer") or rec.get("answer") or rec.get("gt")
        image = load_image(img_path)
        answer = m.caption(image, question)

        rows.append(
            {
                "id": str(rec.get("id", i)),
                "image": img_path.name,
                "question": question,
                "gt": gt_answer,
                "pred": answer,
                "label": None,
                "type": "overall",
                "subset": rec.get("question_type"),
            }
        )
        # full record for the external judge (carry through any extra MMHal fields)
        judge_rec = dict(rec)
        judge_rec["model_answer"] = answer
        judge_rec["image_file"] = img_path.name
        judge_records.append(judge_rec)

    if not rows:
        raise RuntimeError("MMHal produced no rows. Check images under MMHAL_IMAGES.")

    pred_path = save_predictions(rows, BENCH, model, method)

    # Drop a judge-ready json next to the predictions for the external GPT-4 step.
    judge_path = pred_path.with_name(pred_path.stem + "__judge_input.json")
    with open(judge_path, "w", encoding="utf-8") as f:
        json.dump(judge_records, f, ensure_ascii=False, indent=2)
    _LOG.warning(
        "MMHal needs an external GPT-4 judge. Wrote judge input -> %s . "
        "score/hallucination_rate below are PLACEHOLDERS (0.0).",
        judge_path,
    )

    extra = {
        "score": 0.0,                 # placeholder: GPT-4 average score (0..6)
        "hallucination_rate": 0.0,    # placeholder: fraction judged hallucinatory
        "needs_external_judge": 1.0,  # flag for viz/run_all to surface
    }
    return compute_and_save_metrics(BENCH, model, method, rows, extra_metrics=extra)


def main() -> None:
    parser = argparse.ArgumentParser(description="MMHal-Bench eval (GPT-4 judged)")
    cfg = load_config()
    add_common_eval_args(parser, cfg.mllm.key)
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    run(args.model, args.method, args.limit, cfg)


if __name__ == "__main__":
    main()
