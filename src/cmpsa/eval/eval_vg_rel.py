"""Visual Genome relation evaluation (self-built, yes/no relation probing).

Reads ``paths.VG_REL_JSONL`` (produced by ``cmpsa.data.build_vg_rel``). Each row is a
yes/no relation question over a VG image, e.g. *"Is the man riding the horse?"* with a
"yes"/"no" label. We call ``method.answer_yes_no`` and report relation Accuracy / F1
(every row carries ``type == "relation"``).

The jsonl is read leniently so it works regardless of the exact field names the build
script emits. Expected / accepted keys per row:

    id / question_id           -> id
    image / image_path / image_id  -> image (resolved via paths.vg_image when an id)
    question / text / query    -> question
    label / answer / gt        -> "yes"/"no" (1/0 also accepted)

Run::

    python -m cmpsa.eval.eval_vg_rel --model llava-1.5-7b --limit 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
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

_LOG = get_logger("cmpsa.eval.vg_rel")
BENCH = "vg_rel"


def _to_yesno(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    s = str(value).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "yes"
    if s in ("no", "n", "false", "0"):
        return "no"
    return None


def _resolve_image(row: Dict[str, Any]):
    """Resolve a VG image path from a path field or a numeric image id."""
    for key in ("image_path", "image"):
        v = row.get(key)
        if v is None:
            continue
        p = Path(str(v))
        if p.exists():
            return p
        # maybe it's a bare file name -> try VG image dirs
        cand = paths.VG_100K / p.name
        if cand.exists():
            return cand
        cand2 = paths.VG_100K_2 / p.name
        if cand2.exists():
            return cand2
    # numeric image id
    for key in ("image_id", "img_id"):
        if row.get(key) is not None:
            return paths.vg_image(row[key])
    # last resort: "image" might itself be an id
    if isinstance(row.get("image"), (int,)) or (
        isinstance(row.get("image"), str) and str(row["image"]).isdigit()
    ):
        return paths.vg_image(row["image"])
    return None


def _load_items(limit: Optional[int]) -> List[Dict[str, Any]]:
    if not paths.VG_REL_JSONL.exists():
        raise RuntimeError(
            f"VG relation file missing: {paths.VG_REL_JSONL}. "
            "Build it first with `python -m cmpsa.data.build_vg_rel`."
        )
    rows = list(read_jsonl(paths.VG_REL_JSONL))
    return apply_limit(rows, limit)


def run(model: str, method: str, limit: Optional[int], cfg: Any) -> Dict[str, Any]:
    """Functional entry point used by ``run_all`` and ``__main__``."""
    set_seed(getattr(cfg, "seed", 42))
    items = _load_items(limit)
    if not items:
        raise RuntimeError("VG relation eval produced no items.")

    m = build_method_for(model, method, cfg)

    rows: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        img_path = _resolve_image(it)
        if img_path is None or not Path(img_path).exists():
            _LOG.warning("VG image not found for row %s (skipping)", it.get("id", i))
            continue
        question = it.get("question") or it.get("text") or it.get("query")
        gt = _to_yesno(it.get("label", it.get("answer", it.get("gt"))))
        image = load_image(img_path)
        pred, _conf = m.answer_yes_no(image, question)
        rows.append(
            {
                "id": str(it.get("id", it.get("question_id", i))),
                "image": Path(img_path).name,
                "question": question,
                "gt": gt,
                "pred": pred,
                "label": (1 if gt == "yes" else 0) if gt in ("yes", "no") else None,
                "type": "relation",
                "subset": it.get("predicate"),
            }
        )

    if not rows:
        raise RuntimeError("VG relation eval produced no usable rows (images missing?).")

    save_predictions(rows, BENCH, model, method)
    return compute_and_save_metrics(BENCH, model, method, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual Genome relation yes/no eval")
    cfg = load_config()
    add_common_eval_args(parser, cfg.mllm.key)
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    run(args.model, args.method, args.limit, cfg)


if __name__ == "__main__":
    main()
