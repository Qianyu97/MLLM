"""MME evaluation (perception + cognition subtasks, yes/no).

Reads ``paths.mme_parquets()`` (the two test shards). Each MME image has TWO yes/no
questions sharing the same ``question_id`` (e.g. ``artwork/10002``). For every
question we call ``method.answer_yes_no`` and compute, per category:

    acc   = correct individual questions / total individual questions
    acc+  = images where BOTH questions are correct / total images
    score = (acc + acc_plus) * 100        # the standard MME per-task score

We report per-category score plus the MME aggregate totals:
    perception_total  = sum of scores over the 10 perception categories
    cognition_total   = sum of scores over the 4 cognition categories
    mme_total         = perception_total + cognition_total

Run::

    python -m cmpsa.eval.eval_mme --model llava-1.5-7b --limit 8
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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

_LOG = get_logger("cmpsa.eval.mme")
BENCH = "mme"

# Official MME task groups.
PERCEPTION = [
    "existence", "count", "position", "color",
    "posters", "celebrity", "scene", "landmark", "artwork", "OCR",
]
COGNITION = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]


def _norm_answer(value: Any) -> Optional[str]:
    s = str(value).strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    return None


def _load_items(limit: Optional[int]) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq  # lazy: data dependency only

    files = paths.mme_parquets()
    if not files:
        raise RuntimeError(f"No MME parquet shards under {paths.MME_DATA}.")

    items: List[Dict[str, Any]] = []
    for f in files:
        table = pq.read_table(f)
        qids = table.column("question_id").to_pylist()
        questions = table.column("question").to_pylist()
        answers = table.column("answer").to_pylist()
        cats = table.column("category").to_pylist()
        images = table.column("image").to_pylist()
        for i in range(len(qids)):
            img = images[i]
            img_bytes = img.get("bytes") if isinstance(img, dict) else None
            img_name = img.get("path") if isinstance(img, dict) else f"mme_{i}.jpg"
            items.append(
                {
                    "question_id": qids[i],
                    "question": questions[i],
                    "gt": _norm_answer(answers[i]),
                    "category": cats[i],
                    "image_bytes": img_bytes,
                    "image_name": img_name,
                }
            )
    # For a meaningful smoke test keep whole image-pairs together: limit by pairs.
    if limit is not None and limit > 0:
        kept: List[Dict[str, Any]] = []
        seen_pairs = set()
        for it in items:
            key = (it["category"], it["question_id"])
            if key not in seen_pairs and len(seen_pairs) >= limit:
                continue
            seen_pairs.add(key)
            kept.append(it)
        items = kept
    return items


def _score_per_category(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float, float]]:
    """Return {category -> (acc, acc_plus, score)}."""
    from cmpsa.eval.common import _norm_yesno

    # group by (category, question_id) for acc+, and track per-category totals
    pair_correct: Dict[Tuple[str, Any], List[bool]] = defaultdict(list)
    cat_q_total: Dict[str, int] = defaultdict(int)
    cat_q_correct: Dict[str, int] = defaultdict(int)

    for r in rows:
        cat = r["subset"]  # we stash category in subset
        if r["gt"] not in ("yes", "no"):
            continue
        correct = _norm_yesno(r["pred"]) == r["gt"]
        cat_q_total[cat] += 1
        if correct:
            cat_q_correct[cat] += 1
        pair_correct[(cat, r["id"].rsplit("#", 1)[0])].append(bool(correct))

    cat_pair_total: Dict[str, int] = defaultdict(int)
    cat_pair_correct: Dict[str, int] = defaultdict(int)
    for (cat, _qid), flags in pair_correct.items():
        cat_pair_total[cat] += 1
        if all(flags):
            cat_pair_correct[cat] += 1

    out: Dict[str, Tuple[float, float, float]] = {}
    for cat in cat_q_total:
        acc = cat_q_correct[cat] / cat_q_total[cat] if cat_q_total[cat] else 0.0
        acc_plus = (
            cat_pair_correct[cat] / cat_pair_total[cat] if cat_pair_total[cat] else 0.0
        )
        score = (acc + acc_plus) * 100.0
        out[cat] = (acc, acc_plus, score)
    return out


def run(model: str, method: str, limit: Optional[int], cfg: Any) -> Dict[str, Any]:
    """Functional entry point used by ``run_all`` and ``__main__``."""
    set_seed(getattr(cfg, "seed", 42))
    items = _load_items(limit)
    if not items:
        raise RuntimeError("MME produced no items.")

    m = build_method_for(model, method, cfg)

    rows: List[Dict[str, Any]] = []
    pair_seen: Dict[Tuple[str, Any], int] = defaultdict(int)
    for it in items:
        if it["image_bytes"] is None:
            _LOG.warning("MME item without image bytes (skipping): %s", it["question_id"])
            continue
        image = load_image(it["image_bytes"])
        pred, _conf = m.answer_yes_no(image, it["question"])
        # make the row id unique while preserving the pair group prefix (qid#n)
        pair_key = (it["category"], it["question_id"])
        idx = pair_seen[pair_key]
        pair_seen[pair_key] += 1
        cat_type = "object" if it["category"] in ("existence", "count") else "overall"
        rows.append(
            {
                "id": f"{it['question_id']}#{idx}",
                "image": it["image_name"],
                "question": it["question"],
                "gt": it["gt"],
                "pred": pred,
                "label": (1 if it["gt"] == "yes" else 0) if it["gt"] in ("yes", "no") else None,
                "type": cat_type,
                "subset": it["category"],
            }
        )

    save_predictions(rows, BENCH, model, method)

    per_cat = _score_per_category(rows)
    extra: Dict[str, float] = {}
    perception_total = 0.0
    cognition_total = 0.0
    for cat, (acc, acc_plus, score) in sorted(per_cat.items()):
        extra[f"{cat}_acc"] = float(acc)
        extra[f"{cat}_acc_plus"] = float(acc_plus)
        extra[f"{cat}_score"] = float(score)
        if cat in PERCEPTION:
            perception_total += score
        elif cat in COGNITION:
            cognition_total += score
    extra["perception_total"] = float(perception_total)
    extra["cognition_total"] = float(cognition_total)
    extra["mme_total"] = float(perception_total + cognition_total)

    return compute_and_save_metrics(BENCH, model, method, rows, extra_metrics=extra)


def main() -> None:
    parser = argparse.ArgumentParser(description="MME yes/no perception+cognition eval")
    cfg = load_config()
    add_common_eval_args(parser, cfg.mllm.key)
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    run(args.model, args.method, args.limit, cfg)


if __name__ == "__main__":
    main()
