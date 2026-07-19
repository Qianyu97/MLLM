"""AMBER evaluation (discriminative yes/no + generative CHAIR-style).

``--task {generative, discriminative, all}`` (default: all).

Discriminative
    Reads the per-dimension question files
    (``AMBER_Q_EXISTENCE`` / ``AMBER_Q_ATTRIBUTE`` / ``AMBER_Q_RELATION``),
    each a list of {id, query, image}. We call ``method.answer_yes_no`` and tag every
    item with ``type`` = object (existence) / attribute / relation, scoring
    Accuracy / F1 overall and per dimension. The yes/no truth is taken from
    ``AMBER_ANN`` (annotations.json, where ``type == "discriminative"``).

Generative
    Reads ``AMBER_Q_GENERATIVE`` ({id, query="Describe this image."}) and the
    object word lists ``truth`` / ``hallu`` from ``AMBER_ANN`` (type "generative").
    We caption each image and compute the AMBER generative metrics following the
    official ``inference.py`` logic:

        CHAIR (Hal-i) = mentioned-hallucinatory-objects / mentioned-objects
        Cover         = mentioned-truth-objects / all-truth-objects
        Hal (Hal-s)   = fraction of responses containing >=1 hallucinatory object
        Cog           = mentioned "cognition" hallucinations / all-truth-objects
                        (here approximated as hallucinated objects that ALSO appear in
                         the hallu list -- the AMBER cognition word set)

Run::

    python -m cmpsa.eval.eval_amber --task all --model llava-1.5-7b --limit 6
"""
from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, load_json, set_seed

from cmpsa.eval.common import (
    add_common_eval_args,
    apply_limit,
    build_method_for,
    compute_and_save_metrics,
    load_image,
    save_predictions,
)

_LOG = get_logger("cmpsa.eval.amber")
BENCH = "amber"

# dimension -> standard prediction "type"
_DIM_TYPE = {"existence": "object", "attribute": "attribute", "relation": "relation"}


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def _load_annotations() -> Dict[int, Dict[str, Any]]:
    """{id -> annotation dict} from AMBER annotations.json."""
    if not paths.AMBER_ANN.exists():
        raise RuntimeError(f"AMBER annotations missing: {paths.AMBER_ANN}")
    anns = load_json(paths.AMBER_ANN)
    out: Dict[int, Dict[str, Any]] = {}
    for a in anns:
        out[int(a["id"])] = a
    return out


def _amber_image_path(image_field: Optional[str], item_id: Optional[int]):
    """Resolve an AMBER image path from the item's ``image`` field or its id."""
    if image_field:
        return paths.AMBER_IMAGES / image_field
    return paths.AMBER_IMAGES / f"AMBER_{int(item_id)}.jpg"


# --------------------------------------------------------------------------- #
# Discriminative task
# --------------------------------------------------------------------------- #
def _truth_to_yesno(value: Any) -> Optional[str]:
    s = str(value).strip().lower()
    if s in ("yes", "no"):
        return s
    return None


def _run_discriminative(
    m: Any, anns: Dict[int, Dict[str, Any]], limit: Optional[int]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dim_files = {
        "existence": paths.AMBER_Q_EXISTENCE,
        "attribute": paths.AMBER_Q_ATTRIBUTE,
        "relation": paths.AMBER_Q_RELATION,
    }
    for dim, qpath in dim_files.items():
        if not qpath.exists():
            _LOG.warning("AMBER %s questions missing: %s (skipping)", dim, qpath)
            continue
        items = apply_limit(load_json(qpath), limit)
        for it in items:
            qid = int(it["id"])
            ann = anns.get(qid, {})
            gt = _truth_to_yesno(ann.get("truth")) if ann else None
            img_path = _amber_image_path(it.get("image"), qid)
            if not img_path.exists():
                _LOG.warning("AMBER image missing: %s (skipping id=%s)", img_path, qid)
                continue
            image = load_image(img_path)
            pred, _conf = m.answer_yes_no(image, it["query"])
            rows.append(
                {
                    "id": f"disc_{qid}",
                    "image": img_path.name,
                    "question": it["query"],
                    "gt": gt,
                    "pred": pred,
                    "label": (1 if gt == "yes" else 0) if gt in ("yes", "no") else None,
                    "type": _DIM_TYPE[dim],
                    "subset": "discriminative",
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Generative task
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return [t for t in text.split() if t]


def _mention_set(text: str, vocabulary: List[str]) -> set:
    """Return which vocabulary words (single or multi-word) appear in ``text``."""
    norm = " " + " ".join(_tokenize(text)) + " "
    found = set()
    for word in vocabulary:
        w = word.strip().lower()
        if not w:
            continue
        if " " in w:
            if f" {w} " in norm:
                found.add(w)
        else:
            if f" {w} " in norm:
                found.add(w)
    return found


def _run_generative(
    m: Any, anns: Dict[int, Dict[str, Any]], limit: Optional[int]
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not paths.AMBER_Q_GENERATIVE.exists():
        raise RuntimeError(f"AMBER generative queries missing: {paths.AMBER_Q_GENERATIVE}")
    items = apply_limit(load_json(paths.AMBER_Q_GENERATIVE), limit)

    rows: List[Dict[str, Any]] = []
    sum_cover = 0.0
    n_cover = 0
    n_mentioned_total = 0
    n_hallu_mentions = 0
    n_cog_mentions = 0
    n_with_hallu = 0
    n_resp = 0

    for it in items:
        qid = int(it["id"])
        ann = anns.get(qid, {})
        truth_words = [str(w).lower() for w in ann.get("truth", [])]
        hallu_words = [str(w).lower() for w in ann.get("hallu", [])]

        img_path = _amber_image_path(it.get("image"), qid)
        if not img_path.exists():
            _LOG.warning("AMBER image missing: %s (skipping id=%s)", img_path, qid)
            continue

        image = load_image(img_path)
        caption = m.caption(image, it.get("query", "Describe this image."))

        mentioned_truth = _mention_set(caption, truth_words)
        mentioned_hallu = _mention_set(caption, hallu_words)
        n_mentioned = len(mentioned_truth) + len(mentioned_hallu)

        n_resp += 1
        if truth_words:
            sum_cover += len(mentioned_truth) / len(truth_words)
            n_cover += 1
        n_mentioned_total += n_mentioned
        n_hallu_mentions += len(mentioned_hallu)
        # "cognition" hallucinations: hallucinatory words that are in the hallu list.
        n_cog_mentions += len(mentioned_hallu)
        if mentioned_hallu:
            n_with_hallu += 1

        rows.append(
            {
                "id": f"gen_{qid}",
                "image": img_path.name,
                "question": it.get("query", "Describe this image."),
                "gt": ",".join(sorted(truth_words)),
                "pred": caption,
                "label": None,
                "type": "object",
                "subset": "generative",
            }
        )

    chair = (n_hallu_mentions / n_mentioned_total) if n_mentioned_total else 0.0
    cover = (sum_cover / n_cover) if n_cover else 0.0
    hal = (n_with_hallu / n_resp) if n_resp else 0.0
    cog = (n_cog_mentions / n_mentioned_total) if n_mentioned_total else 0.0
    metrics = {
        "amber_chair": float(chair),   # hallucination rate over mentioned objects
        "amber_cover": float(cover),
        "amber_hal": float(hal),
        "amber_cog": float(cog),
    }
    return rows, metrics


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(model: str, method: str, limit: Optional[int], cfg: Any, task: str = "all") -> Dict[str, Any]:
    """Functional entry point used by ``run_all`` and ``__main__``.

    ``task`` in {"generative","discriminative","all"}; ``run_all`` calls with the
    default "all".
    """
    set_seed(getattr(cfg, "seed", 42))
    anns = _load_annotations()
    m = build_method_for(model, method, cfg)

    rows: List[Dict[str, Any]] = []
    extra: Dict[str, float] = {}

    if task in ("discriminative", "all"):
        rows.extend(_run_discriminative(m, anns, limit))
    if task in ("generative", "all"):
        gen_rows, gen_metrics = _run_generative(m, anns, limit)
        rows.extend(gen_rows)
        extra.update(gen_metrics)

    if not rows:
        raise RuntimeError(f"AMBER produced no rows for task={task}. Check data paths.")

    save_predictions(rows, BENCH, model, method)
    return compute_and_save_metrics(BENCH, model, method, rows, extra_metrics=extra)


def main() -> None:
    parser = argparse.ArgumentParser(description="AMBER hallucination eval")
    cfg = load_config()
    add_common_eval_args(parser, cfg.mllm.key)
    parser.add_argument(
        "--task", choices=["generative", "discriminative", "all"], default="all"
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    run(args.model, args.method, args.limit, cfg, task=args.task)


if __name__ == "__main__":
    main()
