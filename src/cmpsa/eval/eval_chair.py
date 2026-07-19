"""CHAIR evaluation (Caption Hallucination Assessment).

On a subset of COCO val2014 images we generate a caption with ``method.caption`` and
score object hallucination with the CHAIR metric:

    CHAIR-i = (# hallucinated object mentions) / (# all object mentions)
    CHAIR-s = (# captions with >=1 hallucinated object) / (# captions)

Ground-truth objects per image come from the COCO ``instances_val2014`` annotations
(80 categories) *and* the gold captions, mapped through a synonym dictionary so that
e.g. "man"/"woman"/"boy" all resolve to the COCO category "person".

This is a compact, self-contained CHAIR implementation. The synonym table below is a
minimal version of the one shipped with the official ``chair.py``
(https://github.com/LisaAnne/Hallucination). For paper-grade numbers you can drop in
the official ``chair.py`` and call its ``CHAIR`` class instead -- the prediction rows
written here already contain everything that file needs (image id + generated caption).

Run::

    python -m cmpsa.eval.eval_chair --model llava-1.5-7b --method vanilla --limit 4
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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

_LOG = get_logger("cmpsa.eval.chair")
BENCH = "chair"

DEFAULT_PROMPT = "Describe this image in detail."

# --------------------------------------------------------------------------- #
# Minimal COCO synonym table (subset of the official chair.py mapping).
# Maps a surface word -> canonical COCO category. Replace with the official
# synonyms.txt for full coverage.
# --------------------------------------------------------------------------- #
_SYNONYMS: Dict[str, str] = {
    # people
    "person": "person", "girl": "person", "boy": "person", "man": "person",
    "woman": "person", "men": "person", "women": "person", "people": "person",
    "child": "person", "kid": "person", "baby": "person", "lady": "person",
    "guy": "person", "player": "person", "rider": "person",
    # vehicles
    "bicycle": "bicycle", "bike": "bicycle",
    "car": "car", "automobile": "car",
    "motorcycle": "motorcycle", "motorbike": "motorcycle",
    "airplane": "airplane", "plane": "airplane", "aeroplane": "airplane",
    "bus": "bus", "train": "train", "truck": "truck",
    "boat": "boat", "ship": "boat",
    # outdoor
    "traffic light": "traffic light", "fire hydrant": "fire hydrant",
    "stop sign": "stop sign", "parking meter": "parking meter", "bench": "bench",
    # animals
    "bird": "bird", "cat": "cat", "kitten": "cat", "dog": "dog", "puppy": "dog",
    "horse": "horse", "sheep": "sheep", "lamb": "sheep", "cow": "cow", "cattle": "cow",
    "elephant": "elephant", "bear": "bear", "zebra": "zebra", "giraffe": "giraffe",
    # accessories
    "backpack": "backpack", "umbrella": "umbrella", "handbag": "handbag",
    "purse": "handbag", "tie": "tie", "suitcase": "suitcase", "luggage": "suitcase",
    # sports
    "frisbee": "frisbee", "skis": "skis", "ski": "skis", "snowboard": "snowboard",
    "sports ball": "sports ball", "ball": "sports ball", "kite": "kite",
    "baseball bat": "baseball bat", "bat": "baseball bat", "baseball glove": "baseball glove",
    "skateboard": "skateboard", "surfboard": "surfboard", "tennis racket": "tennis racket",
    "racket": "tennis racket",
    # kitchen
    "bottle": "bottle", "wine glass": "wine glass", "cup": "cup", "mug": "cup",
    "fork": "fork", "knife": "knife", "spoon": "spoon", "bowl": "bowl",
    # food
    "banana": "banana", "apple": "apple", "sandwich": "sandwich", "orange": "orange",
    "broccoli": "broccoli", "carrot": "carrot", "hot dog": "hot dog", "hotdog": "hot dog",
    "pizza": "pizza", "donut": "donut", "doughnut": "donut", "cake": "cake",
    # furniture
    "chair": "chair", "couch": "couch", "sofa": "couch",
    "potted plant": "potted plant", "plant": "potted plant",
    "bed": "bed", "dining table": "dining table", "table": "dining table",
    "toilet": "toilet",
    # electronics
    "tv": "tv", "television": "tv", "laptop": "laptop", "mouse": "mouse",
    "remote": "remote", "keyboard": "keyboard", "cell phone": "cell phone",
    "phone": "cell phone", "smartphone": "cell phone",
    # appliances
    "microwave": "microwave", "oven": "oven", "toaster": "toaster", "sink": "sink",
    "refrigerator": "refrigerator", "fridge": "refrigerator",
    # indoor
    "book": "book", "clock": "clock", "vase": "vase", "scissors": "scissors",
    "teddy bear": "teddy bear", "teddy": "teddy bear",
    "hair drier": "hair drier", "hairdryer": "hair drier", "hair dryer": "hair drier",
    "toothbrush": "toothbrush",
}

# Multi-word synonyms checked first (longest match wins).
_MULTI_WORD = sorted([k for k in _SYNONYMS if " " in k], key=lambda s: -len(s))


def _tokenize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_objects(text: str) -> Set[str]:
    """Return the set of COCO categories mentioned in ``text``.

    Multi-word phrases are matched first (longest-first) and their span is removed
    from the text before single-token matching, so e.g. "teddy bear" maps only to
    the "teddy bear" category and does not also spuriously fire "bear". This mirrors
    the official chair.py ``double_word_dict`` handling.
    """
    norm = " " + _tokenize(text) + " "
    found: Set[str] = set()
    # multi-word phrases first; consume their span to avoid sub-word double counts
    for phrase in _MULTI_WORD:
        token = f" {phrase} "
        if token in norm:
            found.add(_SYNONYMS[phrase])
            norm = norm.replace(token, "  ")
    # single tokens over the remaining text
    for tok in norm.split():
        if tok in _SYNONYMS:
            found.add(_SYNONYMS[tok])
    return found


def _count_object_mentions(text: str) -> List[str]:
    """Return the list (with multiplicity per distinct category) of mentioned cats.

    CHAIR counts unique objects *per caption*, so we return the unique set as a list.
    """
    return sorted(_extract_objects(text))


# --------------------------------------------------------------------------- #
# Ground-truth objects from COCO annotations.
# --------------------------------------------------------------------------- #
def _load_gt_objects() -> Tuple[Dict[int, Set[str]], List[int]]:
    """Build {image_id -> set(COCO category names)} from instances + captions.

    Returns the map and the list of image ids that have annotations (the eval pool).
    """
    inst = load_json(paths.COCO_INSTANCES_VAL2014)
    cat_id_to_name = {c["id"]: c["name"] for c in inst["categories"]}

    gt: Dict[int, Set[str]] = defaultdict(set)
    for ann in inst["annotations"]:
        name = cat_id_to_name.get(ann["category_id"])
        if name:
            gt[ann["image_id"]].add(name)

    # Augment with objects mentioned in the gold captions (chair.py does this too).
    try:
        caps = load_json(paths.COCO_CAPTIONS_VAL2014)
        for ann in caps["annotations"]:
            objs = _extract_objects(ann["caption"])
            if objs:
                gt[ann["image_id"]].update(objs)
    except Exception as exc:  # captions optional
        _LOG.warning("could not load COCO captions for GT augmentation: %s", exc)

    image_ids = sorted(gt.keys())
    return gt, image_ids


def run(model: str, method: str, limit: Optional[int], cfg: Any) -> Dict[str, Any]:
    """Functional entry point used by ``run_all`` and ``__main__``."""
    set_seed(getattr(cfg, "seed", 42))

    if not paths.COCO_INSTANCES_VAL2014.exists():
        raise RuntimeError(
            f"COCO instances annotation missing: {paths.COCO_INSTANCES_VAL2014}. "
            "CHAIR needs the COCO val2014 instance annotations."
        )

    gt_objects, image_ids = _load_gt_objects()
    image_ids = apply_limit(image_ids, limit)
    if not image_ids:
        raise RuntimeError("No COCO val2014 images with annotations found for CHAIR.")

    prompt = getattr(getattr(cfg, "eval", object()), "chair_prompt", DEFAULT_PROMPT)
    m = build_method_for(model, method, cfg)

    rows: List[Dict[str, Any]] = []
    n_hallucinated_mentions = 0
    n_total_mentions = 0
    n_hallucinated_caps = 0
    n_caps = 0

    for image_id in image_ids:
        img_path = paths.coco_val2014_image(image_id)
        if not img_path.exists():
            _LOG.warning("image missing, skipping: %s", img_path)
            continue
        image = load_image(img_path)
        caption = m.caption(image, prompt)

        mentioned = _count_object_mentions(caption)
        gold = gt_objects.get(image_id, set())
        hallucinated = [o for o in mentioned if o not in gold]

        n_caps += 1
        n_total_mentions += len(mentioned)
        n_hallucinated_mentions += len(hallucinated)
        if hallucinated:
            n_hallucinated_caps += 1

        rows.append(
            {
                "id": str(image_id),
                "image": img_path.name,
                "question": prompt,
                "gt": ",".join(sorted(gold)),
                "pred": caption,
                "label": None,
                "type": "object",
                "subset": None,
            }
        )

    chair_i = (n_hallucinated_mentions / n_total_mentions) if n_total_mentions else 0.0
    chair_s = (n_hallucinated_caps / n_caps) if n_caps else 0.0
    extra = {
        "chair_i": float(chair_i),
        "chair_s": float(chair_s),
        "avg_objects_per_caption": float(n_total_mentions / n_caps) if n_caps else 0.0,
        "n_captions": float(n_caps),
    }

    save_predictions(rows, BENCH, model, method)
    return compute_and_save_metrics(BENCH, model, method, rows, extra_metrics=extra)


def main() -> None:
    parser = argparse.ArgumentParser(description="CHAIR caption hallucination eval")
    cfg = load_config()
    add_common_eval_args(parser, cfg.mllm.key)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="captioning prompt")
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
    # stash the prompt onto cfg.eval for run()
    try:
        cfg.eval.chair_prompt = args.prompt
    except Exception:
        pass
    run(args.model, args.method, args.limit, cfg)


if __name__ == "__main__":
    main()
