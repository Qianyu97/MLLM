"""Build an internal *attribute probe* subset from Visual Genome attributes.

WARNING / SCOPE
---------------
Like the relation probe, this is an **internal diagnostic** built from noisy,
crowdsourced Visual Genome attribute annotations. It is not a curated benchmark.
Use it to track attribute-hallucination behavior during development, not to make
headline claims.

What it does
------------
1. Read :data:`cmpsa.paths.VG_ATTRIBUTES` (objects each carry a free-text
   ``attributes`` list) and the object ``names``.
2. For an object that carries a *recognized* attribute (a color / material /
   simple state we know how to negate), build a **counterfactual minimal pair**:
   * positive: ``Is the <obj> <attr>?`` -> ``yes``
   * negative: replace ``<attr>`` with a same-class but *different* attribute via
     :data:`ATTR_ANTONYM` (e.g. ``red`` <-> ``blue``, ``wooden`` -> ``metal``)
     -> ``Is the <obj> <other>?`` -> ``no``.
3. Deduplicate and write standardized rows to :data:`cmpsa.paths.VG_ATTR_JSONL`.

Output row schema::

    {"id","image","question":"Is the <obj> <attr>?",
     "gt":"yes"/"no","type":"attribute","subset":"vg_attr"}

Run::

    python -m cmpsa.data.build_vg_attr --size 2000
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, save_json, set_seed, write_jsonl

LOGGER = get_logger("cmpsa.data.build_vg_attr")

# --------------------------------------------------------------------------- #
# Attribute classes. Each maps an attribute -> the set of *same-class* values it
# can be swapped against to produce a plausible-but-false negative.
# --------------------------------------------------------------------------- #
_COLORS = ["red", "blue", "green", "yellow", "black", "white",
           "brown", "orange", "purple", "pink", "gray", "grey"]
_MATERIALS = ["wooden", "metal", "metallic", "plastic", "glass",
              "leather", "stone", "concrete", "ceramic", "paper"]
_SIZES = ["big", "small", "large", "tiny", "huge", "little"]
_STATES_OPEN = {"open": "closed", "closed": "open"}
_STATES_ON = {"on": "off", "off": "on"}
_STATES_EMPTY = {"empty": "full", "full": "empty"}
_STATES_WET = {"wet": "dry", "dry": "wet"}

# ATTR_ANTONYM: direct opposite where one exists; else handled by same-class swap.
ATTR_ANTONYM: dict[str, str] = {}
ATTR_ANTONYM.update(_STATES_OPEN)
ATTR_ANTONYM.update(_STATES_ON)
ATTR_ANTONYM.update(_STATES_EMPTY)
ATTR_ANTONYM.update(_STATES_WET)
ATTR_ANTONYM.update({
    "tall": "short", "short": "tall",
    "long": "short", "old": "new", "new": "old",
    "clean": "dirty", "dirty": "clean",
    "light": "dark", "dark": "light",
})

# Same-class pools for swap when there is no single antonym (colors / materials).
_CLASS_POOLS: list[list[str]] = [_COLORS, _MATERIALS, _SIZES]


def _negate_attribute(attr: str, rng: random.Random) -> str | None:
    """Return a same-class but different attribute, or None if we can't negate."""
    a = attr.strip().lower()
    if a in ATTR_ANTONYM:
        return ATTR_ANTONYM[a]
    for pool in _CLASS_POOLS:
        if a in pool:
            choices = [x for x in pool if x != a and x not in ("grey", "gray")]
            # keep gray/grey from colliding as duplicates
            if a in ("gray", "grey"):
                choices = [x for x in pool if x not in ("gray", "grey")]
            if choices:
                return rng.choice(choices)
    return None


def _object_name(obj: dict) -> str | None:
    names = obj.get("names")
    if isinstance(names, list) and names:
        n = str(names[0]).strip().lower()
        return n or None
    if isinstance(obj.get("name"), str):
        return obj["name"].strip().lower() or None
    return None


def build(size: int, seed: int) -> int:
    src = paths.VG_ATTRIBUTES
    if not src.exists():
        LOGGER.error("VG attributes.json not found: %s", src)
        return 0

    LOGGER.info("loading %s (large file, please wait)...", src)
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    LOGGER.info("loaded %d image entries", len(data))

    paths.VG_ATTR_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    rows: list[dict] = []
    seen: set[tuple] = set()
    per_image_cap = 5
    pair_target = max(1, size // 2)

    for entry in data:
        if len(seen) >= pair_target:
            break
        image_id = entry.get("image_id")
        if image_id is None:
            continue
        kept = 0
        for obj in entry.get("attributes", []):
            if kept >= per_image_cap or len(seen) >= pair_target:
                break
            name = _object_name(obj)
            attrs = obj.get("attributes")
            if not name or not isinstance(attrs, list) or not attrs:
                continue
            # pick the first attribute we know how to negate
            chosen_attr = None
            neg_attr = None
            for a in attrs:
                if not isinstance(a, str):
                    continue
                cand = _negate_attribute(a, rng)
                if cand is not None and cand != a.strip().lower():
                    chosen_attr = a.strip().lower()
                    neg_attr = cand
                    break
            if chosen_attr is None:
                continue

            key = (image_id, name, chosen_attr)
            if key in seen:
                continue
            seen.add(key)
            kept += 1

            img_str = str(paths.vg_image(image_id))
            base = f"vgattr-{image_id}-{len(seen)}"
            rows.append({
                "id": f"{base}-pos",
                "image": img_str,
                "question": f"Is the {name} {chosen_attr}?",
                "gt": "yes",
                "type": "attribute",
                "subset": "vg_attr",
            })
            rows.append({
                "id": f"{base}-neg",
                "image": img_str,
                "question": f"Is the {name} {neg_attr}?",
                "gt": "no",
                "type": "attribute",
                "subset": "vg_attr",
            })

    uniq: dict[tuple, dict] = {}
    for r in rows:
        uniq[(r["image"], r["question"], r["gt"])] = r
    final = list(uniq.values())[:size]

    n = write_jsonl(final, paths.VG_ATTR_JSONL)
    save_json(
        {"subset": "vg_attr", "n": n, "source": str(src),
         "attr_classes": {"colors": _COLORS, "materials": _MATERIALS,
                          "antonyms": ATTR_ANTONYM},
         "note": "internal probe only; VG attributes are noisy crowdsourced labels."},
        paths.VG_ATTR_DIR / "vg_attr_manifest.json",
    )
    LOGGER.info("VG attribute probe: wrote %d rows -> %s", n, paths.VG_ATTR_JSONL)
    return n


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=cfg.build.vg_attr_size,
                    help="target number of probe rows (pos+neg). "
                         f"default cfg.build.vg_attr_size={cfg.build.vg_attr_size}")
    ap.add_argument("--config", default=None, help="optional config override yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    build(args.size, cfg.seed)


if __name__ == "__main__":
    main()
