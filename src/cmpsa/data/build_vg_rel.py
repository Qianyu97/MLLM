"""Build an internal *relation probe* subset from Visual Genome relationships.

WARNING / SCOPE
---------------
Visual Genome predicate annotations are **crowdsourced and noisy**: predicates
are free-text, inconsistently phrased, and frequently ambiguous about reference
frame (whose left? viewer or subject?). This subset is therefore only an
**internal diagnostic probe** for spatial / semantic relations. It is *not*
equivalent to, nor a drop-in replacement for, MMRel or any curated relation
benchmark. Treat absolute numbers on it as indicative, not authoritative.

What it does
------------
1. Read :data:`cmpsa.paths.VG_RELATIONSHIPS`.
2. Canonicalize high-frequency predicates via :data:`PREDICATE_CANON`
   (e.g. ``on top of`` / ``sitting on`` -> ``on``; ``to the left of`` -> ``left``).
3. For each kept relationship build a **minimal positive/negative pair**: the
   positive is the true triple; the negative flips a directional predicate
   (``on`` <-> ``under``, ``left`` <-> ``right``, ``above`` <-> ``below``,
   ``in front of`` <-> ``behind``) or, when the predicate is non-directional,
   swaps subject/object so the question becomes false.
4. Deduplicate and write standardized rows to :data:`cmpsa.paths.VG_REL_JSONL`.

Output row schema (matches the project's standard probe row)::

    {"id","image","question":"Is the <subj> <rel> the <obj>?",
     "gt":"yes"/"no","type":"relation","subset":"vg_rel"}

Run::

    python -m cmpsa.data.build_vg_rel --size 2000
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, save_json, set_seed, write_jsonl

LOGGER = get_logger("cmpsa.data.build_vg_rel")

# --------------------------------------------------------------------------- #
# Predicate canonicalization: many free-text phrasings -> a small closed set.
# --------------------------------------------------------------------------- #
PREDICATE_CANON: dict[str, str] = {
    # --- on / support ---
    "on": "on", "on top of": "on", "ontop of": "on", "sitting on": "on",
    "standing on": "on", "laying on": "on", "lying on": "on", "resting on": "on",
    "mounted on": "on", "on a": "on", "atop": "on",
    # --- under / below ---
    "under": "under", "underneath": "under", "below": "under", "beneath": "under",
    # --- above ---
    "above": "above", "over": "above", "on top": "above",
    # --- left / right (viewer frame) ---
    "left of": "left", "to the left of": "left", "on the left of": "left",
    "right of": "right", "to the right of": "right", "on the right of": "right",
    # --- front / behind ---
    "in front of": "in front of", "front of": "in front of",
    "behind": "behind", "in back of": "behind", "at the back of": "behind",
    # --- containment / attachment (non-directional, swap to negate) ---
    "in": "in", "inside": "in", "inside of": "in", "within": "in",
    "next to": "next to", "beside": "next to", "near": "next to",
    "by": "next to", "adjacent to": "next to",
    "holding": "holding", "holds": "holding",
    "wearing": "wearing", "wears": "wearing", "has on": "wearing",
    "attached to": "attached to", "connected to": "attached to",
    "covered by": "covered by", "covered with": "covered by",
    "covering": "covering",
}

# Directional predicates -> the predicate that makes the SAME triple false.
DIRECTIONAL_OPPOSITE: dict[str, str] = {
    "on": "under",
    "under": "on",
    "above": "under",
    "left": "right",
    "right": "left",
    "in front of": "behind",
    "behind": "in front of",
    "covered by": "covering",
    "covering": "covered by",
}

# Predicates kept but negated by swapping subject<->object (symmetric phrasing
# would still read false because the entities differ).
SWAP_NEGATE = {"in", "next to", "holding", "wearing", "attached to"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _entity_name(ent: dict) -> str | None:
    """VG subject/object dicts carry either 'name' (str) or 'names' (list)."""
    if not isinstance(ent, dict):
        return None
    if isinstance(ent.get("name"), str) and ent["name"].strip():
        return ent["name"].strip().lower()
    names = ent.get("names")
    if isinstance(names, list) and names:
        first = str(names[0]).strip().lower()
        return first or None
    return None


def _canon_predicate(pred: Any) -> str | None:
    if not isinstance(pred, str):
        return None
    key = pred.strip().lower()
    return PREDICATE_CANON.get(key)


def _question(subj: str, rel: str, obj: str) -> str:
    return f"Is the {subj} {rel} the {obj}?"


def build(size: int) -> int:
    """Generate up to ``size`` pos/neg relation rows from VG and write the jsonl."""
    src = paths.VG_RELATIONSHIPS
    if not src.exists():
        LOGGER.error("VG relationships.json not found: %s", src)
        return 0

    LOGGER.info("loading %s (large file, please wait)...", src)
    # json is fine here; relationships.json is a single big array.
    import json
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    LOGGER.info("loaded %d image entries", len(data))

    paths.VG_REL_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_pos: set[tuple] = set()       # (image_id, subj, rel, obj) dedup
    per_image_cap = 6                  # keep diversity across images
    pos_target = max(1, size // 2)     # we emit one pos + one neg per accepted triple

    # Process images in order; sampling is implicitly spread because we cap per image.
    for entry in data:
        if len(seen_pos) >= pos_target:
            break
        image_id = entry.get("image_id")
        if image_id is None:
            continue
        kept_this_image = 0
        for rel in entry.get("relationships", []):
            if kept_this_image >= per_image_cap or len(seen_pos) >= pos_target:
                break
            crel = _canon_predicate(rel.get("predicate"))
            if crel is None:
                continue
            subj = _entity_name(rel.get("subject", {}))
            obj = _entity_name(rel.get("object", {}))
            if not subj or not obj or subj == obj:
                continue

            key = (image_id, subj, crel, obj)
            if key in seen_pos:
                continue

            # Build the negative.
            if crel in DIRECTIONAL_OPPOSITE:
                neg_rel, neg_subj, neg_obj = DIRECTIONAL_OPPOSITE[crel], subj, obj
            elif crel in SWAP_NEGATE:
                neg_rel, neg_subj, neg_obj = crel, obj, subj
            else:
                # non-directional and not in swap set: skip (can't reliably negate)
                continue

            seen_pos.add(key)
            kept_this_image += 1
            img_str = str(paths.vg_image(image_id))
            base = f"vgrel-{image_id}-{len(seen_pos)}"

            rows.append({
                "id": f"{base}-pos",
                "image": img_str,
                "question": _question(subj, crel, obj),
                "gt": "yes",
                "type": "relation",
                "subset": "vg_rel",
            })
            rows.append({
                "id": f"{base}-neg",
                "image": img_str,
                "question": _question(neg_subj, neg_rel, neg_obj),
                "gt": "no",
                "type": "relation",
                "subset": "vg_rel",
            })

    # Final dedup on the (image, question, gt) signature.
    uniq: dict[tuple, dict] = {}
    for r in rows:
        uniq[(r["image"], r["question"], r["gt"])] = r
    final = list(uniq.values())[:size]

    n = write_jsonl(final, paths.VG_REL_JSONL)
    # tiny provenance sidecar
    save_json(
        {"subset": "vg_rel", "n": n, "source": str(src),
         "canon_predicates": sorted(set(PREDICATE_CANON.values())),
         "note": "internal probe only; VG predicates are noisy; not MMRel."},
        paths.VG_REL_DIR / "vg_rel_manifest.json",
    )
    LOGGER.info("VG relation probe: wrote %d rows -> %s", n, paths.VG_REL_JSONL)
    return n


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=cfg.build.vg_rel_size,
                    help="target number of probe rows (pos+neg). "
                         f"default cfg.build.vg_rel_size={cfg.build.vg_rel_size}")
    ap.add_argument("--config", default=None, help="optional config override yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    build(args.size)


if __name__ == "__main__":
    main()
