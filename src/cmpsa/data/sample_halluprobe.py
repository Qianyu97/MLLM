"""Sample the HalluProbe-VL manifest (the self-built probe split).

The manifest is a *plan*, not the final labeled data: it records which images
are selected for each sub-pool so the heavier generation / annotation steps can
run reproducibly later.

Three pools (sizes default to ``cfg.build.*``):

* ``--coco`` images: stratified sample from **COCO 2017 val** by object
  category. IMPORTANT: we deliberately use COCO **2017** val (5000 imgs), which
  is disjoint from the COCO **2014** val used by POPE / CHAIR. Sampling from a
  different split prevents test-set leakage into our internal probe.

* ``--vg`` images: sample from Visual Genome by *relationship density* (images
  with more annotated relationships, which stress relation grounding).

* ``--adv`` adversarial: a placeholder pool. We only record intended slots here;
  the adversarial images are produced later by **scene-graph counterfactual
  editing + InstructPix2Pix** (e.g. recolor / remove / move an object so the
  caption-implied scene graph changes), then human-checked. This script just
  reserves the manifest entries and documents the generation recipe.

Output: a single JSON manifest under :data:`cmpsa.paths.HALLUPROBE_SPLITS`.

Run::

    python -m cmpsa.data.sample_halluprobe                  # uses cfg.build.*
    python -m cmpsa.data.sample_halluprobe --coco 200 --vg 100 --adv 50
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, save_json, set_seed

LOGGER = get_logger("cmpsa.data.sample_halluprobe")


# --------------------------------------------------------------------------- #
# COCO 2017 val, stratified by object category
# --------------------------------------------------------------------------- #
def sample_coco(n: int, seed: int) -> list[dict]:
    """Stratified-by-category sample of COCO 2017 *val* image ids."""
    ann_path = paths.COCO_INSTANCES_VAL2017
    if not ann_path.exists():
        LOGGER.warning(
            "COCO val2017 instances json not found at %s — emitting empty COCO pool. "
            "(Expected file: instances_val2017.json next to the val2014 annotations.)",
            ann_path,
        )
        return []

    try:
        from pycocotools.coco import COCO  # local import per contract
    except Exception:
        LOGGER.warning(
            "pycocotools not installed; falling back to raw-json parse. "
            "Install with `pip install pycocotools` for the standard API."
        )
        return _sample_coco_rawjson(ann_path, n, seed)

    import random
    rng = random.Random(seed)
    coco = COCO(str(ann_path))
    cat_ids = coco.getCatIds()
    cats = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}

    # bucket image ids per category
    by_cat: dict[int, list[int]] = {cid: coco.getImgIds(catIds=[cid]) for cid in cat_ids}
    return _stratified_pick(by_cat, cats, n, rng)


def _sample_coco_rawjson(ann_path: Path, n: int, seed: int) -> list[dict]:
    """No-pycocotools fallback using the raw instances json."""
    import random
    rng = random.Random(seed)
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    by_cat: dict[int, set] = defaultdict(set)
    for a in data.get("annotations", []):
        by_cat[a["category_id"]].add(a["image_id"])
    by_cat = {k: sorted(v) for k, v in by_cat.items()}
    return _stratified_pick(by_cat, cats, n, rng)


def _stratified_pick(by_cat, cats, n, rng) -> list[dict]:
    """Round-robin across categories until ``n`` unique images are chosen."""
    order = list(by_cat.keys())
    rng.shuffle(order)
    for cid in order:
        rng.shuffle(by_cat[cid])
    chosen: dict[int, dict] = {}
    cursor = {cid: 0 for cid in order}
    progress = True
    while len(chosen) < n and progress:
        progress = False
        for cid in order:
            if len(chosen) >= n:
                break
            imgs = by_cat[cid]
            i = cursor[cid]
            if i < len(imgs):
                cursor[cid] += 1
                progress = True
                img_id = imgs[i]
                if img_id not in chosen:
                    chosen[img_id] = {
                        "pool": "coco_val2017",
                        "image_id": int(img_id),
                        # COCO2017 val filenames are zero-padded 12-digit .jpg
                        "image": str(paths.COCO_VAL2017 / f"{int(img_id):012d}.jpg"),
                        "strat_category": cats.get(cid, str(cid)),
                    }
    LOGGER.info("COCO val2017 stratified sample: %d images across %d categories",
                len(chosen), len(order))
    return list(chosen.values())


# --------------------------------------------------------------------------- #
# VG by relationship density
# --------------------------------------------------------------------------- #
def sample_vg(n: int, seed: int) -> list[dict]:
    src = paths.VG_RELATIONSHIPS
    if not src.exists():
        LOGGER.warning("VG relationships.json missing (%s) — empty VG pool.", src)
        return []
    import random
    rng = random.Random(seed)
    LOGGER.info("loading %s for relationship-density ranking...", src)
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    dens = [(e.get("image_id"), len(e.get("relationships", []))) for e in data
            if e.get("image_id") is not None]
    dens.sort(key=lambda t: t[1], reverse=True)
    # take a generous top band, then sample to avoid always picking the same head
    band = dens[: max(n * 4, n)]
    rng.shuffle(band)
    picks = band[:n]
    rows = [{
        "pool": "vg_reldensity",
        "image_id": int(img_id),
        "image": str(paths.vg_image(img_id)),
        "n_relationships": int(cnt),
    } for img_id, cnt in picks]
    LOGGER.info("VG relationship-density sample: %d images (density range %d..%d)",
                len(rows),
                picks[-1][1] if picks else 0,
                picks[0][1] if picks else 0)
    return rows


# --------------------------------------------------------------------------- #
# Adversarial placeholder
# --------------------------------------------------------------------------- #
def sample_adv(n: int) -> list[dict]:
    """Reserve adversarial slots; images generated later (see module docstring)."""
    rows = [{
        "pool": "adversarial",
        "slot": i,
        "image": None,                 # to be filled by the generation pipeline
        "status": "placeholder",
        "recipe": "scene-graph counterfactual edit + InstructPix2Pix, human-verified",
    } for i in range(n)]
    LOGGER.info("adversarial pool: reserved %d placeholder slots", n)
    return rows


def build_manifest(n_coco: int, n_vg: int, n_adv: int, seed: int) -> Path:
    paths.HALLUPROBE_SPLITS.mkdir(parents=True, exist_ok=True)
    paths.HALLUPROBE_ANN.mkdir(parents=True, exist_ok=True)
    coco = sample_coco(n_coco, seed)
    vg = sample_vg(n_vg, seed)
    adv = sample_adv(n_adv)
    manifest = {
        "name": "halluprobe_vl",
        "seed": seed,
        "note": "Self-built probe manifest (a sampling plan, not final labels). "
                "COCO pool uses COCO2017 val to avoid overlap with POPE/CHAIR "
                "(COCO2014 val). Adversarial pool is generated later via "
                "scene-graph counterfactual + InstructPix2Pix.",
        "requested": {"coco": n_coco, "vg": n_vg, "adv": n_adv},
        "counts": {"coco": len(coco), "vg": len(vg), "adv": len(adv)},
        "pools": {"coco_val2017": coco, "vg_reldensity": vg, "adversarial": adv},
    }
    out = paths.HALLUPROBE_SPLITS / "halluprobe_manifest.json"
    save_json(manifest, out)
    LOGGER.info("wrote HalluProbe manifest (coco=%d vg=%d adv=%d) -> %s",
                len(coco), len(vg), len(adv), out)
    _write_stats(manifest)
    return out


def _safe_len_json(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        obj = json.loads(text)
        return len(obj) if isinstance(obj, list) else 1
    except json.JSONDecodeError:
        return sum(1 for line in text.splitlines() if line.strip())
    except Exception:
        return 0


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    except Exception:
        return 0


def _question_lengths(limit: int = 5000) -> list[int]:
    """Collect a compact real question-length sample from available benchmarks."""
    out: list[int] = []

    def add(text) -> None:
        if text and len(out) < limit:
            out.append(len(str(text).split()))

    # POPE json/jsonl files.
    for p in paths.POPE_SUBSETS.values():
        if len(out) >= limit or not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            try:
                obj = json.loads(text)
                rows = obj if isinstance(obj, list) else [obj]
            except json.JSONDecodeError:
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            for row in rows:
                add(row.get("text") or row.get("question"))
                if len(out) >= limit:
                    break
        except Exception:
            continue

    # HallusionBench / MME parquet questions.
    for pq_path in [paths.HALLUSION_PARQUET, *paths.mme_parquets()]:
        if len(out) >= limit or not pq_path.exists():
            continue
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(pq_path))
            for batch in pf.iter_batches(batch_size=512, columns=["question"]):
                for q in batch.to_pydict().get("question", []):
                    add(q)
                    if len(out) >= limit:
                        break
                if len(out) >= limit:
                    break
        except Exception:
            continue
    return out


def _write_stats(manifest: dict) -> Path:
    """Write dataset statistics consumed by cmpsa.viz.make_figures."""
    counts = manifest.get("counts", {})
    halluprobe_total = int(counts.get("coco", 0)) + int(counts.get("vg", 0)) + int(counts.get("adv", 0))
    # These are the planned HalluProbe label counts from the adjusted proposal.
    # They are kept separate from image-pool counts in the benchmark-size panel.
    label_scale = halluprobe_total / 3500 if halluprobe_total else 1.0
    planned_labels = {
        "object": int(round(12000 * label_scale)),
        "attribute": int(round(11500 * label_scale)),
        "relation": int(round(8000 * label_scale)),
    }
    stats = {
        "halluprobe_categories": planned_labels,
        "benchmark_sizes": {
            "POPE": sum(_safe_len_json(p) for p in paths.POPE_SUBSETS.values()),
            "AMBER": len(list(paths.AMBER_IMAGES.glob("*.jpg"))) if paths.AMBER_IMAGES.exists() else 0,
            "HallusionBench": _parquet_rows(paths.HALLUSION_PARQUET),
            "MMHal": len(list(paths.MMHAL_IMAGES.glob("*.jpg"))) if paths.MMHAL_IMAGES.exists() else 0,
            "MME": sum(_parquet_rows(p) for p in paths.mme_parquets()),
            "VG-Rel": _count_lines(paths.VG_REL_JSONL),
            "VG-Attr": _count_lines(paths.VG_ATTR_JSONL),
            "HalluProbe": halluprobe_total,
        },
        "question_lengths": _question_lengths(),
        "manifest_counts": counts,
        "note": "HalluProbe category counts are planned label counts from the adjusted CMPSA proposal; benchmark sizes are real local counts.",
    }
    out = paths.HALLUPROBE_ANN / "stats.json"
    save_json(stats, out)
    LOGGER.info("wrote HalluProbe/dataset stats -> %s", out)
    return out


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", type=int, default=cfg.build.halluprobe_coco,
                    help=f"COCO2017-val images (default cfg.build.halluprobe_coco={cfg.build.halluprobe_coco})")
    ap.add_argument("--vg", type=int, default=cfg.build.halluprobe_vg,
                    help=f"VG images by rel density (default cfg.build.halluprobe_vg={cfg.build.halluprobe_vg})")
    ap.add_argument("--adv", type=int, default=cfg.build.halluprobe_adv,
                    help=f"adversarial placeholder slots (default cfg.build.halluprobe_adv={cfg.build.halluprobe_adv})")
    ap.add_argument("--config", default=None, help="optional config override yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    build_manifest(args.coco, args.vg, args.adv, cfg.seed)


if __name__ == "__main__":
    main()
