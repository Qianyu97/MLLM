"""Decode embedded image bytes out of the project parquet files.

Three sources are handled, selected with ``--which``:

* ``hallusion``  HallusionBench: write each embedded PNG under
  :data:`cmpsa.paths.HALLUSION_IMAGES` and emit one standardized meta row per
  question to :data:`cmpsa.paths.HALLUSION_META` (jsonl). Standard meta keys:
  ``question / gt_answer / category / filename`` (plus a few useful extras).

* ``rlhf``       RLHF-V: the ``text`` column is a JSON *string* with keys
  ``question / chosen / rejected``; image bytes live in ``image.bytes``.

* ``rlaif``      RLAIF-V: ``chosen`` / ``rejected`` are their own columns and
  ``question`` is a plain column; image bytes live in ``image.bytes``.

Both preference sources are normalized into one unified jsonl
(``derived/preferences/<source>.jsonl``) consumed by ``train_hhd``. Images are
only materialized for the preference sets when ``--export-images`` is given
(they are big: RLAIF-V is ~12 GB across 14 shards).

This script uses **pyarrow only** to read parquet and **PIL** to (re)encode
images. It must run on a CPU box, so it never imports torch / transformers.

Run::

    python -m cmpsa.data.parquet_to_images --which all --limit 50
    python -m cmpsa.data.parquet_to_images --which hallusion
    python -m cmpsa.data.parquet_to_images --which rlaif --export-images --limit 100
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed, write_jsonl

LOGGER = get_logger("cmpsa.data.parquet_to_images")

# Unified preference jsonl lives under derived/ next to the other built subsets.
PREF_DIR = paths.DERIVED / "preferences"


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _iter_parquet_rows(path: Path, limit: int | None = None) -> Iterator[dict]:
    """Yield rows of a parquet file as plain python dicts (streamed by row group)."""
    pf = pq.ParquetFile(str(path))
    seen = 0
    for batch in pf.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield row
            seen += 1
            if limit is not None and seen >= limit:
                return


def _image_bytes(cell: Any) -> bytes | None:
    """Extract raw image bytes from a HF ``image`` struct ({bytes,path}) or raw bytes."""
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
    return None


def _save_png(raw: bytes, out_path: Path) -> bool:
    """Re-encode arbitrary image bytes to PNG at ``out_path``. Returns success."""
    try:
        from PIL import Image  # local import: PIL is a data-prep dependency, not torch
    except Exception as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "Pillow (PIL) is required to decode/encode images: pip install pillow"
        ) from exc
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        LOGGER.warning("could not decode image bytes for %s: %s", out_path.name, exc)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return True


# --------------------------------------------------------------------------- #
# HallusionBench
# --------------------------------------------------------------------------- #
def convert_hallusion(limit: int | None = None) -> int:
    """Decode HallusionBench parquet -> PNG images + standardized meta jsonl."""
    src = paths.HALLUSION_PARQUET
    if not src.exists():
        LOGGER.error("HallusionBench parquet not found: %s", src)
        return 0

    paths.HALLUSION_IMAGES.mkdir(parents=True, exist_ok=True)
    meta_rows: list[dict] = []
    n_img = 0

    for i, row in enumerate(_iter_parquet_rows(src, limit)):
        # Build a stable filename. Prefer the dataset-provided "filename"; else
        # synthesize from category/set/figure ids so we never collide.
        filename = row.get("filename")
        if not filename:
            cat = str(row.get("category", "cat"))
            sid = str(row.get("set_id", "s"))
            fid = str(row.get("figure_id", "f"))
            qid = str(row.get("question_id", i))
            filename = f"{cat}_{sid}_{fid}_{qid}.png"
        # normalize to .png basename only (HallusionBench filenames are relative paths)
        stem = Path(str(filename)).stem
        png_name = f"{stem}.png"

        raw = _image_bytes(row.get("image"))
        img_written = False
        if raw is not None:
            out_path = paths.HALLUSION_IMAGES / png_name
            if not out_path.exists():
                img_written = _save_png(raw, out_path)
            else:
                img_written = True
            if img_written:
                n_img += 1
        else:
            # non-image samples exist in HallusionBench (text-only set); keep meta.
            LOGGER.debug("row %d has no embedded image (filename=%s)", i, filename)

        gt_raw = row.get("gt_answer")
        meta_rows.append(
            {
                "filename": png_name,
                "question": row.get("question"),
                "gt_answer": str(gt_raw) if gt_raw is not None else None,
                "gt_answer_details": row.get("gt_answer_details"),
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "set_id": row.get("set_id"),
                "figure_id": row.get("figure_id"),
                "question_id": row.get("question_id"),
                "has_image": bool(img_written),
            }
        )

    n_meta = write_jsonl(meta_rows, paths.HALLUSION_META)
    LOGGER.info(
        "HallusionBench: wrote %d images -> %s, %d meta rows -> %s",
        n_img, paths.HALLUSION_IMAGES, n_meta, paths.HALLUSION_META,
    )
    return n_meta


# --------------------------------------------------------------------------- #
# Preference sets (RLHF-V / RLAIF-V) -> unified jsonl
# --------------------------------------------------------------------------- #
def _unified_pref_row(
    source: str, idx: int, question: str | None, chosen: str | None,
    rejected: str | None, image_rel: str | None,
) -> dict:
    """One unified preference record consumed by train_hhd."""
    return {
        "id": f"{source}-{idx}",
        "source": source,
        "image": image_rel,          # relative png name if exported, else None
        "question": question,
        "chosen": chosen,
        "rejected": rejected,
    }


def convert_rlhf(limit: int | None = None, export_images: bool = False) -> int:
    """RLHF-V parquet -> unified preference jsonl (text column is a JSON string)."""
    src = paths.RLHF_V_PARQUET
    if not src.exists():
        LOGGER.error("RLHF-V parquet not found: %s", src)
        return 0

    img_dir = PREF_DIR / "rlhf_v_images"
    rows: list[dict] = []
    n_img = 0

    for i, row in enumerate(_iter_parquet_rows(src, limit)):
        text = row.get("text")
        question = chosen = rejected = None
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                question = parsed.get("question")
                chosen = parsed.get("chosen")
                rejected = parsed.get("rejected")
            except Exception as exc:
                LOGGER.warning("RLHF-V row %d: bad text JSON: %s", i, exc)
        elif isinstance(text, dict):  # tolerate already-parsed dicts
            question = text.get("question")
            chosen = text.get("chosen")
            rejected = text.get("rejected")

        image_rel = None
        if export_images:
            raw = _image_bytes(row.get("image"))
            if raw is not None:
                name = f"rlhf_v_{row.get('idx', i)}.png"
                if _save_png(raw, img_dir / name):
                    image_rel = str((img_dir / name))
                    n_img += 1

        rows.append(_unified_pref_row("rlhf_v", row.get("idx", i),
                                      question, chosen, rejected, image_rel))

    out = PREF_DIR / "rlhf_v.jsonl"
    n = write_jsonl(rows, out)
    LOGGER.info("RLHF-V: %d preference pairs -> %s (images exported: %d)", n, out, n_img)
    return n


def convert_rlaif(limit: int | None = None, export_images: bool = False) -> int:
    """RLAIF-V parquet shards -> unified preference jsonl (separate columns)."""
    shards = paths.rlaif_v_parquets()
    if not shards:
        LOGGER.error("no RLAIF-V parquet shards under %s", paths.RLAIF_V_DIR)
        return 0

    img_dir = PREF_DIR / "rlaif_v_images"
    rows: list[dict] = []
    n_img = 0
    seen = 0

    for shard in shards:
        if limit is not None and seen >= limit:
            break
        remaining = None if limit is None else (limit - seen)
        for row in _iter_parquet_rows(shard, remaining):
            idx = row.get("idx", seen)
            image_rel = None
            if export_images:
                raw = _image_bytes(row.get("image"))
                if raw is not None:
                    name = f"rlaif_v_{idx}.png"
                    if _save_png(raw, img_dir / name):
                        image_rel = str((img_dir / name))
                        n_img += 1
            rows.append(_unified_pref_row(
                "rlaif_v", idx, row.get("question"),
                row.get("chosen"), row.get("rejected"), image_rel))
            seen += 1
            if limit is not None and seen >= limit:
                break

    out = PREF_DIR / "rlaif_v.jsonl"
    n = write_jsonl(rows, out)
    LOGGER.info("RLAIF-V: %d preference pairs from %d shard(s) -> %s (images: %d)",
                n, len(shards), out, n_img)
    return n


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["hallusion", "rlhf", "rlaif", "all"],
                    default="all", help="which parquet source(s) to decode")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per source (smoke test). default: all rows")
    ap.add_argument("--export-images", action="store_true",
                    help="also materialize preference-set images (large!)")
    ap.add_argument("--config", default=None, help="optional config override yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    PREF_DIR.mkdir(parents=True, exist_ok=True)

    which = args.which
    if which in ("hallusion", "all"):
        convert_hallusion(args.limit)
    if which in ("rlhf", "all"):
        convert_rlhf(args.limit, args.export_images)
    if which in ("rlaif", "all"):
        convert_rlaif(args.limit, args.export_images)


if __name__ == "__main__":
    main()
