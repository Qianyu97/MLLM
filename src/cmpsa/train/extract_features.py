"""Stage 0 -- feature extraction / caching.

Pre-compute and cache the per-patch visual features and per-token text features
that every downstream training stage consumes, so that the heavy backbones only
run once.

Visual backbone : CLIP-ViT-L/14-336 (``cfg.visual_backbone.key``).
    We keep ``last_hidden_state`` -> ``[N_patches(+cls), feature_dim]`` per image.
    Cached under :data:`cmpsa.paths.CLIP_FEATURES` as ``<image_id>.pt`` (torch) or
    ``<image_id>.npy`` (numpy) depending on ``--format``.

Text backbone : ``cfg.text_backbone.key`` (Llama-2-7B by default).
    We keep the per-token hidden states of the last layer ->
    ``[N_tokens, feature_dim]`` per caption / sentence.
    Cached under :data:`cmpsa.paths.LLAMA_FEATURES`.

Run as::

    python -m cmpsa.train.extract_features --backbone clip --split train2017 --limit 8
    python -m cmpsa.train.extract_features --backbone llama --split train2017 --limit 8

``torch`` / ``transformers`` are imported lazily inside the worker functions, so
``--help`` works without a GPU and ``ast.parse`` succeeds with the libs absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed

LOG = get_logger("cmpsa.extract_features")

# Splits we know how to enumerate -> a directory of jpg/png images.
_IMAGE_SPLITS = {
    "val2014": paths.COCO_VAL2014,
    "val2017": paths.COCO_VAL2017,
    "train2017": paths.COCO_TRAIN2017,
    "vg": paths.VG_100K,
    "vg2": paths.VG_100K_2,
}


# --------------------------------------------------------------------------- #
# Helpers (no heavy imports)
# --------------------------------------------------------------------------- #
def _image_id_from_path(p: Path) -> str:
    """Stable id used as the cache filename stem.

    For COCO ``COCO_val2014_000000123456.jpg`` -> ``000000123456`` (the numeric
    tail); for VG ``12345.jpg`` -> ``12345``. Falls back to the file stem.
    """
    stem = p.stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            return tail  # keep zero-padding so the id round-trips to the file name
    return stem


def iter_split_images(split: str, limit: int | None = None) -> Iterator[Path]:
    """Yield image paths for an image split, sorted for determinism."""
    if split not in _IMAGE_SPLITS:
        raise ValueError(
            f"unknown image split {split!r}; known: {sorted(_IMAGE_SPLITS)}"
        )
    root = _IMAGE_SPLITS[split]
    if not root.exists():
        LOG.warning("split dir does not exist: %s", root)
        return
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in sorted(root.glob(ext)):
            yield p
            n += 1
            if limit is not None and n >= limit:
                return


def _coco_caption_iter(captions_json: Path, limit: int | None) -> Iterator[tuple[str, str]]:
    """Yield ``(caption_id, text)`` pairs from a COCO captions annotation file."""
    if not captions_json.exists():
        LOG.warning("captions file missing: %s", captions_json)
        return
    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data.get("annotations", []) if isinstance(data, dict) else []
    for i, a in enumerate(anns):
        if limit is not None and i >= limit:
            return
        yield str(a.get("id", i)), str(a.get("caption", "")).strip()


def iter_split_texts(split: str, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield ``(text_id, text)`` for the text backbone.

    For COCO splits we read the matching captions annotation; this gives a
    self-contained, dependency-free text source for the text backbone cache.
    """
    cap = {
        "val2014": paths.COCO_CAPTIONS_VAL2014,
        "val2017": paths.COCO_CAPTIONS_VAL2017,
        "train2017": paths.COCO_CAPTIONS_TRAIN2017,
    }.get(split)
    if cap is None:
        LOG.warning("no caption source wired for split %r; nothing to extract", split)
        return
    yield from _coco_caption_iter(cap, limit)


# --------------------------------------------------------------------------- #
# Backbone loaders (lazy heavy imports)
# --------------------------------------------------------------------------- #
def _resolve_local_dir(cfg, model_key: str) -> Path:
    """Resolve the on-disk weight dir for a model registry key."""
    entry = getattr(cfg.models, model_key, None)
    if entry is None:
        raise KeyError(f"model key {model_key!r} not in configs/models.yaml")
    return paths.MODELS_ROOT / entry.local_dir


def _load_clip(cfg):
    """Load CLIP processor + vision model (lazy import)."""
    import torch  # noqa: F401  (imported for device side-effects/checks)
    from transformers import CLIPImageProcessor, CLIPVisionModel

    key = cfg.visual_backbone.key
    local = _resolve_local_dir(cfg, key)
    src = str(local) if local.exists() else getattr(cfg.models, key).hf_id
    LOG.info("loading CLIP vision backbone from %s", src)
    processor = CLIPImageProcessor.from_pretrained(src)
    model = CLIPVisionModel.from_pretrained(src)
    model.eval()
    return processor, model


def _load_text_backbone(cfg):
    """Load the text backbone tokenizer + model (lazy import)."""
    import torch  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    key = cfg.text_backbone.key
    local = _resolve_local_dir(cfg, key)
    src = str(local) if local.exists() else getattr(cfg.models, key).hf_id
    LOG.info("loading text backbone from %s", src)
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(src, output_hidden_states=True)
    model.eval()
    return tok, model


def _device(cfg):
    import torch

    want = getattr(cfg, "device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        LOG.warning("cfg.device=cuda but no GPU visible; falling back to cpu")
        return "cpu"
    return want


def _save_feat(arr, out: Path, fmt: str) -> None:
    """Persist a feature tensor as .pt or .npy."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        import numpy as np

        np.save(out.with_suffix(".npy"), arr.detach().cpu().numpy())
    else:
        import torch

        torch.save(arr.detach().cpu(), out.with_suffix(".pt"))


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #
def extract_clip(cfg, split: str, limit: int | None, fmt: str, overwrite: bool) -> int:
    """Cache per-patch CLIP ``last_hidden_state`` features for an image split."""
    import torch
    from PIL import Image

    out_dir = paths.CLIP_FEATURES / split
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_split_images(split, limit))
    if not images:
        LOG.warning("no images found for split %r under %s", split, _IMAGE_SPLITS.get(split))
        return 0

    processor, model = _load_clip(cfg)
    dev = _device(cfg)
    model.to(dev)

    n_done = 0
    for p in images:
        img_id = _image_id_from_path(p)
        suffix = ".npy" if fmt == "npy" else ".pt"
        out = out_dir / (img_id + suffix)
        if out.exists() and not overwrite:
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:  # corrupt / unreadable image -> skip
            LOG.warning("skip unreadable image %s: %s", p, e)
            continue
        inputs = processor(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        # [1, N_patches(+cls), feature_dim] -> [N_patches(+cls), feature_dim]
        feats = outputs.last_hidden_state.squeeze(0)
        _save_feat(feats, out_dir / img_id, fmt)
        n_done += 1
        if n_done % 100 == 0:
            LOG.info("clip: cached %d images", n_done)
    LOG.info("clip: done, cached %d feature files under %s", n_done, out_dir)
    return n_done


def extract_text(cfg, split: str, limit: int | None, fmt: str, overwrite: bool) -> int:
    """Cache per-token last-layer hidden states for the text backbone."""
    import torch

    out_dir = paths.LLAMA_FEATURES / split
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = list(iter_split_texts(split, limit))
    if not texts:
        LOG.warning("no texts found for split %r", split)
        return 0

    tok, model = _load_text_backbone(cfg)
    dev = _device(cfg)
    model.to(dev)

    n_done = 0
    for text_id, text in texts:
        if not text:
            continue
        suffix = ".npy" if fmt == "npy" else ".pt"
        out = out_dir / (text_id + suffix)
        if out.exists() and not overwrite:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
        with torch.no_grad():
            outputs = model(**enc)
        # last hidden state -> [1, N_tokens, feature_dim] -> [N_tokens, feature_dim]
        hs = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else outputs.last_hidden_state
        feats = hs.squeeze(0)
        _save_feat(feats, out_dir / text_id, fmt)
        n_done += 1
        if n_done % 200 == 0:
            LOG.info("text: cached %d texts", n_done)
    LOG.info("text: done, cached %d feature files under %s", n_done, out_dir)
    return n_done


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract & cache CLIP per-patch / text per-token features for CMPSA.",
    )
    p.add_argument(
        "--backbone",
        choices=["clip", "llama"],
        required=True,
        help="clip -> visual per-patch features; llama -> text per-token features "
        "(uses cfg.text_backbone, which may be any HF text model).",
    )
    p.add_argument(
        "--split",
        default="train2017",
        help="image/text split to extract: val2014|val2017|train2017|vg|vg2|... "
        "(image splits for clip; coco caption splits for the text backbone).",
    )
    p.add_argument("--limit", type=int, default=None, help="cap #items for a smoke test")
    p.add_argument("--format", choices=["pt", "npy"], default="pt", help="cache file format")
    p.add_argument("--overwrite", action="store_true", help="recompute even if cache exists")
    p.add_argument("--config", default=None, help="optional YAML override merged on default.yaml")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    paths.ensure_dirs()

    LOG.info("extract_features: backbone=%s split=%s limit=%s format=%s",
             args.backbone, args.split, args.limit, args.format)

    if args.backbone == "clip":
        n = extract_clip(cfg, args.split, args.limit, args.format, args.overwrite)
    else:  # "llama" -> generic text backbone from cfg.text_backbone
        n = extract_text(cfg, args.split, args.limit, args.format, args.overwrite)

    LOG.info("extract_features: finished, %d files written", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
