"""Build real PSAS embeddings for the t-SNE paper figure.

The figure renderer expects ``cache/psas_tsne.npz`` with:

    emb      [N, D] float32 PSAS means
    labels   [N]    integer semantic class ids
    classes  [C]    class names

This script creates that file from real cached CLIP/LLaMA features and the
trained PVE/PLE heads.  It pairs COCO images with their captions, assigns each
pair a COCO object-category label, and writes both the visual PSAS point and the
language PSAS point with that semantic label.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.models.pgd_decode import _merge_peft_linear_state
from cmpsa.models.pve_ple import PLEHead, PVEHead
from cmpsa.utils import get_logger, set_seed

LOG = get_logger("cmpsa.viz.build_psas_tsne")


_SPLIT_FILES = {
    "train2017": (paths.COCO_INSTANCES_TRAIN2017, paths.COCO_CAPTIONS_TRAIN2017),
    "val2017": (paths.COCO_INSTANCES_VAL2017, paths.COCO_CAPTIONS_VAL2017),
    "val2014": (paths.COCO_INSTANCES_VAL2014, paths.COCO_CAPTIONS_VAL2014),
}


def _image_stem(image_id: int | str) -> str:
    return f"{int(image_id):012d}"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_heads(cfg, device: str):
    pj = cfg.projection
    pve = PVEHead(
        cfg.visual_backbone.feature_dim,
        pj.psas_dim,
        pj.hidden_dim,
        pj.min_logvar,
        pj.max_logvar,
    ).to(device).eval()
    ple = PLEHead(
        cfg.text_backbone.feature_dim,
        pj.psas_dim,
        pj.hidden_dim,
        pj.min_logvar,
        pj.max_logvar,
    ).to(device).eval()
    return pve, ple


def _load_projection_state(cfg, device: str):
    import torch

    pve, ple = _build_heads(cfg, device)
    for name in ("cmota.pt", "pretrain_proj.pt"):
        ckpt = paths.CKPT_DIR / name
        if not ckpt.exists():
            continue
        state = torch.load(ckpt, map_location="cpu")
        lora_rank = state.get("lora_rank")
        pve_state = _merge_peft_linear_state(state.get("pve", {}), lora_rank)
        ple_state = _merge_peft_linear_state(state.get("ple", {}), lora_rank)
        pve.load_state_dict(pve_state, strict=False)
        ple.load_state_dict(ple_state, strict=False)
        LOG.info("loaded projection heads from %s", ckpt)
        return pve, ple
    raise FileNotFoundError(f"no projection checkpoint under {paths.CKPT_DIR}")


def _load_coco_pairs(split: str, top_classes: int, per_class: int, seed: int):
    if split not in _SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; known={sorted(_SPLIT_FILES)}")
    instances_path, captions_path = _SPLIT_FILES[split]
    instances = _load_json(instances_path)
    captions = _load_json(captions_path)

    cat_by_id = {int(c["id"]): str(c["name"]) for c in instances.get("categories", [])}
    image_cats: dict[int, set[str]] = defaultdict(set)
    for ann in instances.get("annotations", []):
        cid = ann.get("category_id")
        iid = ann.get("image_id")
        if cid in cat_by_id and iid is not None:
            image_cats[int(iid)].add(cat_by_id[int(cid)])

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    text_cache = paths.LLAMA_FEATURES / split
    for ann in captions.get("annotations", []):
        iid = ann.get("image_id")
        tid = str(ann.get("id", ""))
        if iid is not None and tid and (text_cache / f"{tid}.pt").exists():
            captions_by_image[int(iid)].append(tid)

    clip_cache = paths.CLIP_FEATURES / split
    eligible: list[tuple[int, str, str]] = []
    counts: Counter[str] = Counter()
    for iid, cats in image_cats.items():
        if not cats:
            continue
        stem = _image_stem(iid)
        if not (clip_cache / f"{stem}.pt").exists():
            continue
        if not captions_by_image.get(iid):
            continue
        for c in cats:
            counts[c] += 1

    selected = [c for c, _ in counts.most_common(top_classes)]
    selected_set = set(selected)
    for iid, cats in image_cats.items():
        cats2 = [c for c in cats if c in selected_set]
        if not cats2:
            continue
        stem = _image_stem(iid)
        if not (clip_cache / f"{stem}.pt").exists() or not captions_by_image.get(iid):
            continue
        # Assign a deterministic primary label among selected classes.
        label = sorted(cats2, key=lambda c: (-counts[c], c))[0]
        eligible.append((iid, label, captions_by_image[iid][0]))

    rng = random.Random(seed)
    by_class: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for row in eligible:
        by_class[row[1]].append(row)
    rows: list[tuple[int, str, str]] = []
    for cname in selected:
        pool = by_class.get(cname, [])
        rng.shuffle(pool)
        rows.extend(pool[:per_class])
    rng.shuffle(rows)
    LOG.info("selected %d image-caption pairs from %s across classes=%s",
             len(rows), split, selected)
    return rows, selected


def _pool_feature(x, kind: str):
    """Pool cached token/patch features to one vector."""
    if kind == "clip" and x.ndim == 2 and x.shape[0] > 1:
        x = x[1:]  # drop CLS
    if x.ndim == 2:
        return x.float().mean(dim=0, keepdim=True).unsqueeze(0)
    if x.ndim == 1:
        return x.float().view(1, 1, -1)
    raise ValueError(f"unsupported feature shape {tuple(x.shape)}")


def build_embeddings(
    config: str | None,
    split: str,
    top_classes: int,
    per_class: int,
    out_path: Path,
) -> Path:
    import torch

    cfg = load_config(config)
    set_seed(int(getattr(cfg, "seed", 42)))
    device = "cuda" if getattr(cfg, "device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"
    pve, ple = _load_projection_state(cfg, device)

    rows, classes = _load_coco_pairs(
        split=split,
        top_classes=top_classes,
        per_class=per_class,
        seed=int(getattr(cfg, "seed", 42)),
    )
    if not rows:
        raise RuntimeError("no eligible cached COCO image-caption pairs for PSAS t-SNE")

    labels_by_class = {c: i for i, c in enumerate(classes)}
    emb: list[np.ndarray] = []
    labels: list[int] = []
    modalities: list[str] = []

    clip_cache = paths.CLIP_FEATURES / split
    text_cache = paths.LLAMA_FEATURES / split
    with torch.inference_mode():
        for iid, cname, text_id in rows:
            stem = _image_stem(iid)
            v = torch.load(clip_cache / f"{stem}.pt", map_location="cpu")
            t = torch.load(text_cache / f"{text_id}.pt", map_location="cpu")
            v_in = _pool_feature(v, "clip").to(device)
            t_in = _pool_feature(t, "text").to(device)
            v_mu, _ = pve(v_in)
            t_mu, _ = ple(t_in)
            emb.append(v_mu.squeeze(0).squeeze(0).detach().cpu().float().numpy())
            labels.append(labels_by_class[cname])
            modalities.append("visual")
            emb.append(t_mu.squeeze(0).squeeze(0).detach().cpu().float().numpy())
            labels.append(labels_by_class[cname])
            modalities.append("text")

    arr = np.stack(emb).astype("float32")
    lab = np.asarray(labels, dtype="int64")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        emb=arr,
        labels=lab,
        classes=np.asarray(classes, dtype=object),
        modalities=np.asarray(modalities, dtype=object),
        split=split,
    )
    LOG.info("wrote real PSAS t-SNE embeddings -> %s shape=%s", out_path, arr.shape)
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build real PSAS embeddings for fig_psas_tsne.")
    p.add_argument("--config", default=None, help="optional config override")
    p.add_argument("--split", default="val2017", choices=sorted(_SPLIT_FILES))
    p.add_argument("--top-classes", type=int, default=6)
    p.add_argument("--per-class", type=int, default=80)
    p.add_argument("--out", default=str(paths.CACHE / "psas_tsne.npz"))
    return p


def main() -> int:
    args = build_argparser().parse_args()
    build_embeddings(
        config=args.config,
        split=args.split,
        top_classes=args.top_classes,
        per_class=args.per_class,
        out_path=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
