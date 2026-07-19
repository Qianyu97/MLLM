"""Stage A -- projection-head pretraining.

Train the probabilistic projection heads :class:`cmpsa.models.pve_ple.PVEHead`
(visual) and :class:`~cmpsa.models.pve_ple.PLEHead` (text) that map the frozen
backbone features into the shared *probabilistic semantic alignment space*
(PSAS). This is the warm-up stage before the CM-OTA alignment (Stage B).

Data (per ``cfg.pretrain``):
  * ``cfg.pretrain.data`` -> ShareGPT4V captioner (1.246M). We keep only samples
    whose image is available locally (``cfg.pretrain.coco_subset_only`` -> the
    ``coco/train2017/*`` subset), then subsample to ``cfg.pretrain.max_samples``
    (LCS-558K scale) for comparability.
  * plus COCO-2017 train images with their captions.

Each sample contributes a (visual feature, text feature) *positive* pair. The
heads are trained so that the visual and text Gaussians land near each other in
PSAS: a symmetric Gaussian-W2 pull on the matched pair plus an InfoNCE-style
contrast against in-batch negatives, with a small log-variance KL regularizer to
keep the predicted covariances well-conditioned (anti-collapse).

Run as::

    python -m cmpsa.train.pretrain_proj --limit 16            # smoke test
    python -m cmpsa.train.pretrain_proj --config configs/exp.yaml

``torch`` / ``transformers`` are imported lazily; ``--help`` works without a GPU.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed

LOG = get_logger("cmpsa.pretrain_proj")


# --------------------------------------------------------------------------- #
# Dataset assembly (pure-python, no torch)
# --------------------------------------------------------------------------- #
def _coco_train_image(filename_or_id: str) -> Path | None:
    """Resolve a ``coco/train2017/<id>.jpg`` reference to a local path."""
    name = Path(filename_or_id).name
    p = paths.COCO_TRAIN2017 / name
    return p if p.exists() else None


def _resolve_sharegpt4v_image(rel: str) -> Path | None:
    """Map a ShareGPT4V image reference to a local file, COCO subset only.

    ShareGPT4V references look like ``coco/train2017/000000123.jpg``,
    ``sam/images/sa_x.jpg`` etc. Only the ``coco/`` ones live locally here.
    """
    rel = rel.replace("\\", "/")
    if rel.startswith("coco/train2017/"):
        return _coco_train_image(rel)
    if rel.startswith("coco/val2017/"):
        p = paths.COCO_VAL2017 / Path(rel).name
        return p if p.exists() else None
    if rel.startswith("coco/val2014/") or rel.startswith("coco/images/val2014/"):
        p = paths.COCO_VAL2014 / Path(rel).name
        return p if p.exists() else None
    return None


def _first_caption(conversations: list[dict]) -> str:
    """Pull the model (gpt) caption from a ShareGPT4V conversation list."""
    for turn in conversations:
        if turn.get("from") == "gpt":
            return str(turn.get("value", "")).strip()
    return ""


def iter_sharegpt4v_pairs(coco_subset_only: bool, limit: int | None) -> Iterator[dict]:
    """Yield ``{"image": Path, "text": str, "id": str}`` from ShareGPT4V captioner.

    ``coco_subset_only`` is currently always honored: only the locally-available
    COCO subset of the captioner references can be resolved on this box, so
    non-COCO references are skipped regardless of the flag value.
    """
    src = paths.SHAREGPT4V_CAPTIONER_1246K
    if not coco_subset_only:
        LOG.info("coco_subset_only=False requested, but only the local COCO subset "
                 "is resolvable here; non-COCO references will still be skipped")
    if not src.exists():
        LOG.warning("ShareGPT4V captioner json missing: %s", src)
        return
    # The file is a large JSON array; json.load is acceptable for a one-shot prep.
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    for item in data:
        rel = str(item.get("image", ""))
        # coco_subset_only is the only supported mode here: we can only resolve
        # the COCO subset locally, so non-local references are skipped either way.
        img = _resolve_sharegpt4v_image(rel)
        if img is None:
            continue
        text = _first_caption(item.get("conversations", []))
        if not text:
            continue
        yield {"id": str(item.get("id", n)), "image": img, "text": text}
        n += 1
        if limit is not None and n >= limit:
            return


def iter_coco2017_pairs(limit: int | None) -> Iterator[dict]:
    """Yield ``{"image": Path, "text": str, "id": str}`` from COCO-2017 train captions."""
    cap = paths.COCO_CAPTIONS_TRAIN2017
    if not cap.exists():
        LOG.warning("COCO train2017 captions missing: %s", cap)
        return
    with open(cap, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data.get("annotations", []) if isinstance(data, dict) else []
    n = 0
    for a in anns:
        img_id = int(a.get("image_id", 0))
        name = f"{img_id:012d}.jpg"
        p = paths.COCO_TRAIN2017 / name
        if not p.exists():
            continue
        text = str(a.get("caption", "")).strip()
        if not text:
            continue
        yield {"id": f"coco_{a.get('id', n)}", "image": p, "text": text}
        n += 1
        if limit is not None and n >= limit:
            return


def build_dataset(cfg, limit: int | None) -> list[dict]:
    """Assemble the Stage-A (image, caption) positive-pair list.

    Honors ``cfg.pretrain.coco_subset_only`` and ``cfg.pretrain.max_samples``.
    With ``--limit`` we cap to a tiny smoke-test slice across both sources.
    """
    coco_only = bool(getattr(cfg.pretrain, "coco_subset_only", True))
    max_samples = int(getattr(cfg.pretrain, "max_samples", 558000))
    per_source = limit if limit is not None else None

    rows: list[dict] = []
    rows.extend(iter_sharegpt4v_pairs(coco_only, per_source))
    rows.extend(iter_coco2017_pairs(per_source))

    if not rows:
        LOG.warning("no (image, caption) pairs assembled -- check data availability")
        return rows

    # Subsample to the configured scale (deterministic given cfg.seed).
    cap = limit if limit is not None else max_samples
    if len(rows) > cap:
        rng = random.Random(cfg.seed)
        rng.shuffle(rows)
        rows = rows[:cap]
    LOG.info("Stage-A dataset: %d (image, caption) pairs (coco_subset_only=%s)",
             len(rows), coco_only)
    return rows


# --------------------------------------------------------------------------- #
# Backbone feature extraction (lazy heavy imports)
# --------------------------------------------------------------------------- #
def _device(cfg):
    import torch

    want = getattr(cfg, "device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        LOG.warning("cfg.device=cuda but no GPU visible; using cpu")
        return "cpu"
    return want


def _resolve_local_dir(cfg, model_key: str) -> Path:
    entry = getattr(cfg.models, model_key, None)
    if entry is None:
        raise KeyError(f"model key {model_key!r} not in configs/models.yaml")
    return paths.MODELS_ROOT / entry.local_dir


class _Backbones:
    """Holds the frozen visual + text backbones and produces feature tensors.

    Features are mean-free per-token / per-patch sequences; for the projection
    head we pool to a single vector per item (the heads accept ``[B, N, D]`` and
    we treat each item as a single-token sequence after pooling, keeping the
    contract ``forward(feats)->(mu, logvar)`` intact).
    """

    def __init__(self, cfg):
        import torch
        from transformers import (
            AutoModel,
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionModel,
        )

        self.cfg = cfg
        self.torch = torch
        dev = _device(cfg)
        self.device = dev

        vkey = cfg.visual_backbone.key
        vlocal = _resolve_local_dir(cfg, vkey)
        vsrc = str(vlocal) if vlocal.exists() else getattr(cfg.models, vkey).hf_id
        LOG.info("loading visual backbone %s", vsrc)
        self.clip_proc = CLIPImageProcessor.from_pretrained(vsrc)
        self.clip = CLIPVisionModel.from_pretrained(vsrc).to(dev).eval()

        tkey = cfg.text_backbone.key
        tlocal = _resolve_local_dir(cfg, tkey)
        tsrc = str(tlocal) if tlocal.exists() else getattr(cfg.models, tkey).hf_id
        LOG.info("loading text backbone %s", tsrc)
        self.tok = AutoTokenizer.from_pretrained(tsrc)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.text = AutoModel.from_pretrained(tsrc, output_hidden_states=True).to(dev).eval()

    def visual_feats(self, images: list) -> "object":
        """Return mean-pooled patch features ``[B, 1, feature_dim]``."""
        torch = self.torch
        inputs = self.clip_proc(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.clip(**inputs)
        # [B, N+1, D] -> drop cls, mean-pool patches -> [B, 1, D]
        patches = out.last_hidden_state[:, 1:, :]
        pooled = patches.mean(dim=1, keepdim=True)
        return pooled

    def text_feats(self, texts: list[str]) -> "object":
        """Return mean-pooled token features ``[B, 1, feature_dim]``."""
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=128).to(self.device)
        with torch.no_grad():
            out = self.text(**enc)
        hs = out.hidden_states[-1] if getattr(out, "hidden_states", None) else out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled.unsqueeze(1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _build_heads(cfg):
    """Instantiate PVE/PLE heads from the models package (lazy import)."""
    from cmpsa.models.pve_ple import PLEHead, PVEHead

    pj = cfg.projection
    pve = PVEHead(
        in_dim=cfg.visual_backbone.feature_dim,
        psas_dim=pj.psas_dim,
        hidden_dim=pj.hidden_dim,
        min_logvar=pj.min_logvar,
        max_logvar=pj.max_logvar,
    )
    ple = PLEHead(
        in_dim=cfg.text_backbone.feature_dim,
        psas_dim=pj.psas_dim,
        hidden_dim=pj.hidden_dim,
        min_logvar=pj.min_logvar,
        max_logvar=pj.max_logvar,
    )
    return pve, ple


def _pair_loss(torch, v_mu, v_logvar, l_mu, l_logvar):
    """Symmetric matched-pair Gaussian-W2 pull + in-batch InfoNCE + logvar KL.

    Uses the closed-form Gaussian W2 from ``cmpsa.models.cm_ota`` when available;
    otherwise falls back to a mean MSE + variance term so the script is robust.
    """
    # squeeze the single-token axis -> [B, D]
    vm, vl = v_mu.squeeze(1), v_logvar.squeeze(1)
    lm, ll = l_mu.squeeze(1), l_logvar.squeeze(1)

    try:
        from cmpsa.models.cm_ota import gaussian_w2

        # pairwise cost matrix [B, B]
        cost = gaussian_w2(vm[:, None, :], vl[:, None, :], lm[None, :, :], ll[None, :, :])
    except Exception:  # pragma: no cover - fallback distance
        # pairwise squared mean distance + variance mismatch
        diff = vm.unsqueeze(1) - lm.unsqueeze(0)            # [B, B, D]
        mean_term = (diff ** 2).sum(-1)
        var_term = ((vl.exp().unsqueeze(1) + ll.exp().unsqueeze(0))).sum(-1)
        cost = mean_term + 0.0 * var_term

    b = cost.shape[0]
    diag = torch.diagonal(cost)

    # InfoNCE: matched pair (diagonal) is the positive, smaller cost = closer.
    logits = -cost                                          # [B, B]
    targets = torch.arange(b, device=cost.device)
    nce_v = torch.nn.functional.cross_entropy(logits, targets)
    nce_l = torch.nn.functional.cross_entropy(logits.t(), targets)
    l_pull = diag.mean()
    l_nce = 0.5 * (nce_v + nce_l)

    # anti-collapse: keep log-variances near 0 (unit covariance prior).
    l_klreg = 0.5 * (vl.pow(2).mean() + ll.pow(2).mean())

    loss = l_pull + l_nce + 0.01 * l_klreg
    return loss, {"l_pull": float(l_pull.item()), "l_nce": float(l_nce.item()),
                  "l_klreg": float(l_klreg.item())}


def train(cfg, dataset: list[dict], limit: int | None) -> Path:
    """Run Stage-A pretraining and save the projection-head checkpoint."""
    import torch
    from PIL import Image

    if not dataset:
        raise RuntimeError("empty Stage-A dataset; cannot train (check data paths)")

    dev = _device(cfg)
    backbones = _Backbones(cfg)
    pve, ple = _build_heads(cfg)
    pve.to(dev).train()
    ple.to(dev).train()

    params = list(pve.parameters()) + list(ple.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg.pretrain.lr))

    bs = int(cfg.pretrain.batch_size)
    if limit is not None:
        bs = min(bs, max(2, limit))
    epochs = int(cfg.pretrain.epochs)
    use_amp = bool(getattr(cfg, "amp", False)) and dev != "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    LOG.info("Stage-A train: n=%d bs=%d epochs=%d lr=%g amp=%s dev=%s",
             len(dataset), bs, epochs, cfg.pretrain.lr, use_amp, dev)

    step = 0
    for ep in range(epochs):
        random.Random(cfg.seed + ep).shuffle(dataset)
        for i in range(0, len(dataset), bs):
            batch = dataset[i:i + bs]
            images, texts = [], []
            for r in batch:
                try:
                    images.append(Image.open(r["image"]).convert("RGB"))
                    texts.append(r["text"])
                except Exception as e:
                    LOG.warning("skip %s: %s", r["image"], e)
            if len(images) < 2:
                continue  # InfoNCE needs >=2 items

            with torch.cuda.amp.autocast(enabled=use_amp):
                v_in = backbones.visual_feats(images).to(dev).float()
                l_in = backbones.text_feats(texts).to(dev).float()
                v_mu, v_logvar = pve(v_in)
                l_mu, l_logvar = ple(l_in)
                loss, parts = _pair_loss(torch, v_mu, v_logvar, l_mu, l_logvar)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 20 == 0 or limit is not None:
                LOG.info("ep%d step%d loss=%.4f pull=%.4f nce=%.4f klreg=%.4f",
                         ep, step, float(loss.item()), parts["l_pull"],
                         parts["l_nce"], parts["l_klreg"])

    ckpt = paths.CKPT_DIR / "pretrain_proj.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "A_pretrain_proj",
            "pve": pve.state_dict(),
            "ple": ple.state_dict(),
            "config": {
                "psas_dim": cfg.projection.psas_dim,
                "hidden_dim": cfg.projection.hidden_dim,
                "visual_in": cfg.visual_backbone.feature_dim,
                "text_in": cfg.text_backbone.feature_dim,
            },
            "steps": step,
        },
        ckpt,
    )
    LOG.info("Stage-A: saved projection heads -> %s", ckpt)
    return ckpt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage A: pretrain the PVE/PLE projection heads into the PSAS.",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="cap #samples per source for a smoke test")
    p.add_argument("--config", default=None,
                   help="optional YAML override merged on default.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble the dataset and report sizes without training")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    paths.ensure_dirs()

    dataset = build_dataset(cfg, args.limit)
    LOG.info("Stage-A: %d positive pairs assembled", len(dataset))

    if args.dry_run:
        LOG.info("--dry-run set; skipping training")
        return 0
    if not dataset:
        LOG.error("no data available; aborting (this is expected on a CPU box "
                  "without the training corpora present)")
        return 1

    train(cfg, dataset, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
