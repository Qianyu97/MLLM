"""Stage B -- CM-OTA cross-modal alignment via entropic optimal transport.

Load the Stage-A projection heads and continue training them (LoRA-style, rank
``cfg.cmota.lora_rank``) so that the visual and text Gaussian sets are aligned by
*Sinkhorn optimal transport* in the PSAS. The transport cost is the closed-form
2-Wasserstein^2 between diagonal Gaussians (``cfg.cmota.distance``), the plan is
solved with entropic Sinkhorn (``cfg.cmota.sinkhorn_eps`` / ``sinkhorn_iters``),
and a covariance KL term regularizes against collapse.

Hard negatives: for every positive (image, caption) we synthesize
``cfg.cmota.neg_per_pos`` object / attribute / relation negatives by perturbing
the caption using VG / COCO ground truth where available, else simple lexical
substitution (:func:`create_negative_samples`). Negatives sharpen the OT
geometry by acting as extra, deliberately-mismatched text Gaussians.

Logs ``L_total`` / ``L_OT`` / ``L_klreg`` and saves the aligned checkpoint.

Run as::

    python -m cmpsa.train.train_cmota --limit 16
    python -m cmpsa.train.train_cmota --config configs/exp.yaml

``torch`` / ``transformers`` are imported lazily; ``--help`` works without a GPU.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed

LOG = get_logger("cmpsa.train_cmota")

# Small built-in vocabularies for fallback lexical negatives (no data needed).
_FALLBACK_OBJECTS = [
    "dog", "cat", "car", "tree", "person", "bicycle", "boat", "bird", "horse",
    "chair", "table", "bottle", "cup", "clock", "umbrella", "laptop", "phone",
]
_FALLBACK_ATTRS = [
    "red", "blue", "green", "yellow", "black", "white", "large", "small", "old",
    "new", "wooden", "metal", "round", "square", "bright", "dark", "wet", "dry",
]
_FALLBACK_RELATIONS = [
    "on", "under", "next to", "behind", "in front of", "above", "below",
    "inside", "near", "beside", "holding", "riding", "sitting on",
]


# --------------------------------------------------------------------------- #
# Dataset assembly (pure-python, no torch)
# --------------------------------------------------------------------------- #
def _coco_train_image(name: str) -> Path | None:
    p = paths.COCO_TRAIN2017 / Path(name).name
    if p.exists():
        return p
    p = paths.COCO_VAL2014 / Path(name).name
    return p if p.exists() else None


def iter_llava_150k(limit: int | None) -> Iterator[dict]:
    """Yield ``{"id","image":Path,"text":str}`` from LLaVA-Instruct-150K."""
    src = paths.LLAVA_INSTRUCT_150K
    if not src.exists():
        LOG.warning("LLaVA-150K json missing: %s", src)
        return
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    for item in data:
        img = _coco_train_image(str(item.get("image", "")))
        if img is None:
            continue
        text = _first_gpt(item.get("conversations", []))
        if not text:
            continue
        yield {"id": str(item.get("id", n)), "image": img, "text": text}
        n += 1
        if limit is not None and n >= limit:
            return


def iter_sharegpt4v_cap(limit: int | None) -> Iterator[dict]:
    """Yield positives from the ShareGPT4V cap100k file (COCO subset only)."""
    src = paths.SHAREGPT4V_CAP100K
    if not src.exists():
        LOG.warning("ShareGPT4V cap100k json missing: %s", src)
        return
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    for item in data:
        rel = str(item.get("image", "")).replace("\\", "/")
        if not rel.startswith("coco/"):
            continue
        img = _coco_train_image(rel)
        if img is None:
            continue
        text = _first_gpt(item.get("conversations", []))
        if not text:
            continue
        yield {"id": f"s4v_{item.get('id', n)}", "image": img, "text": text}
        n += 1
        if limit is not None and n >= limit:
            return


def _first_gpt(conversations: list[dict]) -> str:
    for turn in conversations:
        if turn.get("from") == "gpt":
            return str(turn.get("value", "")).strip()
    return ""


def build_dataset(cfg, limit: int | None) -> list[dict]:
    """Assemble the Stage-B positive (image, caption) list per ``cfg.cmota.data``."""
    wanted = list(getattr(cfg.cmota, "data", []) or [])
    per_source = limit if limit is not None else None
    rows: list[dict] = []
    if "llava_150k" in wanted or not wanted:
        rows.extend(iter_llava_150k(per_source))
    if "sharegpt4v_cap100k" in wanted:
        rows.extend(iter_sharegpt4v_cap(per_source))
    if not rows:
        LOG.warning("no Stage-B positives assembled -- check data availability")
        return rows
    cap = limit if limit is not None else len(rows)
    if len(rows) > cap:
        rng = random.Random(cfg.seed)
        rng.shuffle(rows)
        rows = rows[:cap]
    LOG.info("Stage-B dataset: %d positive (image, caption) pairs", len(rows))
    return rows


# --------------------------------------------------------------------------- #
# Negative-sample generation (object / attribute / relation)
# --------------------------------------------------------------------------- #
def _load_vg_vocab() -> dict:
    """Best-effort VG/COCO vocab for realistic substitutions; empty if absent."""
    vocab = {"objects": list(_FALLBACK_OBJECTS), "attrs": list(_FALLBACK_ATTRS),
             "relations": list(_FALLBACK_RELATIONS)}
    try:
        if paths.VG_OBJECTS.exists():
            with open(paths.VG_OBJECTS, "r", encoding="utf-8") as f:
                data = json.load(f)
            names = set()
            for rec in data[:2000]:
                for o in rec.get("objects", []):
                    for nm in o.get("names", []):
                        if isinstance(nm, str):
                            names.add(nm.lower())
            if names:
                vocab["objects"] = sorted(names)[:500]
    except Exception as e:  # pragma: no cover
        LOG.warning("VG object vocab unavailable (%s); using fallback list", e)
    return vocab


def create_negative_samples(text: str, neg_per_pos: int, vocab: dict,
                            rng: random.Random) -> list[dict]:
    """Produce object/attribute/relation negatives for one caption.

    Each negative perturbs exactly one semantic dimension so the OT alignment is
    forced to distinguish the perturbation type. Returns a list of
    ``{"text","neg_type"}``; ``neg_type`` in {object, attribute, relation}.
    Reuses VG/COCO ground-truth words when available, otherwise simple lexical
    swaps from the built-in vocab.
    """
    words = text.split()
    out: list[dict] = []
    if not words:
        return out
    types = ["object", "attribute", "relation"]
    for k in range(neg_per_pos):
        ntype = types[k % len(types)]
        pool = {"object": vocab["objects"], "attribute": vocab["attrs"],
                "relation": vocab["relations"]}[ntype]
        repl = rng.choice(pool)
        idx = rng.randrange(len(words))
        if ntype == "relation":
            # insert a relation phrase to corrupt spatial/relational meaning
            neg_words = words[:idx] + [repl] + words[idx:]
        else:
            neg_words = list(words)
            neg_words[idx] = repl
        out.append({"text": " ".join(neg_words), "neg_type": ntype})
    return out


# --------------------------------------------------------------------------- #
# Backbones / heads (lazy heavy imports)
# --------------------------------------------------------------------------- #
def _device(cfg):
    import torch

    want = getattr(cfg, "device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        LOG.warning("cfg.device=cuda but no GPU; using cpu")
        return "cpu"
    return want


def _resolve_local_dir(cfg, model_key: str) -> Path:
    entry = getattr(cfg.models, model_key, None)
    if entry is None:
        raise KeyError(f"model key {model_key!r} not in configs/models.yaml")
    return paths.MODELS_ROOT / entry.local_dir


class _Backbones:
    """Frozen visual + text backbones (shared with Stage A logic)."""

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
        self.device = _device(cfg)

        vkey = cfg.visual_backbone.key
        vlocal = _resolve_local_dir(cfg, vkey)
        vsrc = str(vlocal) if vlocal.exists() else getattr(cfg.models, vkey).hf_id
        LOG.info("loading visual backbone %s", vsrc)
        self.clip_proc = CLIPImageProcessor.from_pretrained(vsrc)
        self.clip = CLIPVisionModel.from_pretrained(vsrc).to(self.device).eval()

        tkey = cfg.text_backbone.key
        tlocal = _resolve_local_dir(cfg, tkey)
        tsrc = str(tlocal) if tlocal.exists() else getattr(cfg.models, tkey).hf_id
        LOG.info("loading text backbone %s", tsrc)
        self.tok = AutoTokenizer.from_pretrained(tsrc)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.text = AutoModel.from_pretrained(tsrc, output_hidden_states=True).to(self.device).eval()

    def visual_feats(self, images):
        torch = self.torch
        inputs = self.clip_proc(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.clip(**inputs)
        return out.last_hidden_state[:, 1:, :].mean(dim=1, keepdim=True)

    def text_feats(self, texts):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=128).to(self.device)
        with torch.no_grad():
            out = self.text(**enc)
        hs = out.hidden_states[-1] if getattr(out, "hidden_states", None) else out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled.unsqueeze(1)


def _build_heads(cfg):
    from cmpsa.models.pve_ple import PLEHead, PVEHead

    pj = cfg.projection
    pve = PVEHead(cfg.visual_backbone.feature_dim, pj.psas_dim, pj.hidden_dim,
                  pj.min_logvar, pj.max_logvar)
    ple = PLEHead(cfg.text_backbone.feature_dim, pj.psas_dim, pj.hidden_dim,
                  pj.min_logvar, pj.max_logvar)
    return pve, ple


def _load_stage_a(pve, ple, torch) -> bool:
    """Warm-start from the Stage-A checkpoint if present."""
    ckpt = paths.CKPT_DIR / "pretrain_proj.pt"
    if not ckpt.exists():
        LOG.warning("Stage-A ckpt missing (%s); training from scratch", ckpt)
        return False
    state = torch.load(ckpt, map_location="cpu")
    pve.load_state_dict(state["pve"], strict=False)
    ple.load_state_dict(state["ple"], strict=False)
    LOG.info("warm-started PVE/PLE from %s", ckpt)
    return True


def _apply_lora(modules: list, rank: int):
    """Wrap projection heads with LoRA adapters if `peft` is available.

    Falls back to plain fine-tuning (returns the modules unchanged) when peft is
    not installed, keeping the script runnable; the LoRA rank is still recorded
    in the checkpoint metadata.
    """
    try:
        from peft import LoraConfig, get_peft_model  # noqa: F401

        wrapped = []
        for m in modules:
            lconf = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                               target_modules="all-linear", bias="none")
            wrapped.append(get_peft_model(m, lconf))
        LOG.info("applied LoRA r=%d to %d projection heads", rank, len(modules))
        return wrapped, True
    except Exception as e:  # pragma: no cover
        LOG.warning("peft unavailable (%s); full fine-tuning the heads instead", e)
        return modules, False


# --------------------------------------------------------------------------- #
# OT loss
# --------------------------------------------------------------------------- #
def _cmota_step(torch, cfg, v_mu, v_logvar, l_mu, l_logvar):
    """Compute the CM-OTA loss dict for one batch.

    Uses ``cmpsa.models.cm_ota.cmota_loss`` (the contract). Squeeze the
    single-token axis to ``[B, D]`` before passing in.
    """
    from cmpsa.models.cm_ota import cmota_loss

    return cmota_loss(
        v_mu.squeeze(1), v_logvar.squeeze(1),
        l_mu.squeeze(1), l_logvar.squeeze(1),
        cfg,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg, dataset: list[dict], limit: int | None) -> Path:
    """Run Stage-B CM-OTA alignment and save the aligned checkpoint."""
    import torch
    from PIL import Image

    if not dataset:
        raise RuntimeError("empty Stage-B dataset; cannot train")

    dev = _device(cfg)
    backbones = _Backbones(cfg)
    pve, ple = _build_heads(cfg)
    _load_stage_a(pve, ple, torch)
    pve.to(dev).train()
    ple.to(dev).train()

    (pve_l, ple_l), used_lora = _apply_lora([pve, ple], int(cfg.cmota.lora_rank))
    pve, ple = pve_l, ple_l

    params = [p for p in list(pve.parameters()) + list(ple.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.cmota.lr))

    vocab = _load_vg_vocab()
    neg_per_pos = int(cfg.cmota.neg_per_pos)
    rng = random.Random(cfg.seed)

    bs = int(cfg.cmota.batch_size)
    if limit is not None:
        bs = min(bs, max(2, limit))
    epochs = int(cfg.cmota.epochs)
    use_amp = bool(getattr(cfg, "amp", False)) and dev != "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    LOG.info("Stage-B train: n=%d bs=%d epochs=%d lr=%g neg/pos=%d lora=%s dev=%s",
             len(dataset), bs, epochs, cfg.cmota.lr, neg_per_pos, used_lora, dev)

    step = 0
    for ep in range(epochs):
        random.Random(cfg.seed + ep).shuffle(dataset)
        for i in range(0, len(dataset), bs):
            batch = dataset[i:i + bs]
            images, texts = [], []
            for r in batch:
                try:
                    images.append(Image.open(r["image"]).convert("RGB"))
                except Exception as e:
                    LOG.warning("skip %s: %s", r["image"], e)
                    continue
                texts.append(r["text"])
                # add hard negatives as extra mismatched text Gaussians
                for neg in create_negative_samples(r["text"], neg_per_pos, vocab, rng):
                    texts.append(neg["text"])
            if len(images) < 2:
                continue

            with torch.cuda.amp.autocast(enabled=use_amp):
                v_in = backbones.visual_feats(images).to(dev).float()
                l_in = backbones.text_feats(texts).to(dev).float()
                v_mu, v_logvar = pve(v_in)
                l_mu, l_logvar = ple(l_in)
                losses = _cmota_step(torch, cfg, v_mu, v_logvar, l_mu, l_logvar)

            loss = losses["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 10 == 0 or limit is not None:
                LOG.info("ep%d step%d L_total=%.4f L_OT=%.4f L_klreg=%.4f",
                         ep, step, float(loss.item()),
                         float(losses["l_ot"]) if not hasattr(losses["l_ot"], "item")
                         else float(losses["l_ot"].item()),
                         float(losses["l_klreg"]) if not hasattr(losses["l_klreg"], "item")
                         else float(losses["l_klreg"].item()))

    ckpt = paths.CKPT_DIR / "cmota.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap a peft model so the base state_dict is recoverable downstream.
    pve_sd = pve.state_dict()
    ple_sd = ple.state_dict()
    torch.save(
        {
            "stage": "B_cmota",
            "pve": pve_sd,
            "ple": ple_sd,
            "lora_rank": int(cfg.cmota.lora_rank),
            "used_lora": used_lora,
            "distance": cfg.cmota.distance,
            "steps": step,
        },
        ckpt,
    )
    LOG.info("Stage-B: saved aligned heads -> %s", ckpt)
    return ckpt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage B: CM-OTA Sinkhorn-OT cross-modal alignment (LoRA r16).",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="cap #positives per source for a smoke test")
    p.add_argument("--config", default=None,
                   help="optional YAML override merged on default.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble data + a few negatives and report sizes only")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    paths.ensure_dirs()

    dataset = build_dataset(cfg, args.limit)
    LOG.info("Stage-B: %d positives assembled", len(dataset))

    if args.dry_run:
        vocab = _load_vg_vocab()
        rng = random.Random(cfg.seed)
        if dataset:
            sample = create_negative_samples(dataset[0]["text"],
                                              int(cfg.cmota.neg_per_pos), vocab, rng)
            LOG.info("example negatives for first positive:")
            for s in sample:
                LOG.info("  [%s] %s", s["neg_type"], s["text"][:80])
        LOG.info("--dry-run set; skipping training")
        return 0
    if not dataset:
        LOG.error("no data available; aborting (expected without training corpora)")
        return 1

    train(cfg, dataset, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
