"""HHD detection evaluation — held-out AUC / F1 for OLD / ALD / RLD (Tab.5, Fig.10).

This module was missing entirely in the prior run (contribution C3 had no
empirical support: zero AUC/ROC code anywhere in src). It scores the *raw*
detector signals against benchmark ground truth:

* **OLD** on POPE (9,000 balanced object-existence yes/no questions):
  probe = the object phrase in "Is there a(n) X in the image?";
  score = :class:`cmpsa.models.hhd.ObjectDetector` MC existence probability of
  the probe under the per-patch visual PSAS.
* **ALD** on AMBER-attribute (7,628 questions, gt from annotations.json):
  probe = (noun, attr) parsed from "Is the NOUN ATTR in this image?";
  scores = (a) AttributeDetector KL-inconsistency of "ATTR NOUN" against the
  noun-grounded patch (inverted so higher = supported), (b) OLD-style phrase
  support of "ATTR NOUN" (secondary signal).
* **RLD** on AMBER-relation (1,664, "direct contact between the X and Y") and
  VG-Rel (2,000 internal probes, "Is the S PRED (the) O?"):
  scores = (a) RelationDetector surrogate over (subj, obj) grounding,
  (b) OLD-style phrase support of "S PRED O" (secondary signal).

For every bed it reports ROC-AUC, average precision, best-F1 over a threshold
sweep, n and class balance, and dumps per-item scores for the Fig.10 ROC.

Run::

    python -m cmpsa.eval.eval_hhd                     # all beds
    python -m cmpsa.eval.eval_hhd --benchmarks pope --limit 200
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, load_json, set_seed

LOG = get_logger("cmpsa.eval.eval_hhd")

OUT_METRICS = "hhd_detection"

# VG-Rel predicates (canon list mirrors data/build_vg_rel.py)
_VG_PREDICATES = [
    "sitting on", "standing on", "lying on", "in front of", "next to",
    "on top of", "attached to", "walking on", "leaning on", "parked on",
    "on", "under", "above", "below", "behind", "near", "beside", "inside",
    "wearing", "holding", "riding", "carrying", "eating", "watching",
    "hanging on", "covered by", "covering", "against", "at", "over", "in",
]


# --------------------------------------------------------------------------- #
# Probe parsing (formats verified against the actual benchmark files)
# --------------------------------------------------------------------------- #
def parse_pope(question: str) -> str | None:
    m = re.search(r"[Ii]s there (?:a|an) (.+?) in (?:the|this) image", question)
    return m.group(1).strip().lower() if m else None


def parse_amber_attr(question: str) -> tuple[str, str] | None:
    """'Is the NOUN ATTR in this image?' -> (noun, attr); noun may be multiword."""
    m = re.search(r"[Ii]s the (.+?) in this image", question)
    if not m:
        return None
    words = m.group(1).strip().lower().split()
    if len(words) < 2:
        return None
    return " ".join(words[:-1]), words[-1]


def parse_amber_rel(question: str) -> tuple[str, str, str] | None:
    """'Is there direct contact between the X and Y?' -> (X, 'touching', Y)."""
    m = re.search(r"between the (.+?) and (?:the )?(.+?)\?", question)
    if not m:
        return None
    return m.group(1).strip().lower(), "touching", m.group(2).strip().lower()


def parse_vg_rel(question: str) -> tuple[str, str, str] | None:
    """'Is the S PRED (the) O?' with PRED from the canon predicate list."""
    m = re.search(r"[Ii]s the (.+?)\?$", question.strip())
    if not m:
        return None
    body = m.group(1).strip().lower()
    for pred in sorted(_VG_PREDICATES, key=len, reverse=True):
        marker = f" {pred} "
        if marker in f" {body} ":
            left, _, right = f" {body} ".partition(marker)
            subj = left.strip()
            obj = right.strip()
            if obj.startswith("the "):
                obj = obj[4:]
            if subj and obj:
                return subj, pred, obj
    return None


# --------------------------------------------------------------------------- #
# PSAS encoders (CLIP patches + text-backbone phrases through PVE/PLE)
# --------------------------------------------------------------------------- #
class PSASEncoder:
    """Frozen CLIP + text backbone + trained PVE/PLE heads, with caching."""

    def __init__(self, cfg, ckpt_path: Path | None = None):
        import torch
        from transformers import (AutoModel, AutoTokenizer,
                                  CLIPImageProcessor, CLIPVisionModel)
        from cmpsa.models.pve_ple import PLEHead, PVEHead

        self.torch = torch
        self.cfg = cfg
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        pj = cfg.projection
        self.pve = PVEHead(cfg.visual_backbone.feature_dim, pj.psas_dim,
                           pj.hidden_dim, pj.min_logvar, pj.max_logvar).to(self.dev).eval()
        self.ple = PLEHead(cfg.text_backbone.feature_dim, pj.psas_dim,
                           pj.hidden_dim, pj.min_logvar, pj.max_logvar).to(self.dev).eval()
        self.ckpt_used = self._load_heads(ckpt_path)

        def _local(key):
            entry = getattr(cfg.models, key)
            p = paths.MODELS_ROOT / entry.local_dir
            return str(p) if p.exists() else entry.hf_id

        vsrc = _local(cfg.visual_backbone.key)
        LOG.info("loading CLIP from %s", vsrc)
        self.clip_proc = CLIPImageProcessor.from_pretrained(vsrc)
        self.clip = CLIPVisionModel.from_pretrained(
            vsrc, torch_dtype=torch.float16).to(self.dev).eval()

        tsrc = _local(cfg.text_backbone.key)
        LOG.info("loading text backbone from %s", tsrc)
        self.tok = AutoTokenizer.from_pretrained(tsrc)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.text = AutoModel.from_pretrained(
            tsrc, torch_dtype=torch.float16, output_hidden_states=True).to(self.dev).eval()

        self._img_cache: dict[str, tuple] = {}
        self._txt_cache: dict[str, tuple] = {}

    def _load_heads(self, ckpt_path: Path | None) -> str:
        torch = self.torch
        candidates = ([ckpt_path] if ckpt_path else
                      [paths.CKPT_DIR / "cmota.pt", paths.CKPT_DIR / "pretrain_proj.pt"])
        for ck in candidates:
            if ck is None or not Path(ck).exists():
                continue
            state = torch.load(ck, map_location="cpu", weights_only=False)
            pve_sd, ple_sd = state.get("pve", {}), state.get("ple", {})
            if any(k.startswith("base_model.model.") for k in pve_sd):
                from cmpsa.models.pgd_decode import _merge_peft_linear_state
                rank = state.get("lora_rank")
                pve_sd = _merge_peft_linear_state(pve_sd, rank)
                ple_sd = _merge_peft_linear_state(ple_sd, rank)
            m1, _ = self.pve.load_state_dict(pve_sd, strict=False)
            m2, _ = self.ple.load_state_dict(ple_sd, strict=False)
            if m1 or m2:
                raise RuntimeError(f"head load from {ck} left missing keys: {m1 or m2}")
            LOG.info("PSAS heads loaded from %s (stage=%s)", ck, state.get("stage"))
            return str(ck)
        raise RuntimeError(
            "no projection checkpoint found (results/checkpoints/{cmota,pretrain_proj}.pt); "
            "refusing to evaluate random heads")

    def visual(self, img_path: Path):
        """Per-patch visual PSAS Gaussians [576, D] (CLS dropped), cached."""
        key = str(img_path)
        if key in self._img_cache:
            return self._img_cache[key]
        torch = self.torch
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        inp = self.clip_proc(images=img, return_tensors="pt").to(self.dev, torch.float16)
        with torch.inference_mode():
            patch = self.clip(**inp).last_hidden_state[:, 1:, :].float()
            mu, lv = self.pve(patch)
        out = (mu.squeeze(0), lv.squeeze(0))
        if len(self._img_cache) < 4096:
            self._img_cache[key] = out
        return out

    def phrase(self, text: str):
        """Mean-pooled text PSAS Gaussian [D] for a short phrase, cached."""
        key = text.lower()
        if key in self._txt_cache:
            return self._txt_cache[key]
        torch = self.torch
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=16).to(self.dev)
        with torch.inference_mode():
            hs = self.text(**enc).hidden_states[-1].float()
            pooled = hs.mean(dim=1)
            mu, lv = self.ple(pooled)
        out = (mu.squeeze(0), lv.squeeze(0))
        self._txt_cache[key] = out
        return out


# --------------------------------------------------------------------------- #
# Detection metrics
# --------------------------------------------------------------------------- #
def detection_metrics(labels: list[int], scores: list[float]) -> dict:
    """AUC / AP / best-F1 over a threshold sweep (higher score = positive/yes)."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    out: dict[str, Any] = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else None}
    if len(set(y.tolist())) < 2:
        out.update({"auc": None, "ap": None, "best_f1": None,
                    "note": "single-class ground truth; AUC undefined"})
        return out
    out["auc"] = float(roc_auc_score(y, s))
    out["ap"] = float(average_precision_score(y, s))
    best_f1, best_thr = 0.0, None
    for thr in np.quantile(s, np.linspace(0.02, 0.98, 49)):
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if tp == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    out["best_f1"] = round(best_f1, 4)
    out["best_thr"] = best_thr
    return out


# --------------------------------------------------------------------------- #
# Benchmark beds
# --------------------------------------------------------------------------- #
def _amber_truth() -> dict[int, str]:
    anns = load_json(paths.AMBER_ANN)
    return {int(a["id"]): str(a.get("truth", "")).strip().lower() for a in anns}


def _resolve_vg_image(image_field: str) -> Path | None:
    name = Path(image_field).name
    for root in (paths.VG_100K, paths.VG_100K_2,
                 Path(r"G:\cmpsa_data\basic\visual_genome\images\VG_100K"),
                 Path(r"G:\cmpsa_data\basic\visual_genome\images\VG_100K_2")):
        p = root / name
        if p.exists():
            return p
    return None


def _load_jsonl(path) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_pope(enc: PSASEncoder, limit: int | None) -> tuple[list, dict]:
    from cmpsa.models.hhd import ObjectDetector

    old = ObjectDetector(enc.cfg)
    rows = []
    for subset, qfile in paths.POPE_SUBSETS.items():
        items = _load_jsonl(qfile)   # POPE files are JSONL, not a JSON array
        if limit:
            items = items[:limit]
        for it in items:
            probe = parse_pope(it["text"])
            if probe is None:
                continue
            img = paths.POPE_IMAGE_DIR / it["image"]
            if not img.exists():
                continue
            v = enc.visual(img)
            l_mu, l_lv = enc.phrase(probe)
            s = old.score(l_mu, l_lv, v[0], v[1])
            rows.append({"bench": "pope", "subset": subset, "id": it["question_id"],
                         "probe": probe, "label": 1 if it["label"] == "yes" else 0,
                         "old_score": s})
    variants = {"old_mc_existence": detection_metrics(
        [r["label"] for r in rows], [r["old_score"] for r in rows])}
    return rows, variants


def run_amber_attr(enc: PSASEncoder, limit: int | None) -> tuple[list, dict]:
    from cmpsa.models.hhd import AttributeDetector, ObjectDetector

    ald = AttributeDetector(enc.cfg)
    old = ObjectDetector(enc.cfg)
    truth = _amber_truth()
    items = load_json(paths.AMBER_Q_ATTRIBUTE)
    if limit:
        items = items[:limit]
    rows = []
    torch = enc.torch
    for it in items:
        gt = truth.get(int(it["id"]))
        if gt not in ("yes", "no"):
            continue
        parsed = parse_amber_attr(it["query"])
        if parsed is None:
            continue
        noun, attr = parsed
        img = paths.AMBER_IMAGES / it["image"]
        if not img.exists():
            continue
        v_mu, v_lv = enc.visual(img)
        n_mu, _ = enc.phrase(noun)
        p_mu, p_lv = enc.phrase(f"{attr} {noun}")
        j = int(((v_mu - n_mu[None, :]) ** 2).sum(dim=-1).argmin())
        kl_incons = ald.score(p_mu, p_lv, v_mu[j], v_lv[j])
        support = old.score(p_mu, p_lv, v_mu, v_lv)
        rows.append({"bench": "amber_attr", "id": it["id"], "probe": f"{attr} {noun}",
                     "label": 1 if gt == "yes" else 0,
                     "ald_support": 1.0 - kl_incons, "old_phrase": support})
    labels = [r["label"] for r in rows]
    variants = {
        "ald_kl_support": detection_metrics(labels, [r["ald_support"] for r in rows]),
        "old_phrase_support": detection_metrics(labels, [r["old_phrase"] for r in rows]),
    }
    return rows, variants


def _rel_rows(enc: PSASEncoder, items: Iterable[dict], bench: str,
              parser, get_img, get_gt) -> list:
    from cmpsa.models.hhd import ObjectDetector, RelationDetector

    rld = RelationDetector(enc.cfg, rel_head=None)
    old = ObjectDetector(enc.cfg)
    rows = []
    for it in items:
        gt = get_gt(it)
        if gt not in ("yes", "no"):
            continue
        parsed = parser(it)
        if parsed is None:
            continue
        subj, pred, obj = parsed
        img = get_img(it)
        if img is None or not img.exists():
            continue
        v_mu, v_lv = enc.visual(img)
        s_mu, s_lv = enc.phrase(subj)
        o_mu, o_lv = enc.phrase(obj)
        ph_mu, ph_lv = enc.phrase(f"{subj} {pred} {obj}")
        surrogate = rld.score(s_mu, s_lv, o_mu, o_lv, v_mu, v_lv)
        support = old.score(ph_mu, ph_lv, v_mu, v_lv)
        rows.append({"bench": bench, "id": it.get("id"),
                     "probe": f"{subj}|{pred}|{obj}",
                     "label": 1 if gt == "yes" else 0,
                     "rld_surrogate": surrogate, "old_phrase": support})
    return rows


def run_amber_rel(enc: PSASEncoder, limit: int | None) -> tuple[list, dict]:
    truth = _amber_truth()
    items = load_json(paths.AMBER_Q_RELATION)
    if limit:
        items = items[:limit]
    rows = _rel_rows(
        enc, items, "amber_rel",
        parser=lambda it: parse_amber_rel(it["query"]),
        get_img=lambda it: paths.AMBER_IMAGES / it["image"],
        get_gt=lambda it: truth.get(int(it["id"])),
    )
    labels = [r["label"] for r in rows]
    variants = {
        "rld_surrogate": detection_metrics(labels, [r["rld_surrogate"] for r in rows]),
        "old_phrase_support": detection_metrics(labels, [r["old_phrase"] for r in rows]),
    }
    return rows, variants


def run_vg_rel(enc: PSASEncoder, limit: int | None) -> tuple[list, dict]:
    items = []
    with open(paths.VG_REL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    if limit:
        items = items[:limit]
    rows = _rel_rows(
        enc, items, "vg_rel",
        parser=lambda it: parse_vg_rel(it["question"]),
        get_img=lambda it: _resolve_vg_image(it["image"]),
        get_gt=lambda it: str(it.get("gt", "")).lower(),
    )
    labels = [r["label"] for r in rows]
    variants = {
        "rld_surrogate": detection_metrics(labels, [r["rld_surrogate"] for r in rows]),
        "old_phrase_support": detection_metrics(labels, [r["old_phrase"] for r in rows]),
    }
    return rows, variants


_BEDS = {
    "pope": ("OLD", run_pope),
    "amber_attr": ("ALD", run_amber_attr),
    "amber_rel": ("RLD", run_amber_rel),
    "vg_rel": ("RLD", run_vg_rel),
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HHD OLD/ALD/RLD held-out detection eval "
                                            "(AUC / AP / best-F1; Tab.5, Fig.10).")
    p.add_argument("--benchmarks", default="pope,amber_attr,amber_rel,vg_rel")
    p.add_argument("--limit", type=int, default=None, help="cap items per bed")
    p.add_argument("--ckpt", default=None, help="projection checkpoint (default: "
                                                "results/checkpoints/cmota.pt)")
    p.add_argument("--config", default=None)
    p.add_argument("--tag", default="default", help="output filename tag")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    paths.ensure_dirs()

    enc = PSASEncoder(cfg, Path(args.ckpt) if args.ckpt else None)
    all_metrics: dict[str, Any] = {"ckpt": enc.ckpt_used, "tag": args.tag}

    pred_dir = paths.PRED_DIR / OUT_METRICS
    pred_dir.mkdir(parents=True, exist_ok=True)
    met_dir = paths.METRICS_DIR / OUT_METRICS
    met_dir.mkdir(parents=True, exist_ok=True)

    for bed in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        if bed not in _BEDS:
            LOG.warning("unknown bed %r (known: %s)", bed, sorted(_BEDS))
            continue
        layer, fn = _BEDS[bed]
        LOG.info("=== %s (%s) ===", bed, layer)
        rows, variants = fn(enc, args.limit)
        with open(pred_dir / f"{bed}__{args.tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_metrics[bed] = {"layer": layer, **variants}
        for vname, m in variants.items():
            LOG.info("%s / %s: AUC=%s AP=%s bestF1=%s n=%s pos=%.3f",
                     bed, vname, m.get("auc"), m.get("ap"), m.get("best_f1"),
                     m.get("n"), m.get("pos_rate") or -1)

    out = met_dir / f"detection__{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
