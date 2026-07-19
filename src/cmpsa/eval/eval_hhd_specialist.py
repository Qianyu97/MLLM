"""Specialist HHD detection eval -> official Tab.5 (AUC / AP / best-F1) + Fig.10 ROC.

Per-layer specialist grounding (validated in the Phase-1 pilot; the PSAS-heuristic
detectors in eval_hhd score ~chance, which is itself evidence that the learned PSAS
is not the grounding engine):

* OLD (object)   -> CLIP zero-shot sim(image, "a photo of a X")           [POPE]
* ALD (attribute)-> Grounding-DINO noun box -> crop -> CLIP contrastive
                    sim(crop,"attr noun") - sim(crop,"noun")               [AMBER-attr]
* RLD (relation) -> Grounding-DINO two boxes -> geometry
                    contact = box overlap/min-area                        [AMBER-rel]
                    direction = signed relative position                 [VG-rel]

Run::  python -m cmpsa.eval.eval_hhd_specialist            # full scale
       python -m cmpsa.eval.eval_hhd_specialist --limit 300 --tag quick
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, load_json, set_seed
from cmpsa.eval.eval_hhd import (parse_pope, parse_amber_attr, parse_amber_rel,
                                 parse_vg_rel, _amber_truth, _resolve_vg_image,
                                 _load_jsonl, detection_metrics)

LOG = get_logger("cmpsa.eval.eval_hhd_specialist")
OUT = "hhd_detection_specialist"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_VG_DIR = {"on": ("dy", +1), "above": ("dy", +1), "over": ("dy", +1), "on top of": ("dy", +1),
           "under": ("dy", -1), "below": ("dy", -1), "beneath": ("dy", -1),
           "left": ("dx", +1), "right": ("dx", -1)}


class Grounder:
    """CLIP (object/attribute) + Grounding-DINO (regions/relations)."""

    def __init__(self, cfg):
        from transformers import (CLIPModel, CLIPProcessor,
                                  AutoProcessor, GroundingDinoForObjectDetection)
        self.cfg = cfg
        cdir = str(paths.MODELS_ROOT / getattr(cfg.models, cfg.visual_backbone.key).local_dir)
        self.clip = CLIPModel.from_pretrained(cdir, torch_dtype=torch.float16).to(DEV).eval()
        self.cproc = CLIPProcessor.from_pretrained(cdir)
        gdir = str(paths.MODELS_ROOT.parent / "tools" / "grounding_dino")
        self.gdproc = AutoProcessor.from_pretrained(gdir)
        self.gdino = GroundingDinoForObjectDetection.from_pretrained(gdir).to(DEV).eval()
        self._imgc: dict[str, torch.Tensor] = {}
        self._txc: dict[str, torch.Tensor] = {}
        self._boxc: dict[tuple, list | None] = {}

    @torch.no_grad()
    def img_emb(self, path):
        k = str(path)
        if k not in self._imgc:
            im = Image.open(path).convert("RGB")
            pin = self.cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
            v = self.clip.visual_projection(self.clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output)
            self._imgc[k] = F.normalize(v, dim=-1)
        return self._imgc[k]

    @torch.no_grad()
    def crop_emb(self, im):
        pin = self.cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
        v = self.clip.visual_projection(self.clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output)
        return F.normalize(v, dim=-1)

    @torch.no_grad()
    def txt_emb(self, t):
        if t not in self._txc:
            tin = self.cproc(text=[t], return_tensors="pt", padding=True).to(DEV)
            f = self.clip.text_projection(self.clip.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output)
            self._txc[t] = F.normalize(f, dim=-1)
        return self._txc[t]

    @torch.no_grad()
    def box(self, image, phrase, path_key=None):
        ck = (str(path_key), phrase.lower()) if path_key else None
        if ck and ck in self._boxc:
            return self._boxc[ck]
        inp = self.gdproc(images=image, text=phrase.lower().strip() + ".", return_tensors="pt").to(DEV)
        out = self.gdino(**inp)
        res = None
        for kw in ("threshold", "box_threshold"):
            try:
                res = self.gdproc.post_process_grounded_object_detection(
                    out, inp["input_ids"], **{kw: 0.15}, text_threshold=0.15,
                    target_sizes=[image.size[::-1]])[0]
                break
            except TypeError:
                continue
        b = None if (res is None or len(res["scores"]) == 0) else [float(v) for v in res["boxes"][int(res["scores"].argmax())]]
        if ck:
            self._boxc[ck] = b
        return b

    # -- grounding scores --
    def old_object(self, img_path, obj):
        return float((self.img_emb(img_path) @ self.txt_emb(f"a photo of a {obj}").T).item())

    def ald_attribute(self, img_path, noun, attr):
        image = Image.open(img_path).convert("RGB")
        b = self.box(image, noun, img_path)
        region = self._crop(image, b) if b else image
        ci = self.crop_emb(region)
        return float((ci @ self.txt_emb(f"a photo of a {attr} {noun}").T).item()
                     - (ci @ self.txt_emb(f"a photo of a {noun}").T).item())

    def rld_contact(self, img_path, subj, obj):
        image = Image.open(img_path).convert("RGB")
        bs = self.box(image, subj, img_path); bo = self.box(image, obj, img_path)
        if not bs or not bo:
            return 0.0
        return self._overlap(bs, bo)

    def rld_direction(self, img_path, subj, pred, obj):
        image = Image.open(img_path).convert("RGB"); W, H = image.size
        bs = self.box(image, subj, img_path); bo = self.box(image, obj, img_path)
        if not bs or not bo:
            return 0.0
        g = self._geom(bs, bo, W, H)
        key = next((k for k in _VG_DIR if k in pred), None)
        if key:
            comp, sign = _VG_DIR[key]
            return sign * g[comp]
        return g["iou"]

    @staticmethod
    def _crop(image, b, pad=0.1):
        W, H = image.size; x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0
        x0 = max(0, x0 - pad*w); y0 = max(0, y0 - pad*h); x1 = min(W, x1 + pad*w); y1 = min(H, y1 + pad*h)
        return image if (x1 - x0 < 5 or y1 - y0 < 5) else image.crop((x0, y0, x1, y1))

    @staticmethod
    def _overlap(b1, b2):
        ax0, ay0, ax1, ay1 = b1; bx0, by0, bx1, by1 = b2
        iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih; a1 = (ax1-ax0)*(ay1-ay0); a2 = (bx1-bx0)*(by1-by0)
        return inter / (min(a1, a2) + 1e-6)

    @staticmethod
    def _geom(b1, b2, W, H):
        ax0, ay0, ax1, ay1 = b1; bx0, by0, bx1, by1 = b2
        iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
        inter = iw*ih; a1 = (ax1-ax0)*(ay1-ay0); a2 = (bx1-bx0)*(by1-by0)
        acx, acy = (ax0+ax1)/2, (ay0+ay1)/2; bcx, bcy = (bx0+bx1)/2, (by0+by1)/2
        return dict(iou=inter/(a1+a2-inter+1e-6), dy=(bcy-acy)/H, dx=(bcx-acx)/W)


def run_pope(g, limit):
    rows = []
    for subset, qf in paths.POPE_SUBSETS.items():
        items = _load_jsonl(qf)
        if limit: items = items[:limit]
        for it in items:
            obj = parse_pope(it["text"]); img = paths.POPE_IMAGE_DIR / it["image"]
            if not obj or not img.exists(): continue
            rows.append({"label": 1 if it["label"] == "yes" else 0, "score": g.old_object(img, obj)})
    return rows


def run_amber_attr(g, limit):
    truth = _amber_truth(); items = load_json(paths.AMBER_Q_ATTRIBUTE)
    if limit: items = items[:limit]
    rows = []
    for it in items:
        gt = truth.get(int(it["id"])); p = parse_amber_attr(it["query"]); img = paths.AMBER_IMAGES / it["image"]
        if gt not in ("yes", "no") or not p or not img.exists(): continue
        noun, attr = p
        rows.append({"label": 1 if gt == "yes" else 0, "score": g.ald_attribute(img, noun, attr)})
    return rows


def run_amber_rel(g, limit):
    truth = _amber_truth(); items = load_json(paths.AMBER_Q_RELATION)
    if limit: items = items[:limit]
    rows = []
    for it in items:
        gt = truth.get(int(it["id"])); p = parse_amber_rel(it["query"]); img = paths.AMBER_IMAGES / it["image"]
        if gt not in ("yes", "no") or not p or not img.exists(): continue
        s_, _, o_ = p
        rows.append({"label": 1 if gt == "yes" else 0, "score": g.rld_contact(img, s_, o_)})
    return rows


def run_vg_rel(g, limit):
    rows = []
    items = [json.loads(l) for l in open(paths.VG_REL_JSONL, encoding="utf-8")]
    if limit: items = items[:limit]
    for it in items:
        p = parse_vg_rel(it["question"]); img = _resolve_vg_image(it["image"]); gt = str(it.get("gt", "")).lower()
        if gt not in ("yes", "no") or not p or not img or not img.exists(): continue
        s_, pred, o_ = p
        rows.append({"label": 1 if gt == "yes" else 0, "score": g.rld_direction(img, s_, pred, o_)})
    return rows


_BEDS = {"pope": ("OLD", run_pope), "amber_attr": ("ALD", run_amber_attr),
         "amber_rel": ("RLD", run_amber_rel), "vg_rel": ("RLD", run_vg_rel)}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Specialist HHD detection eval (Tab.5)")
    ap.add_argument("--benchmarks", default="pope,amber_attr,amber_rel,vg_rel")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config); set_seed(cfg.seed); paths.ensure_dirs()

    g = Grounder(cfg)
    pred_dir = paths.PRED_DIR / OUT; pred_dir.mkdir(parents=True, exist_ok=True)
    met_dir = paths.METRICS_DIR / OUT; met_dir.mkdir(parents=True, exist_ok=True)
    out = {"tag": args.tag, "grounder": "CLIP-OLD / GDINO-crop-ALD / GDINO-geom-RLD"}
    for bed in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        if bed not in _BEDS:
            continue
        layer, fn = _BEDS[bed]
        LOG.info("=== %s (%s) ===", bed, layer)
        rows = fn(g, args.limit)
        with open(pred_dir / f"{bed}__{args.tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        m = detection_metrics([r["label"] for r in rows], [r["score"] for r in rows])
        out[bed] = {"layer": layer, **m}
        LOG.info("%s (%s): AUC=%s AP=%s bestF1=%s n=%s pos=%.3f",
                 bed, layer, m.get("auc"), m.get("ap"), m.get("best_f1"), m.get("n"), m.get("pos_rate") or -1)
    p = met_dir / f"detection_specialist__{args.tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    LOG.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
