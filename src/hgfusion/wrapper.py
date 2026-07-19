# -*- coding: utf-8 -*-
"""hgfusion — a plug-and-play hallucination wrapper around any multimodal LLM.

The wrapper never touches the backbone's weights, gradients, or decoding state.
It consumes only (image, output text) — plus, optionally, whatever extra access
the host system can offer. That gives three access levels, each validated in the
paper on four backbones (LLaVA-1.5/1.6, InstructBLIP, Qwen-VL-Chat):

  L0  verify_caption(...) / revise_caption(..., strategy="remove")
      needs: image + output text only. Wraps ANY backbone, incl. API-only models.
  L1  revise_caption(..., strategy="rewrite", rewrite_fn=<your callable>)
      needs: the ability to prompt the backbone once more (and a backbone that
      follows instructions — InstructBLIP does not, see the paper).
  L2  calibrate(...) then answer(..., p_yes=...)
      needs: the backbone's first-token probability p_yes (open-weight deployments).

Quickstart (L0):

    from hgfusion import HGFWrapper
    w = HGFWrapper(models_root="/path/to/models")     # CLIP + Grounding-DINO dir
    report = w.verify_caption("cat.jpg", "A cat sits by a red vase.")
    clean  = w.revise_caption("cat.jpg", "A cat sits by a red vase.").text

Weights: models_root must contain clip-vit-l14-336/ and grounding_dino/
(fetch with scripts/download_weights.py --group core). Models load lazily on
first use, so constructing the wrapper is free.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from PIL import Image

# COCO-synonym object extraction — the vocabulary the pipeline was validated on.
from cmpsa.eval.eval_chair import _extract_objects

ImageLike = Union[str, Path, Image.Image]


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def sentence_remove(caption: str, flagged: Iterable[str],
                    extract: Callable[[str], set] = _extract_objects) -> str:
    """L0 revision: drop every sentence that mentions a flagged object.

    Deterministic and backbone-agnostic; validated on all four backbones
    (CHAIR-i -20%..-41%). Falls back to the original caption if removal would
    delete everything.
    """
    fset = set(flagged)
    if not fset:
        return caption
    parts = re.split(r"(?<=[.!?])\s+", caption.strip())
    kept = [s for s in parts if not (set(extract(s)) & fset)]
    return (" ".join(kept).strip()) or caption


@dataclass
class CaptionReport:
    """What the detector saw in one caption."""
    objects: Dict[str, float]          # mentioned object -> grounding score
    flagged: List[str]                 # objects below threshold (likely hallucinated)
    threshold: float

    @property
    def clean(self) -> bool:
        return not self.flagged


@dataclass
class RevisedCaption:
    text: str                          # the corrected caption
    report: CaptionReport              # the detection evidence behind it
    strategy: str                      # "remove" | "rewrite" | "none"


@dataclass
class FusionCalibration:
    """Parameters of the calibrated decision fusion (Eqs. 12-13 of the paper)."""
    mu_p: float; sd_p: float
    mu_g: float; sd_g: float
    lam: float
    tau: float                          # decision threshold on z (matched yes-ratio)


class HGFWrapper:
    """Hierarchical-grounding-fusion wrapper. See module docstring for levels."""

    def __init__(self, models_root: Optional[Union[str, Path]] = None,
                 device: str = "cuda", gdino_threshold: float = 0.30):
        if models_root is None:
            models_root = os.environ.get("CMPSA_MODELS_ROOT") or os.path.join(
                os.environ.get("CMPSA_DATA_ROOT", os.getcwd()), "models")
        self.models_root = Path(models_root)
        self.device = device
        self.gdino_threshold = float(gdino_threshold)
        self._gdino = None
        self._gdproc = None
        self._clip = None
        self._cproc = None
        self._img_cache: Dict[str, object] = {}
        self.calibration: Optional[FusionCalibration] = None

    # ------------------------------------------------------------ model loading
    def _gdino_dir(self) -> Path:
        for cand in (self.models_root / "grounding_dino",
                     self.models_root.parent / "tools" / "grounding_dino"):
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"Grounding-DINO not found under {self.models_root} "
            "(run scripts/download_weights.py --group core)")

    def _ensure_gdino(self):
        if self._gdino is None:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection
            d = str(self._gdino_dir())
            self._gdproc = AutoProcessor.from_pretrained(d)
            self._gdino = (GroundingDinoForObjectDetection.from_pretrained(d)
                           .to(self.device).eval())

    def _ensure_clip(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            d = str(self.models_root / "clip-vit-l14-336")
            dtype = torch.float16 if "cuda" in self.device else torch.float32
            self._clip = (CLIPModel.from_pretrained(d, torch_dtype=dtype)
                          .to(self.device).eval())
            self._cproc = CLIPProcessor.from_pretrained(d)

    # ------------------------------------------------------------ scores
    def gdino_score(self, image: ImageLike, obj: str) -> float:
        """Region-level presence score s_obj (the precise, revision-driving source)."""
        import torch
        self._ensure_gdino()
        im = _to_pil(image)
        inp = self._gdproc(images=im, text=obj.lower() + ".",
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._gdino(**inp)
        for kw in ("threshold", "box_threshold"):     # transformers-version compat
            try:
                res = self._gdproc.post_process_grounded_object_detection(
                    out, inp["input_ids"], **{kw: 0.05}, text_threshold=0.05,
                    target_sizes=[im.size[::-1]])[0]
                break
            except TypeError:
                continue
        return float(res["scores"].max()) if len(res["scores"]) else 0.0

    def clip_score(self, image: ImageLike, obj: str) -> float:
        """Global zero-shot presence score g_obj (the L2 fusion source)."""
        import torch
        import torch.nn.functional as F
        self._ensure_clip()
        im = _to_pil(image)
        key = getattr(image, "filename", None) or str(image)
        if key not in self._img_cache:
            pin = self._cproc(images=im, return_tensors="pt").to(self.device)
            pin["pixel_values"] = pin["pixel_values"].to(self._clip.dtype)
            with torch.no_grad():
                feat = self._clip.visual_projection(
                    self._clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output)
            self._img_cache[key] = F.normalize(feat, dim=-1)
        tin = self._cproc(text=[f"a photo of a {obj}"], return_tensors="pt",
                          padding=True).to(self.device)
        with torch.no_grad():
            tf = self._clip.text_projection(
                self._clip.text_model(input_ids=tin["input_ids"],
                                      attention_mask=tin["attention_mask"]).pooler_output)
        tf = F.normalize(tf, dim=-1)
        return float((self._img_cache[key] @ tf.T).item())

    # ------------------------------------------------------------ L0: detect
    def verify_caption(self, image: ImageLike, caption: str,
                       objects: Optional[Sequence[str]] = None,
                       threshold: Optional[float] = None) -> CaptionReport:
        """Score every object the caption mentions; flag the unsupported ones.

        `objects` overrides extraction (pass your own list for non-COCO vocab);
        by default the COCO-synonym extractor used throughout the paper is applied.
        """
        thr = self.gdino_threshold if threshold is None else float(threshold)
        objs = list(objects) if objects is not None else sorted(_extract_objects(caption))
        im = _to_pil(image)
        scores = {o: self.gdino_score(im, o) for o in objs}
        flagged = [o for o, s in scores.items() if s < thr]
        return CaptionReport(objects=scores, flagged=flagged, threshold=thr)

    # ------------------------------------------------------------ L0/L1: revise
    def revise_caption(self, image: ImageLike, caption: str,
                       strategy: str = "remove",
                       rewrite_fn: Optional[Callable[[ImageLike, str, List[str]], str]] = None,
                       objects: Optional[Sequence[str]] = None,
                       threshold: Optional[float] = None) -> RevisedCaption:
        """Detect-then-revise. strategy="remove" is L0; "rewrite" is L1 and needs
        rewrite_fn(image, caption, flagged) -> new caption (your backbone call)."""
        report = self.verify_caption(image, caption, objects=objects, threshold=threshold)
        if not report.flagged:
            return RevisedCaption(text=caption, report=report, strategy="none")
        if strategy == "remove":
            return RevisedCaption(text=sentence_remove(caption, report.flagged),
                                  report=report, strategy="remove")
        if strategy == "rewrite":
            if rewrite_fn is None:
                raise ValueError('strategy="rewrite" (L1) needs rewrite_fn — '
                                 "a callable that re-prompts YOUR backbone.")
            return RevisedCaption(text=rewrite_fn(image, caption, report.flagged),
                                  report=report, strategy="rewrite")
        raise ValueError(f"unknown strategy: {strategy!r} (use 'remove' or 'rewrite')")

    # ------------------------------------------------------------ L2: decision fusion
    def calibrate(self, records: Sequence[tuple], lam_grid: Optional[Sequence[float]] = None
                  ) -> FusionCalibration:
        """Fit the fusion on held-out records [(p_yes, g, label), ...], label in {0,1}.

        Standardizes both sources, grid-searches the single weight lambda for best
        accuracy at a threshold matching the raw model's own yes-ratio (so the
        fused rule cannot win by just answering 'no' more often — Eq. 13).
        """
        import numpy as np
        arr = np.asarray(records, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 20:
            raise ValueError("need >=20 records of (p_yes, g, label)")
        p, g, y = arr[:, 0], arr[:, 1], arr[:, 2]
        mu_p, sd_p = float(p.mean()), float(p.std() + 1e-8)
        mu_g, sd_g = float(g.mean()), float(g.std() + 1e-8)
        pz, gz = (p - mu_p) / sd_p, (g - mu_g) / sd_g
        yes_ratio = float((p >= 0.5).mean())          # the raw model's own base rate
        best = None
        for lam in (lam_grid if lam_grid is not None else np.linspace(0, 3, 31)):
            z = pz + lam * gz
            tau = float(np.quantile(z, 1 - yes_ratio))
            acc = float(((z >= tau).astype(int) == y).mean())
            if best is None or acc > best[0]:
                best = (acc, float(lam), tau)
        _, lam, tau = best
        self.calibration = FusionCalibration(mu_p, sd_p, mu_g, sd_g, lam, tau)
        return self.calibration

    def answer(self, image: ImageLike, question: str, p_yes: float,
               obj: Optional[str] = None) -> dict:
        """Fused yes/no decision (L2). p_yes is YOUR backbone's first-token
        probability of 'yes'; obj overrides extraction from the question."""
        if self.calibration is None:
            raise RuntimeError("call calibrate(records) first (or set .calibration)")
        if obj is None:
            found = sorted(_extract_objects(question))
            if not found:
                raise ValueError("could not extract an object from the question; pass obj=")
            obj = found[0]
        g = self.clip_score(image, obj)
        c = self.calibration
        z = (p_yes - c.mu_p) / c.sd_p + c.lam * (g - c.mu_g) / c.sd_g
        return {"answer": "yes" if z >= c.tau else "no",
                "z": z, "g": g, "p_yes": p_yes, "object": obj}
