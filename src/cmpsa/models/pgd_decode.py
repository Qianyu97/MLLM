"""Live PGD decoding integration for llava-hf models.

This module wires CMPSA's HHD/PGD signals into generation rather than using a
post-hoc pass-through wrapper. It implements two interventions:

* caption/free-form generation: a HuggingFace ``LogitsProcessor`` inspects the
  current top-k next-token candidates, projects semantic candidates into PSAS,
  runs HHD, and subtracts ``lambda * drift`` from flagged token logits.
* yes/no scoring: the question's semantic probes are grounded against the image
  with HHD; unsupported probes penalize the "Yes" logit before the Yes/No
  softmax.

The implementation is intentionally top-k rather than full-vocabulary. It is
the practical decoding path for one GPU: every step touches only the candidates
the model is already likely to emit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cmpsa import paths
from cmpsa.utils import get_logger, load_json

LOG = get_logger("cmpsa.models.pgd_decode")


_ATTR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
    "pink", "brown", "gray", "grey", "large", "small", "big", "tiny", "tall",
    "short", "long", "round", "square", "old", "new", "young", "wooden",
    "metal", "plastic", "bright", "dark", "shiny", "wet", "dry", "open",
    "closed", "empty", "full", "striped", "spotted", "clear", "dirty",
}
_REL_WORDS = {
    "on", "under", "above", "below", "behind", "near", "beside", "inside",
    "front", "left", "right", "holding", "riding", "wearing", "sitting",
    "standing", "lying", "between", "over", "beneath", "atop", "next",
}
_COMMON_OBJECTS = {
    "person", "man", "woman", "boy", "girl", "people", "child", "dog", "cat",
    "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "car", "bus", "truck", "train", "boat", "bicycle", "motorcycle", "plane",
    "chair", "couch", "table", "bed", "bench", "bottle", "cup", "bowl",
    "fork", "knife", "spoon", "apple", "banana", "orange", "pizza", "cake",
    "book", "clock", "vase", "laptop", "phone", "keyboard", "remote", "tv",
    "umbrella", "backpack", "bag", "ball", "kite", "skateboard", "surfboard",
}
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "there", "here",
    "is", "are", "was", "were", "be", "being", "been", "to", "of", "in",
    "for", "with", "and", "or", "but", "as", "at", "by", "from", "it",
    "its", "image", "picture", "photo", "scene", "yes", "no", "not",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass(frozen=True)
class SemanticCandidate:
    vocab_id: int
    surface: str
    kind: str


def _get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _resolve_model_source(cfg, model_key: str) -> str:
    entry = getattr(cfg.models, model_key)
    local = paths.MODELS_ROOT / entry.local_dir
    return str(local) if local.exists() else entry.hf_id


def _merge_peft_linear_state(state: dict[str, Any], lora_rank: int | None = None) -> dict[str, Any]:
    """Convert a PEFT-wrapped projection-head state dict into plain Linear keys."""
    if not any(k.startswith("base_model.model.") for k in state):
        return state
    import torch

    rank = int(lora_rank or 16)
    scale = 2.0  # train_cmota uses lora_alpha=2*r, hence alpha/r = 2.
    out: dict[str, Any] = {}
    modules = ["trunk.0", "trunk.2", "mu_head", "logvar_head"]
    for mod in modules:
        base = f"base_model.model.{mod}"
        w_key = f"{base}.base_layer.weight"
        b_key = f"{base}.base_layer.bias"
        if w_key not in state:
            continue
        weight = state[w_key].clone()
        a_key = f"{base}.lora_A.default.weight"
        b_lora_key = f"{base}.lora_B.default.weight"
        if a_key in state and b_lora_key in state:
            update = torch.matmul(state[b_lora_key], state[a_key]) * scale
            if update.shape == weight.shape:
                weight = weight + update
        out[f"{mod}.weight"] = weight
        if b_key in state:
            out[f"{mod}.bias"] = state[b_key]
    return out


class CMPSAPGDDecoder:
    """PGD-aware decoder for an already-loaded llava-hf MLLM wrapper."""

    def __init__(self, mllm, cfg, hhd, pgd):
        if getattr(mllm, "family", None) != "llava-hf":
            raise NotImplementedError("CMPSA live PGD decoding currently supports llava-hf models only.")
        self.mllm = mllm
        self.cfg = cfg
        self.hhd = hhd
        self.pgd = pgd
        self.model = mllm.model
        self.processor = mllm.processor
        self.tokenizer = self.processor.tokenizer
        self.device = self.model.device

        p = getattr(cfg, "pgd", None)
        self.top_k = int(_get(p, "decode_top_k", 24))
        self.max_semantic = int(_get(p, "decode_max_semantic", 12))
        self.yesno_weight = float(_get(p, "yesno_weight", 1.0))
        self.caption_enabled = bool(_get(p, "caption_decode", True))
        self.yesno_enabled = bool(_get(p, "yesno_decode", True))

        self._loaded = False
        self._text_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.object_words, self.object_phrases = self._load_object_vocab()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, image, prompt: str, max_new_tokens: int = 64) -> str:
        if not self.caption_enabled:
            return self.mllm.generate(image, prompt, max_new_tokens=max_new_tokens)
        self._ensure_loaded()

        import torch
        from transformers import LogitsProcessorList

        full_prompt = self.mllm._format_prompt(prompt)
        inputs = self.processor(images=image, text=full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        visual_psas = self.visual_psas(image)
        logits_processor = LogitsProcessorList([
            _CMPSALogitsProcessor(self, visual_psas=visual_psas)
        ])

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                num_beams=1,
                logits_processor=logits_processor,
            )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1]
        return text.strip()

    def score_yes_no(self, image, question: str) -> tuple[str, float]:
        if not self.yesno_enabled:
            return self.mllm.score_yes_no(image, question)
        self._ensure_loaded()

        import torch

        prompt = (f"{question} Please answer this question with one word, "
                  f"either Yes or No.")
        full_prompt = self.mllm._format_prompt(prompt)
        inputs = self.processor(images=image, text=full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        logits = out.scores[0][0].clone()
        risk = self.question_yes_risk(image, question)
        if risk > 0:
            drift = torch.zeros_like(logits)
            yes_ids = self.mllm._token_ids(self.tokenizer, ["Yes", "yes", "▁Yes"])
            for tid in yes_ids:
                if 0 <= tid < drift.numel():
                    drift[tid] = max(float(drift[tid]), risk * self.yesno_weight)
            logits = self.pgd.reweight(logits, drift)

        yes_ids = self.mllm._token_ids(self.tokenizer, ["Yes", "yes", "▁Yes"])
        no_ids = self.mllm._token_ids(self.tokenizer, ["No", "no", "▁No"])
        logit_yes = logits[yes_ids].max() if yes_ids else logits.new_tensor(-1e30)
        logit_no = logits[no_ids].max() if no_ids else logits.new_tensor(-1e30)
        probs = torch.softmax(torch.stack([logit_yes, logit_no]), dim=0)
        p_yes = float(probs[0])
        return ("yes" if p_yes >= 0.5 else "no"), p_yes

    # ------------------------------------------------------------------ #
    # PSAS stack
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
        from cmpsa.models.pve_ple import PLEHead, PVEHead
        from cmpsa.models.hhd import HHD

        pj = self.cfg.projection
        self.pve = PVEHead(
            self.cfg.visual_backbone.feature_dim,
            pj.psas_dim,
            pj.hidden_dim,
            pj.min_logvar,
            pj.max_logvar,
        ).to(self.device).eval()
        self.ple = PLEHead(
            self.cfg.text_backbone.feature_dim,
            pj.psas_dim,
            pj.hidden_dim,
            pj.min_logvar,
            pj.max_logvar,
        ).to(self.device).eval()

        self._load_projection_heads(torch)

        vsrc = _resolve_model_source(self.cfg, self.cfg.visual_backbone.key)
        tsrc = _resolve_model_source(self.cfg, self.cfg.text_backbone.key)
        LOG.info("CMPSA PGD: loading visual PSAS backbone from %s", vsrc)
        self.clip_proc = CLIPImageProcessor.from_pretrained(vsrc)
        self.clip = CLIPVisionModel.from_pretrained(vsrc).to(self.device).eval()
        LOG.info("CMPSA PGD: loading text PSAS backbone from %s", tsrc)
        self.text_tok = AutoTokenizer.from_pretrained(tsrc)
        if self.text_tok.pad_token is None and self.text_tok.eos_token is not None:
            self.text_tok.pad_token = self.text_tok.eos_token
        self.text_model = AutoModel.from_pretrained(tsrc, output_hidden_states=True).to(self.device).eval()

        rel_head = self._load_relation_head(torch)
        self.hhd = HHD(self.cfg, rel_head=rel_head)
        self._loaded = True

    def _load_projection_heads(self, torch) -> None:
        for name in ("cmota.pt", "pretrain_proj.pt"):
            ckpt = paths.CKPT_DIR / name
            if not ckpt.exists():
                continue
            state = torch.load(ckpt, map_location="cpu")
            lora_rank = state.get("lora_rank")
            pve_state = _merge_peft_linear_state(state.get("pve", {}), lora_rank)
            ple_state = _merge_peft_linear_state(state.get("ple", {}), lora_rank)
            pve_missing, pve_unexpected = self.pve.load_state_dict(pve_state, strict=False)
            ple_missing, ple_unexpected = self.ple.load_state_dict(ple_state, strict=False)
            LOG.info("CMPSA PGD: loaded PVE/PLE from %s (pve missing=%d unexpected=%d; "
                     "ple missing=%d unexpected=%d)",
                     ckpt, len(pve_missing), len(pve_unexpected),
                     len(ple_missing), len(ple_unexpected))
            return
        LOG.warning("CMPSA PGD: no projection checkpoint found; using random PVE/PLE heads")

    def _load_relation_head(self, torch):
        ckpt = paths.CKPT_DIR / "hhd.pt"
        if not ckpt.exists():
            return None
        state = torch.load(ckpt, map_location="cpu")
        rel_state = state.get("rel_head")
        if not rel_state:
            return None
        psas = int(self.cfg.projection.psas_dim)
        rel_head = torch.nn.Sequential(
            torch.nn.Linear(3 * psas, psas),
            torch.nn.GELU(),
            torch.nn.Linear(psas, 1),
        ).to(self.device).eval()
        rel_head.load_state_dict(rel_state, strict=False)
        LOG.info("CMPSA PGD: loaded HHD relation head from %s", ckpt)
        return rel_head

    def visual_psas(self, image):
        import torch

        self._ensure_loaded()
        inputs = self.clip_proc(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.clip(**inputs)
            patch = out.last_hidden_state[:, 1:, :].float()
            mu, logvar = self.pve(patch)
        return mu.squeeze(0).detach(), logvar.squeeze(0).detach()

    def text_psas(self, surfaces: list[str], kinds: list[str] | None = None):
        import torch

        self._ensure_loaded()
        kinds = kinds or ["object"] * len(surfaces)
        cached: list[tuple[Any, Any] | None] = []
        missing_surfaces: list[str] = []
        missing_keys: list[tuple[str, str]] = []
        for surface, kind in zip(surfaces, kinds):
            key = (kind, surface.lower())
            val = self._text_cache.get(key)
            cached.append(val)
            if val is None:
                missing_surfaces.append(surface)
                missing_keys.append(key)

        if missing_surfaces:
            enc = self.text_tok(
                missing_surfaces,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16,
            ).to(self.device)
            with torch.inference_mode():
                out = self.text_model(**enc)
                hs = out.hidden_states[-1] if getattr(out, "hidden_states", None) else out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                mu, logvar = self.ple(pooled)
            for key, m, lv in zip(missing_keys, mu, logvar):
                self._text_cache[key] = (m.detach(), lv.detach())

        mus, logvars = [], []
        for surface, kind in zip(surfaces, kinds):
            m, lv = self._text_cache[(kind, surface.lower())]
            mus.append(m.to(self.device))
            logvars.append(lv.to(self.device))
        return torch.stack(mus, dim=0), torch.stack(logvars, dim=0)

    # ------------------------------------------------------------------ #
    # HHD/PGD scoring
    # ------------------------------------------------------------------ #
    def candidates_from_logits(self, scores) -> list[SemanticCandidate]:
        import torch

        k = min(self.top_k, int(scores.shape[-1]))
        top = torch.topk(scores, k=k, dim=-1)
        out: list[SemanticCandidate] = []
        seen = set()
        for tid in top.indices.detach().cpu().tolist():
            if tid in seen:
                continue
            seen.add(tid)
            surface = self._clean_surface(self.tokenizer.decode([int(tid)], skip_special_tokens=True))
            kind = self.classify_surface(surface)
            if kind is None:
                continue
            out.append(SemanticCandidate(vocab_id=int(tid), surface=surface, kind=kind))
            if len(out) >= self.max_semantic:
                break
        return out

    def drift_for_candidates(self, candidates: list[SemanticCandidate], visual_psas, vocab_size: int):
        import torch

        drift = torch.zeros(vocab_size, dtype=torch.float32, device=self.device)
        if not candidates:
            return drift
        surfaces = [c.surface for c in candidates]
        kinds = [c.kind for c in candidates]
        l_mu, l_logvar = self.text_psas(surfaces, kinds)
        tokens = []
        for i, kind in enumerate(kinds):
            if kind == "relation":
                tokens.append({"type": "relation", "subj_index": i, "obj_index": i})
            else:
                tokens.append({"type": kind, "ple_index": i})
        with torch.inference_mode():
            detections = self.hhd.detect(tokens, visual_psas, (l_mu, l_logvar))
        for cand, det in zip(candidates, detections):
            risk = self._risk_from_detection(det)
            if det.get("flag", False) and risk > 0:
                drift[cand.vocab_id] = max(float(drift[cand.vocab_id]), risk)
        return drift

    def question_yes_risk(self, image, question: str) -> float:
        probes = self.extract_question_probes(question)
        if not probes:
            return 0.0
        visual = self.visual_psas(image)
        surfaces = [p.surface for p in probes]
        kinds = [p.kind for p in probes]
        l_mu, l_logvar = self.text_psas(surfaces, kinds)
        tokens = []
        for i, kind in enumerate(kinds):
            if kind == "relation":
                tokens.append({"type": "relation", "subj_index": i, "obj_index": i})
            else:
                tokens.append({"type": kind, "ple_index": i})
        dets = self.hhd.detect(tokens, visual, (l_mu, l_logvar))
        if not dets:
            return 0.0
        return max(self._risk_from_detection(d) for d in dets)

    @staticmethod
    def _risk_from_detection(det: dict[str, Any]) -> float:
        t = det.get("type", "object")
        s = float(det.get("score", 0.0))
        if t == "attribute":
            return max(0.0, min(1.0, s))
        return max(0.0, min(1.0, 1.0 - s))

    # ------------------------------------------------------------------ #
    # Semantic typing
    # ------------------------------------------------------------------ #
    def _load_object_vocab(self) -> tuple[set[str], set[str]]:
        words = set(_COMMON_OBJECTS)
        phrases = set()
        for ann in (paths.COCO_INSTANCES_VAL2017, paths.COCO_INSTANCES_VAL2014):
            if not ann.exists():
                continue
            try:
                for cat in load_json(ann).get("categories", []):
                    name = str(cat.get("name", "")).lower()
                    if not name:
                        continue
                    phrases.add(name)
                    words.update(_WORD_RE.findall(name))
            except Exception as exc:  # noqa: BLE001
                LOG.warning("CMPSA PGD: could not read COCO object vocab %s: %s", ann, exc)
        words = {w for w in words if len(w) > 1 and w not in _STOPWORDS}
        return words, phrases

    @staticmethod
    def _clean_surface(surface: str) -> str:
        surface = surface.replace("▁", " ").strip().lower()
        surface = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", surface)
        return surface

    def classify_surface(self, surface: str) -> str | None:
        s = self._clean_surface(surface)
        if not s or len(s) < 2 or s in _STOPWORDS:
            return None
        if s in _REL_WORDS:
            return "relation"
        if s in _ATTR_WORDS:
            return "attribute"
        if s in self.object_words or s in self.object_phrases:
            return "object"
        return None

    def extract_question_probes(self, question: str) -> list[SemanticCandidate]:
        q = question.lower()
        out: list[SemanticCandidate] = []
        seen: set[tuple[str, str]] = set()

        for phrase in sorted(self.object_phrases, key=len, reverse=True):
            if " " in phrase and re.search(rf"\b{re.escape(phrase)}\b", q):
                key = ("object", phrase)
                if key not in seen:
                    out.append(SemanticCandidate(-1, phrase, "object"))
                    seen.add(key)

        for word in _WORD_RE.findall(q):
            w = word.lower()
            kind = None
            if w in _REL_WORDS:
                kind = "relation"
            elif w in _ATTR_WORDS:
                kind = "attribute"
            elif w in self.object_words:
                kind = "object"
            if kind is None:
                continue
            key = (kind, w)
            if key in seen:
                continue
            out.append(SemanticCandidate(-1, w, kind))
            seen.add(key)
            if len(out) >= self.max_semantic:
                break
        return out


class _CMPSALogitsProcessor:
    """Top-k semantic HHD/PGD logits processor."""

    def __init__(self, decoder: CMPSAPGDDecoder, visual_psas):
        self.decoder = decoder
        self.visual_psas = visual_psas

    def __call__(self, input_ids, scores):
        if scores.shape[0] != 1:
            return scores
        candidates = self.decoder.candidates_from_logits(scores[0])
        if not candidates:
            return scores
        drift = self.decoder.drift_for_candidates(
            candidates,
            self.visual_psas,
            vocab_size=int(scores.shape[-1]),
        ).to(dtype=scores.dtype, device=scores.device)
        if bool((drift != 0).any()):
            scores = scores.clone()
            scores[0] = self.decoder.pgd.reweight(scores[0], drift)
        return scores
