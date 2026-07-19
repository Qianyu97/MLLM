"""Stage C -- HHD detector supervision from preference pairs.

Supervise the three Hierarchical Hallucination Detectors
(:class:`cmpsa.models.hhd.ObjectDetector`,
:class:`~cmpsa.models.hhd.AttributeDetector`,
:class:`~cmpsa.models.hhd.RelationDetector`) using the *chosen* / *rejected*
preference pairs from RLAIF-V (primary, ~12 GB) and RLHF-V (auxiliary).

Label construction
------------------
For each preference pair we treat ``chosen`` as faithful and ``rejected`` as
containing localized hallucinations. We diff the two responses at the token /
segment level: tokens that appear in ``rejected`` but not in the aligned span of
``chosen`` are the *hallucinated* segment and receive label ``1``; tokens shared
with ``chosen`` receive label ``0``. We additionally classify each hallucinated
segment into object / attribute / relation by matching against the
object / attribute / relation vocabularies, producing per-detector token-level
binary targets.

Data reading
------------
RLAIF-V / RLHF-V parquet files embed image bytes. We read them directly with
``pyarrow`` (or via the jsonl exported by ``parquet_to_images`` if present). For
HHD training we only need text + a single pooled visual feature per image, so we
keep the pipeline light.

Run as::

    python -m cmpsa.train.train_hhd --limit 32
    python -m cmpsa.train.train_hhd --config configs/exp.yaml

``torch`` / ``transformers`` are imported lazily; ``--help`` works without a GPU.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, read_jsonl, set_seed

LOG = get_logger("cmpsa.train_hhd")

# Lightweight relation/attribute cue lists for segment typing (no data needed).
_RELATION_CUES = {
    "on", "under", "above", "below", "behind", "next", "near", "beside",
    "inside", "in", "front", "left", "right", "holding", "riding", "wearing",
    "sitting", "standing", "lying", "between", "over", "beneath", "atop",
}
_ATTRIBUTE_CUES = {
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
    "pink", "brown", "gray", "grey", "large", "small", "big", "tiny", "tall",
    "short", "long", "round", "square", "old", "new", "young", "wooden",
    "metal", "plastic", "bright", "dark", "shiny", "wet", "dry", "open",
    "closed", "empty", "full", "striped", "spotted",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


# --------------------------------------------------------------------------- #
# Preference-pair reading (pure-python, no torch)
# --------------------------------------------------------------------------- #
def _read_parquet_pairs(parquet: Path, has_struct_text: bool,
                        limit: int | None) -> Iterator[dict]:
    """Yield ``{"question","chosen","rejected","image_bytes"}`` from one parquet.

    ``has_struct_text=True`` -> RLHF-V layout (``text`` is a JSON string).
    ``has_struct_text=False`` -> RLAIF-V layout (flat columns).
    """
    import pyarrow.parquet as pq

    if not parquet.exists():
        LOG.warning("parquet missing: %s", parquet)
        return
    pf = pq.ParquetFile(parquet)
    seen = 0
    for batch in pf.iter_batches(batch_size=256):
        cols = batch.to_pydict()
        n = len(cols.get("image", cols.get("question", [])))
        for i in range(n):
            if has_struct_text:
                try:
                    t = json.loads(cols["text"][i])
                except Exception:
                    continue
                q, chosen, rejected = t.get("question"), t.get("chosen"), t.get("rejected")
            else:
                q = cols.get("question", [None] * n)[i]
                chosen = cols.get("chosen", [None] * n)[i]
                rejected = cols.get("rejected", [None] * n)[i]
            if not chosen or not rejected:
                continue
            img = cols.get("image", [None] * n)[i]
            img_bytes = None
            if isinstance(img, dict):
                img_bytes = img.get("bytes")
            yield {"question": q, "chosen": chosen, "rejected": rejected,
                   "image_bytes": img_bytes}
            seen += 1
            if limit is not None and seen >= limit:
                return


def _read_jsonl_pairs(jsonl: Path, limit: int | None) -> Iterator[dict]:
    """Yield pairs from a jsonl exported by parquet_to_images (if present)."""
    n = 0
    for row in read_jsonl(jsonl):
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not chosen or not rejected:
            # RLHF-V style nested text
            txt = row.get("text")
            if isinstance(txt, str):
                try:
                    t = json.loads(txt)
                    chosen, rejected = t.get("chosen"), t.get("rejected")
                    row["question"] = t.get("question")
                except Exception:
                    pass
        if not chosen or not rejected:
            continue
        img_path = row.get("image") or row.get("image_path")
        yield {"question": row.get("question"), "chosen": chosen,
               "rejected": rejected, "image_path": img_path, "image_bytes": None}
        n += 1
        if limit is not None and n >= limit:
            return


def iter_preference_pairs(cfg, limit: int | None) -> Iterator[dict]:
    """Iterate RLAIF-V (primary) then RLHF-V (auxiliary) preference pairs.

    Prefers a ``parquet_to_images`` jsonl export when available; otherwise reads
    the parquet files directly with pyarrow.
    """
    wanted = list(getattr(cfg.hhd, "data", []) or ["rlaif_v", "rlhf_v"])
    remaining = limit

    if "rlaif_v" in wanted:
        # jsonl export first
        jsonl = paths.RLAIF_V_DIR / "rlaif_v.jsonl"
        if jsonl.exists():
            yield from _take(_read_jsonl_pairs(jsonl, remaining), remaining)
        else:
            for pq_file in paths.rlaif_v_parquets():
                if remaining is not None and remaining <= 0:
                    break
                for row in _read_parquet_pairs(pq_file, has_struct_text=False, limit=remaining):
                    yield row
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            break

    if "rlhf_v" in wanted and (limit is None or remaining is None or remaining > 0):
        jsonl = paths.RLHF_V_PARQUET.parent / "rlhf_v.jsonl"
        rem2 = remaining
        if jsonl.exists():
            yield from _take(_read_jsonl_pairs(jsonl, rem2), rem2)
        else:
            yield from _read_parquet_pairs(paths.RLHF_V_PARQUET, has_struct_text=True,
                                           limit=rem2)


def _take(it: Iterator[dict], n: int | None) -> Iterator[dict]:
    if n is None:
        yield from it
        return
    c = 0
    for x in it:
        if c >= n:
            return
        yield x
        c += 1


# --------------------------------------------------------------------------- #
# Segment-level -> token-level label construction
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _segment_type(token: str) -> str:
    """Heuristically type a hallucinated token: relation | attribute | object."""
    if token in _RELATION_CUES:
        return "relation"
    if token in _ATTRIBUTE_CUES:
        return "attribute"
    return "object"


def build_token_labels(chosen: str, rejected: str) -> list[dict]:
    """Diff chosen vs rejected -> per-token labels on the rejected response.

    Tokens of ``rejected`` not present in ``chosen`` are flagged hallucination
    (label 1) and typed; shared tokens are faithful (label 0). Returns one dict
    per rejected token: ``{"token","label","type"}``.
    """
    chosen_set = set(_tokens(chosen))
    out: list[dict] = []
    for tok in _tokens(rejected):
        if tok in chosen_set:
            out.append({"token": tok, "label": 0, "type": _segment_type(tok)})
        else:
            out.append({"token": tok, "label": 1, "type": _segment_type(tok)})
    return out


def make_examples(cfg, limit: int | None) -> list[dict]:
    """Materialize Stage-C training examples with token-level labels."""
    examples: list[dict] = []
    for pair in iter_preference_pairs(cfg, limit):
        labels = build_token_labels(pair["chosen"], pair["rejected"])
        if not labels:
            continue
        examples.append({
            "question": pair.get("question"),
            "rejected": pair["rejected"],
            "chosen": pair["chosen"],
            "image_bytes": pair.get("image_bytes"),
            "image_path": pair.get("image_path"),
            "token_labels": labels,
        })
        if limit is not None and len(examples) >= limit:
            break
    # quick label stats
    by_type = {"object": [0, 0], "attribute": [0, 0], "relation": [0, 0]}
    for ex in examples:
        for tl in ex["token_labels"]:
            by_type[tl["type"]][tl["label"]] += 1
    LOG.info("Stage-C examples=%d  per-type (neg/pos): obj=%s attr=%s rel=%s",
             len(examples), by_type["object"], by_type["attribute"], by_type["relation"])
    return examples


# --------------------------------------------------------------------------- #
# Visual / text features (lazy heavy imports)
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


def _load_image(ex) -> "object":
    """Decode an example's image to a PIL.Image (RGB) or None."""
    from PIL import Image

    if ex.get("image_bytes"):
        try:
            return Image.open(io.BytesIO(ex["image_bytes"])).convert("RGB")
        except Exception:
            return None
    p = ex.get("image_path")
    if p and Path(p).exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None
    return None


class _Encoders:
    """Frozen visual + text backbones to feed the detectors during supervision."""

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
        self.clip_proc = CLIPImageProcessor.from_pretrained(vsrc)
        self.clip = CLIPVisionModel.from_pretrained(vsrc).to(self.device).eval()

        tkey = cfg.text_backbone.key
        tlocal = _resolve_local_dir(cfg, tkey)
        tsrc = str(tlocal) if tlocal.exists() else getattr(cfg.models, tkey).hf_id
        self.tok = AutoTokenizer.from_pretrained(tsrc)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.text = AutoModel.from_pretrained(tsrc, output_hidden_states=True).to(self.device).eval()

    def visual_psas(self, image, pve):
        torch = self.torch
        inputs = self.clip_proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.clip(**inputs)
        patch = out.last_hidden_state[:, 1:, :].mean(dim=1, keepdim=True).float()
        mu, logvar = pve(patch)
        return mu, logvar

    def text_psas(self, tokens: list[str], ple):
        """Per-token language PSAS Gaussians, shaped ``[N_tokens, psas_dim]``.

        Each token string is encoded independently (as a padded batch) and its
        sub-word hidden states are mean-pooled, so there is exactly one Gaussian
        per input token and ``ple_index`` lines up with the token list used by
        :meth:`cmpsa.models.hhd.HHD.detect`.
        """
        torch = self.torch
        if not tokens:
            raise ValueError("text_psas received an empty token list")
        enc = self.tok(list(tokens), return_tensors="pt", padding=True,
                       truncation=True, max_length=16).to(self.device)
        with torch.no_grad():
            out = self.text(**enc)
        hs = out.hidden_states[-1] if getattr(out, "hidden_states", None) else out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()              # [N, seq, 1]
        pooled = (hs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [N, D]
        mu, logvar = ple(pooled[None])                                 # [1, N, psas]
        return mu[0], logvar[0]                                        # [N, psas]


def _build_heads_and_load(cfg, torch):
    """Instantiate PVE/PLE and load the Stage-B (or Stage-A) checkpoint."""
    from cmpsa.models.pve_ple import PLEHead, PVEHead

    pj = cfg.projection
    pve = PVEHead(cfg.visual_backbone.feature_dim, pj.psas_dim, pj.hidden_dim,
                  pj.min_logvar, pj.max_logvar)
    ple = PLEHead(cfg.text_backbone.feature_dim, pj.psas_dim, pj.hidden_dim,
                  pj.min_logvar, pj.max_logvar)
    for name in ("cmota.pt", "pretrain_proj.pt"):
        ckpt = paths.CKPT_DIR / name
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            pve.load_state_dict(state["pve"], strict=False)
            ple.load_state_dict(state["ple"], strict=False)
            LOG.info("loaded projection heads from %s", ckpt)
            break
    else:
        LOG.warning("no projection ckpt found; using randomly-initialized heads")
    return pve, ple


# --------------------------------------------------------------------------- #
# Detector training
# --------------------------------------------------------------------------- #
def _build_hhd(cfg):
    from cmpsa.models.hhd import HHD

    return HHD(cfg)


class _ScoreCalibrator:
    """Per-type Platt calibrator turning a raw detector score into a logit.

    ``logit = scale_t * (score - 0.5) + bias_t``. The raw [0,1] detector score is
    a fixed feature; Stage-C learns, per type, the affine mapping it to a
    hallucination logit. The *sign* of ``scale`` is learned, so OLD/RLD (low score
    = hallucination) and ALD (high score = hallucination) are handled
    automatically. This calibrates the non-parametric OLD/ALD judgment rules'
    decision boundary on labelled data while PVE/PLE stay frozen from Stage-A/B.
    """

    TYPES = ("object", "attribute", "relation")

    def __init__(self, torch, device):
        self.torch = torch
        self.scale = torch.nn.Parameter(torch.ones(len(self.TYPES), device=device))
        self.bias = torch.nn.Parameter(torch.zeros(len(self.TYPES), device=device))

    def parameters(self):
        return [self.scale, self.bias]

    def logit(self, score, ttype):
        i = self.TYPES.index(ttype if ttype in self.TYPES else "object")
        s = self.torch.as_tensor(float(score), dtype=self.scale.dtype,
                                 device=self.scale.device)
        return (self.scale[i] * (s - 0.5) + self.bias[i]).reshape(1)

    def state(self):
        return {"types": list(self.TYPES),
                "scale": self.scale.detach().cpu().tolist(),
                "bias": self.bias.detach().cpu().tolist()}


def _dict_tokens(token_labels: list[dict]) -> list[dict]:
    """Convert token-level labels to the dict tokens that HHD.detect() expects.

    object/attribute -> ``{"type","ple_index"}``; a relation cue word additionally
    takes its neighbouring tokens as subject/object endpoints.
    """
    n = len(token_labels)
    out = []
    for k, tl in enumerate(token_labels):
        ttype = tl.get("type", "object")
        d = {"type": ttype, "ple_index": k}
        if ttype == "relation":
            d["subj_index"] = max(0, k - 1)
            d["obj_index"] = min(n - 1, k + 1)
        out.append(d)
    return out


def train(cfg, examples: list[dict], limit: int | None) -> Path:
    """Stage-C: supervise the HHD detectors on token-level hallucination labels.

    Trainable components: (i) a per-type score calibrator for the non-parametric
    OLD/ALD judgment rules (Platt scaling of their scores) and (ii) the RLD
    relation head, trained differentiably on relation-typed tokens. PVE/PLE are
    frozen. The loop fails loudly instead of silently no-op'ing if examples cannot
    be encoded or zero optimisation steps run.
    """
    import torch
    from cmpsa.models.hhd import HHD

    if not examples:
        raise RuntimeError("empty Stage-C example set; cannot train")

    dev = _device(cfg)
    encoders = _Encoders(cfg)
    pve, ple = _build_heads_and_load(cfg, torch)
    pve.to(dev).eval()
    ple.to(dev).eval()

    psas = int(cfg.projection.psas_dim)
    rel_head = torch.nn.Sequential(
        torch.nn.Linear(3 * psas, psas),
        torch.nn.GELU(),
        torch.nn.Linear(psas, 1),
    ).to(dev).train()
    hhd = HHD(cfg, rel_head=rel_head)
    calib = _ScoreCalibrator(torch, dev)

    params = calib.parameters() + list(rel_head.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg.hhd.lr))
    bce = torch.nn.BCEWithLogitsLoss()
    epochs = int(cfg.hhd.epochs)
    LOG.info("Stage-C calibrate: n=%d epochs=%d lr=%g dev=%s",
             len(examples), epochs, cfg.hhd.lr, dev)

    step = 0
    encode_fail = 0
    for ep in range(epochs):
        random.Random(cfg.seed + ep).shuffle(examples)
        for ex in examples:
            image = _load_image(ex)
            if image is None:
                continue
            tok_labels = ex["token_labels"]
            tokens = [tl["token"] for tl in tok_labels]
            try:
                v = encoders.visual_psas(image, pve)            # ([1,1,psas], ...)
                t = encoders.text_psas(tokens, ple)             # ([N,psas], [N,psas])
            except Exception as e:
                encode_fail += 1
                if step == 0 and encode_fail >= 16:
                    raise RuntimeError(
                        f"Stage-C: the first {encode_fail} examples all failed to "
                        f"encode (last error: {e}); aborting instead of training on "
                        "nothing. Check backbone weights under MODELS_ROOT.") from e
                LOG.warning("encode failed (%d): %s", encode_fail, e)
                continue

            preds = hhd.detect(_dict_tokens(tok_labels), v, t)
            t_mu = t[0].reshape(-1, psas).detach()
            v_mean = v[0].reshape(-1, psas).mean(dim=0).detach()

            losses = []
            for k, (tl, pr) in enumerate(zip(tok_labels, preds)):
                if tl.get("type") == "relation" and t_mu.shape[0] > 0:
                    s_idx = max(0, k - 1)
                    o_idx = min(t_mu.shape[0] - 1, k + 1)
                    feat = torch.cat([t_mu[s_idx], t_mu[o_idx], v_mean], dim=-1)
                    logit = rel_head(feat).reshape(1)             # differentiable RLD head
                else:
                    logit = calib.logit(pr["score"], pr["type"])  # differentiable in scale/bias
                target = torch.as_tensor(float(tl["label"]), device=dev,
                                         dtype=logit.dtype).reshape(1)
                losses.append(bce(logit, target))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0 or limit is not None:
                LOG.info("ep%d step%d L_hhd=%.4f calib_scale=%s", ep, step,
                         float(loss.item()),
                         [round(x, 3) for x in calib.scale.detach().cpu().tolist()])

    if step == 0:
        raise RuntimeError("Stage-C ran 0 optimisation steps (all examples skipped); "
                           "refusing to save a no-op checkpoint.")
    return _save_ckpt(cfg, hhd, torch, steps=step, calibrator=calib, rel_head=rel_head)


def _save_ckpt(cfg, hhd, torch, steps: int, calibrator=None, rel_head=None) -> Path:
    """Persist the Stage-C calibrator + relation head + thresholds."""
    ckpt = paths.CKPT_DIR / "hhd.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {"stage": "C_hhd", "steps": steps,
             "thresholds": {"tau_obj": cfg.hhd.tau_obj, "tau_attr": cfg.hhd.tau_attr,
                            "tau_rel": cfg.hhd.tau_rel}}
    if calibrator is not None:
        state["calibrator"] = calibrator.state()
    if rel_head is not None:
        state["rel_head"] = rel_head.state_dict()
    try:
        if hasattr(hhd, "state_dict"):
            state["hhd"] = hhd.state_dict()
    except Exception as e:  # pragma: no cover
        LOG.warning("hhd.state_dict() unavailable: %s", e)
    torch.save(state, ckpt)
    LOG.info("Stage-C: saved HHD (calibrator+rel_head, %d steps) -> %s", steps, ckpt)
    return ckpt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage C: supervise HHD detectors from RLAIF-V/RLHF-V preference pairs.",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="cap #preference pairs for a smoke test")
    p.add_argument("--config", default=None,
                   help="optional YAML override merged on default.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="build token-level labels and print stats without training")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    paths.ensure_dirs()

    examples = make_examples(cfg, args.limit)
    LOG.info("Stage-C: %d labeled examples", len(examples))

    if args.dry_run:
        if examples:
            ex = examples[0]
            LOG.info("example rejected: %s", ex["rejected"][:120])
            flagged = [tl for tl in ex["token_labels"] if tl["label"] == 1]
            LOG.info("flagged tokens (%d): %s", len(flagged),
                     [(t["token"], t["type"]) for t in flagged[:10]])
        LOG.info("--dry-run set; skipping training")
        return 0
    if not examples:
        LOG.error("no preference pairs available; aborting "
                  "(expected without RLAIF-V/RLHF-V present)")
        return 1

    train(cfg, examples, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
