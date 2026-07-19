"""MLLM wrapper: a thin, uniform interface over multimodal LLMs.

The wrapper exposes two operations used throughout evaluation:

* :meth:`MLLM.generate` — free-form caption / answer generation.
* :meth:`MLLM.score_yes_no` — for yes/no benchmarks (POPE / AMBER discriminative
  / HallusionBench): compares the model's first-token probability of ``"Yes"``
  vs ``"No"`` and returns the chosen label plus the probability of "yes".

``load_mllm`` resolves model weights from ``paths.MODELS_ROOT/<local_dir>`` when
that directory exists locally, otherwise from the HuggingFace id
``cfg.models.<key>.hf_id``.  The **llava-hf** family is implemented concretely
with ``transformers`` (``AutoProcessor`` + ``LlavaForConditionalGeneration``).
Other model keys raise :class:`NotImplementedError` naming the key.

``torch`` / ``transformers`` are imported lazily *inside* ``load_mllm`` /
methods, so importing this module (and ``--help``) works without them.
"""
from __future__ import annotations

import argparse

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_model_source(model_key: str, cfg) -> tuple:
    """Return ``(source, entry)`` where ``source`` is a local dir or an HF id.

    Prefers a local directory under ``MODELS_ROOT/<local_dir>`` if it exists,
    otherwise falls back to the HuggingFace id from the registry.
    """
    try:
        entry = getattr(cfg.models, model_key)
    except AttributeError as e:
        raise NotImplementedError(
            f"Unknown model key {model_key!r}: not present in configs/models.yaml. "
            f"Add an entry (hf_id / local_dir / role) before loading it."
        ) from e

    local_dir = getattr(entry, "local_dir", None)
    hf_id = getattr(entry, "hf_id", None)
    if local_dir:
        local_path = paths.MODELS_ROOT / local_dir
        if local_path.exists():
            return str(local_path), entry
    if hf_id:
        return hf_id, entry
    raise FileNotFoundError(
        f"Model {model_key!r} has neither a local dir ({paths.MODELS_ROOT / (local_dir or '?')}) "
        f"nor a usable hf_id. Download the weights or set models.{model_key}.hf_id."
    )


def _is_llava_hf(model_key: str, entry) -> bool:
    """Heuristic: is this a llava-hf (transformers LlavaForConditionalGeneration) model?"""
    hf_id = (getattr(entry, "hf_id", "") or "").lower()
    key = model_key.lower()
    # llava-1.5-* use Llava; llava-1.6 (next) uses a different class -> handled separately.
    return ("llava-hf/llava-1.5" in hf_id) or key.startswith("llava-1.5")


def _is_llava_next(model_key: str, entry) -> bool:
    hf_id = (getattr(entry, "hf_id", "") or "").lower()
    key = model_key.lower()
    return ("llava-v1.6" in hf_id) or ("llava-1.6" in key) or ("llava-next" in hf_id)


# --------------------------------------------------------------------------- #
# MLLM wrapper
# --------------------------------------------------------------------------- #
class MLLM:
    """Uniform wrapper around a loaded multimodal model.

    Attributes
    ----------
    model : the underlying ``transformers`` model (or backend handle).
    processor : the matching processor / tokenizer+image processor.
    key : str — the registry key this wrapper was loaded from.
    """

    def __init__(self, model, processor, key: str, family: str = "llava-hf", cfg=None):
        self.model = model
        self.processor = processor
        self.key = key
        self.family = family
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _format_prompt(self, prompt: str) -> str:
        """Wrap a user prompt into the llava-1.5 chat template with an <image>."""
        # llava-1.5 expects: "USER: <image>\n{prompt} ASSISTANT:"
        return f"USER: <image>\n{prompt} ASSISTANT:"

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, image, prompt, max_new_tokens: int = 64) -> str:
        """Generate free-form text for ``(image, prompt)``.

        ``image`` is a PIL.Image (RGB).  Returns only the assistant's reply
        (the prompt prefix is stripped).
        """
        import torch  # lazy

        full_prompt = self._format_prompt(prompt)
        inputs = self.processor(images=image, text=full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                num_beams=1,
            )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        # Strip everything up to and including the assistant marker.
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1]
        return text.strip()

    # ------------------------------------------------------------------ #
    # Yes/No scoring
    # ------------------------------------------------------------------ #
    def score_yes_no(self, image, question) -> tuple:
        """Return ``(label, p_yes)`` for a yes/no ``question`` about ``image``.

        Compares the **first generated token's** logit probability of "Yes" vs
        "No".  ``label`` is ``"yes"`` if P(Yes) >= P(No) else ``"no"``; ``p_yes``
        is the normalised P(Yes) over the {Yes, No} pair in [0, 1].
        """
        import torch  # lazy

        prompt = (f"{question} Please answer this question with one word, "
                  f"either Yes or No.")
        full_prompt = self._format_prompt(prompt)
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
        # scores[0]: logits for the first generated token, shape [1, vocab].
        logits = out.scores[0][0]
        tok = self.processor.tokenizer

        yes_ids = self._token_ids(tok, ["Yes", "yes", "▁Yes"])
        no_ids = self._token_ids(tok, ["No", "no", "▁No"])

        logit_yes = logits[yes_ids].max() if len(yes_ids) else logits.new_tensor(-1e30)
        logit_no = logits[no_ids].max() if len(no_ids) else logits.new_tensor(-1e30)

        pair = torch.stack([logit_yes, logit_no])
        probs = torch.softmax(pair, dim=0)
        p_yes = float(probs[0])
        label = "yes" if p_yes >= 0.5 else "no"
        return label, p_yes

    @staticmethod
    def _token_ids(tokenizer, candidates) -> list:
        """Collect the vocab ids for a set of candidate surface forms.

        Tries the raw token, a leading-space variant, and ``convert_tokens_to_ids``;
        de-duplicates and drops the unk id.
        """
        ids = set()
        unk = getattr(tokenizer, "unk_token_id", None)
        for c in candidates:
            for variant in (c, " " + c.lstrip("▁ ")):
                try:
                    enc = tokenizer.encode(variant, add_special_tokens=False)
                except Exception:
                    enc = []
                if enc:
                    ids.add(int(enc[0]))
            try:
                tid = tokenizer.convert_tokens_to_ids(c)
                if tid is not None and tid != unk and tid >= 0:
                    ids.add(int(tid))
            except Exception:
                pass
        if unk is not None:
            ids.discard(int(unk))
        return sorted(ids)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_mllm(model_key: str, cfg) -> MLLM:
    """Load a multimodal model by registry key and wrap it in :class:`MLLM`.

    The llava-1.5 family is implemented concretely with ``transformers``.
    Other keys raise :class:`NotImplementedError` naming the key (so the
    evaluation harness degrades with a clear message instead of a cryptic error).
    """
    log = get_logger("cmpsa.models.mllm_wrapper")
    source, entry = _resolve_model_source(model_key, cfg)

    if _is_llava_hf(model_key, entry):
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers + torch are required to load the llava-hf model. "
                "Install them (and a CUDA build of torch for GPU inference)."
            ) from e

        log.info("Loading llava-hf model %r from %s", model_key, source)
        device = getattr(cfg, "device", "cuda")
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32

        processor = AutoProcessor.from_pretrained(source)
        model = LlavaForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to("cuda" if use_cuda else "cpu")
        model.eval()
        return MLLM(model, processor, key=model_key, family="llava-hf", cfg=cfg)

    if _is_llava_next(model_key, entry):
        raise NotImplementedError(
            f"Model key {model_key!r} (llava-1.6 / llava-next) is not implemented yet. "
            f"It needs transformers' LlavaNextForConditionalGeneration + "
            f"LlavaNextProcessor and a different prompt template. Use a 'llava-1.5-*' "
            f"key for now, or extend load_mllm() to add the llava-next branch."
        )

    raise NotImplementedError(
        f"Model key {model_key!r} is not implemented in load_mllm(). "
        f"Currently only the llava-1.5 (llava-hf) family is supported. "
        f"To add it, implement a loader branch for {model_key!r} "
        f"(hf_id={getattr(entry, 'hf_id', None)!r}, "
        f"role={getattr(entry, 'role', None)!r}); models like instructblip-7b, "
        f"qwen-vl-chat, sharegpt4v-7b, llava-1.6-vicuna-7b, minigpt-v2 each need "
        f"their own processor/model classes and prompt template."
    )


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLLM wrapper — load + tiny generation test.")
    p.add_argument("--model", default=None, help="Model key (default: cfg.mllm.key).")
    p.add_argument("--config", default=None, help="Path to an override YAML config.")
    p.add_argument("--image", default=None, help="Optional image path for a generation test.")
    p.add_argument("--prompt", default="Describe this image.", help="Prompt for the test.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log = get_logger("cmpsa.models.mllm_wrapper")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    model_key = args.model or cfg.mllm.key
    log.info("Resolving model %r ...", model_key)
    try:
        source, entry = _resolve_model_source(model_key, cfg)
        log.info("Source resolved: %s (hf_id=%s, local_dir=%s)",
                 source, getattr(entry, "hf_id", None), getattr(entry, "local_dir", None))
    except Exception as e:
        log.error("Could not resolve model source: %s", e)
        return

    try:
        mllm = load_mllm(model_key, cfg)
    except NotImplementedError as e:
        log.warning("Not implemented: %s", e)
        return
    except Exception as e:
        log.error("Failed to load model (weights likely absent): %s", e)
        return

    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        log.info("generate -> %r", mllm.generate(img, args.prompt, cfg.eval.max_new_tokens))
        label, p_yes = mllm.score_yes_no(img, "Is there a person in the image?")
        log.info("score_yes_no -> label=%s p_yes=%.4f", label, p_yes)
    else:
        log.info("Model loaded OK (key=%s, family=%s). Pass --image to run generation.",
                 mllm.key, mllm.family)


if __name__ == "__main__":
    main()
