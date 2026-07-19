"""Method factory: wraps an :class:`MLLM` into an evaluation strategy.

Two methods are provided:

* ``"vanilla"`` — pass-through to the underlying MLLM (no intervention).
* ``"cmpsa"`` — live HHD + PGD decoding: free-form generation uses a top-k
  HuggingFace ``LogitsProcessor`` that consults HHD before each next-token
  decision; yes/no scoring adjusts the first-token Yes/No logits from the
  question's grounded semantic probes.

Contract (CROSS-FILE INTERFACES)::

    build_method(method_key, mllm, cfg) -> Method
    class Method:
        .answer_yes_no(image, question) -> (str, float)
        .caption(image, prompt) -> str

``torch`` is imported lazily (only the cmpsa method touches it).
"""
from __future__ import annotations

import argparse

from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed


# --------------------------------------------------------------------------- #
# Base Method
# --------------------------------------------------------------------------- #
class Method:
    """Base evaluation strategy — pure pass-through to the wrapped MLLM."""

    def __init__(self, mllm, cfg, key: str = "vanilla"):
        self.mllm = mllm
        self.cfg = cfg
        self.key = key

    def answer_yes_no(self, image, question) -> tuple:
        """Return ``(label, p_yes)`` for a yes/no question (delegates to MLLM)."""
        return self.mllm.score_yes_no(image, question)

    def caption(self, image, prompt) -> str:
        """Return a free-form caption / answer (delegates to MLLM)."""
        max_new = getattr(getattr(self.cfg, "eval", None), "max_new_tokens", 64)
        return self.mllm.generate(image, prompt, max_new_tokens=max_new)


# --------------------------------------------------------------------------- #
# CMPSA method (HHD + PGD reference wrapper)
# --------------------------------------------------------------------------- #
class CMPSAMethod(Method):
    """CMPSA strategy: MLLM generation guarded by live HHD/PGD decoding."""

    def __init__(self, mllm, cfg, key: str = "cmpsa"):
        super().__init__(mllm, cfg, key=key)
        # Lazy-build HHD + PGD so importing this module stays torch-free.
        from cmpsa.models.hhd import HHD
        from cmpsa.models.pgd import PGD
        self.hhd = HHD(cfg)
        self.pgd = PGD(self.hhd, cfg)
        self._decoder = None

    def _live_decoder(self):
        """Build the live PGD decoder on first use.

        The dummy CLI smoke test does not have a real MLLM backend; in that case
        return ``None`` so the smoke test still exercises the factory.
        """
        if self._decoder is not None:
            return self._decoder
        if not hasattr(self.mllm, "model") or not hasattr(self.mllm, "processor"):
            return None
        from cmpsa.models.pgd_decode import CMPSAPGDDecoder

        self._decoder = CMPSAPGDDecoder(self.mllm, self.cfg, self.hhd, self.pgd)
        return self._decoder

    # ------------------------------------------------------------------ #
    def caption(self, image, prompt) -> str:
        """Generate with live top-k HHD/PGD logit reweighting when supported."""
        self.pgd.reset()
        dec = self._live_decoder()
        max_new = getattr(self.cfg.eval, "max_new_tokens", 64)
        if dec is None:
            return self.mllm.generate(image, prompt, max_new_tokens=max_new)
        return dec.generate(image, prompt, max_new_tokens=max_new)

    def answer_yes_no(self, image, question) -> tuple:
        """Yes/No answer with first-token PGD reweighting when supported."""
        self.pgd.reset()
        dec = self._live_decoder()
        if dec is None:
            return self.mllm.score_yes_no(image, question)
        return dec.score_yes_no(image, question)

    # ------------------------------------------------------------------ #
    def apply_pgd(self, logits, drift):
        """Expose the PGD reweighting for the decoding loop / unit tests."""
        return self.pgd.reweight(logits, drift)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_METHOD_REGISTRY = {
    "vanilla": Method,
    "cmpsa": CMPSAMethod,
}


def build_method(method_key: str, mllm, cfg) -> Method:
    """Build a :class:`Method` for ``method_key`` wrapping ``mllm``.

    Known keys: ``"vanilla"`` (pass-through) and ``"cmpsa"`` (HHD + PGD wrapper).
    Unknown keys raise ``NotImplementedError`` naming the key.
    """
    cls = _METHOD_REGISTRY.get(method_key)
    if cls is None:
        raise NotImplementedError(
            f"Unknown method key {method_key!r}. Available: {sorted(_METHOD_REGISTRY)}. "
            f"(SOTA baselines like vcd/opera/m3id register their own Method subclass.)"
        )
    return cls(mllm, cfg, key=method_key)


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Method factory — build a method (no weights needed).")
    p.add_argument("--method", default="vanilla", help="Method key (vanilla|cmpsa).")
    p.add_argument("--config", default=None, help="Path to an override YAML config.")
    return p


class _DummyMLLM:
    """Stand-in MLLM so the factory smoke test runs without model weights."""
    key = "dummy"
    family = "dummy"

    def generate(self, image, prompt, max_new_tokens=64):
        return "(dummy caption)"

    def score_yes_no(self, image, question):
        return "yes", 0.5


def main() -> None:
    args = _build_parser().parse_args()
    log = get_logger("cmpsa.models.methods")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    method = build_method(args.method, _DummyMLLM(), cfg)
    log.info("Built method key=%s class=%s", method.key, type(method).__name__)
    log.info("caption(dummy) -> %r", method.caption(None, "Describe this image."))
    log.info("answer_yes_no(dummy) -> %s", method.answer_yes_no(None, "Is there a cat?"))
    if isinstance(method, CMPSAMethod):
        log.info("CMPSA hooks present: hhd=%s pgd=%s",
                 type(method.hhd).__name__, type(method.pgd).__name__)


if __name__ == "__main__":
    main()
