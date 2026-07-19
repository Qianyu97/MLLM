"""PGD: Probability-Guided Decoding (training-free, inference-time).

PGD intervenes on the MLLM's next-token logits using the *drift* signal from the
HHD detectors.  Conceptually we want to down-weight token probabilities that the
detectors flag as drifting away from the visual evidence::

    p'(token) ∝ p(token) * exp(-lambda * drift(token))

In logit space this is the numerically stable form::

    logits' = logits + log( exp(-lambda * drift) )
            = logits - lambda * drift

so we never exponentiate large magnitudes before the model's own softmax.  When
``cfg.pgd.adaptive_trigger`` is set we only apply the shift on steps where some
HHD detector actually fires (otherwise we leave the distribution untouched to
avoid degrading clean generations).  ``rollback`` supports re-generating from a
saved decoder state when a *severe* object hallucination is detected.

Contract (CROSS-FILE INTERFACES)::

    class PGD:
        __init__(self, hhd, cfg)
        reweight(self, logits, drift) -> logits   # logits + log(exp(-lambda*drift))
        rollback() helper

``torch`` is imported lazily; the module is import-clean without it.
"""
from __future__ import annotations

import argparse

from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed


def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for PGD reweighting. Install torch to use this module."
        ) from e


class PGD:
    """Probability-Guided Decoding controller.

    Parameters
    ----------
    hhd : HHD | None
        The hierarchical hallucination detector.  May be ``None`` for the pure
        logit-reweighting use-case (when ``drift`` is supplied externally).
    cfg : namespace
        Uses ``cfg.pgd.{lambda_decay, adaptive_trigger, rollback, max_rollback}``.
    """

    def __init__(self, hhd, cfg):
        self.hhd = hhd
        self.cfg = cfg
        p = cfg.pgd
        self.lambda_decay = float(p.lambda_decay)
        self.adaptive_trigger = bool(p.adaptive_trigger)
        self.rollback_enabled = bool(p.rollback)
        self.max_rollback = int(p.max_rollback)

        # Rollback bookkeeping.
        self._rollback_count = 0
        self._checkpoints: list = []

    # ------------------------------------------------------------------ #
    # Core reweighting
    # ------------------------------------------------------------------ #
    def reweight(self, logits, drift):
        """Apply the probability-guided shift to ``logits``.

        Implements ``logits' = logits + log(exp(-lambda * drift))`` in the
        numerically stable additive form ``logits - lambda * drift`` (the
        ``log(exp(...))`` round-trip is avoided to prevent overflow for large
        ``lambda * drift``).

        Parameters
        ----------
        logits : Tensor [..., V]
            Next-token logits from the MLLM.
        drift : Tensor broadcastable to ``logits`` | float
            Per-token drift in [0, inf).  A scalar applies a uniform shift
            (a no-op on the softmax, since a constant cancels), so meaningful
            use supplies a per-vocab drift vector ``[V]`` or ``[..., V]``.

        Returns
        -------
        Tensor
            The reweighted logits (same shape as ``logits``).
        """
        torch = _torch()
        if not torch.is_tensor(drift):
            drift = torch.as_tensor(drift, dtype=logits.dtype, device=logits.device)
        drift = drift.to(dtype=logits.dtype, device=logits.device)

        # Adaptive trigger: skip the intervention when no drift is present
        # (all-zero drift => detectors did not fire).  This keeps clean steps
        # bit-identical to vanilla decoding.
        if self.adaptive_trigger and bool((drift == 0).all()):
            return logits

        # logits + log(exp(-lambda * drift)) == logits - lambda * drift
        return logits - self.lambda_decay * drift

    # ------------------------------------------------------------------ #
    # Adaptive trigger from HHD detections
    # ------------------------------------------------------------------ #
    def drift_from_detections(self, detections, vocab_size, token_to_vocab=None):
        """Build a per-vocab drift vector ``[vocab_size]`` from HHD flags.

        Each flagged detection contributes drift proportional to its score on
        the vocab id it refers to (via ``token_to_vocab[detection_index]``).
        Unflagged or unmapped tokens get zero drift.  This is the bridge used by
        the decoding loop to turn token-level detections into a logit shift.
        """
        torch = _torch()
        drift = torch.zeros(vocab_size)
        if not detections:
            return drift
        for k, det in enumerate(detections):
            if not det.get("flag", False):
                continue
            vid = None
            if token_to_vocab is not None:
                vid = token_to_vocab.get(det.get("index", k)) if isinstance(token_to_vocab, dict) \
                    else (token_to_vocab[k] if k < len(token_to_vocab) else None)
            if vid is None:
                continue
            # Object flags use (1 - existence prob) as drift; attribute flags use
            # the inconsistency score directly; relation flags use (1 - prob).
            ttype = det.get("type", "object")
            s = float(det.get("score", 0.0))
            mag = s if ttype == "attribute" else (1.0 - s)
            drift[int(vid)] = max(float(drift[int(vid)]), max(0.0, mag))
        return drift

    def should_trigger(self, detections) -> bool:
        """True if any detector fired (used to gate the intervention)."""
        return any(d.get("flag", False) for d in (detections or []))

    def is_severe_object(self, detections) -> bool:
        """Severe object hallucination => candidate for rollback-regenerate."""
        for d in detections or []:
            if d.get("type") == "object" and d.get("flag", False):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Rollback hooks
    # ------------------------------------------------------------------ #
    def push_checkpoint(self, state) -> None:
        """Save a decoder state (e.g. past_key_values + generated ids) for rollback."""
        self._checkpoints.append(state)

    def can_rollback(self) -> bool:
        """Rollback is allowed if enabled, under the budget, and a state exists."""
        return (self.rollback_enabled
                and self._rollback_count < self.max_rollback
                and len(self._checkpoints) > 0)

    def rollback(self):
        """Pop and return the last saved decoder state, incrementing the counter.

        Returns ``None`` when rollback is disabled / budget exhausted / no state.
        The caller restores the returned state and re-generates the offending
        span (typically with a stronger ``lambda_decay`` or banned tokens).
        """
        if not self.can_rollback():
            return None
        self._rollback_count += 1
        return self._checkpoints.pop()

    def reset(self) -> None:
        """Reset per-sequence rollback bookkeeping (call once per new generation)."""
        self._rollback_count = 0
        self._checkpoints = []


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PGD reweighting — smoke test.")
    p.add_argument("--config", default=None, help="Path to an override YAML config.")
    p.add_argument("--vocab", type=int, default=10, help="Toy vocab size.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log = get_logger("cmpsa.models.pgd")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    torch = _torch()
    logits = torch.randn(args.vocab)
    drift = torch.zeros(args.vocab)
    drift[0] = 1.0  # pretend vocab id 0 is a flagged hallucinated token

    pgd = PGD(hhd=None, cfg=cfg)
    new_logits = pgd.reweight(logits, drift)
    log.info("lambda_decay=%.2f adaptive=%s rollback=%s max_rollback=%d",
             pgd.lambda_decay, pgd.adaptive_trigger, pgd.rollback_enabled, pgd.max_rollback)
    log.info("delta on flagged token id0 = %.4f (expected -%.4f)",
             float(new_logits[0] - logits[0]), pgd.lambda_decay)
    log.info("delta on clean token id1 = %.4f (expected 0)",
             float(new_logits[1] - logits[1]))

    # rollback demo
    pgd.reset()
    pgd.push_checkpoint({"step": 3})
    log.info("can_rollback=%s rollback()=%s", pgd.can_rollback(), pgd.rollback())


if __name__ == "__main__":
    main()
