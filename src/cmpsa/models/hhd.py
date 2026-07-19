"""HHD: Hierarchical Hallucination Detection.

Three detectors operate on the PSAS Gaussians produced by the PVE (visual) and
PLE (language) heads:

* :class:`ObjectDetector` — **OLD** (Object-Level Detection).  Estimates the
  *existence probability* of an object token by Monte-Carlo sampling from the
  language token's posterior and measuring how well it is supported by the
  visual distribution (soft match probability).  Flags a hallucination when the
  existence probability falls below ``cfg.hhd.tau_obj``.

* :class:`AttributeDetector` — **ALD** (Attribute-Level Detection).  Uses the
  attribute-conditioned KL divergence between the language attribute Gaussian
  and its grounded visual Gaussian; a large KL means the attribute is not
  visually supported.  Flags when ``score > cfg.hhd.tau_attr`` (the *score*
  reported is a [0,1] inconsistency normalised from the KL).

* :class:`RelationDetector` — **RLD** (Relation-Level Detection).  A relation
  probability head scores how plausible the (subject, predicate, object)
  relation is given the joint visual context; flags when probability is below
  ``cfg.hhd.tau_rel``.

``HHD`` is the unified wrapper.  ``detect()`` returns a list of standardised
dicts ``{"type", "score", "flag"}`` — one entry per analysed token / relation.

``torch`` is imported lazily; the module is import-clean without it.
"""
from __future__ import annotations

import argparse

from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed


# --------------------------------------------------------------------------- #
# Lazy torch
# --------------------------------------------------------------------------- #
def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the HHD detectors. Install torch to use this module."
        ) from e


def _as_tensor(x):
    """Coerce ``x`` to a float tensor (no-op if already a tensor)."""
    torch = _torch()
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
class ObjectDetector:
    """OLD — object existence via Monte-Carlo existence probability.

    Score = P(object token is supported by the visual evidence), estimated by
    drawing ``mc_samples`` reparameterised samples from the *language* token's
    posterior ``N(l_mu, l_var)`` and measuring the average soft match to the
    visual token distribution(s) via a Gaussian likelihood kernel.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tau = float(cfg.hhd.tau_obj)
        self.mc_samples = int(cfg.hhd.mc_samples)

    def score(self, l_mu, l_logvar, v_mu, v_logvar) -> float:
        """Monte-Carlo existence probability in [0, 1].

        ``l_*`` are the language token Gaussian params ``[D]`` (or ``[1, D]``);
        ``v_*`` are visual token Gaussians ``[Nv, D]``.  We sample from the
        language posterior and, for each sample, take the *max* soft-match
        probability over visual tokens (existence = is it supported by *any*
        visual region), then average over samples.
        """
        torch = _torch()
        l_mu = _as_tensor(l_mu).reshape(-1)
        l_logvar = _as_tensor(l_logvar).reshape(-1)
        v_mu = _as_tensor(v_mu).reshape(-1, l_mu.shape[-1])
        v_logvar = _as_tensor(v_logvar).reshape(-1, l_mu.shape[-1])

        std = torch.exp(0.5 * l_logvar)
        d = l_mu.shape[-1]
        # MC samples from the language posterior: [S, D]
        eps = torch.randn(self.mc_samples, d, device=l_mu.device, dtype=l_mu.dtype)
        samples = l_mu[None, :] + eps * std[None, :]

        # Gaussian likelihood of each sample under each visual token, using the
        # visual variance as the kernel bandwidth.  We use a normalised radial
        # similarity in [0,1]: exp(-0.5 * mean_d (x-mu_v)^2 / var_v).
        v_var = torch.exp(v_logvar).clamp_min(1e-6)                  # [Nv, D]
        # [S, Nv, D]
        diff2 = (samples[:, None, :] - v_mu[None, :, :]) ** 2
        quad = (diff2 / v_var[None, :, :]).mean(dim=-1)             # [S, Nv]
        sim = torch.exp(-0.5 * quad)                                # [S, Nv] in (0,1]
        per_sample = sim.max(dim=1).values                         # existence per sample
        return float(per_sample.mean().clamp(0.0, 1.0))

    def is_hallucination(self, *args, **kwargs) -> bool:
        """True when existence probability < tau_obj."""
        return self.score(*args, **kwargs) < self.tau


class AttributeDetector:
    """ALD — attribute-conditioned KL inconsistency.

    Score = normalised KL( language-attribute Gaussian || grounded-visual
    Gaussian ), mapped to [0, 1] via ``1 - exp(-kl)`` so larger means more
    inconsistent (more likely hallucinated).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tau = float(cfg.hhd.tau_attr)

    def _diag_kl(self, mu1, logvar1, mu2, logvar2):
        """KL( N1 || N2 ) for diagonal Gaussians, summed over D."""
        torch = _torch()
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2).clamp_min(1e-6)
        kl = 0.5 * ((logvar2 - logvar1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1.0)
        return kl.sum(dim=-1)

    def score(self, l_mu, l_logvar, v_mu, v_logvar) -> float:
        """Attribute inconsistency in [0, 1] (1 = strongly inconsistent)."""
        torch = _torch()
        l_mu = _as_tensor(l_mu).reshape(-1)
        l_logvar = _as_tensor(l_logvar).reshape(-1)
        v_mu = _as_tensor(v_mu).reshape(-1)
        v_logvar = _as_tensor(v_logvar).reshape(-1)
        kl = self._diag_kl(l_mu, l_logvar, v_mu, v_logvar)
        # Normalise per-dimension so the threshold is dimension-agnostic.
        kl_norm = kl / max(1, l_mu.shape[-1])
        return float((1.0 - torch.exp(-kl_norm)).clamp(0.0, 1.0))

    def is_hallucination(self, *args, **kwargs) -> bool:
        """True when attribute inconsistency score > tau_attr."""
        return self.score(*args, **kwargs) > self.tau


class RelationDetector:
    """RLD — relation plausibility via a (lazy) relation probability head.

    Score = P(relation holds | subject Gaussian, object Gaussian, visual
    context).  Without a trained head we fall back to a deterministic,
    interpretable surrogate: the relation is plausible when both the subject and
    object tokens are visually grounded (high pairwise match) — implemented as a
    sigmoid over the negative mean W2 distance between the relation's endpoints
    and the visual evidence.  A trained ``rel_head`` (an ``nn.Module`` mapping a
    concatenated feature to a logit) may be supplied to override the surrogate.
    """

    def __init__(self, cfg, rel_head=None):
        self.cfg = cfg
        self.tau = float(cfg.hhd.tau_rel)
        self.rel_head = rel_head

    def score(self, subj_mu, subj_logvar, obj_mu, obj_logvar,
              v_mu, v_logvar) -> float:
        """Relation plausibility probability in [0, 1]."""
        torch = _torch()
        subj_mu = _as_tensor(subj_mu).reshape(-1)
        obj_mu = _as_tensor(obj_mu).reshape(-1)
        v_mu = _as_tensor(v_mu).reshape(-1, subj_mu.shape[-1])

        if self.rel_head is not None:
            feat = torch.cat([subj_mu, obj_mu, v_mu.mean(dim=0)], dim=-1)
            logit = self.rel_head(feat).reshape(())
            return float(torch.sigmoid(logit))

        # Surrogate: how close (in mean L2) are subject & object to their nearest
        # visual region; map the average distance through a sigmoid so closer =>
        # higher plausibility.
        def _nearest_dist(m):
            return ((v_mu - m[None, :]) ** 2).sum(dim=-1).min()
        d_subj = _nearest_dist(subj_mu)
        d_obj = _nearest_dist(obj_mu)
        avg_d = 0.5 * (d_subj + d_obj)
        # Normalise by feature dim so the scale is comparable across psas_dim.
        avg_d = avg_d / max(1, subj_mu.shape[-1])
        return float(torch.sigmoid(-avg_d))

    def is_hallucination(self, *args, **kwargs) -> bool:
        """True when relation plausibility < tau_rel."""
        return self.score(*args, **kwargs) < self.tau


# --------------------------------------------------------------------------- #
# Unified HHD
# --------------------------------------------------------------------------- #
class HHD:
    """Hierarchical Hallucination Detector — unifies OLD / ALD / RLD.

    Parameters
    ----------
    cfg : namespace
        Uses ``cfg.hhd.{mc_samples, tau_obj, tau_attr, tau_rel}``.
    rel_head : optional nn.Module
        Trained relation probability head (passed to :class:`RelationDetector`).
    """

    def __init__(self, cfg, rel_head=None):
        self.cfg = cfg
        self.obj = ObjectDetector(cfg)
        self.attr = AttributeDetector(cfg)
        self.rel = RelationDetector(cfg, rel_head=rel_head)

    def detect(self, tokens, visual_psas, text_psas) -> list:
        """Run the hierarchical detectors over decoded ``tokens``.

        Parameters
        ----------
        tokens : list[dict]
            One entry per analysed unit, e.g.::

                {"type": "object",    "ple_index": i}
                {"type": "attribute", "ple_index": i, "v_index": j}
                {"type": "relation",  "subj_index": s, "obj_index": o}

            ``ple_index`` indexes into ``text_psas``; ``v_index`` (optional)
            indexes a specific visual token in ``visual_psas`` (else all visual
            tokens are considered).
        visual_psas : tuple(v_mu, v_logvar)
            Visual PSAS Gaussians ``[Nv, D]``.
        text_psas : tuple(l_mu, l_logvar)
            Language PSAS Gaussians ``[Nl, D]``.

        Returns
        -------
        list[dict]
            Each ``{"type": str, "score": float, "flag": bool}`` plus an echoed
            ``"index"`` / token reference for traceability.
        """
        v_mu, v_logvar = visual_psas
        l_mu, l_logvar = text_psas
        v_mu = _as_tensor(v_mu)
        v_logvar = _as_tensor(v_logvar)
        l_mu = _as_tensor(l_mu)
        l_logvar = _as_tensor(l_logvar)

        out = []
        for tok in tokens:
            ttype = tok.get("type", "object")
            if ttype == "object":
                i = int(tok.get("ple_index", 0))
                score = self.obj.score(l_mu[i], l_logvar[i], v_mu, v_logvar)
                flag = score < self.obj.tau
            elif ttype == "attribute":
                i = int(tok.get("ple_index", 0))
                if "v_index" in tok:
                    j = int(tok["v_index"])
                    vm, vlv = v_mu[j], v_logvar[j]
                else:
                    # Use the nearest visual token (by mean L2) as the grounding.
                    j = int(((v_mu - l_mu[i][None, :]) ** 2).sum(dim=-1).argmin())
                    vm, vlv = v_mu[j], v_logvar[j]
                score = self.attr.score(l_mu[i], l_logvar[i], vm, vlv)
                flag = score > self.attr.tau
            elif ttype == "relation":
                s = int(tok.get("subj_index", 0))
                o = int(tok.get("obj_index", 0))
                score = self.rel.score(l_mu[s], l_logvar[s], l_mu[o], l_logvar[o],
                                       v_mu, v_logvar)
                flag = score < self.rel.tau
            else:
                # Unknown type -> no detection, neutral score.
                score, flag = 0.0, False

            entry = {"type": ttype, "score": float(score), "flag": bool(flag)}
            if "ple_index" in tok:
                entry["index"] = int(tok["ple_index"])
            out.append(entry)
        return out


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HHD detectors — smoke test.")
    p.add_argument("--config", default=None, help="Path to an override YAML config.")
    p.add_argument("--nv", type=int, default=8, help="Number of visual tokens.")
    p.add_argument("--nl", type=int, default=6, help="Number of language tokens.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log = get_logger("cmpsa.models.hhd")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    torch = _torch()
    d = cfg.projection.psas_dim
    v_mu = torch.randn(args.nv, d)
    v_logvar = torch.zeros(args.nv, d)
    l_mu = torch.randn(args.nl, d)
    l_logvar = torch.zeros(args.nl, d)

    hhd = HHD(cfg)
    tokens = [
        {"type": "object", "ple_index": 0},
        {"type": "attribute", "ple_index": 1},
        {"type": "relation", "subj_index": 0, "obj_index": 2},
    ]
    res = hhd.detect(tokens, (v_mu, v_logvar), (l_mu, l_logvar))
    for r in res:
        log.info("detect: type=%s score=%.4f flag=%s "
                 "(tau_obj=%.2f tau_attr=%.2f tau_rel=%.2f mc=%d)",
                 r["type"], r["score"], r["flag"],
                 cfg.hhd.tau_obj, cfg.hhd.tau_attr, cfg.hhd.tau_rel, cfg.hhd.mc_samples)


if __name__ == "__main__":
    main()
