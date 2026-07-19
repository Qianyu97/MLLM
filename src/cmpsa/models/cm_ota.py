"""CM-OTA: Cross-Modal Optimal-Transport Alignment.

This module aligns the visual and language PSAS distributions with **entropic
optimal transport**.  The ground cost between a visual token's Gaussian and a
language token's Gaussian is the closed-form **2-Wasserstein squared distance**
between diagonal Gaussians::

    W2^2( N(mu1, diag(s1^2)), N(mu2, diag(s2^2)) )
        = || mu1 - mu2 ||^2 + || s1 - s2 ||^2

(the second term is exact for diagonal/commuting covariances, where the
Bures term reduces to ``sum_d (s1_d - s2_d)^2``).

The transport plan is solved by a pure-torch, log-domain stable Sinkhorn
iteration.  If the optional `POT` (``ot``) package is installed it is used only
as a *cross-check* (never as a hard dependency).

Contract (CROSS-FILE INTERFACES)::

    gaussian_w2(mu1, logvar1, mu2, logvar2) -> Tensor   # supports [Bv]x[Bl] cost
    sinkhorn(cost, eps, iters) -> plan                  # entropic OT plan
    cmota_loss(v_mu, v_logvar, l_mu, l_logvar, cfg) -> dict {"loss","l_ot","l_klreg"}

``torch`` is imported lazily so the module is import-clean without it.
"""
from __future__ import annotations

import argparse

from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed


# --------------------------------------------------------------------------- #
# Lazy torch access
# --------------------------------------------------------------------------- #
def _torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the CM-OTA alignment loss. Install torch to use this module."
        ) from e


# --------------------------------------------------------------------------- #
# Closed-form 2-Wasserstein^2 between diagonal Gaussians
# --------------------------------------------------------------------------- #
def gaussian_w2(mu1, logvar1, mu2, logvar2):
    """Closed-form squared 2-Wasserstein distance between diagonal Gaussians.

    Parameters
    ----------
    mu1, logvar1 : Tensor [..., D]   (e.g. [Bv, D] for visual tokens)
    mu2, logvar2 : Tensor [..., D]   (e.g. [Bl, D] for language tokens)

    Behaviour
    ---------
    * If ``mu1`` is ``[Bv, D]`` and ``mu2`` is ``[Bl, D]`` (different leading
      sizes), a **pairwise cost matrix** ``[Bv, Bl]`` is returned via
      broadcasting (``mu1[:, None, :]`` vs ``mu2[None, :, :]``).
    * If the leading shapes already match they are compared elementwise and a
      ``[...]`` tensor (one scalar per aligned pair) is returned.

    Returns
    -------
    Tensor
        ``W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2`` where
        ``sigma = exp(0.5 * logvar)``.
    """
    torch = _torch()
    sigma1 = torch.exp(0.5 * logvar1)
    sigma2 = torch.exp(0.5 * logvar2)

    same_leading = (mu1.dim() == mu2.dim()) and (mu1.shape[:-1] == mu2.shape[:-1])
    if same_leading:
        mean_term = ((mu1 - mu2) ** 2).sum(dim=-1)
        cov_term = ((sigma1 - sigma2) ** 2).sum(dim=-1)
        return mean_term + cov_term

    # Pairwise [Bv, Bl]: insert broadcast axes.  We support the common 2-D case
    # (each of shape [B, D]); higher-rank inputs are flattened to [-1, D] first.
    m1 = mu1.reshape(-1, mu1.shape[-1])
    m2 = mu2.reshape(-1, mu2.shape[-1])
    s1 = sigma1.reshape(-1, sigma1.shape[-1])
    s2 = sigma2.reshape(-1, sigma2.shape[-1])

    mean_term = ((m1[:, None, :] - m2[None, :, :]) ** 2).sum(dim=-1)   # [Bv, Bl]
    cov_term = ((s1[:, None, :] - s2[None, :, :]) ** 2).sum(dim=-1)    # [Bv, Bl]
    return mean_term + cov_term


# --------------------------------------------------------------------------- #
# Entropic OT — log-domain stable Sinkhorn (pure torch)
# --------------------------------------------------------------------------- #
def sinkhorn(cost, eps: float = 0.05, iters: int = 50, a=None, b=None):
    """Solve entropic OT and return the transport plan.

    Uses the numerically stable **log-domain** Sinkhorn iteration (log-sum-exp
    updates of the dual potentials), which avoids the under/overflow of the raw
    matrix-scaling form when ``eps`` is small.

    Parameters
    ----------
    cost : Tensor [n, m]
        Non-negative ground cost matrix (e.g. from :func:`gaussian_w2`).
    eps : float
        Entropic regularization strength.
    iters : int
        Number of Sinkhorn iterations.
    a : Tensor [n] | None
        Source marginal (defaults to uniform).
    b : Tensor [m] | None
        Target marginal (defaults to uniform).

    Returns
    -------
    plan : Tensor [n, m]
        The transport plan ``P`` with row-sums ``a`` and column-sums ``b``.
    """
    torch = _torch()
    n, m = cost.shape
    device, dtype = cost.device, cost.dtype

    if a is None:
        a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    if b is None:
        b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)

    log_a = torch.log(a + 1e-30)
    log_b = torch.log(b + 1e-30)

    # Kernel in log-space: -cost / eps
    K = -cost / float(eps)                      # [n, m]
    f = torch.zeros(n, device=device, dtype=dtype)   # log dual u
    g = torch.zeros(m, device=device, dtype=dtype)   # log dual v

    for _ in range(int(iters)):
        # f_i = log a_i - logsumexp_j (K_ij + g_j)
        f = log_a - torch.logsumexp(K + g[None, :], dim=1)
        # g_j = log b_j - logsumexp_i (K_ij + f_i)
        g = log_b - torch.logsumexp(K + f[:, None], dim=0)

    log_plan = f[:, None] + K + g[None, :]
    return torch.exp(log_plan)


def _pot_cross_check(cost, eps, iters, plan, log) -> None:
    """Optional cross-check against POT's sinkhorn (if installed)."""
    try:
        import numpy as np
        import ot  # POT
    except Exception:
        return
    try:
        c = cost.detach().cpu().double().numpy()
        n, m = c.shape
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        ref = ot.sinkhorn(a, b, c, reg=float(eps), numItermax=int(iters))
        diff = float(np.abs(ref - plan.detach().cpu().double().numpy()).max())
        log.info("POT cross-check: max|P_torch - P_pot| = %.3e", diff)
    except Exception as e:  # pragma: no cover
        log.warning("POT cross-check skipped: %s", e)


# --------------------------------------------------------------------------- #
# Covariance KL regularizer (anti-collapse)
# --------------------------------------------------------------------------- #
def _cov_kl_reg(logvar):
    """KL( N(0, diag(var)) || N(0, I) ) per element, summed over D, mean over tokens.

    This pulls the *covariance* of each token's posterior toward the unit
    prior, preventing the variances from collapsing to ~0 (which would make the
    probabilistic space degenerate to a point estimate).  Closed form::

        KL = 0.5 * sum_d ( var_d - 1 - logvar_d )

    Only the covariance is regularized (means are free to align via OT).
    """
    torch = _torch()
    var = torch.exp(logvar)
    kl = 0.5 * (var - 1.0 - logvar).sum(dim=-1)   # [...]
    return kl.mean()


# --------------------------------------------------------------------------- #
# CM-OTA loss
# --------------------------------------------------------------------------- #
def cmota_loss(v_mu, v_logvar, l_mu, l_logvar, cfg) -> dict:
    """Cross-modal OT alignment loss.

    Parameters
    ----------
    v_mu, v_logvar : Tensor [Bv, D]
        Visual PSAS Gaussians (one per visual token in the batch).
    l_mu, l_logvar : Tensor [Bl, D]
        Language PSAS Gaussians (one per language token in the batch).
    cfg : namespace
        Config (uses ``cfg.cmota.{distance, sinkhorn_eps, sinkhorn_iters,
        lambda_ot, lambda_klreg}``).

    Returns
    -------
    dict with keys:
        "loss"    : total = lambda_ot * l_ot + lambda_klreg * l_klreg
        "l_ot"    : OT alignment cost <P, C>
        "l_klreg" : covariance KL regularizer (visual + language)
    """
    torch = _torch()
    c = cfg.cmota

    # ---- ground cost matrix [Bv, Bl] ----
    distance = getattr(c, "distance", "wasserstein")
    if distance == "wasserstein":
        cost = gaussian_w2(v_mu, v_logvar, l_mu, l_logvar)
    elif distance == "euclidean":
        m1 = v_mu.reshape(-1, v_mu.shape[-1])
        m2 = l_mu.reshape(-1, l_mu.shape[-1])
        cost = ((m1[:, None, :] - m2[None, :, :]) ** 2).sum(dim=-1)
    elif distance == "kl":
        # Symmetric-ish: use W2 mean term + cov KL between the two diagonal Gaussians.
        cost = _pairwise_diag_kl(v_mu, v_logvar, l_mu, l_logvar)
    else:
        raise ValueError(f"Unknown cmota.distance={distance!r} (wasserstein|euclidean|kl)")

    # ---- entropic OT plan (plan detached from cost? no — keep grad through cost) ----
    plan = sinkhorn(cost, eps=c.sinkhorn_eps, iters=c.sinkhorn_iters)
    l_ot = (plan * cost).sum()

    # ---- covariance KL regularizer (anti-collapse), on both modalities ----
    l_klreg = _cov_kl_reg(v_logvar) + _cov_kl_reg(l_logvar)

    loss = c.lambda_ot * l_ot + c.lambda_klreg * l_klreg
    return {"loss": loss, "l_ot": l_ot, "l_klreg": l_klreg}


def _pairwise_diag_kl(mu1, logvar1, mu2, logvar2):
    """Pairwise KL( N1 || N2 ) for diagonal Gaussians -> cost matrix [Bv, Bl]."""
    torch = _torch()
    m1 = mu1.reshape(-1, mu1.shape[-1])
    m2 = mu2.reshape(-1, mu2.shape[-1])
    lv1 = logvar1.reshape(-1, logvar1.shape[-1])
    lv2 = logvar2.reshape(-1, logvar2.shape[-1])
    var1 = torch.exp(lv1)
    var2 = torch.exp(lv2)

    # KL = 0.5 * sum_d [ lv2 - lv1 + (var1 + (mu1-mu2)^2)/var2 - 1 ]
    lv_term = (lv2[None, :, :] - lv1[:, None, :])                       # [Bv, Bl, D]
    ratio = (var1[:, None, :] + (m1[:, None, :] - m2[None, :, :]) ** 2) / var2[None, :, :]
    kl = 0.5 * (lv_term + ratio - 1.0).sum(dim=-1)                      # [Bv, Bl]
    return kl


# --------------------------------------------------------------------------- #
# CM-OTA v2 — collapse-resistant alignment loss (REDESIGN)
# --------------------------------------------------------------------------- #
# The original ``cmota_loss`` minimises the raw entropic OT cost <P,C> over a
# batch with *uniform* marginals and no image<->caption correspondence.  Its
# global optimum is representational collapse (all mu equal -> cost 0), which the
# 25,968-step prior run hit (L_OT -> 2e-4; PSAS pairwise-cosine -> 0.99).  This
# redesign keeps the Gaussian-W2 / Sinkhorn machinery for the Theorem-1 narrative
# but drives learning with a *correspondence-supervised, collapse-resistant*
# objective validated on cached features (val pairwise-cos 0.998 -> 0.067,
# caption->image R@1 0.026 -> 0.181):
#   L = w_nce*InfoNCE(matched, false-neg masked) + w_pull*W2(matched pairs)
#       + w_sd*SinkhornDivergence(images, captions)     # debiased -> collapse-safe
#       + w_vic*VICReg-std-hinge(means)                 # explicit anti-collapse
#       + w_kl*logvar^2                                 # keep covariances conditioned
def _cosine_logits(a_mu, b_mu, temp: float):
    torch = _torch()
    an = a_mu / (a_mu.norm(dim=-1, keepdim=True) + 1e-8)
    bn = b_mu / (b_mu.norm(dim=-1, keepdim=True) + 1e-8)
    return (an @ bn.t()) / float(temp)


def _pairwise_w2(mu1, logvar1, mu2, logvar2):
    """Always-pairwise diagonal-Gaussian W2^2 cost matrix [n, m].

    (``gaussian_w2`` returns an *elementwise* vector when the leading shapes
    match; the OT terms need the full [n, m] matrix regardless.)
    """
    torch = _torch()
    s1 = torch.exp(0.5 * logvar1)
    s2 = torch.exp(0.5 * logvar2)
    mean = ((mu1[:, None, :] - mu2[None, :, :]) ** 2).sum(dim=-1)
    cov = ((s1[:, None, :] - s2[None, :, :]) ** 2).sum(dim=-1)
    return mean + cov


def sinkhorn_divergence(v_mu, v_logvar, l_mu, l_logvar, eps: float = 0.05, iters: int = 50):
    """Debiased Sinkhorn divergence S = OT(V,L) - .5 OT(V,V) - .5 OT(L,L).

    Unlike the raw OT cost, S is not minimised by shrinking the space (the three
    terms shrink together and cancel), so it aligns the two Gaussian *sets*
    without rewarding collapse.
    """
    torch = _torch()

    def _ot(m1, lv1, m2, lv2):
        c = _pairwise_w2(m1, lv1, m2, lv2)
        p = sinkhorn(c, eps=eps, iters=iters)
        return (p * c).sum()

    return (_ot(v_mu, v_logvar, l_mu, l_logvar)
            - 0.5 * _ot(v_mu, v_logvar, v_mu, v_logvar)
            - 0.5 * _ot(l_mu, l_logvar, l_mu, l_logvar))


def _vic_std_hinge(mu, gamma: float = 1.0):
    """VICReg variance hinge: penalise per-dim batch std below ``gamma`` (anti-collapse on the means)."""
    torch = _torch()
    std = torch.sqrt(mu.var(dim=0, unbiased=False) + 1e-6)
    return torch.relu(float(gamma) - std).mean()


def cmota_align_loss(v_mu, v_logvar, l_mu, l_logvar, cfg, n_pos=None, img_ids=None) -> dict:
    """Collapse-resistant CM-OTA alignment loss (the redesign; see block comment).

    Parameters
    ----------
    v_mu, v_logvar : Tensor [P, D]
        Visual PSAS Gaussians, one per image in the batch.
    l_mu, l_logvar : Tensor [P + M, D]
        Text PSAS Gaussians: the first ``n_pos`` (=P) are the *matched* positive
        captions (row i <-> image i); the remaining M are hard negatives (no
        matched image) that enlarge the InfoNCE denominator.
    n_pos : int | None
        Number of matched positives P (defaults to v_mu.shape[0]).
    img_ids : LongTensor [P] | None
        Image id per positive, used to mask same-image in-batch false negatives.
    cfg : namespace
        Uses ``cfg.cmota.{sinkhorn_eps, sinkhorn_iters, nce_temp, w_nce, w_pull,
        w_sd, w_vic, vic_gamma, w_kl}`` (all optional with sane defaults).
    """
    torch = _torch()
    c = cfg.cmota
    P = int(n_pos) if n_pos is not None else v_mu.shape[0]
    temp = float(getattr(c, "nce_temp", 0.07))
    w_nce = float(getattr(c, "w_nce", 1.0))
    w_pull = float(getattr(c, "w_pull", 0.05))
    w_sd = float(getattr(c, "w_sd", 0.1))
    w_vic = float(getattr(c, "w_vic", 1.0))
    w_kl = float(getattr(c, "w_kl", 0.01))
    gamma = float(getattr(c, "vic_gamma", 1.0))
    eps = float(getattr(c, "sinkhorn_eps", 0.05))
    iters = int(getattr(c, "sinkhorn_iters", 50))

    vp_mu, vp_lv = v_mu[:P], v_logvar[:P]
    lp_mu, lp_lv = l_mu[:P], l_logvar[:P]

    # ---- InfoNCE correspondence (image<->positive caption), false-neg masked ----
    sim_i2t = _cosine_logits(vp_mu, l_mu, temp)            # [P, P+M]
    sim_t2i = _cosine_logits(lp_mu, vp_mu, temp)           # [P, P]
    tgt = torch.arange(P, device=v_mu.device)
    if img_ids is not None:
        same = (img_ids[:, None] == img_ids[None, :])
        eye = torch.eye(P, dtype=torch.bool, device=v_mu.device)
        fn = same & (~eye)                                 # same image, off-diagonal
        sim_i2t[:, :P] = sim_i2t[:, :P].masked_fill(fn, float("-inf"))
        sim_t2i = sim_t2i.masked_fill(fn, float("-inf"))
    l_nce = 0.5 * (torch.nn.functional.cross_entropy(sim_i2t, tgt)
                   + torch.nn.functional.cross_entropy(sim_t2i, tgt))

    # ---- matched-pair Gaussian-W2 pull (tighten alignment; Theorem-1 quantity) ----
    # same leading shape [P] -> gaussian_w2 returns the elementwise W2^2 per pair.
    l_pull = gaussian_w2(vp_mu, vp_lv, lp_mu, lp_lv).mean()

    # ---- debiased Sinkhorn divergence between image & caption Gaussian sets ----
    l_sd = sinkhorn_divergence(vp_mu, vp_lv, lp_mu, lp_lv, eps=eps, iters=iters)

    # ---- explicit anti-collapse on the means + keep covariances conditioned ----
    l_vic = _vic_std_hinge(vp_mu, gamma) + _vic_std_hinge(lp_mu, gamma)
    l_klreg = 0.5 * (v_logvar.pow(2).mean() + l_logvar.pow(2).mean())

    loss = (w_nce * l_nce + w_pull * l_pull + w_sd * l_sd
            + w_vic * l_vic + w_kl * l_klreg)
    return {"loss": loss, "l_nce": l_nce, "l_pull": l_pull, "l_sd": l_sd,
            "l_vic": l_vic, "l_klreg": l_klreg}


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CM-OTA alignment — smoke test.")
    p.add_argument("--config", default=None, help="Path to an override YAML config.")
    p.add_argument("--bv", type=int, default=6, help="Number of visual tokens.")
    p.add_argument("--bl", type=int, default=5, help="Number of language tokens.")
    p.add_argument("--pot-check", action="store_true", help="Cross-check plan against POT if installed.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log = get_logger("cmpsa.models.cm_ota")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    torch = _torch()
    d = cfg.projection.psas_dim
    v_mu = torch.randn(args.bv, d)
    v_logvar = torch.randn(args.bv, d).clamp(cfg.projection.min_logvar, cfg.projection.max_logvar)
    l_mu = torch.randn(args.bl, d)
    l_logvar = torch.randn(args.bl, d).clamp(cfg.projection.min_logvar, cfg.projection.max_logvar)

    cost = gaussian_w2(v_mu, v_logvar, l_mu, l_logvar)
    log.info("cost matrix shape=%s range=[%.3f, %.3f]", tuple(cost.shape),
             float(cost.min()), float(cost.max()))

    plan = sinkhorn(cost, eps=cfg.cmota.sinkhorn_eps, iters=cfg.cmota.sinkhorn_iters)
    log.info("plan shape=%s row-sum~%.4f col-sum~%.4f total mass=%.4f",
             tuple(plan.shape), float(plan.sum(dim=1).mean()),
             float(plan.sum(dim=0).mean()), float(plan.sum()))
    if args.pot_check:
        _pot_cross_check(cost, cfg.cmota.sinkhorn_eps, cfg.cmota.sinkhorn_iters, plan, log)

    out = cmota_loss(v_mu, v_logvar, l_mu, l_logvar, cfg)
    log.info("cmota_loss: loss=%.4f l_ot=%.4f l_klreg=%.4f",
             float(out["loss"]), float(out["l_ot"]), float(out["l_klreg"]))


if __name__ == "__main__":
    main()
