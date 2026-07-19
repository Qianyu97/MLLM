"""PVE / PLE probabilistic projection heads.

The Probabilistic Visual Encoder head (``PVEHead``) and the Probabilistic
Language Encoder head (``PLEHead``) each map a backbone feature tensor into the
shared *Probabilistic Semantic Alignment Space* (PSAS).  Instead of producing a
single point per token they produce a **diagonal Gaussian** ``N(mu, diag(var))``
per token, parameterised by ``mu`` and a (clamped) ``logvar``.

Both heads are a simple two-layer MLP trunk followed by two linear projections
(one for ``mu`` and one for ``logvar``).  ``logvar`` is clamped to
``[min_logvar, max_logvar]`` to avoid posterior collapse (var -> 0) or blow-up.

Contract (CROSS-FILE INTERFACES)::

    class PVEHead(nn.Module):
        __init__(self, in_dim, psas_dim, hidden_dim, min_logvar, max_logvar)
        forward(self, feats) -> (mu, logvar)   # feats [B,N,in_dim] -> [B,N,psas_dim]
    class PLEHead(nn.Module):  # identical signature
    reparameterize(mu, logvar) -> sample

``torch`` is imported lazily inside :func:`_torch` / class bodies so that the
module is import-clean (and ``--help`` works) on a box without torch installed.
"""
from __future__ import annotations

import argparse

from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed


# --------------------------------------------------------------------------- #
# Lazy torch access
# --------------------------------------------------------------------------- #
def _torch():
    """Import torch on demand with a clear error message if it is missing."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        return torch, nn
    except Exception as e:  # pragma: no cover - exercised only without torch
        raise RuntimeError(
            "PyTorch is required to build / run the PVE/PLE projection heads. "
            "Install torch (and a CUDA build if you have a GPU) to use this module."
        ) from e


def _make_head_class():
    """Construct the concrete ``_GaussianHead`` ``nn.Module`` subclass.

    We build the class inside a function so that ``import torch`` only happens
    when a head is actually instantiated, keeping the module import torch-free.
    """
    torch, nn = _torch()

    class _GaussianHead(nn.Module):
        """Two-layer MLP trunk + (mu, logvar) projection into PSAS.

        Parameters
        ----------
        in_dim : int
            Dimensionality of the incoming backbone features.
        psas_dim : int
            Dimensionality of the shared probabilistic semantic alignment space.
        hidden_dim : int
            Width of the MLP trunk.
        min_logvar, max_logvar : float
            Clamp range applied to the predicted log-variance.
        """

        def __init__(self, in_dim, psas_dim, hidden_dim, min_logvar, max_logvar):
            super().__init__()
            self.in_dim = int(in_dim)
            self.psas_dim = int(psas_dim)
            self.hidden_dim = int(hidden_dim)
            self.min_logvar = float(min_logvar)
            self.max_logvar = float(max_logvar)

            # Two-layer MLP trunk (Linear -> GELU -> Linear -> GELU).
            self.trunk = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
            # Separate heads for the mean and the log-variance.
            self.mu_head = nn.Linear(self.hidden_dim, self.psas_dim)
            self.logvar_head = nn.Linear(self.hidden_dim, self.psas_dim)

        def forward(self, feats):
            """Project ``feats`` [B, N, in_dim] (or [N, in_dim]) into PSAS.

            Returns
            -------
            (mu, logvar) : tuple of Tensor
                Same leading shape as ``feats`` with last dim ``psas_dim``.
                ``logvar`` is clamped to ``[min_logvar, max_logvar]``.
            """
            h = self.trunk(feats)
            mu = self.mu_head(h)
            logvar = self.logvar_head(h)
            logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
            return mu, logvar

    return _GaussianHead


# Cache the constructed class so repeated instantiation is cheap.
_HEAD_CLASS_CACHE = {}


def _head_class():
    if "cls" not in _HEAD_CLASS_CACHE:
        _HEAD_CLASS_CACHE["cls"] = _make_head_class()
    return _HEAD_CLASS_CACHE["cls"]


# --------------------------------------------------------------------------- #
# Public head classes
# --------------------------------------------------------------------------- #
def PVEHead(in_dim, psas_dim, hidden_dim, min_logvar, max_logvar):
    """Probabilistic Visual Encoder head.

    Implemented as a thin factory that instantiates the (lazily built) shared
    Gaussian-head ``nn.Module``.  The returned object *is* an ``nn.Module``
    instance, so ``isinstance(head, nn.Module)`` holds and it can be added to
    parameter lists / optimizers as usual.
    """
    return _head_class()(in_dim, psas_dim, hidden_dim, min_logvar, max_logvar)


def PLEHead(in_dim, psas_dim, hidden_dim, min_logvar, max_logvar):
    """Probabilistic Language Encoder head (identical architecture to PVE)."""
    return _head_class()(in_dim, psas_dim, hidden_dim, min_logvar, max_logvar)


def reparameterize(mu, logvar):
    """Reparameterisation trick: sample ``z = mu + sigma * eps``.

    ``sigma = exp(0.5 * logvar)`` and ``eps ~ N(0, I)``.  In eval mode the
    caller may simply use ``mu`` directly; this helper always draws a sample.
    """
    torch, _ = _torch()
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PVE/PLE probabilistic projection heads — smoke test."
    )
    p.add_argument("--config", default=None, help="Path to an override YAML config.")
    p.add_argument("--in-dim", type=int, default=None,
                   help="Input feature dim (default: visual_backbone.feature_dim).")
    p.add_argument("--tokens", type=int, default=4, help="Num tokens N for the smoke test.")
    p.add_argument("--batch", type=int, default=2, help="Batch size B for the smoke test.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log = get_logger("cmpsa.models.pve_ple")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    in_dim = args.in_dim if args.in_dim is not None else cfg.visual_backbone.feature_dim
    pj = cfg.projection
    log.info("Building PVE/PLE heads: in_dim=%d psas_dim=%d hidden_dim=%d",
             in_dim, pj.psas_dim, pj.hidden_dim)

    torch, _ = _torch()
    pve = PVEHead(in_dim, pj.psas_dim, pj.hidden_dim, pj.min_logvar, pj.max_logvar)
    feats = torch.randn(args.batch, args.tokens, in_dim)
    mu, logvar = pve(feats)
    z = reparameterize(mu, logvar)
    log.info("PVE forward OK: feats=%s -> mu=%s logvar=%s sample=%s",
             tuple(feats.shape), tuple(mu.shape), tuple(logvar.shape), tuple(z.shape))
    log.info("logvar range after clamp: [%.3f, %.3f] (clamp=[%.1f, %.1f])",
             float(logvar.detach().min()), float(logvar.detach().max()),
             pj.min_logvar, pj.max_logvar)


if __name__ == "__main__":
    main()
