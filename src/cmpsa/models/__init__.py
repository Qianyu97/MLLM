"""CMPSA model module.

This package holds the probabilistic projection heads (PVE / PLE), the
cross-modal optimal-transport alignment loss (CM-OTA), the hierarchical
hallucination detectors (HHD), the probability-guided decoding controller
(PGD), the MLLM wrapper, and the high-level evaluation ``Method`` factory.

Heavy dependencies (``torch`` / ``transformers``) are imported *lazily* inside
functions / methods so that ``import cmpsa.models`` (and ``--help`` of any CLI)
works on a CPU-only box without those packages installed.  Only public symbols
that do not force a top-level torch import are re-exported here.
"""
from __future__ import annotations

# NOTE: do NOT import the submodules eagerly here.  ``pve_ple`` / ``cm_ota`` etc.
# define ``nn.Module`` subclasses at *call* time via lazy imports, so importing
# this package is cheap and torch-free.  Callers import the concrete symbols
# directly, e.g. ``from cmpsa.models.cm_ota import cmota_loss``.

__all__ = [
    "pve_ple",
    "cm_ota",
    "hhd",
    "pgd",
    "mllm_wrapper",
    "methods",
]
