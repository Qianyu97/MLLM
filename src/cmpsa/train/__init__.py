"""CMPSA training subpackage.

Stages
------
- :mod:`cmpsa.train.extract_features`  cache per-patch CLIP features and
  per-token text-backbone features under ``paths.CLIP_FEATURES`` /
  ``paths.LLAMA_FEATURES``.
- :mod:`cmpsa.train.pretrain_proj`  Stage A: pretrain the PVE/PLE projection
  heads that map visual/text features into the PSAS.
- :mod:`cmpsa.train.train_cmota`    Stage B: Sinkhorn-OT cross-modal alignment
  (LoRA fine-tuning of the projection heads) with hard negatives.
- :mod:`cmpsa.train.train_hhd`      Stage C: supervise the HHD detectors with
  RLAIF-V / RLHF-V preference pairs.

All entry points expose an ``argparse`` CLI under ``if __name__ == '__main__'``
and are runnable with ``--limit`` for a smoke test. Heavy GPU imports
(torch / transformers) are performed lazily inside functions so that
``python -m cmpsa.train.<module> --help`` works on a CPU-only box.
"""
