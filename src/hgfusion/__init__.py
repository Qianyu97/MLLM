# -*- coding: utf-8 -*-
"""hgfusion — plug-and-play hallucination wrapper for multimodal LLMs.

    from hgfusion import HGFWrapper
    w = HGFWrapper(models_root="/path/to/models")
    clean = w.revise_caption("img.jpg", caption).text          # L0
"""
from .wrapper import (CaptionReport, FusionCalibration, HGFWrapper,
                      RevisedCaption, sentence_remove)

__all__ = ["HGFWrapper", "CaptionReport", "RevisedCaption",
           "FusionCalibration", "sentence_remove"]
__version__ = "1.0.0"
