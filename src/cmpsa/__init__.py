"""CMPSA — Cross-Modal Probabilistic Semantic Alignment.

Reference implementation / experiment harness for the RC-4 paper:
"A Probabilistic Distribution Alignment Method for Detecting and Mitigating
Hierarchical Hallucinations in Multimodal Foundation Models".

Subpackages
-----------
- :mod:`cmpsa.paths`   single source of truth for data / result paths
- :mod:`cmpsa.config`  YAML config loader
- :mod:`cmpsa.utils`   logging / io / seeding helpers
- :mod:`cmpsa.data`    data preparation & verification
- :mod:`cmpsa.models`  PVE/PLE, CM-OTA, HHD, PGD modules
- :mod:`cmpsa.train`   feature extraction & training stages A/B/C
- :mod:`cmpsa.eval`    per-benchmark evaluation + orchestration
- :mod:`cmpsa.viz`     paper tables & figures
"""

__version__ = "0.1.0"
