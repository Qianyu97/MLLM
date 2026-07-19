"""Single source of truth for every data / result path in the CMPSA project.

The data and this project live together under one folder (``cmpsa_data/``) so the
whole thing can be rsync'd to a server as a unit::

    cmpsa_data/                <- DATA_ROOT (default)
    ├── basic/  benchmarks/  training/  tools/  halluprobe_vl/  cache/  derived/
    └── cmpsa_project/         <- PROJECT_ROOT (this package lives here)
        └── src/cmpsa/paths.py

Override the roots with environment variables when layouts differ on the server:

    CMPSA_DATA_ROOT     absolute path to the data root      (default: parent of project)
    CMPSA_MODELS_ROOT   where MLLM / backbone weights live  (default: DATA_ROOT/models)
    CMPSA_RESULTS_ROOT  where run artefacts are written     (default: PROJECT_ROOT/results)
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
_PATHS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _PATHS_FILE.parents[2]            # .../cmpsa_project
_DEFAULT_DATA_ROOT = _PATHS_FILE.parents[3]      # .../cmpsa_data

DATA_ROOT = Path(os.environ.get("CMPSA_DATA_ROOT", str(_DEFAULT_DATA_ROOT))).resolve()
MODELS_ROOT = Path(os.environ.get("CMPSA_MODELS_ROOT", str(DATA_ROOT / "models"))).resolve()
RESULTS_ROOT = Path(os.environ.get("CMPSA_RESULTS_ROOT", str(PROJECT_ROOT / "results"))).resolve()

# --------------------------------------------------------------------------- #
# A. Basic image / scene-graph sources
# --------------------------------------------------------------------------- #
BASIC = DATA_ROOT / "basic"

COCO = BASIC / "coco"
COCO_IMAGES = COCO / "images"
COCO_TRAIN2017 = COCO_IMAGES / "train2017"
COCO_VAL2017 = COCO_IMAGES / "val2017"
COCO_VAL2014 = COCO_IMAGES / "val2014"
COCO_ANN = COCO / "annotations"
COCO_INSTANCES_VAL2014 = COCO_ANN / "instances_val2014.json"
COCO_CAPTIONS_VAL2014 = COCO_ANN / "captions_val2014.json"
COCO_INSTANCES_VAL2017 = COCO_ANN / "instances_val2017.json"
COCO_CAPTIONS_VAL2017 = COCO_ANN / "captions_val2017.json"
COCO_INSTANCES_TRAIN2017 = COCO_ANN / "instances_train2017.json"
COCO_CAPTIONS_TRAIN2017 = COCO_ANN / "captions_train2017.json"

VG = BASIC / "visual_genome"
VG_OBJECTS = VG / "objects.json"
VG_RELATIONSHIPS = VG / "relationships.json"
VG_ATTRIBUTES = VG / "attributes.json"
VG_IMAGES = VG / "images"
VG_100K = VG_IMAGES / "VG_100K"
VG_100K_2 = VG_IMAGES / "VG_100K_2"

LVIS = BASIC / "lvis"
LVIS_TRAIN = LVIS / "lvis_v1_train.json"
LVIS_VAL = LVIS / "lvis_v1_val.json"

# --------------------------------------------------------------------------- #
# B. Evaluation benchmarks
# --------------------------------------------------------------------------- #
BENCH = DATA_ROOT / "benchmarks"

POPE_DIR = BENCH / "pope"
POPE_OUTPUT_COCO = POPE_DIR / "output" / "coco"
POPE_RANDOM = POPE_OUTPUT_COCO / "coco_pope_random.json"
POPE_POPULAR = POPE_OUTPUT_COCO / "coco_pope_popular.json"
POPE_ADVERSARIAL = POPE_OUTPUT_COCO / "coco_pope_adversarial.json"
POPE_SUBSETS = {"random": POPE_RANDOM, "popular": POPE_POPULAR, "adversarial": POPE_ADVERSARIAL}
# POPE questions reference COCO val2014 file names (COCO_val2014_<id>.jpg)
POPE_IMAGE_DIR = COCO_VAL2014

MME_DIR = BENCH / "mme"
MME_DATA = MME_DIR / "data"               # contains test-0000{0,1}-of-00002.parquet

HALLUSION_DIR = BENCH / "hallusion_bench"
HALLUSION_PARQUET = HALLUSION_DIR / "data" / "image-00000-of-00001.parquet"
HALLUSION_NONIMAGE_PARQUET = HALLUSION_DIR / "data" / "non_image-00000-of-00001.parquet"
HALLUSION_IMAGES = HALLUSION_DIR / "images"     # produced by parquet_to_images
HALLUSION_META = HALLUSION_DIR / "meta.jsonl"   # produced by parquet_to_images

MMHAL_DIR = BENCH / "mmhal_bench"
MMHAL_IMAGES = MMHAL_DIR / "images"

AMBER_DIR = BENCH / "amber"
AMBER_IMAGES = AMBER_DIR / "image"              # AMBER_1.jpg .. AMBER_1004.jpg
AMBER_DATA = AMBER_DIR / "data"
AMBER_ANN = AMBER_DATA / "annotations.json"
AMBER_RELATION = AMBER_DATA / "relation.json"
AMBER_QUERY = AMBER_DATA / "query"
AMBER_Q_GENERATIVE = AMBER_QUERY / "query_generative.json"
AMBER_Q_DISCRIMINATIVE = AMBER_QUERY / "query_discriminative.json"
AMBER_Q_EXISTENCE = AMBER_QUERY / "query_discriminative-existence.json"
AMBER_Q_ATTRIBUTE = AMBER_QUERY / "query_discriminative-attribute.json"
AMBER_Q_RELATION = AMBER_QUERY / "query_discriminative-relation.json"

CHAIR_DIR = BENCH / "chair"                     # evaluation code; images = COCO val2014
MMSAP_DIR = BENCH / "mm_sap"                    # NOTE: empty / missing in this download

# --------------------------------------------------------------------------- #
# C. Training data (image bytes are embedded inside the parquet files)
# --------------------------------------------------------------------------- #
TRAIN = DATA_ROOT / "training"

LLAVA_150K = TRAIN / "llava_150k"
LLAVA_INSTRUCT_150K = LLAVA_150K / "llava_instruct_150k.json"
LLAVA_MIX665K = LLAVA_150K / "llava_v1_5_mix665k.json"

SHAREGPT4V = TRAIN / "sharegpt4v"
SHAREGPT4V_CAPTIONER_1246K = SHAREGPT4V / "share-captioner_coco_lcs_sam_1246k_1107.json"
SHAREGPT4V_CAP100K = SHAREGPT4V / "sharegpt4v_instruct_gpt4-vision_cap100k.json"
SHAREGPT4V_MIX665K = SHAREGPT4V / "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"

RLHF_V_PARQUET = TRAIN / "rlhf_v" / "RLHF-V-Dataset.parquet"
RLAIF_V_DIR = TRAIN / "rlaif_v"                 # RLAIF-V-Dataset_000.parquet .. _013.parquet

# --------------------------------------------------------------------------- #
# D. Auto-labeling tool model weights
# --------------------------------------------------------------------------- #
TOOLS = DATA_ROOT / "tools"
GROUNDING_DINO = TOOLS / "grounding_dino"
SAM2 = TOOLS / "sam2"
RAM_PLUS = TOOLS / "ram_plus"
RELTR = TOOLS / "reltr"

# --------------------------------------------------------------------------- #
# Self-built / derived / cache
# --------------------------------------------------------------------------- #
HALLUPROBE = DATA_ROOT / "halluprobe_vl"
HALLUPROBE_IMAGES = HALLUPROBE / "images"
HALLUPROBE_ANN = HALLUPROBE / "annotations"
HALLUPROBE_SPLITS = HALLUPROBE / "splits"

DERIVED = DATA_ROOT / "derived"
VG_REL_DIR = DERIVED / "vg_rel"
VG_REL_JSONL = VG_REL_DIR / "vg_rel.jsonl"
VG_ATTR_DIR = DERIVED / "vg_attr"
VG_ATTR_JSONL = VG_ATTR_DIR / "vg_attr.jsonl"

CACHE = DATA_ROOT / "cache"
CLIP_FEATURES = CACHE / "clip_features"
LLAMA_FEATURES = CACHE / "llama_features"

# --------------------------------------------------------------------------- #
# Results layout
# --------------------------------------------------------------------------- #
CKPT_DIR = RESULTS_ROOT / "checkpoints"
PRED_DIR = RESULTS_ROOT / "predictions"
METRICS_DIR = RESULTS_ROOT / "metrics"
FIGURES_DIR = RESULTS_ROOT / "figures"
TABLES_DIR = RESULTS_ROOT / "tables"
LOG_DIR = RESULTS_ROOT / "logs"

# Directories that scripts are allowed to create.
WRITABLE_DIRS = [
    HALLUPROBE_IMAGES, HALLUPROBE_ANN, HALLUPROBE_SPLITS,
    VG_REL_DIR, VG_ATTR_DIR, HALLUSION_IMAGES,
    CLIP_FEATURES, LLAMA_FEATURES,
    CKPT_DIR, PRED_DIR, METRICS_DIR, FIGURES_DIR, TABLES_DIR, LOG_DIR,
]


def ensure_dirs() -> None:
    """Create all writable output directories (idempotent)."""
    for d in WRITABLE_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def pred_path(benchmark: str, model: str, method: str) -> Path:
    """Standardized prediction file: results/predictions/<bench>/<model>__<method>.jsonl"""
    out = PRED_DIR / benchmark
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{model}__{method}.jsonl"


def metrics_path(benchmark: str, model: str, method: str) -> Path:
    """Standardized metrics file: results/metrics/<bench>/<model>__<method>.json"""
    out = METRICS_DIR / benchmark
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{model}__{method}.json"


def mme_parquets() -> list[Path]:
    return sorted(MME_DATA.glob("test-*.parquet"))


def rlaif_v_parquets() -> list[Path]:
    return sorted(RLAIF_V_DIR.glob("RLAIF-V-Dataset_*.parquet"))


def coco_val2014_image(image_id: int | str) -> Path:
    """Resolve a COCO val2014 image path from an integer id or a file name."""
    if isinstance(image_id, str) and image_id.endswith(".jpg"):
        return COCO_VAL2014 / image_id
    return COCO_VAL2014 / f"COCO_val2014_{int(image_id):012d}.jpg"


def vg_image(image_id: int | str) -> Path:
    """Resolve a Visual Genome image path; images are split across VG_100K(_2)."""
    name = f"{int(image_id)}.jpg"
    p = VG_100K / name
    if p.exists():
        return p
    return VG_100K_2 / name
