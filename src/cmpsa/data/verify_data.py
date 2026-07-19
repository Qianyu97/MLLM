"""Verify the on-disk CMPSA data inventory and report a ✓ / ⚠️ / ✗ table.

This is the one data script that must *really run* on the prep box, so it uses
**pyarrow / json / pathlib only** — no torch, no transformers, no PIL.

Checks performed
----------------
* COCO image-count sanity: train2017 == 118287, val2017 == 5000, val2014 == 40504.
* Visual Genome: objects.json / relationships.json / attributes.json present.
* Benchmarks: key files for POPE, MME, HallusionBench, AMBER, MMHal, CHAIR.
* MM-SAP: explicitly reported as MISSING (it is absent from this download).
* Parquet files (HallusionBench, RLHF-V, RLAIF-V, MME) open with pyarrow; we
  report row count and column names.
* Training parquet *count* (RLAIF-V shards).

Severity:
  ✓  ok            (counts as pass)
  ⚠️  warning       (soft / optional / not-blocking)
  ✗  hard failure  (blocking — script exits non-zero)

Run::

    python -m cmpsa.data.verify_data
    python -m cmpsa.data.verify_data --fast    # skip slow image-dir counting
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger

LOGGER = get_logger("cmpsa.data.verify_data")

OK, WARN, FAIL = "✓", "⚠️", "✗"

# Expected COCO image counts (the standard dataset cardinalities).
EXPECT_COUNTS = {
    "COCO train2017": (paths.COCO_TRAIN2017, 118287),
    "COCO val2017": (paths.COCO_VAL2017, 5000),
    "COCO val2014": (paths.COCO_VAL2014, 40504),
}


class Report:
    """Accumulate (severity, name, detail) rows and render a final table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.hard_fail = False

    def add(self, sev: str, name: str, detail: str = "") -> None:
        if sev == FAIL:
            self.hard_fail = True
        self.rows.append((sev, name, detail))

    def ok(self, name, detail=""):   self.add(OK, name, detail)
    def warn(self, name, detail=""): self.add(WARN, name, detail)
    def fail(self, name, detail=""): self.add(FAIL, name, detail)

    def render(self) -> str:
        wname = max((len(r[1]) for r in self.rows), default=8) + 2
        lines = ["", "=" * 78, "CMPSA DATA VERIFICATION", "=" * 78]
        for sev, name, detail in self.rows:
            lines.append(f"  {sev}  {name.ljust(wname)} {detail}")
        n_ok = sum(1 for r in self.rows if r[0] == OK)
        n_warn = sum(1 for r in self.rows if r[0] == WARN)
        n_fail = sum(1 for r in self.rows if r[0] == FAIL)
        lines.append("-" * 78)
        lines.append(f"  {OK} {n_ok} ok   {WARN} {n_warn} warn   {FAIL} {n_fail} fail")
        lines.append("=" * 78)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual check helpers
# --------------------------------------------------------------------------- #
def _count_images(d: Path) -> int:
    if not d.exists():
        return -1
    n = 0
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            n += 1
    return n


def check_coco_counts(rep: Report, fast: bool) -> None:
    for name, (d, expected) in EXPECT_COUNTS.items():
        if not d.exists():
            rep.fail(name, f"dir missing: {d}")
            continue
        if fast:
            rep.warn(name, f"(skipped count; expected {expected}) {d}")
            continue
        got = _count_images(d)
        if got == expected:
            rep.ok(name, f"{got} images")
        else:
            rep.fail(name, f"count {got} != expected {expected} ({d})")


def check_exists(rep: Report, name: str, path: Path, hard: bool = True) -> None:
    if path.exists():
        rep.ok(name, str(path))
    elif hard:
        rep.fail(name, f"MISSING: {path}")
    else:
        rep.warn(name, f"missing (optional): {path}")


def check_parquet(rep: Report, name: str, path: Path, hard: bool = True) -> None:
    """Open with pyarrow, report row count + column names."""
    if not path.exists():
        (rep.fail if hard else rep.warn)(name, f"MISSING: {path}")
        return
    try:
        pf = pq.ParquetFile(str(path))
        nrows = pf.metadata.num_rows
        cols = list(pf.schema_arrow.names)
        col_preview = ", ".join(cols[:8]) + (" ..." if len(cols) > 8 else "")
        rep.ok(name, f"{nrows} rows; cols=[{col_preview}]")
    except Exception as exc:
        rep.fail(name, f"pyarrow open failed: {exc}")


def check_vg(rep: Report) -> None:
    for name, p in [("VG objects.json", paths.VG_OBJECTS),
                    ("VG relationships.json", paths.VG_RELATIONSHIPS),
                    ("VG attributes.json", paths.VG_ATTRIBUTES)]:
        check_exists(rep, name, p, hard=True)


def check_benchmarks(rep: Report) -> None:
    # POPE
    for k, p in paths.POPE_SUBSETS.items():
        check_exists(rep, f"POPE {k}", p, hard=True)
    # MME parquets
    mme = paths.mme_parquets()
    if mme:
        for p in mme:
            check_parquet(rep, f"MME {p.name}", p, hard=True)
    else:
        rep.fail("MME parquets", f"none found under {paths.MME_DATA}")
    # HallusionBench
    check_parquet(rep, "HallusionBench parquet", paths.HALLUSION_PARQUET, hard=True)
    check_exists(rep, "HallusionBench meta.jsonl (built)", paths.HALLUSION_META, hard=False)
    # AMBER
    check_exists(rep, "AMBER annotations.json", paths.AMBER_ANN, hard=True)
    check_exists(rep, "AMBER query_generative.json", paths.AMBER_Q_GENERATIVE, hard=True)
    check_exists(rep, "AMBER query_discriminative.json", paths.AMBER_Q_DISCRIMINATIVE, hard=False)
    check_exists(rep, "AMBER images dir", paths.AMBER_IMAGES, hard=True)
    # MMHal
    check_exists(rep, "MMHal images dir", paths.MMHAL_IMAGES, hard=False)
    # CHAIR (reuses COCO val2014 + captions/instances)
    check_exists(rep, "CHAIR captions_val2014.json", paths.COCO_CAPTIONS_VAL2014, hard=True)
    check_exists(rep, "CHAIR instances_val2014.json", paths.COCO_INSTANCES_VAL2014, hard=True)
    # MM-SAP: known missing from this download.
    if paths.MMSAP_DIR.exists() and any(paths.MMSAP_DIR.iterdir()):
        rep.ok("MM-SAP", str(paths.MMSAP_DIR))
    else:
        rep.warn("MM-SAP", "MISSING (known absent from this download; not blocking)")


def check_training(rep: Report) -> None:
    # RLHF-V single parquet
    check_parquet(rep, "RLHF-V parquet", paths.RLHF_V_PARQUET, hard=False)
    # RLAIF-V shard count + open first shard
    shards = paths.rlaif_v_parquets()
    if shards:
        rep.ok("RLAIF-V shard count", f"{len(shards)} parquet shard(s) under {paths.RLAIF_V_DIR}")
        check_parquet(rep, "RLAIF-V shard[0]", shards[0], hard=False)
    else:
        rep.warn("RLAIF-V shards", f"none found under {paths.RLAIF_V_DIR}")
    # LLaVA / ShareGPT4V instruction jsons (optional for data prep)
    check_exists(rep, "LLaVA mix665k", paths.LLAVA_MIX665K, hard=False)
    check_exists(rep, "ShareGPT4V captioner 1246k", paths.SHAREGPT4V_CAPTIONER_1246K, hard=False)


def check_derived(rep: Report) -> None:
    """Built artefacts are soft: they may not exist until build scripts run."""
    check_exists(rep, "derived vg_rel.jsonl (built)", paths.VG_REL_JSONL, hard=False)
    check_exists(rep, "derived vg_attr.jsonl (built)", paths.VG_ATTR_JSONL, hard=False)
    check_exists(rep, "halluprobe manifest (built)",
                 paths.HALLUPROBE_SPLITS / "halluprobe_manifest.json", hard=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(fast: bool) -> int:
    rep = Report()
    rep.ok("DATA_ROOT", str(paths.DATA_ROOT)) if paths.DATA_ROOT.exists() \
        else rep.fail("DATA_ROOT", f"MISSING: {paths.DATA_ROOT}")

    check_coco_counts(rep, fast)
    check_vg(rep)
    check_benchmarks(rep)
    check_training(rep)
    check_derived(rep)

    print(rep.render())
    if rep.hard_fail:
        LOGGER.error("hard data failures present — see ✗ rows above")
        return 1
    return 0


def main() -> None:
    # Make the ✓/✗ report safe on non-UTF-8 consoles (e.g. Windows GBK);
    # harmless on a Linux/UTF-8 server.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cfg = load_config()  # noqa: F841  (loaded for parity / future config use)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true",
                    help="skip slow image-directory counting (counts -> warnings)")
    ap.add_argument("--config", default=None, help="optional config override yaml")
    args = ap.parse_args()
    sys.exit(run(args.fast))


if __name__ == "__main__":
    main()
