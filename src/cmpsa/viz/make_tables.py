"""Aggregate standard metrics JSON into the CMPSA paper tables.

Reads every ``*.json`` under ``paths.METRICS_DIR`` (the standard metrics schema,
see the import contract) and emits the paper tables as **both CSV and LaTeX** to
``paths.TABLES_DIR``:

    Table5  main comparison   methods x {POPE-F1, CHAIR-i, CHAIR-s,
                                        HallusionBench-fAcc, AMBER-Hal, MMHal-Score}
    Table6  HalluProbe        three categories (object / attribute / relation)
                              -- placeholder, awaiting data
    Table7  ablation          variant x {POPE F1, CHAIR-i, HallusionBench}
    Table8  cross-backbone    model  x {key metrics}
    Table9  efficiency        method x {tokens/sec, GPU_mem, latency}

Missing cells are filled with ``"-"``.

Standard metrics JSON schema consumed here::

    {"benchmark": str, "model": str, "method": str, "n": int,
     "metrics": {name: float}, "by_type": {type: {name: float}}}

Each file under ``METRICS_DIR/<bench>/<model>__<method>.json`` contributes one
(benchmark, model, method) record. Metric *names* inside ``metrics`` vary per
benchmark; this module maps the canonical names it needs and tolerates aliases.

CLI::

    python -m cmpsa.viz.make_tables                 # build from real metrics
    python -m cmpsa.viz.make_tables --demo          # synthesize placeholder tables
    python -m cmpsa.viz.make_tables --tables 5 7    # only a subset

This module is plotting/IO only -- it imports pandas / numpy / json and NEVER
imports torch or transformers.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MISSING = "-"  # placeholder for a missing table cell
DEMO_WATERMARK = "DEMO(占位,非真实结果)"

LOG = get_logger("cmpsa.viz.tables")


# --------------------------------------------------------------------------- #
# Metric-name resolution
# --------------------------------------------------------------------------- #
# Real eval modules may write slightly different metric keys. We resolve a
# *canonical* metric by trying a list of aliases (case-insensitive) against the
# flattened metrics dict for a (benchmark, model, method) record.
METRIC_ALIASES: dict[str, list[str]] = {
    "pope_f1": ["f1", "pope_f1", "f1_score"],
    "pope_acc": ["acc", "accuracy", "pope_acc"],
    "chair_i": ["chair_i", "chairi", "chair_instance", "ci"],
    "chair_s": ["chair_s", "chairs", "chair_sentence", "cs"],
    "hallusion_facc": ["facc", "facc", "figure_acc", "hallusion_facc", "f_acc"],
    "hallusion_qacc": ["qacc", "question_acc", "hallusion_qacc"],
    "amber_hal": ["amber_hal", "hal", "hallucination_rate", "chair", "amber_chair"],
    "amber_cover": ["cover", "coverage", "amber_cover"],
    "mmhal_score": ["mmhal_score", "score", "mmhal", "avg_score"],
    "mmhal_halrate": ["hall_rate", "hallucination_rate", "mmhal_halrate"],
    # efficiency
    "tokens_per_sec": ["tokens_per_sec", "tokens/sec", "tok_per_sec", "throughput"],
    "gpu_mem": ["gpu_mem", "gpu_memory_gb", "gpu_mem_gb", "peak_mem_gb", "vram_gb"],
    "latency": ["latency", "latency_ms", "latency_s", "lat"],
}

PERCENT_METRICS = {
    "pope_f1", "pope_acc", "chair_i", "chair_s",
    "hallusion_facc", "hallusion_qacc", "amber_hal", "amber_cover",
    "mmhal_halrate", "yes_ratio",
}


def _resolve_metric(metrics: dict[str, Any], canonical: str) -> float | None:
    """Return the first matching metric value for a canonical name, else None."""
    if not metrics:
        return None
    lower = {str(k).lower(): v for k, v in metrics.items()}
    for alias in METRIC_ALIASES.get(canonical, [canonical]):
        if alias.lower() in lower:
            val = lower[alias.lower()]
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _fmt(value: float | None, decimals: int = 2) -> str:
    """Format a numeric cell; missing -> MISSING placeholder."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return MISSING
    return f"{value:.{decimals}f}"


def _display_value(value: float | None, canonical: str) -> float | None:
    """Convert stored metric values into paper-table display values.

    Evaluation scripts store most rates as fractions in [0, 1], while papers
    usually report F1/accuracy/hallucination rates as percentages. Demo records
    already use percentage-like values, so values > 1 are left untouched.
    """
    if value is None:
        return None
    if canonical in PERCENT_METRICS and abs(value) <= 1.000001:
        return value * 100.0
    return value


def _fmt_metric(value: float | None, canonical: str, decimals: int = 2) -> str:
    """Format a canonical metric for paper display."""
    return _fmt(_display_value(value, canonical), decimals)


# --------------------------------------------------------------------------- #
# Metrics loading
# --------------------------------------------------------------------------- #
def load_all_metrics(metrics_dir: Path | None = None) -> list[dict]:
    """Load every standard metrics JSON under ``metrics_dir`` (recursively).

    Returns a list of records each shaped like::

        {"benchmark","model","method","n","metrics":{...},"by_type":{...},
         "_path": str}
    """
    metrics_dir = Path(metrics_dir) if metrics_dir is not None else paths.METRICS_DIR
    records: list[dict] = []
    if not metrics_dir.exists():
        LOG.warning("metrics dir does not exist: %s", metrics_dir)
        return records
    for jf in sorted(metrics_dir.rglob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as exc:  # noqa: BLE001 -- tolerate a bad file, keep going
            LOG.warning("skip unreadable metrics file %s: %s", jf, exc)
            continue
        if not isinstance(obj, dict) or "metrics" not in obj:
            LOG.warning("skip non-standard metrics file %s", jf)
            continue
        # Fill benchmark from the parent dir name when absent.
        obj.setdefault("benchmark", jf.parent.name)
        obj.setdefault("model", "unknown")
        obj.setdefault("method", "unknown")
        obj.setdefault("metrics", {})
        obj.setdefault("by_type", {})
        obj["_path"] = str(jf)
        records.append(obj)
    LOG.info("loaded %d metrics records from %s", len(records), metrics_dir)
    return records


def index_records(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Index records by (benchmark, model, method) for O(1) lookup."""
    idx: dict[tuple[str, str, str], dict] = {}
    for r in records:
        key = (str(r["benchmark"]).lower(), str(r["model"]), str(r["method"]))
        idx[key] = r
    return idx


def _bench_metric(idx: dict, bench: str, model: str, method: str, canonical: str) -> float | None:
    """Look up a canonical metric for a (bench, model, method) triple."""
    rec = idx.get((bench.lower(), model, method))
    if rec is None:
        return None
    return _resolve_metric(rec.get("metrics", {}), canonical)


def discover_methods(records: list[dict]) -> list[str]:
    """All distinct method names seen, with vanilla/cmpsa floated to the front."""
    seen = []
    for r in records:
        m = str(r["method"])
        if m not in seen:
            seen.append(m)
    order = {"vanilla": 0, "cmpsa": 1}
    return sorted(seen, key=lambda m: (order.get(m, 2), m))


def discover_models(records: list[dict]) -> list[str]:
    seen = []
    for r in records:
        m = str(r["model"])
        if m not in seen:
            seen.append(m)
    return sorted(seen)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_table(df: pd.DataFrame, stem: str, caption: str, label: str,
                tables_dir: Path, demo: bool = False) -> tuple[Path, Path]:
    """Write a DataFrame to ``<stem>.csv`` and ``<stem>.tex`` under ``tables_dir``.

    The index of ``df`` becomes the first column. A ``DEMO`` watermark comment is
    prepended (CSV) / added to the caption (LaTeX) when ``demo`` is True.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{stem}.csv"
    tex_path = tables_dir / f"{stem}.tex"

    # ---- CSV ----
    header_comment = f"# {DEMO_WATERMARK}\n" if demo else ""
    csv_body = df.to_csv(index=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        if header_comment:
            f.write(header_comment)
        f.write(csv_body)

    # ---- LaTeX ----
    tex_caption = caption + (f"  % {DEMO_WATERMARK}" if demo else "")
    tex = _df_to_latex(df, caption=tex_caption, label=label, demo=demo)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    LOG.info("wrote %s and %s", csv_path.name, tex_path.name)
    return csv_path, tex_path


def _latex_escape(text: str) -> str:
    """Escape LaTeX-special characters in a cell / header string."""
    text = str(text)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def _df_to_latex(df: pd.DataFrame, caption: str, label: str, demo: bool) -> str:
    """Hand-rolled booktabs LaTeX (no jinja2 dependency needed by pandas styler)."""
    index_name = df.index.name or "Method"
    cols = list(df.columns)
    ncol = len(cols) + 1
    colspec = "l" + "c" * len(cols)

    lines: list[str] = []
    if demo:
        lines.append(f"% {DEMO_WATERMARK}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    header = " & ".join([_latex_escape(index_name)] + [_latex_escape(c) for c in cols]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for idx_val, row in df.iterrows():
        cells = [_latex_escape(idx_val)] + [_latex_escape(row[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")  # trailing newline
    assert ncol == len(cols) + 1  # silence unused; documents column count
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Table builders (real data)
# --------------------------------------------------------------------------- #
TABLE5_COLUMNS = [
    ("POPE-F1 (%)", "pope", "pope_f1", 2),
    ("CHAIR-i (%)", "chair", "chair_i", 2),
    ("CHAIR-s (%)", "chair", "chair_s", 2),
    ("HallusionBench-fAcc (%)", "hallusionbench", "hallusion_facc", 2),
    ("AMBER-Hal (%)", "amber", "amber_hal", 2),
    ("MMHal-Score", "mmhal", "mmhal_score", 2),
]


def build_table5(records: list[dict], cfg) -> pd.DataFrame:
    """Main comparison table: methods x key hallucination metrics."""
    idx = index_records(records)
    methods = discover_methods(records) or list(getattr(cfg.eval, "methods", []) or ["vanilla", "cmpsa"])
    model = cfg.mllm.key
    data: dict[str, list[str]] = {col: [] for col, *_ in TABLE5_COLUMNS}
    for method in methods:
        for col, bench, canonical, dec in TABLE5_COLUMNS:
            val = _bench_metric(idx, bench, model, method, canonical)
            data[col].append(_fmt_metric(val, canonical, dec))
    df = pd.DataFrame(data, index=pd.Index(methods, name="Method"))
    return df


def build_table6(records: list[dict], cfg) -> pd.DataFrame:
    """Three-category hallucination table (object / attribute / relation).

    DATA-INTEGRITY FIX (2026-07-12): this table previously fell back to AMBER's
    by_type accuracies and presented them *unmarked* under a HalluProbe caption
    (the audit flagged this as data-not-real risk). It now (1) prefers a real
    ``halluprobe`` record, (2) otherwise clearly marks each row's provenance in a
    ``Source`` column as "AMBER-proxy (NOT HalluProbe)", and never silently
    impersonates HalluProbe-VL.
    """
    idx = index_records(records)
    methods = discover_methods(records) or ["vanilla", "cmpsa"]
    model = cfg.mllm.key
    categories = ["object", "attribute", "relation"]
    cols = [f"{c.capitalize()} Acc (%)" for c in categories] + ["Overall Acc (%)", "Source"]
    rows: dict[str, list[str]] = {c: [] for c in cols}
    for method in methods:
        rec = idx.get(("halluprobe", model, method))
        source = "HalluProbe-VL"
        if rec is None:
            rec = idx.get(("amber", model, method))
            source = "AMBER-proxy (NOT HalluProbe)" if rec is not None else "-"
        by_type = (rec or {}).get("by_type", {}) if rec else {}
        overall_metrics = (rec or {}).get("metrics", {}) if rec else {}
        for cat in categories:
            cat_metrics = by_type.get(cat, {})
            val = _resolve_metric(cat_metrics, "pope_acc") if cat_metrics else None
            rows[f"{cat.capitalize()} Acc (%)"].append(_fmt_metric(val, "pope_acc"))
        rows["Overall Acc (%)"].append(_fmt_metric(_resolve_metric(overall_metrics, "pope_acc"), "pope_acc"))
        rows["Source"].append(source)
    df = pd.DataFrame(rows, index=pd.Index(methods, name="Method"))
    return df


TABLE7_COLUMNS = [
    ("POPE F1 (%)", "pope", "pope_f1", 2),
    ("CHAIR-i (%)", "chair", "chair_i", 2),
    ("HallusionBench fAcc (%)", "hallusionbench", "hallusion_facc", 2),
]

# Ablation variants are encoded as method names like "cmpsa-w/o-ota". We surface
# whatever methods exist; the canonical full model is "cmpsa".
ABLATION_DISPLAY = {
    "vanilla": "Vanilla (no CMPSA)",
    "cmpsa": "CMPSA (full)",
    "cmpsa-wo-ota": "w/o CM-OTA",
    "cmpsa-wo-hhd": "w/o HHD",
    "cmpsa-wo-pgd": "w/o PGD",
    "cmpsa-euclidean": "OT dist = Euclidean",
    "cmpsa-kl": "OT dist = KL",
    "cmpsa-deterministic": "Deterministic (no PVE/PLE)",
}


def build_table7(records: list[dict], cfg) -> pd.DataFrame:
    """Ablation table: variant x {POPE F1, CHAIR-i, HallusionBench}."""
    idx = index_records(records)
    model = cfg.mllm.key
    # Ablation methods = any method containing 'cmpsa' (incl. variants) + vanilla.
    methods = discover_methods(records)
    ablation_methods = [m for m in methods if ("cmpsa" in m.lower() or m == "vanilla")]
    if not ablation_methods:
        ablation_methods = ["vanilla", "cmpsa"]
    display_names = [ABLATION_DISPLAY.get(m, m) for m in ablation_methods]
    data: dict[str, list[str]] = {col: [] for col, *_ in TABLE7_COLUMNS}
    for method in ablation_methods:
        for col, bench, canonical, dec in TABLE7_COLUMNS:
            val = _bench_metric(idx, bench, model, method, canonical)
            data[col].append(_fmt_metric(val, canonical, dec))
    df = pd.DataFrame(data, index=pd.Index(display_names, name="Variant"))
    return df


TABLE8_COLUMNS = [
    ("POPE-F1 (%)", "pope", "pope_f1", 2),
    ("CHAIR-i (%)", "chair", "chair_i", 2),
    ("HallusionBench-fAcc (%)", "hallusionbench", "hallusion_facc", 2),
    ("AMBER-Hal (%)", "amber", "amber_hal", 2),
]


def build_table8(records: list[dict], cfg) -> pd.DataFrame:
    """Cross-backbone generalization: models x key metrics (CMPSA method).

    Rows are the distinct models that have results; the column metrics are taken
    from the CMPSA method when available, otherwise vanilla, otherwise MISSING.
    """
    models = discover_models(records)
    if not models:
        models = [cfg.mllm.key]
    idx = index_records(records)
    methods_present = discover_methods(records)
    prefer = "cmpsa" if "cmpsa" in methods_present else (methods_present[0] if methods_present else "vanilla")
    data: dict[str, list[str]] = {col: [] for col, *_ in TABLE8_COLUMNS}
    for model in models:
        for col, bench, canonical, dec in TABLE8_COLUMNS:
            val = _bench_metric(idx, bench, model, prefer, canonical)
            if val is None and prefer != "vanilla":
                val = _bench_metric(idx, bench, model, "vanilla", canonical)
            data[col].append(_fmt_metric(val, canonical, dec))
    df = pd.DataFrame(data, index=pd.Index(models, name="Backbone"))
    return df


TABLE9_COLUMNS = [
    ("Tokens/sec", "tokens_per_sec", 1),
    ("GPU Mem (GB)", "gpu_mem", 2),
    ("Latency (s)", "latency", 3),
]


def build_table9(records: list[dict], cfg) -> pd.DataFrame:
    """Efficiency table: methods x {tokens/sec, GPU_mem, latency}.

    Efficiency numbers live in the metrics dict of *any* benchmark record (eval
    can attach them as extra_metrics). We scan all benchmarks for the model and
    take the first record per method that carries an efficiency metric.
    """
    model = cfg.mllm.key
    methods = discover_methods(records) or ["vanilla", "cmpsa"]
    # Group records by method for this model.
    by_method: dict[str, list[dict]] = {}
    for r in records:
        if str(r["model"]) != model:
            continue
        by_method.setdefault(str(r["method"]), []).append(r)
    data: dict[str, list[str]] = {col: [] for col, *_ in TABLE9_COLUMNS}
    for method in methods:
        recs = by_method.get(method, [])
        for col, canonical, dec in TABLE9_COLUMNS:
            val = None
            for r in recs:
                val = _resolve_metric(r.get("metrics", {}), canonical)
                if val is not None:
                    break
            data[col].append(_fmt(val, dec))
    df = pd.DataFrame(data, index=pd.Index(methods, name="Method"))
    return df


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #
def demo_records(cfg) -> list[dict]:
    """Synthesize standard metrics records so the full table pipeline can run.

    Numbers are plausible-looking but entirely fabricated; every produced table
    is watermarked with ``DEMO_WATERMARK``.
    """
    rng = np.random.default_rng(getattr(cfg, "seed", 42))
    primary = cfg.mllm.key
    models = [primary, "llava-1.5-13b", "instructblip-7b", "qwen-vl-chat"]
    methods = ["vanilla", "cmpsa"]
    ablation_methods = [
        "cmpsa", "cmpsa-wo-ota", "cmpsa-wo-hhd", "cmpsa-wo-pgd",
        "cmpsa-euclidean", "cmpsa-kl", "cmpsa-deterministic",
    ]
    records: list[dict] = []

    def mk(bench: str, model: str, method: str, metrics: dict, by_type=None):
        records.append({
            "benchmark": bench, "model": model, "method": method,
            "n": int(rng.integers(500, 3000)), "metrics": metrics,
            "by_type": by_type or {},
        })

    # cmpsa should look a bit better than vanilla; variants degrade vs full.
    def boost(method: str) -> float:
        if method == "vanilla":
            return 0.0
        if method == "cmpsa":
            return 1.0
        return float(rng.uniform(0.3, 0.85))  # partial ablations

    for model in models:
        base_f1 = float(rng.uniform(83, 87))
        base_chair_i = float(rng.uniform(8, 14))
        base_chair_s = float(rng.uniform(45, 60))
        base_facc = float(rng.uniform(30, 42))
        base_amber = float(rng.uniform(6, 12))
        base_mmhal = float(rng.uniform(2.0, 2.6))
        all_methods = methods if model != primary else sorted(set(methods + ablation_methods))
        for method in all_methods:
            b = boost(method)
            mk("pope", model, method, {
                "f1": base_f1 + b * rng.uniform(2.5, 4.5),
                "acc": base_f1 + b * rng.uniform(2.0, 4.0) - 1.0,
            })
            mk("chair", model, method, {
                "chair_i": max(1.0, base_chair_i - b * rng.uniform(3.0, 6.0)),
                "chair_s": max(5.0, base_chair_s - b * rng.uniform(10.0, 20.0)),
            })
            mk("hallusionbench", model, method, {
                "facc": base_facc + b * rng.uniform(4.0, 8.0),
                "qacc": base_facc + 20 + b * rng.uniform(3.0, 6.0),
            })
            mk("amber", model, method, {
                "amber_hal": max(1.0, base_amber - b * rng.uniform(2.0, 4.0)),
                "cover": float(rng.uniform(48, 56)),
            })
            mk("mmhal", model, method, {
                "mmhal_score": base_mmhal + b * rng.uniform(0.2, 0.5),
                "hall_rate": max(0.1, 0.6 - b * rng.uniform(0.1, 0.25)),
            })
            # efficiency: attach to a synthetic record
            tps = float(rng.uniform(28, 40))
            mk("efficiency", model, method, {
                "tokens_per_sec": tps - (3.0 if method != "vanilla" else 0.0),
                "gpu_mem": float(rng.uniform(15.5, 17.5)) + (0.8 if method != "vanilla" else 0.0),
                "latency": float(rng.uniform(1.6, 2.4)) + (0.25 if method != "vanilla" else 0.0),
            })
            # HalluProbe by_type (object/attribute/relation accuracies)
            mk("halluprobe", model, method, {
                "acc": float(rng.uniform(70, 78)) + b * rng.uniform(3, 6),
            }, by_type={
                "object": {"acc": float(rng.uniform(78, 85)) + b * rng.uniform(2, 5)},
                "attribute": {"acc": float(rng.uniform(68, 76)) + b * rng.uniform(3, 6)},
                "relation": {"acc": float(rng.uniform(60, 70)) + b * rng.uniform(4, 8)},
            })
    return records


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
TABLE_SPECS = {
    "5": ("table5_main", build_table5,
          "Main comparison on hallucination benchmarks.", "tab:main"),
    "6": ("table6_halluprobe", build_table6,
          "Object, attribute, and relation hallucination accuracy.", "tab:halluprobe"),
    "7": ("table7_ablation", build_table7,
          "Ablation study of CMPSA components.", "tab:ablation"),
    "8": ("table8_backbone", build_table8,
          "Cross-backbone generalization.", "tab:backbone"),
    "9": ("table9_efficiency", build_table9,
          "Inference efficiency.", "tab:efficiency"),
}


def make_tables(which: list[str] | None = None, demo: bool = False,
                config: str | None = None, tables_dir: Path | None = None,
                metrics_dir: Path | None = None) -> dict[str, tuple[Path, Path]]:
    """Build the requested tables and write CSV + LaTeX. Returns {id: (csv,tex)}."""
    cfg = load_config(config)
    set_seed(getattr(cfg, "seed", 42))
    tables_dir = Path(tables_dir) if tables_dir is not None else paths.TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    if demo:
        records = demo_records(cfg)
        LOG.info("DEMO mode: synthesized %d placeholder metrics records", len(records))
    else:
        records = load_all_metrics(metrics_dir)
        if not records:
            LOG.warning("no real metrics found under %s -- tables will be all '%s'. "
                        "Use --demo for a visual sample.",
                        metrics_dir or paths.METRICS_DIR, MISSING)

    which = which or list(TABLE_SPECS.keys())
    outputs: dict[str, tuple[Path, Path]] = {}
    for tid in which:
        tid = str(tid)
        if tid not in TABLE_SPECS:
            LOG.warning("unknown table id %s (valid: %s)", tid, ", ".join(TABLE_SPECS))
            continue
        stem, builder, caption, label = TABLE_SPECS[tid]
        df = builder(records, cfg)
        outputs[tid] = write_table(df, stem, caption, label, tables_dir, demo=demo)
    LOG.info("done: %d table(s) -> %s", len(outputs), tables_dir)
    return outputs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cmpsa.viz.make_tables",
        description="Aggregate RESULTS/metrics/** into the CMPSA paper tables (CSV + LaTeX).",
    )
    p.add_argument("--config", default=None, help="optional config YAML override")
    p.add_argument("--demo", action="store_true",
                   help="synthesize placeholder data and watermark the tables")
    p.add_argument("--tables", nargs="*", default=None, metavar="ID",
                   help="subset of table ids to build (5 6 7 8 9); default: all")
    p.add_argument("--tables-dir", default=None,
                   help="override output dir (default: paths.TABLES_DIR)")
    p.add_argument("--metrics-dir", default=None,
                   help="override input metrics dir (default: paths.METRICS_DIR)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.verbose:
        get_logger("cmpsa.viz.tables").setLevel(logging.DEBUG)
    outs = make_tables(
        which=args.tables,
        demo=args.demo,
        config=args.config,
        tables_dir=Path(args.tables_dir) if args.tables_dir else None,
        metrics_dir=Path(args.metrics_dir) if args.metrics_dir else None,
    )
    for tid, (csv_path, tex_path) in sorted(outs.items()):
        print(f"Table{tid}: {csv_path}  |  {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
