"""Render the CMPSA paper figures as PNG + PDF to ``paths.FIGURES_DIR``.

Figures
-------
    fig12_dataset_stats   HalluProbe / benchmark dataset statistics histograms
    fig13_ablation        ablation component bar chart
    fig14_robustness      noise-level vs hallucination-rate curves
    fig_psas_tsne         t-SNE of PSAS embeddings (from cache; --demo synthesizes)
    fig_main_compare      main-metric comparison bar chart

All text is English, the style is paper-ready (no oversized titles, tight
layout, deterministic colors) so figures can be dropped straight into the paper.
Every figure supports ``--demo`` to synthesize placeholder data; demo figures
carry a small ``DEMO(占位,非真实结果)`` watermark.

This module is plotting/IO only -- it imports matplotlib / seaborn / numpy /
pandas / json and NEVER imports torch or transformers. t-SNE uses scikit-learn
if available, otherwise falls back to a deterministic 2-D PCA projection.

CLI::

    python -m cmpsa.viz.make_figures                       # build from real data
    python -m cmpsa.viz.make_figures --demo                # synthesize everything
    python -m cmpsa.viz.make_figures --figures 13 main     # subset
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / server-safe backend
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: E402
    _HAS_SNS = True
except Exception:  # noqa: BLE001
    sns = None
    _HAS_SNS = False

from cmpsa import paths  # noqa: E402
from cmpsa.config import load_config  # noqa: E402
from cmpsa.utils import get_logger, set_seed  # noqa: E402

# Reuse the table metrics loader so figures and tables read the same source.
from cmpsa.viz.make_tables import (  # noqa: E402
    demo_records,
    index_records,
    load_all_metrics,
    _bench_metric,
    _resolve_metric,
    discover_methods,
    _display_value,
)

LOG = get_logger("cmpsa.viz.figures")

# Figures render in English and use an ASCII-safe on-canvas watermark. The full
# Chinese watermark (``DEMO_WATERMARK``) is still used for the table text files.
FIG_WATERMARK = "DEMO (placeholder, not real results)"

# --------------------------------------------------------------------------- #
# Paper style: edit these constants to tune every figure in one place.
# --------------------------------------------------------------------------- #
MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 190.0
FIGURE_DPI = 600
FONT_FAMILY = "Arial"
FONT_FALLBACKS = [FONT_FAMILY, "Helvetica", "DejaVu Sans"]

AXIS_LINEWIDTH = 0.75
GRID_LINEWIDTH = 0.45
LINEWIDTH = 1.6
MARKER_SIZE = 4.0
BAR_EDGE_WIDTH = 0.4

PAPER_RC = {
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FALLBACKS,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": GRID_LINEWIDTH,
    "axes.linewidth": AXIS_LINEWIDTH,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.02,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

# Deterministic, color-blind-friendly palette with clear contrast in print.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9",
           "#E69F00", "#332288", "#999999", "#117733", "#882255"]


def apply_style() -> None:
    plt.rcParams.update(PAPER_RC)
    if _HAS_SNS:
        sns.set_style("whitegrid")
        sns.set_context("paper")
        plt.rcParams.update(PAPER_RC)  # re-assert after seaborn


def _figsize(height_mm: float) -> tuple[float, float]:
    """Return a 190-mm-wide IEEE/PAMI-ready figure size in inches."""
    return FIGURE_WIDTH_MM / MM_PER_INCH, height_mm / MM_PER_INCH


def _watermark(ax, demo: bool) -> None:
    """Stamp a faint DEMO watermark on an axes when in demo mode."""
    if not demo:
        return
    ax.text(0.5, 0.5, FIG_WATERMARK, transform=ax.transAxes,
            fontsize=13, color="0.85", alpha=0.45, ha="center", va="center",
            rotation=20, zorder=0, fontweight="bold")


def _save(fig, stem: str, figures_dir: Path) -> tuple[Path, Path]:
    """Save a figure as PNG and PDF; return both paths."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches=None)
    fig.savefig(pdf, dpi=FIGURE_DPI, bbox_inches=None)
    plt.close(fig)
    LOG.info("wrote %s and %s", png.name, pdf.name)
    return png, pdf


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# --------------------------------------------------------------------------- #
# fig13_ablation: ablation component bar chart
# --------------------------------------------------------------------------- #
def fig13_ablation(records: list[dict], cfg, figures_dir: Path, demo: bool) -> tuple[Path, Path]:
    """Grouped bars: ablation variants vs {POPE-F1, HallusionBench-fAcc}.

    Higher = better for both metrics shown.
    """
    apply_style()
    idx = index_records(records)
    model = cfg.mllm.key
    methods = [m for m in discover_methods(records) if ("cmpsa" in m.lower() or m == "vanilla")]
    if not methods:
        methods = ["vanilla", "cmpsa"]
    display = {
        "vanilla": "Vanilla", "cmpsa": "Full",
        "cmpsa-wo-ota": "w/o OTA", "cmpsa-wo-hhd": "w/o HHD",
        "cmpsa-wo-pgd": "w/o PGD", "cmpsa-euclidean": "Euclid",
        "cmpsa-kl": "KL", "cmpsa-deterministic": "Determ.",
    }
    labels = [display.get(m, m) for m in methods]
    pope = [_bench_metric(idx, "pope", model, m, "pope_f1") for m in methods]
    facc = [_bench_metric(idx, "hallusionbench", model, m, "hallusion_facc") for m in methods]
    pope = [_display_value(v, "pope_f1") for v in pope]
    facc = [_display_value(v, "hallusion_facc") for v in facc]
    pope_v = [np.nan if v is None else v for v in pope]
    facc_v = [np.nan if v is None else v for v in facc]

    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=_figsize(76))
    ax.bar(x - w / 2, pope_v, w, label="POPE-F1", color=_color(0),
           edgecolor="black", linewidth=BAR_EDGE_WIDTH)
    ax.bar(x + w / 2, facc_v, w, label="HallusionBench-fAcc", color=_color(2),
           edgecolor="black", linewidth=BAR_EDGE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.legend(ncol=2, loc="lower right")
    ax.margins(y=0.15)
    _watermark(ax, demo)
    fig.tight_layout()
    return _save(fig, "fig13_ablation", figures_dir)


# --------------------------------------------------------------------------- #
# fig14_robustness: noise level vs hallucination rate
# --------------------------------------------------------------------------- #
def _load_robustness(records: list[dict], cfg) -> dict[str, dict[str, list[float]]] | None:
    """Try to read a robustness sweep from metrics records.

    Convention: records with benchmark == 'robustness' carry metrics
    {"noise": float, "hall_rate": float} and a method name. Returns
    {method: {"noise":[...], "hall":[...]}} or None when absent.
    """
    series: dict[str, dict[str, list[float]]] = {}
    for r in records:
        if str(r.get("benchmark", "")).lower() != "robustness":
            continue
        m = str(r.get("method", "?"))
        noise = _resolve_metric(r.get("metrics", {}), "noise")
        hall = r.get("metrics", {}).get("hall_rate")
        if noise is None or hall is None:
            continue
        d = series.setdefault(m, {"noise": [], "hall": []})
        d["noise"].append(float(noise))
        d["hall"].append(float(hall))
    if not series:
        return None
    for m, d in series.items():
        order = np.argsort(d["noise"])
        d["noise"] = list(np.array(d["noise"])[order])
        d["hall"] = list(np.array(d["hall"])[order])
    return series


def fig14_robustness(records: list[dict], cfg, figures_dir: Path, demo: bool) -> tuple[Path, Path]:
    """Curves: input-noise level (x) vs hallucination rate (y), per method."""
    apply_style()
    series = _load_robustness(records, cfg)
    if series is None:
        if not demo:
            LOG.warning("no robustness sweep in metrics; emitting empty fig14 "
                        "(use --demo for a sample curve)")
        noise = np.linspace(0.0, 0.5, 11)
        rng = np.random.default_rng(getattr(cfg, "seed", 42))
        series = {
            "Vanilla": {"noise": list(noise),
                        "hall": list(0.18 + 0.55 * noise + rng.normal(0, 0.01, noise.size))},
            "CMPSA": {"noise": list(noise),
                      "hall": list(0.10 + 0.28 * noise + rng.normal(0, 0.008, noise.size))},
        }
        if not demo:
            # No real data and not demo: keep axes but draw nothing meaningful.
            series = {}

    fig, ax = plt.subplots(figsize=_figsize(72))
    for i, (method, d) in enumerate(sorted(series.items())):
        hall = [_display_value(float(v), "amber_hal") for v in d["hall"]]
        ax.plot(d["noise"], hall, marker="o", markersize=MARKER_SIZE,
                linewidth=LINEWIDTH, color=_color(i), label=method)
    ax.set_xlabel("Input noise level $\\sigma$")
    ax.set_ylabel("Hallucination rate (%)")
    if series:
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "no robustness data", transform=ax.transAxes,
                ha="center", va="center", color="0.6")
    ax.margins(x=0.02)
    _watermark(ax, demo)
    fig.tight_layout()
    return _save(fig, "fig14_robustness", figures_dir)


# --------------------------------------------------------------------------- #
# fig12_dataset_stats: dataset statistics histograms
# --------------------------------------------------------------------------- #
def _load_dataset_stats(cfg) -> dict[str, dict] | None:
    """Read dataset statistics for HalluProbe / benchmarks if a stats file exists.

    Looks for ``paths.HALLUPROBE_ANN/stats.json`` or
    ``paths.TABLES_DIR/dataset_stats.json``. Expected shape (all optional)::

        {"halluprobe_categories": {"object": n, "attribute": n, "relation": n},
         "benchmark_sizes": {"POPE": n, "AMBER": n, ...},
         "question_lengths": [..ints..]}
    """
    candidates = [
        paths.HALLUPROBE_ANN / "stats.json",
        paths.TABLES_DIR / "dataset_stats.json",
        paths.HALLUPROBE / "stats.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("failed to read stats file %s: %s", c, exc)
    return None


def _demo_dataset_stats(cfg) -> dict[str, Any]:
    rng = np.random.default_rng(getattr(cfg, "seed", 42))
    b = getattr(cfg, "build", None)
    coco = int(getattr(b, "halluprobe_coco", 2000)) if b else 2000
    vg = int(getattr(b, "halluprobe_vg", 1000)) if b else 1000
    adv = int(getattr(b, "halluprobe_adv", 500)) if b else 500
    return {
        "halluprobe_categories": {
            "object": int((coco + vg) * 0.45),
            "attribute": int((coco + vg) * 0.33),
            "relation": int((coco + vg) * 0.22 + adv),
        },
        "benchmark_sizes": {
            "POPE": 9000, "AMBER": 1004, "HallusionBench": 1129,
            "MMHal": 96, "MME": 2374, "HalluProbe": coco + vg + adv,
        },
        "question_lengths": list(rng.integers(5, 25, size=2000).astype(int)),
    }


def fig12_dataset_stats(records, cfg, figures_dir: Path, demo: bool) -> tuple[Path, Path]:
    """Three-panel dataset statistics: HalluProbe categories, benchmark sizes,
    and a question-length histogram."""
    apply_style()
    stats = _load_dataset_stats(cfg)
    if stats is None:
        if not demo:
            LOG.warning("no dataset stats file found; using demo stats for fig12")
        stats = _demo_dataset_stats(cfg)
        demo = True  # data is synthetic -> watermark all panels

    fig, axes = plt.subplots(1, 3, figsize=_figsize(72))

    # Panel 1: HalluProbe categories
    cats = stats.get("halluprobe_categories", {})
    ax = axes[0]
    if cats:
        names = list(cats.keys())
        vals = [cats[k] for k in names]
        ax.bar(names, vals, color=[_color(i) for i in range(len(names))],
               edgecolor="black", linewidth=BAR_EDGE_WIDTH)
        ax.set_ylabel("# samples")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=7)
    ax.set_title("HalluProbe categories")
    _watermark(ax, demo)

    # Panel 2: benchmark sizes
    sizes = stats.get("benchmark_sizes", {})
    ax = axes[1]
    if sizes:
        names = list(sizes.keys())
        vals = [sizes[k] for k in names]
        ax.barh(names, vals, color=_color(0), edgecolor="black",
                linewidth=BAR_EDGE_WIDTH)
        ax.set_xlabel("# items")
        ax.invert_yaxis()
    ax.set_title("Benchmark sizes")
    _watermark(ax, demo)

    # Panel 3: question-length histogram
    qlens = stats.get("question_lengths", [])
    ax = axes[2]
    if qlens:
        ax.hist(qlens, bins=20, color=_color(2), edgecolor="white",
                linewidth=BAR_EDGE_WIDTH)
        ax.set_xlabel("Question length (tokens)")
        ax.set_ylabel("Count")
    ax.set_title("Question lengths")
    _watermark(ax, demo)

    fig.tight_layout()
    return _save(fig, "fig12_dataset_stats", figures_dir)


# --------------------------------------------------------------------------- #
# fig_psas_tsne: t-SNE of PSAS embeddings
# --------------------------------------------------------------------------- #
def _embed_2d(feats: np.ndarray, seed: int) -> np.ndarray:
    """Project [N,D] features to 2-D. Use scikit-learn t-SNE if present, else PCA."""
    feats = np.asarray(feats, dtype=np.float64)
    if feats.shape[0] < 3:
        return np.zeros((feats.shape[0], 2))
    try:
        from sklearn.manifold import TSNE  # local import keeps module light
        perplexity = float(min(30, max(5, feats.shape[0] // 4)))
        ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  random_state=seed, learning_rate="auto")
        return ts.fit_transform(feats)
    except Exception as exc:  # noqa: BLE001 -- fall back to PCA
        LOG.warning("t-SNE unavailable (%s); falling back to PCA projection", exc)
        x = feats - feats.mean(axis=0, keepdims=True)
        # SVD-based PCA, deterministic.
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return (u[:, :2] * s[:2])


def _load_psas_embeddings(cfg) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load cached PSAS embeddings for t-SNE.

    Looks for ``paths.CLIP_FEATURES/psas_tsne.npz`` (or ``LLAMA_FEATURES``) with
    arrays ``emb`` [N,D] and integer ``labels`` [N] plus optional ``classes``.
    Returns (emb, labels, class_names) or None if absent.
    """
    candidates = [
        paths.CLIP_FEATURES / "psas_tsne.npz",
        paths.LLAMA_FEATURES / "psas_tsne.npz",
        paths.CACHE / "psas_tsne.npz",
    ]
    for c in candidates:
        if c.exists():
            try:
                data = np.load(c, allow_pickle=True)
                emb = np.asarray(data["emb"])
                labels = np.asarray(data["labels"]).astype(int)
                classes = list(data["classes"]) if "classes" in data else \
                    [f"class{i}" for i in sorted(set(labels.tolist()))]
                LOG.info("loaded PSAS embeddings from %s shape=%s", c, emb.shape)
                return emb, labels, [str(x) for x in classes]
            except Exception as exc:  # noqa: BLE001
                LOG.warning("failed to load embeddings %s: %s", c, exc)
    return None


def _demo_embeddings(cfg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthesize three clustered Gaussians standing in for object/attr/rel PSAS."""
    rng = np.random.default_rng(getattr(cfg, "seed", 42))
    dim = int(getattr(getattr(cfg, "projection", object()), "psas_dim", 256) or 256)
    classes = ["object", "attribute", "relation"]
    n_per = 120
    centers = rng.normal(0, 4.0, size=(len(classes), dim))
    embs, labs = [], []
    for ci in range(len(classes)):
        pts = rng.normal(0, 1.0, size=(n_per, dim)) + centers[ci]
        embs.append(pts)
        labs.append(np.full(n_per, ci, dtype=int))
    return np.vstack(embs), np.concatenate(labs), classes


def fig_psas_tsne(records, cfg, figures_dir: Path, demo: bool) -> tuple[Path, Path]:
    """2-D t-SNE scatter of PSAS embeddings colored by semantic category."""
    apply_style()
    loaded = None if demo else _load_psas_embeddings(cfg)
    if loaded is None:
        if not demo:
            LOG.warning("no cached PSAS embeddings; synthesizing demo clusters for t-SNE")
        emb, labels, classes = _demo_embeddings(cfg)
        demo = True  # data is synthetic -> always watermark this panel
    else:
        emb, labels, classes = loaded

    xy = _embed_2d(emb, seed=int(getattr(cfg, "seed", 42)))
    fig, ax = plt.subplots(figsize=_figsize(82))
    for ci, cname in enumerate(classes):
        mask = labels == ci
        if not np.any(mask):
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=0.7,
                   color=_color(ci), label=cname, edgecolors="none")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", markerscale=1.5)
    _watermark(ax, demo)
    fig.tight_layout()
    return _save(fig, "fig_psas_tsne", figures_dir)


# --------------------------------------------------------------------------- #
# fig_main_compare: main-metric comparison bars
# --------------------------------------------------------------------------- #
MAIN_COMPARE_SPEC = [
    ("POPE-F1", "pope", "pope_f1", True),          # higher better
    ("HallusionBench-fAcc", "hallusionbench", "hallusion_facc", True),
    ("MMHal-Score", "mmhal", "mmhal_score", True),
    ("CHAIR-i (lower)", "chair", "chair_i", False),  # lower better
    ("AMBER-Hal (lower)", "amber", "amber_hal", False),
]


def fig_main_compare(records: list[dict], cfg, figures_dir: Path, demo: bool) -> tuple[Path, Path]:
    """Grouped bars comparing methods across the headline metrics.

    Metrics with "lower is better" are annotated in their label; we plot raw
    values (no normalization) on a single axis for compactness.
    """
    apply_style()
    idx = index_records(records)
    model = cfg.mllm.key
    methods = discover_methods(records) or ["vanilla", "cmpsa"]
    # Keep main methods compact: vanilla + cmpsa (+ up to 2 baselines).
    main_methods = [m for m in methods if m in ("vanilla", "cmpsa")]
    extras = [m for m in methods if m not in ("vanilla", "cmpsa") and "cmpsa-" not in m]
    main_methods = main_methods + extras[:2]
    if not main_methods:
        main_methods = ["vanilla", "cmpsa"]

    metric_labels = [s[0] for s in MAIN_COMPARE_SPEC]
    x = np.arange(len(metric_labels))
    n = len(main_methods)
    total_w = 0.8
    w = total_w / max(n, 1)

    fig, ax = plt.subplots(figsize=_figsize(78))
    for mi, method in enumerate(main_methods):
        vals = []
        for _, bench, canonical, _ in MAIN_COMPARE_SPEC:
            v = _bench_metric(idx, bench, model, method, canonical)
            vals.append(np.nan if v is None else _display_value(v, canonical))
        offset = (mi - (n - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=method, color=_color(mi),
               edgecolor="black", linewidth=BAR_EDGE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_ylabel("Value (% unless noted)")
    ax.legend(ncol=min(n, 4), loc="upper right")
    ax.margins(y=0.18)
    _watermark(ax, demo)
    fig.tight_layout()
    return _save(fig, "fig_main_compare", figures_dir)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
FIGURE_SPECS: dict[str, Callable] = {
    "12": fig12_dataset_stats,
    "13": fig13_ablation,
    "14": fig14_robustness,
    "tsne": fig_psas_tsne,
    "main": fig_main_compare,
}
# Friendly aliases.
FIGURE_ALIASES = {
    "dataset": "12", "stats": "12",
    "ablation": "13",
    "robustness": "14", "robust": "14",
    "psas": "tsne", "psas_tsne": "tsne",
    "compare": "main", "main_compare": "main",
}


def make_figures(which: list[str] | None = None, demo: bool = False,
                 config: str | None = None, figures_dir: Path | None = None,
                 metrics_dir: Path | None = None) -> dict[str, tuple[Path, Path]]:
    """Render the requested figures (PNG + PDF). Returns {id: (png, pdf)}."""
    cfg = load_config(config)
    set_seed(getattr(cfg, "seed", 42))
    figures_dir = Path(figures_dir) if figures_dir is not None else paths.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    if demo:
        records = demo_records(cfg)
        LOG.info("DEMO mode: synthesized %d placeholder metrics records", len(records))
    else:
        records = load_all_metrics(metrics_dir)

    if which:
        norm = []
        for w in which:
            w = str(w).lower()
            norm.append(FIGURE_ALIASES.get(w, w))
        which = norm
    else:
        which = list(FIGURE_SPECS.keys())

    outputs: dict[str, tuple[Path, Path]] = {}
    for fid in which:
        if fid not in FIGURE_SPECS:
            LOG.warning("unknown figure id %s (valid: %s)", fid, ", ".join(FIGURE_SPECS))
            continue
        builder = FIGURE_SPECS[fid]
        try:
            outputs[fid] = builder(records, cfg, figures_dir, demo)
        except Exception as exc:  # noqa: BLE001 -- never let one figure kill the batch
            LOG.error("figure %s failed: %s", fid, exc, exc_info=True)
    LOG.info("done: %d figure(s) -> %s", len(outputs), figures_dir)
    return outputs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cmpsa.viz.make_figures",
        description="Render the CMPSA paper figures (PNG + PDF) to paths.FIGURES_DIR.",
    )
    p.add_argument("--config", default=None, help="optional config YAML override")
    p.add_argument("--demo", action="store_true",
                   help="synthesize placeholder data and watermark the figures")
    p.add_argument("--figures", nargs="*", default=None, metavar="ID",
                   help="subset of figure ids: 12 13 14 tsne main (or aliases); default: all")
    p.add_argument("--figures-dir", default=None,
                   help="override output dir (default: paths.FIGURES_DIR)")
    p.add_argument("--metrics-dir", default=None,
                   help="override input metrics dir (default: paths.METRICS_DIR)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.verbose:
        get_logger("cmpsa.viz.figures").setLevel(logging.DEBUG)
    outs = make_figures(
        which=args.figures,
        demo=args.demo,
        config=args.config,
        figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        metrics_dir=Path(args.metrics_dir) if args.metrics_dir else None,
    )
    for fid, (png, pdf) in outs.items():
        print(f"fig[{fid}]: {png}  |  {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
