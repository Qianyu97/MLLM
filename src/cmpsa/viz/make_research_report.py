"""Build a compact CMPSA paper-research package from current artifacts.

The report is deliberately conservative: it distinguishes real measured metrics
from pilot/smoke metrics and records sample sizes beside every benchmark. This
prevents tiny debugging runs from being mistaken for final paper numbers while
still giving the manuscript, tables, figures, and experiment status one stable
place to live.

Outputs
-------
    results/paper/CMPSA_experiment_report.md
    results/paper/CMPSA_manuscript_draft.md
    results/paper/experiment_status.json

This module is CPU-only and does not import torch or transformers.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed
from cmpsa.viz.make_tables import (
    _display_value,
    _resolve_metric,
    load_all_metrics,
)

LOG = get_logger("cmpsa.viz.research_report")


PAPER_DIR_NAME = "paper"

KEY_METRICS = {
    "pope": [("F1 (%)", "pope_f1"), ("Acc (%)", "pope_acc"), ("Yes (%)", "yes_ratio")],
    "chair": [("CHAIR-i (%)", "chair_i"), ("CHAIR-s (%)", "chair_s")],
    "amber": [("Acc (%)", "pope_acc"), ("Hal (%)", "amber_hal"), ("Cover (%)", "amber_cover")],
    "hallusionbench": [("aAcc (%)", "pope_acc"), ("qAcc (%)", "hallusion_qacc"), ("fAcc (%)", "hallusion_facc")],
    "mmhal": [("Score", "mmhal_score"), ("Needs judge", "needs_external_judge")],
    "mme": [("MME total", "mme_total"), ("Acc (%)", "pope_acc")],
    "vg_rel": [("Acc (%)", "pope_acc"), ("F1 (%)", "pope_f1")],
    "robustness": [("Noise sigma", "noise"), ("Acc (%)", "pope_acc"), ("Hall. rate (%)", "mmhal_halrate")],
}

PILOT_THRESHOLDS = {
    "pope": 9000,
    "chair": 500,
    "amber": 1000,
    "hallusionbench": 900,
    "mmhal": 96,
    "mme": 2000,
    "vg_rel": 1500,
    "robustness": 900,
}


def _paper_dir(out_dir: Path | None = None) -> Path:
    root = Path(out_dir) if out_dir is not None else paths.RESULTS_ROOT / PAPER_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fmt(value: float | None, canonical: str) -> str:
    value = _display_value(value, canonical)
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _record_row(record: dict[str, Any]) -> dict[str, Any]:
    bench = str(record.get("benchmark", "")).lower()
    metrics = record.get("metrics", {}) or {}
    row = {
        "benchmark": bench,
        "model": record.get("model", "unknown"),
        "method": record.get("method", "unknown"),
        "n": int(record.get("n", 0) or 0),
        "status": "pilot",
        "metrics": {},
        "path": record.get("_path", ""),
    }
    if bench == "robustness" and row["n"] >= PILOT_THRESHOLDS["robustness"]:
        row["status"] = "paper-sweep"
    elif row["n"] >= PILOT_THRESHOLDS.get(bench, 10**9):
        row["status"] = "paper-scale"
    for label, canonical in KEY_METRICS.get(bench, []):
        row["metrics"][label] = _fmt(_resolve_metric(metrics, canonical), canonical)
    return row


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No records yet._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def _artifact_inventory() -> dict[str, Any]:
    def files_under(root: Path, patterns: tuple[str, ...]) -> list[str]:
        if not root.exists():
            return []
        out: list[str] = []
        for pat in patterns:
            out.extend(str(p.relative_to(paths.RESULTS_ROOT)) for p in sorted(root.glob(pat)))
        return out

    checkpoints = []
    for p in sorted(paths.CKPT_DIR.glob("*.pt")):
        st = p.stat()
        checkpoints.append({
            "name": p.name,
            "bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })

    cache_counts: dict[str, int] = {}
    for root in (paths.CLIP_FEATURES, paths.LLAMA_FEATURES):
        if not root.exists():
            continue
        for split in sorted([p for p in root.iterdir() if p.is_dir()]):
            cache_counts[str(split.relative_to(paths.CACHE))] = len(list(split.glob("*.pt")))

    latest_logs = []
    if paths.LOG_DIR.exists():
        logs = sorted(paths.LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in logs[:10]:
            latest_logs.append({
                "name": p.name,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                "bytes": p.stat().st_size,
            })

    return {
        "checkpoints": checkpoints,
        "cache_counts": cache_counts,
        "tables": files_under(paths.TABLES_DIR, ("*.csv", "*.tex")),
        "figures": files_under(paths.FIGURES_DIR, ("*.png", "*.pdf")),
        "latest_logs": latest_logs,
    }


def _records_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_record_row(r) for r in records]
    rows.sort(key=lambda r: (r["benchmark"], r["model"], r["method"]))
    return rows


def build_status(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(paths.DATA_ROOT),
        "models_root": str(paths.MODELS_ROOT),
        "results_root": str(paths.RESULTS_ROOT),
        "gpu_policy": {
            "physical_gpu": 2,
            "cuda_visible_devices": "2",
            "note": "All project runtime scripts source _runtime_env.sh and pin physical GPU 2.",
        },
        "metric_records": _records_summary(records),
        "artifacts": _artifact_inventory(),
        "limitations": [
            "The current cmpsa inference method uses live top-k PGD logit reweighting and first-token Yes/No reweighting; it is practical for one GPU but not a full-vocabulary exhaustive HHD pass.",
            "MMHal score requires an external GPT-4-style judge; current score fields are placeholders when needs_external_judge=1.",
            "The robustness curve is a Gaussian-noise signal-degradation sweep on POPE; it should not be described as semantic adversarial robustness.",
            "Metrics with n far below the benchmark size are pilot/debug measurements, not final paper-scale results.",
        ],
    }


def build_experiment_report(status: dict[str, Any]) -> str:
    rows = []
    for r in status["metric_records"]:
        metrics = "; ".join(f"{k}={v}" for k, v in r["metrics"].items()) or "-"
        rows.append([r["benchmark"], r["model"], r["method"], r["n"], r["status"], metrics])

    ckpt_rows = [
        [c["name"], c["bytes"], c["modified"]]
        for c in status["artifacts"]["checkpoints"]
    ]
    log_rows = [
        [x["name"], x["modified"], x["bytes"]]
        for x in status["artifacts"]["latest_logs"][:8]
    ]
    cache_rows = [
        [k, v] for k, v in sorted(status["artifacts"]["cache_counts"].items())
    ]

    return f"""# CMPSA 实验研究报告

生成时间：{status["generated_at"]}

## 当前结论

本项目已经建立了独立运行环境、GPU 2 固定运行策略、数据准备、三阶段训练、主基准评估、PGD 解码评估、鲁棒性 sweep、真实 PSAS t-SNE 嵌入、论文图表与报告生成流水线。`status=paper-scale` 的主基准可用于论文结果讨论；`status=paper-sweep` 是固定子集上的退化鲁棒性曲线；凡 `status=pilot` 的分数只能用于检查流程与排版。

## GPU 与环境策略

- 数据根目录：`{status["data_root"]}`
- 模型缓存：`{status["models_root"]}`
- 结果目录：`{status["results_root"]}`
- GPU 约束：只使用物理 GPU 2，即 `CUDA_VISIBLE_DEVICES=2`。

## 当前真实指标记录

{_markdown_table(["Benchmark", "Model", "Method", "n", "Status", "Key metrics"], rows)}

## Checkpoints

{_markdown_table(["File", "Bytes", "Modified"], ckpt_rows)}

## Feature Cache

{_markdown_table(["Cache split", "# .pt files"], cache_rows)}

## 最新日志

{_markdown_table(["Log", "Modified", "Bytes"], log_rows)}

## 论文图表产物

- Tables: `{paths.TABLES_DIR}`
- Figures: `{paths.FIGURES_DIR}`
- Paper package: `{paths.RESULTS_ROOT / PAPER_DIR_NAME}`

## 已完成与仍需处理

- 已完成 Stage A/B/C 真实训练，并完成 POPE、CHAIR、AMBER、HallusionBench、MME、VG-Rel、MMHal 的主评估。
- 已完成真实 PGD 解码接入、Gaussian-noise 鲁棒性 sweep，以及真实 `cache/psas_tsne.npz` 嵌入生成；`fig14_robustness` 与 `fig_psas_tsne` 不再是空图或合成 demo。
- 仍需外部 GPT-style judge 才能得到 MMHal 官方分数。
- 仍建议继续做 PGD 参数消融（`lambda_decay/decode_top_k/yesno_weight`）和多主干实验，以提升最终论文说服力。
"""


def build_manuscript_draft(status: dict[str, Any]) -> str:
    metric_rows = []
    for r in status["metric_records"]:
        metrics = "; ".join(f"{k}={v}" for k, v in r["metrics"].items()) or "-"
        metric_rows.append([r["benchmark"], r["method"], r["n"], r["status"], metrics])

    return f"""# Cross-Modal Probabilistic Semantic Alignment for Hierarchical Hallucination Governance

## Abstract

Multimodal large language models often produce object, attribute, and relation hallucinations despite fluent surface language. This paper studies hallucination as a deviation between visual-conditioned and language-conditioned semantic distributions. We propose CMPSA, a unified framework that maps visual and textual representations into a probabilistic semantic alignment space, aligns them with entropic optimal transport, detects hierarchical hallucination risks with a shared conditional-divergence principle, and mitigates risky decoding steps with probability-guided decoding. Experiments are organized over six public benchmarks plus HalluProbe-VL, with object, attribute, and relation results reported separately.

## Contributions

1. We formulate object, attribute, and relation hallucinations as cross-modal conditional distribution drift.
2. We instantiate probabilistic visual/language projection heads and CM-OTA, an optimal-transport alignment objective over diagonal Gaussian semantic embeddings.
3. We define HHD, a hierarchical detector with OLD, ALD, and RLD layers under a common conditional divergence view.
4. We provide a reproducible experiment harness, paper-table generation, and IEEE/PAMI-style figure rendering with 190 mm width and 600 dpi exports.

## Method Overview

CMPSA contains five modules: PVE, PLE, CM-OTA, HHD, and PGD. PVE and PLE project frozen visual and text backbone features into PSAS. CM-OTA minimizes a Sinkhorn-regularized 2-Wasserstein transport cost between the two distributions. HHD evaluates object existence, attribute consistency, and relation plausibility. PGD is designed as an inference-time controller that reweights next-token logits according to HHD drift signals.

## Experimental Protocol

The current implementation uses LLaVA-1.5-7B as the primary MLLM, CLIP-ViT-L/14-336 as the visual backbone, and a Llama-family text backbone for PLE features. Evaluation covers POPE, CHAIR, AMBER, HallusionBench, MMHal-Bench, MME, and VG-Rel. MM-SAP is excluded because it is both missing locally and not aligned with the target hallucination taxonomy.

## Current Empirical Status

{_markdown_table(["Benchmark", "Method", "n", "Status", "Key metrics"], metric_rows)}

The table above is generated from the current result directory at `{status["generated_at"]}`. Rows marked `pilot` are not final paper-scale numbers.

## Figures and Tables

The project generates tables as CSV/LaTeX under `{paths.TABLES_DIR}` and figures as PDF/PNG under `{paths.FIGURES_DIR}`. The plotting configuration centralizes paper width, DPI, font, line width, marker size, and color palette in `src/cmpsa/viz/make_figures.py`.

## Limitations and Next Steps

The current `cmpsa` evaluation method now replaces vanilla decoding with a practical top-k PGD logits processor and applies HHD-grounded first-token reweighting for Yes/No questions. It does not exhaustively score the full vocabulary at each step, so final claims should still rely on calibrated paper-scale evaluations. MMHal also requires an external judge for its official score.
"""


def make_report(config: str | None = None, output_dir: Path | None = None) -> dict[str, Path]:
    cfg = load_config(config)
    set_seed(getattr(cfg, "seed", 42))
    out_dir = _paper_dir(output_dir)
    records = load_all_metrics(paths.METRICS_DIR)
    status = build_status(records)

    status_path = out_dir / "experiment_status.json"
    report_path = out_dir / "CMPSA_experiment_report.md"
    draft_path = out_dir / "CMPSA_manuscript_draft.md"

    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_experiment_report(status), encoding="utf-8")
    draft_path.write_text(build_manuscript_draft(status), encoding="utf-8")

    LOG.info("wrote research report package -> %s", out_dir)
    return {"status": status_path, "report": report_path, "draft": draft_path}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build CMPSA paper research report files.")
    p.add_argument("--config", default=None, help="optional config YAML override")
    p.add_argument("--output-dir", default=None, help="default: results/paper")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    outs = make_report(
        config=args.config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    for key, path in outs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
