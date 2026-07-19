"""Image-degradation robustness evaluation for CMPSA.

This is a real signal-level robustness sweep, not a semantic adversarial set.
It perturbs POPE images with Gaussian noise and measures the object
hallucination false-positive rate on no-object questions, plus standard yes/no
metrics.  Results are written as standard metrics JSON records with
``benchmark == "robustness"`` so ``fig14_robustness`` can render real curves.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.common import _binary_metrics, _norm_yesno, build_method_for, load_image
from cmpsa.eval.eval_pope import _load_pope_items
from cmpsa.utils import get_logger, save_json, set_seed, write_jsonl

LOG = get_logger("cmpsa.eval.robustness")
BENCH = "robustness"


def _add_gaussian_noise(image, sigma: float, rng: np.random.Generator):
    if sigma <= 0:
        return image
    from PIL import Image

    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.clip(arr + rng.normal(0.0, float(sigma), size=arr.shape), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype("uint8"), mode="RGB")


def _false_positive_rate(rows: list[dict[str, Any]]) -> float:
    no_rows = [r for r in rows if _norm_yesno(r.get("gt")) == "no"]
    if not no_rows:
        return 0.0
    fp = sum(1 for r in no_rows if _norm_yesno(r.get("pred")) == "yes")
    return fp / len(no_rows)


def _run_one(
    model: str,
    method: str,
    cfg: Any,
    sigma: float,
    limit_per_subset: int,
    seed: int,
) -> dict[str, Any]:
    items = _load_pope_items(limit_per_subset)
    if not items:
        raise RuntimeError("no POPE items available for robustness eval")
    m = build_method_for(model, method, cfg)
    rng = np.random.default_rng(seed + int(round(sigma * 10000)))

    rows: list[dict[str, Any]] = []
    for it in items:
        image = load_image(it["image_path"])
        image = _add_gaussian_noise(image, sigma=sigma, rng=rng)
        pred, conf = m.answer_yes_no(image, it["question"])
        label = str(it["label"]).strip().lower()
        rows.append(
            {
                "id": f"gaussian{sigma:.3f}_{it['id']}",
                "image": it["image_name"],
                "question": it["question"],
                "gt": label if label in ("yes", "no") else None,
                "pred": pred,
                "confidence": conf,
                "label": 1 if label == "yes" else (0 if label == "no" else None),
                "type": "object",
                "subset": it["subset"],
                "degradation": "gaussian",
                "noise": float(sigma),
            }
        )

    metrics = _binary_metrics(rows) or {}
    hall_rate = _false_positive_rate(rows)
    metrics.update(
        {
            "noise": round(float(sigma), 6),
            "hall_rate": round(float(hall_rate), 6),
            "degradation": "gaussian",
            "limit_per_subset": int(limit_per_subset),
        }
    )

    stem = f"{model}__{method}__gaussian_s{sigma:.3f}".replace(".", "p")
    pred_path = paths.PRED_DIR / BENCH / f"{stem}.jsonl"
    metrics_path = paths.METRICS_DIR / BENCH / f"{stem}.json"
    write_jsonl(rows, pred_path)
    out = {
        "benchmark": BENCH,
        "model": model,
        "method": method,
        "n": len(rows),
        "metrics": metrics,
        "by_type": {"object": metrics},
    }
    save_json(out, metrics_path)
    LOG.info("robustness %s sigma=%.3f n=%d acc=%.4f fpr=%.4f -> %s",
             method, sigma, len(rows), float(metrics.get("accuracy", 0.0)),
             hall_rate, metrics_path)
    return out


def run_sweep(
    model: str,
    methods: list[str],
    cfg: Any,
    sigmas: list[float],
    limit_per_subset: int,
) -> list[dict[str, Any]]:
    set_seed(int(getattr(cfg, "seed", 42)))
    paths.ensure_dirs()
    outputs: list[dict[str, Any]] = []
    for method in methods:
        for sigma in sigmas:
            outputs.append(
                _run_one(
                    model=model,
                    method=method,
                    cfg=cfg,
                    sigma=float(sigma),
                    limit_per_subset=limit_per_subset,
                    seed=int(getattr(cfg, "seed", 42)),
                )
            )
    return outputs


def build_argparser() -> argparse.ArgumentParser:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Run Gaussian-noise robustness sweep on POPE.")
    p.add_argument("--config", default=None, help="optional config override")
    p.add_argument("--model", default=cfg.mllm.key)
    p.add_argument("--methods", default="vanilla,cmpsa")
    p.add_argument("--sigmas", default="0,0.03,0.06,0.10,0.15")
    p.add_argument("--limit-per-subset", type=int, default=300)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    run_sweep(
        model=args.model,
        methods=methods,
        cfg=cfg,
        sigmas=sigmas,
        limit_per_subset=args.limit_per_subset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
