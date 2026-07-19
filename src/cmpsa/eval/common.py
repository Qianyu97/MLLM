"""Shared helpers for every CMPSA evaluation script.

Responsibilities
----------------
* :func:`load_image`            -> robustly open an image as an RGB ``PIL.Image``.
* :func:`build_method_for`      -> lazily load an MLLM + wrap it in a Method,
                                   caching the result *in-process* so repeated
                                   eval calls (e.g. from ``run_all``) reuse weights.
* :func:`save_predictions`      -> write the standard prediction jsonl
                                   (``paths.pred_path`` + ``write_jsonl``).
* :func:`compute_and_save_metrics`
                                -> compute the common yes/no metrics
                                   (Accuracy / Precision / Recall / F1 / Yes-Ratio),
                                   aggregate ``by_type``, merge any extra metrics,
                                   and write the standard metrics json
                                   (``paths.metrics_path``).

GPU / model imports are *lazy*: nothing here imports torch / transformers at module
top level, so ``python -m cmpsa.eval.<m> --help`` works on a CPU-only box and the
data-loading parts of every eval remain importable without weights present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cmpsa import paths
from cmpsa.utils import get_logger, write_jsonl

_LOG = get_logger("cmpsa.eval.common")

# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_image(path_or_pil: Any) -> "Any":
    """Return an RGB :class:`PIL.Image.Image`.

    Accepts an already-opened PIL image, a path (str / :class:`pathlib.Path`),
    or raw ``bytes`` (e.g. parquet image structs). Always converts to ``RGB``.
    """
    from PIL import Image  # local import: PIL is light but keep top-level clean

    # Already a PIL image?
    if hasattr(path_or_pil, "convert") and hasattr(path_or_pil, "size"):
        return path_or_pil.convert("RGB")

    # Raw bytes -> decode in-memory.
    if isinstance(path_or_pil, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(path_or_pil)).convert("RGB")

    p = Path(path_or_pil)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    return Image.open(p).convert("RGB")


# --------------------------------------------------------------------------- #
# Method construction (in-process cache)
# --------------------------------------------------------------------------- #
# Cache keyed by (model_key, method_key) so run_all can evaluate many benchmarks
# with one model load. We deliberately do NOT cache across config objects because
# thresholds (cfg.hhd / cfg.pgd) can change the method behaviour.
_METHOD_CACHE: Dict[Tuple[str, str], Any] = {}
_MLLM_CACHE: Dict[str, Any] = {}


def build_method_for(model_key: str, method_key: str, cfg: Any) -> Any:
    """Load the MLLM ``model_key`` and wrap it with the requested ``method_key``.

    Heavy imports (torch / transformers via ``cmpsa.models.*``) happen *here*,
    lazily, so importing this module never requires a GPU. Loaded methods are
    cached in-process; pass a fresh interpreter to force a reload.

    Raises a clear ``RuntimeError`` (not a crash) when the model backend cannot
    be loaded, e.g. weights absent or torch/transformers missing.
    """
    cache_key = (model_key, method_key)
    if cache_key in _METHOD_CACHE:
        return _METHOD_CACHE[cache_key]

    # Lazy import of the model stack so this file stays import-clean without GPU.
    try:
        from cmpsa.models.mllm_wrapper import load_mllm
        from cmpsa.models.methods import build_method
    except Exception as exc:  # pragma: no cover - depends on sibling agents / torch
        raise RuntimeError(
            "Could not import the CMPSA model stack "
            "(cmpsa.models.mllm_wrapper / cmpsa.models.methods). "
            "Make sure torch + transformers are installed and the models package "
            f"is present. Underlying error: {exc}"
        ) from exc

    _LOG.info("loading MLLM '%s' and building method '%s' ...", model_key, method_key)
    if model_key in _MLLM_CACHE:
        mllm = _MLLM_CACHE[model_key]
        _LOG.info("reusing cached MLLM '%s' for method '%s'", model_key, method_key)
    else:
        try:
            mllm = load_mllm(model_key, cfg)
        except NotImplementedError:
            # Re-raise NotImplementedError untouched: it names the unsupported key.
            raise
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Model weights for '{model_key}' were not found. Download them under "
                f"MODELS_ROOT ({paths.MODELS_ROOT}) first. Original error: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Failed to load MLLM '{model_key}'. This usually means the weights are "
                f"missing or the GPU/transformers environment is not set up. "
                f"Original error: {exc}"
            ) from exc
        _MLLM_CACHE[model_key] = mllm

    method = build_method(method_key, mllm, cfg)
    _METHOD_CACHE[cache_key] = method
    _LOG.info("method '%s' ready for model '%s'.", method_key, model_key)
    return method


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
_ROW_KEYS = ("id", "image", "question", "gt", "pred", "label", "type", "subset")


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a raw row into the standard prediction-jsonl schema (all keys present)."""
    out: Dict[str, Any] = {
        "id": str(row.get("id", "")),
        "image": str(row.get("image", "")) if row.get("image") is not None else "",
        "question": row.get("question"),
        "gt": row.get("gt"),
        "pred": "" if row.get("pred") is None else str(row.get("pred")),
        "label": row.get("label"),
        "type": row.get("type", "overall"),
        "subset": row.get("subset"),
    }
    return out


def save_predictions(rows: List[Dict[str, Any]], bench: str, model: str, method: str) -> Path:
    """Write ``rows`` to ``paths.pred_path(bench, model, method)`` as standard jsonl.

    Returns the path written. Rows are normalized to the standard schema so that
    every key (id/image/question/gt/pred/label/type/subset) is present.
    """
    out_path = paths.pred_path(bench, model, method)
    norm = [_normalize_row(r) for r in rows]
    n = write_jsonl(norm, out_path)
    _LOG.info("wrote %d predictions -> %s", n, out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _norm_yesno(value: Any) -> Optional[str]:
    """Map a free-form answer to 'yes'/'no' (or None if undecidable)."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in ("yes", "y", "true", "1"):
        return "yes"
    if s in ("no", "n", "false", "0"):
        return "no"
    # Look at the leading token / contained word for generative answers.
    head = s.split()[0].strip(".,!:;\"'")
    if head in ("yes", "yeah", "yep", "sure", "correct", "true"):
        return "yes"
    if head in ("no", "nope", "nah", "false", "incorrect"):
        return "no"
    if "yes" in s and "no" not in s:
        return "yes"
    if "no" in s and "yes" not in s:
        return "no"
    return None


def _binary_metrics(rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Compute yes/no Accuracy/Precision/Recall/F1/Yes-Ratio over rows with gt.

    Positive class is "yes". Returns ``None`` if there are no usable yes/no pairs
    (e.g. a pure-captioning benchmark), so callers can rely on extra metrics.
    """
    tp = fp = tn = fn = 0
    n_pred_yes = 0
    n_total = 0
    for r in rows:
        gt = _norm_yesno(r.get("gt"))
        pred = _norm_yesno(r.get("pred"))
        if gt is None:
            continue
        n_total += 1
        if pred == "yes":
            n_pred_yes += 1
        # Treat an unparseable prediction as the wrong answer (counts against acc).
        pred_label = pred if pred is not None else ("no" if gt == "yes" else "yes")
        if gt == "yes" and pred_label == "yes":
            tp += 1
        elif gt == "no" and pred_label == "yes":
            fp += 1
        elif gt == "no" and pred_label == "no":
            tn += 1
        elif gt == "yes" and pred_label == "no":
            fn += 1

    if n_total == 0:
        return None

    acc = (tp + tn) / n_total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    yes_ratio = n_pred_yes / n_total
    return {
        "accuracy": round(acc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "yes_ratio": round(yes_ratio, 6),
    }


def compute_and_save_metrics(
    bench: str,
    model: str,
    method: str,
    rows: List[Dict[str, Any]],
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the standard metrics for ``rows`` and write the standard metrics json.

    * Built-in yes/no metrics (Accuracy/Precision/Recall/F1/Yes-Ratio) are computed
      whenever the rows carry yes/no ground truth.
    * ``by_type`` aggregates the same yes/no metrics per ``row["type"]``
      (object/attribute/relation/overall).
    * ``extra_metrics`` is merged into the top-level ``metrics`` dict (benchmark
      specific scores like CHAIR-i, fAcc, AMBER Cover/Hal, MME score, etc.).

    Returns the metrics dict (also written to ``paths.metrics_path``).
    """
    from cmpsa.utils import save_json  # local import keeps module top-level clean

    metrics: Dict[str, float] = {}
    overall = _binary_metrics(rows)
    if overall is not None:
        metrics.update(overall)

    # by_type aggregation (only meaningful when there is yes/no gt per type).
    by_type: Dict[str, Dict[str, float]] = {}
    types = sorted({str(r.get("type", "overall")) for r in rows})
    if len(types) > 1 or (types and types[0] != "overall"):
        for t in types:
            sub = [r for r in rows if str(r.get("type", "overall")) == t]
            sub_metrics = _binary_metrics(sub)
            if sub_metrics is not None:
                by_type[t] = sub_metrics

    if extra_metrics:
        # Round floats for stable JSON; leave non-floats as-is.
        for k, v in extra_metrics.items():
            metrics[k] = round(v, 6) if isinstance(v, float) else v

    out = {
        "benchmark": bench,
        "model": model,
        "method": method,
        "n": len(rows),
        "metrics": metrics,
        "by_type": by_type,
    }
    out_path = paths.metrics_path(bench, model, method)
    save_json(out, out_path)
    _LOG.info("wrote metrics -> %s  (%s)", out_path, _fmt_metrics(metrics))
    return out


def _fmt_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "(no scalar metrics)"


# --------------------------------------------------------------------------- #
# Misc helpers reused by the eval scripts
# --------------------------------------------------------------------------- #
def apply_limit(items: List[Any], limit: Optional[int]) -> List[Any]:
    """Truncate ``items`` to ``limit`` (for smoke tests). ``None``/<=0 -> no limit."""
    if limit is not None and limit > 0:
        return list(items)[:limit]
    return list(items)


def add_common_eval_args(parser, cfg_default_model: str) -> None:
    """Register the shared ``--model/--method/--limit/--config`` CLI arguments."""
    parser.add_argument("--model", default=cfg_default_model, help="MLLM key (configs/models.yaml)")
    parser.add_argument("--method", default="vanilla", help="method key (vanilla | cmpsa | ...)")
    parser.add_argument("--limit", type=int, default=None, help="limit #items (smoke test)")
    parser.add_argument("--config", default=None, help="optional override yaml on top of default")
