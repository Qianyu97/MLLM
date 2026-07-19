"""Run several CMPSA benchmarks for one model across several methods.

Each ``eval_<bench>.py`` exposes ``run(model, method, limit, cfg) -> metrics`` and
this driver dispatches to them by benchmark name. Because methods are cached
in-process by :func:`cmpsa.eval.common.build_method_for`, the (heavy) model weights
are loaded at most once per (model, method) and reused across benchmarks.

Examples::

    # smoke test: 4 items per benchmark, both methods, default benchmark list
    python -m cmpsa.eval.run_all --model llava-1.5-7b --methods vanilla,cmpsa --limit 4

    # just POPE + AMBER, vanilla only
    python -m cmpsa.eval.run_all --benchmarks pope,amber --methods vanilla
"""
from __future__ import annotations

import argparse
import traceback
from typing import Any, Callable, Dict, List, Optional

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import get_logger, set_seed

# Import the per-benchmark run() entry points lazily-safe (these modules are
# import-clean without a GPU; torch only loads inside build_method_for at call time).
from cmpsa.eval import (
    eval_amber,
    eval_chair,
    eval_hallusionbench,
    eval_mme,
    eval_mmhal,
    eval_pope,
    eval_vg_rel,
)

_LOG = get_logger("cmpsa.eval.run_all")

# benchmark name -> callable(model, method, limit, cfg) -> metrics dict
BENCH_DISPATCH: Dict[str, Callable[..., Dict[str, Any]]] = {
    "pope": eval_pope.run,
    "chair": eval_chair.run,
    "amber": eval_amber.run,           # task defaults to "all"
    "hallusionbench": eval_hallusionbench.run,
    "mmhal": eval_mmhal.run,
    "mme": eval_mme.run,
    "vg_rel": eval_vg_rel.run,
}


def _resolve_benchmarks(arg: Optional[str], cfg: Any) -> List[str]:
    if arg:
        names = [b.strip() for b in arg.split(",") if b.strip()]
    else:
        names = list(getattr(cfg.eval, "benchmarks", list(BENCH_DISPATCH.keys())))
    unknown = [b for b in names if b not in BENCH_DISPATCH]
    if unknown:
        raise SystemExit(
            f"Unknown benchmark(s): {unknown}. Known: {sorted(BENCH_DISPATCH)}"
        )
    return names


def _resolve_methods(arg: Optional[str], cfg: Any) -> List[str]:
    if arg:
        return [m.strip() for m in arg.split(",") if m.strip()]
    return list(getattr(cfg.eval, "methods", ["vanilla"]))


def run_all(
    model: str,
    methods: List[str],
    benchmarks: List[str],
    limit: Optional[int],
    cfg: Any,
) -> Dict[str, Dict[str, Any]]:
    """Run every (method, benchmark) pair; collect a summary.

    Failures on one benchmark are caught and logged so the rest still run; the
    summary records the error string for that cell.
    """
    paths.ensure_dirs()
    set_seed(getattr(cfg, "seed", 42))

    summary: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        for bench in benchmarks:
            key = f"{bench}/{method}"
            _LOG.info("=== running %s : model=%s ===", key, model)
            try:
                metrics = BENCH_DISPATCH[bench](model, method, limit, cfg)
                summary[key] = {"status": "ok", "metrics": metrics.get("metrics", {})}
            except NotImplementedError as exc:
                _LOG.error("%s -> NotImplemented: %s", key, exc)
                summary[key] = {"status": "not_implemented", "error": str(exc)}
            except RuntimeError as exc:
                # Clear, expected failure (missing weights / data) -> do not crash all.
                _LOG.error("%s -> error: %s", key, exc)
                summary[key] = {"status": "error", "error": str(exc)}
            except Exception as exc:  # pragma: no cover - unexpected
                _LOG.error("%s -> unexpected error:\n%s", key, traceback.format_exc())
                summary[key] = {"status": "error", "error": str(exc)}
    _print_summary(model, summary)
    return summary


def _print_summary(model: str, summary: Dict[str, Dict[str, Any]]) -> None:
    _LOG.info("------------- run_all summary (model=%s) -------------", model)
    for key, info in summary.items():
        status = info["status"]
        if status == "ok":
            metrics = info["metrics"]
            head = {
                k: metrics[k]
                for k in ("accuracy", "f1", "chair_i", "amber_cover", "mme_total", "aAcc")
                if k in metrics
            }
            _LOG.info("  %-28s OK   %s", key, head or "(see metrics json)")
        else:
            _LOG.info("  %-28s %s  %s", key, status.upper(), info.get("error", "")[:120])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CMPSA benchmarks for one model")
    cfg = load_config()
    parser.add_argument("--model", default=cfg.mllm.key, help="MLLM key (configs/models.yaml)")
    parser.add_argument(
        "--methods", default=None,
        help="comma list, e.g. 'vanilla,cmpsa' (default: cfg.eval.methods)",
    )
    parser.add_argument(
        "--benchmarks", default=None,
        help="comma list (default: cfg.eval.benchmarks). "
             f"Known: {','.join(BENCH_DISPATCH)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="limit #items per bench (smoke)")
    parser.add_argument("--config", default=None, help="override yaml on top of default")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)

    methods = _resolve_methods(args.methods, cfg)
    benchmarks = _resolve_benchmarks(args.benchmarks, cfg)
    _LOG.info("model=%s methods=%s benchmarks=%s limit=%s",
              args.model, methods, benchmarks, args.limit)
    run_all(args.model, methods, benchmarks, args.limit, cfg)


if __name__ == "__main__":
    main()
