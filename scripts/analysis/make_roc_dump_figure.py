"""Fig.10: OLD/ALD/RLD detection ROC curves from the full-scale specialist eval dumps."""
import sys, json, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from cmpsa import paths

PRED = paths.PRED_DIR / "hhd_detection_specialist"
BEDS = [("pope", "OLD (object) — POPE", "#2563eb"),
        ("amber_attr", "ALD (attribute) — AMBER", "#16a34a"),
        ("amber_rel", "RLD (relation-contact) — AMBER", "#ea580c"),
        ("vg_rel", "RLD (relation-direction) — VG-Rel", "#9333ea")]

fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="chance (0.50)")
for bed, name, color in BEDS:
    fp = PRED / f"{bed}__full.jsonl"
    if not fp.exists():
        print("missing", fp); continue
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    y = np.array([r["label"] for r in rows]); s = np.array([r["score"] for r in rows], dtype=float)
    if len(set(y.tolist())) < 2:
        continue
    fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc:.3f}, n={len(y)})")
    print(f"{bed}: AUC={auc:.3f} n={len(y)}")

ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("HHD hierarchical detection (OLD / ALD / RLD)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
ax.grid(alpha=0.25)
fig.tight_layout()
out = paths.FIGURES_DIR; out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig10_hhd_roc.{ext}", bbox_inches="tight")
print("wrote", out / "fig10_hhd_roc.pdf")
