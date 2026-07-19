# -*- coding: utf-8 -*-
"""Demo: wrap ANY captioning model with hgfusion, in three access levels.

    python examples/wrap_demo.py --image path/to/img.jpg \
        --caption "A dog sits at a table. A cat sleeps on the floor."

L0 needs nothing from the backbone (the caption can even come from an API-only
model). L1 shows the rewrite hook. L2 shows calibration + fused answering.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo checkout
from hgfusion import HGFWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--caption", required=True,
                help="the backbone's caption for the image (from ANY model)")
ap.add_argument("--models-root", default=None,
                help="dir holding clip-vit-l14-336/ and grounding_dino/ "
                     "(default: $CMPSA_MODELS_ROOT)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--thr", type=float, default=0.30)
args = ap.parse_args()

w = HGFWrapper(models_root=args.models_root, device=args.device,
               gdino_threshold=args.thr)

# ---------------------------------------------------------------- L0: verify
print("=== L0 · verify_caption (image + text only) ===")
report = w.verify_caption(args.image, args.caption)
for obj, score in sorted(report.objects.items(), key=lambda kv: kv[1]):
    mark = "FLAG" if obj in report.flagged else "ok  "
    print(f"  [{mark}] {obj:<15} s_obj = {score:.3f}   (threshold {report.threshold})")

# ---------------------------------------------------------------- L0: revise
print("\n=== L0 · revise_caption(strategy='remove') — backbone-agnostic ===")
rev = w.revise_caption(args.image, args.caption, strategy="remove")
print("  before:", args.caption)
print("  after :", rev.text)

# ---------------------------------------------------------------- L1: rewrite hook
print("\n=== L1 · revise_caption(strategy='rewrite') — plug in YOUR backbone ===")
def my_backbone_rewrite(image, caption, flagged):
    """Replace this stub with one more call to your own model, e.g.:
       prompt = f"{caption}\\nThe following are NOT in the image: {', '.join(flagged)}. Rewrite."
       return my_model.generate(image, prompt)"""
    return f"<your model's rewrite of the caption without {', '.join(flagged)}>"
rev1 = w.revise_caption(args.image, args.caption, strategy="rewrite",
                        rewrite_fn=my_backbone_rewrite)
print("  rewrite_fn returned:", rev1.text)

# ---------------------------------------------------------------- L2: decision fusion
print("\n=== L2 · calibrate + answer (needs your backbone's p_yes) ===")
# In practice, collect REAL (p_yes, clip_g, label) triples on a held-out split of
# your data: p_yes from your backbone, clip_g from w.clip_score(image, obj).
# Here: a synthetic set only to demonstrate the API shape. Its value ranges mimic
# real ones (raw CLIP image-text cosines live around 0.15-0.30) so the demo answer
# is sensible — but do NOT ship synthetic calibration in a real system.
import random
random.seed(0)
records = []
for _ in range(200):
    y = random.random() < 0.5
    # an uncertain backbone (overlapping p) + an informative grounding signal:
    # the regime where fusion pays off (cf. the residual view, paper Sec. 3.6)
    p = min(1, max(0, random.gauss(0.60 if y else 0.42, 0.18)))
    g = random.gauss(0.27 if y else 0.17, 0.030)
    records.append((p, g, int(y)))
cal = w.calibrate(records)
print(f"  calibrated: lambda={cal.lam:.2f}  tau={cal.tau:.3f}")
ans = w.answer(args.image, "Is there a dog in the image?", p_yes=0.62)
print(f"  Q: Is there a dog?  p_yes=0.62  g={ans['g']:.3f}  ->  {ans['answer']}  (z={ans['z']:.2f})")
