"""Detection eval on the self-built HalluProbe-VL (object COCO2017-val + attr/rel VG).
Reuses the specialist Grounder. Reports per-type AUC/F1 for the unified diagnostic."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
from pathlib import Path
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_hhd import parse_amber_attr, detection_metrics  # reuse
from cmpsa.eval.eval_hhd_specialist import Grounder
from cmpsa.eval.eval_hhd import parse_vg_rel

cfg = load_config()
g = Grounder(cfg)
OUT = paths.HALLUPROBE

# ---- object (COCO2017-val), CLIP OLD grounding ----
obj_rows = [json.loads(l) for l in open(OUT/"probes"/"object.jsonl", encoding="utf-8")]
import re
recs = []
for it in obj_rows:
    m = re.search(r"[Ii]s there a (.+?) in the image", it["text"])
    if not m: continue
    obj = m.group(1)
    p = Path(it["image"])
    if not p.exists(): continue
    recs.append((1 if it["label"]=="yes" else 0, g.old_object(p, obj)))
mo = detection_metrics([r[0] for r in recs], [r[1] for r in recs])
print(f"HalluProbe OBJECT (OLD, CLIP): AUC={mo.get('auc'):.3f} F1={mo.get('best_f1')} n={mo.get('n')}")

# ---- attribute (VG-Attr), ALD grounding ----
def resolve_vg(image_field):
    from cmpsa.eval.eval_hhd import _resolve_vg_image
    return _resolve_vg_image(image_field)
attr_rows = [json.loads(l) for l in open(paths.VG_ATTR_JSONL, encoding="utf-8")]
ar = []
for it in attr_rows:
    q = it.get("question",""); gt = str(it.get("gt","")).lower()
    m = re.search(r"[Ii]s the (.+?) (.+?)\?", q)
    img = resolve_vg(it["image"])
    if not m or gt not in ("yes","no") or img is None or not img.exists(): continue
    noun, attr = m.group(1).strip(), m.group(2).strip()
    ar.append((1 if gt=="yes" else 0, g.ald_attribute(img, noun, attr)))
if ar:
    ma = detection_metrics([r[0] for r in ar], [r[1] for r in ar])
    print(f"HalluProbe ATTRIBUTE (ALD, VG): AUC={ma.get('auc'):.3f} F1={ma.get('best_f1')} n={ma.get('n')}")

# ---- relation (VG-Rel), RLD direction grounding ----
rel_rows = [json.loads(l) for l in open(paths.VG_REL_JSONL, encoding="utf-8")]
rr = []
for it in rel_rows:
    p = parse_vg_rel(it["question"]); gt = str(it.get("gt","")).lower(); img = resolve_vg(it["image"])
    if not p or gt not in ("yes","no") or img is None or not img.exists(): continue
    s_, pred, o_ = p
    rr.append((1 if gt=="yes" else 0, g.rld_direction(img, s_, pred, o_)))
if rr:
    mr = detection_metrics([r[0] for r in rr], [r[1] for r in rr])
    print(f"HalluProbe RELATION (RLD, VG): AUC={mr.get('auc'):.3f} F1={mr.get('best_f1')} n={mr.get('n')}")

out = {"object": mo, "attribute": ma if ar else None, "relation": mr if rr else None}
(paths.METRICS_DIR/"halluprobe").mkdir(parents=True, exist_ok=True)
json.dump(out, open(paths.METRICS_DIR/"halluprobe"/"detection.json","w"), indent=2)
print("wrote", paths.METRICS_DIR/"halluprobe"/"detection.json")
