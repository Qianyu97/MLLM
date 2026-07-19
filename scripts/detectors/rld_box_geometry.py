"""RLD prototype WITHOUT RelTR: use Grounding-DINO boxes + geometry.
- AMBER-rel ('direct contact between X and Y'): contact <-> boxes overlap/adjacent.
- VG-rel (directional 'X on/under/left/right Y'): relative box position matches predicate.
Report AUC vs whole-image CLIP baseline (0.59 / 0.51)."""
import sys, json, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_hhd import parse_amber_rel, parse_vg_rel, _amber_truth, _resolve_vg_image

cfg = load_config()
TOOLS = paths.MODELS_ROOT.parent / "tools"
DEV = "cuda"
gdproc = AutoProcessor.from_pretrained(str(TOOLS / "grounding_dino"))
gdino = GroundingDinoForObjectDetection.from_pretrained(str(TOOLS / "grounding_dino")).to(DEV).eval()


@torch.no_grad()
def box(image, phrase):
    inp = gdproc(images=image, text=phrase.lower().strip() + ".", return_tensors="pt").to(DEV)
    out = gdino(**inp)
    for kw in ("threshold", "box_threshold"):
        try:
            res = gdproc.post_process_grounded_object_detection(
                out, inp["input_ids"], **{kw: 0.15}, text_threshold=0.15,
                target_sizes=[image.size[::-1]])[0]
            break
        except TypeError:
            continue
    if len(res["scores"]) == 0:
        return None
    return [float(v) for v in res["boxes"][int(res["scores"].argmax())]]


def geom(b1, b2, W, H):
    """features between two boxes, normalized by image size."""
    ax0, ay0, ax1, ay1 = b1; bx0, by0, bx1, by1 = b2
    # intersection / IoU
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    a1 = (ax1 - ax0) * (ay1 - ay0); a2 = (bx1 - bx0) * (by1 - by0)
    iou = inter / (a1 + a2 - inter + 1e-6)
    # center-gap (normalized), edge gap
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    cgap = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / ((W ** 2 + H ** 2) ** 0.5)
    return dict(iou=iou, inter_norm=inter / (min(a1, a2) + 1e-6), cgap=cgap,
                dy=(bcy - acy) / H, dx=(bcx - acx) / W)


# ---- AMBER-rel: contact ----
truth = _amber_truth()
items = json.load(open(paths.AMBER_Q_RELATION, encoding="utf-8"))
random.Random(0).shuffle(items)
y, iou_s, ovl_s, prox_s = [], [], [], []
nb = 0
for it in items:
    gt = truth.get(int(it["id"])); p = parse_amber_rel(it["query"])
    img = paths.AMBER_IMAGES / it["image"]
    if gt not in ("yes", "no") or not p or not img.exists():
        continue
    s_, _, o_ = p
    image = Image.open(img).convert("RGB"); W, H = image.size
    bs, bo = box(image, s_), box(image, o_)
    if bs is None or bo is None:
        g = dict(iou=0, inter_norm=0, cgap=1.0)
    else:
        nb += 1; g = geom(bs, bo, W, H)
    iou_s.append(g["iou"]); ovl_s.append(g["inter_norm"]); prox_s.append(-g["cgap"])
    y.append(1 if gt == "yes" else 0)
    if len(y) >= 800:
        break
print(f"[AMBER-rel contact] n={len(y)} pos={sum(y)/len(y):.3f} both_boxes={nb}/{len(y)}  (CLIP baseline 0.59)")
for nm, s in [("IoU", iou_s), ("overlap/min-area", ovl_s), ("proximity(-center gap)", prox_s)]:
    print(f"   AUC box-{nm:22s} = {roc_auc_score(y, s):.4f}")

# ---- VG-rel: directional ----
_DIR = {"on": ("dy", +1), "above": ("dy", +1), "over": ("dy", +1), "on top of": ("dy", +1),
        "under": ("dy", -1), "below": ("dy", -1), "beneath": ("dy", -1),
        "left": ("dx", +1), "right": ("dx", -1)}
items = [json.loads(l) for l in open(paths.VG_REL_JSONL, encoding="utf-8")]
random.Random(0).shuffle(items)
y2, dir_s, prox2 = [], [], []
nb2 = 0
for it in items:
    p = parse_vg_rel(it["question"]); img = _resolve_vg_image(it["image"])
    gt = str(it.get("gt", "")).lower()
    if gt not in ("yes", "no") or not p or not img or not img.exists():
        continue
    s_, pred, o_ = p
    image = Image.open(img).convert("RGB"); W, H = image.size
    bs, bo = box(image, s_), box(image, o_)
    if bs is None or bo is None:
        dir_s.append(0.0); prox2.append(-1.0); y2.append(1 if gt == "yes" else 0); continue
    nb2 += 1; g = geom(bs, bo, W, H)
    key = next((k for k in _DIR if k in pred), None)
    if key:
        comp, sign = _DIR[key]
        dir_s.append(sign * g[comp])   # subj above obj => dy>0 for 'on'
    else:
        dir_s.append(g["iou"])          # non-directional -> proximity
    prox2.append(-g["cgap"])
    y2.append(1 if gt == "yes" else 0)
    if len(y2) >= 800:
        break
print(f"\n[VG-rel directional] n={len(y2)} pos={sum(y2)/len(y2):.3f} both_boxes={nb2}/{len(y2)}  (CLIP baseline 0.51)")
for nm, s in [("directional match", dir_s), ("proximity", prox2)]:
    print(f"   AUC box-{nm:22s} = {roc_auc_score(y2, s):.4f}")
