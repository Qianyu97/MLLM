"""CLIP zero-shot per HHD layer: OLD(POPE done=0.82), ALD(AMBER-attr), RLD(AMBER-rel, VG-rel).
Uses CONTRASTIVE probes where natural (attr: attr vs its antonym-ish 'not'; rel: contact phrase).
Reports AUC of CLIP image-text similarity vs yes/no ground truth."""
import sys, json, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPProcessor
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_hhd import (parse_amber_attr, parse_amber_rel, parse_vg_rel,
                                 _load_jsonl, _amber_truth, _resolve_vg_image)

cfg = load_config()
cdir = str(paths.MODELS_ROOT / cfg.models.__dict__['clip-vit-l14-336'].local_dir)
model = CLIPModel.from_pretrained(cdir, torch_dtype=torch.float16).to("cuda").eval()
proc = CLIPProcessor.from_pretrained(cdir)

_imgc = {}
def img_e(path):
    k = str(path)
    if k not in _imgc:
        im = Image.open(path).convert("RGB")
        pin = proc(images=im, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            _imgc[k] = F.normalize(model.visual_projection(model.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)
    return _imgc[k]

_txc = {}
def txt_e(t):
    if t not in _txc:
        tin = proc(text=[t], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            _txc[t] = F.normalize(model.text_projection(model.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output), dim=-1)
    return _txc[t]


def auc_of(pairs):
    """pairs: list of (image_path, positive_prompt, label). score = sim(img, prompt)."""
    y, s = [], []
    for img, prompt, lab in pairs:
        s.append((img_e(img) @ txt_e(prompt).T).item()); y.append(lab)
    return roc_auc_score(y, s), len(y), sum(y) / len(y)


# ---- ALD: AMBER attribute ----
truth = _amber_truth()
attr_pairs = []
for it in _load_jsonl(paths.AMBER_Q_ATTRIBUTE.with_suffix(".json")) if False else json.load(open(paths.AMBER_Q_ATTRIBUTE, encoding="utf-8")):
    gt = truth.get(int(it["id"]))
    p = parse_amber_attr(it["query"])
    img = paths.AMBER_IMAGES / it["image"]
    if gt in ("yes", "no") and p and img.exists():
        noun, attr = p
        attr_pairs.append((img, f"a photo of a {attr} {noun}", 1 if gt == "yes" else 0))
    if len(attr_pairs) >= 1500:
        break
print("ALD AMBER-attr  AUC=%.4f  n=%d pos=%.3f" % auc_of(attr_pairs))

# ---- RLD: AMBER relation (contact) ----
rel_pairs = []
for it in json.load(open(paths.AMBER_Q_RELATION, encoding="utf-8")):
    gt = truth.get(int(it["id"]))
    p = parse_amber_rel(it["query"])
    img = paths.AMBER_IMAGES / it["image"]
    if gt in ("yes", "no") and p and img.exists():
        s_, _, o_ = p
        rel_pairs.append((img, f"a photo of a {s_} touching a {o_}", 1 if gt == "yes" else 0))
print("RLD AMBER-rel   AUC=%.4f  n=%d pos=%.3f" % auc_of(rel_pairs))

# ---- RLD: VG-rel (directional predicate) ----
vg_pairs = []
for line in open(paths.VG_REL_JSONL, encoding="utf-8"):
    it = json.loads(line)
    p = parse_vg_rel(it["question"])
    img = _resolve_vg_image(it["image"])
    gt = str(it.get("gt", "")).lower()
    if gt in ("yes", "no") and p and img and img.exists():
        s_, pred, o_ = p
        vg_pairs.append((img, f"a photo of a {s_} {pred} a {o_}", 1 if gt == "yes" else 0))
    if len(vg_pairs) >= 1500:
        break
print("RLD VG-rel      AUC=%.4f  n=%d pos=%.3f" % auc_of(vg_pairs))
