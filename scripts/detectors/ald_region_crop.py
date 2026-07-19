"""ALD prototype: does region-crop CLIP beat whole-image CLIP for attributes?
Grounding-DINO detects the noun's box -> crop -> CLIP attribute scoring on the crop.
Compares several formulations on AMBER-attribute vs the whole-image baseline (0.60)."""
import sys, json, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                          CLIPModel, CLIPProcessor)
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_hhd import parse_amber_attr, _amber_truth

cfg = load_config()
M = paths.MODELS_ROOT
DEV = "cuda"

gdproc = AutoProcessor.from_pretrained(str(M.parent / "tools" / "grounding_dino"))
gdino = GroundingDinoForObjectDetection.from_pretrained(str(M.parent / "tools" / "grounding_dino")).to(DEV).eval()
clip = CLIPModel.from_pretrained(str(M / "clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M / "clip-vit-l14-336"))


@torch.no_grad()
def detect_box(image, noun):
    text = noun.lower().strip() + "."
    inp = gdproc(images=image, text=text, return_tensors="pt").to(DEV)
    out = gdino(**inp)
    try:
        res = gdproc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=0.2, text_threshold=0.2,
            target_sizes=[image.size[::-1]])[0]
    except TypeError:
        res = gdproc.post_process_grounded_object_detection(
            out, inp["input_ids"], box_threshold=0.2, text_threshold=0.2,
            target_sizes=[image.size[::-1]])[0]
    if len(res["scores"]) == 0:
        return None
    i = int(res["scores"].argmax())
    return [float(v) for v in res["boxes"][i]]


@torch.no_grad()
def clip_img_emb(im):
    pin = cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
    return F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)

_txt = {}
@torch.no_grad()
def clip_txt_emb(t):
    if t not in _txt:
        tin = cproc(text=[t], return_tensors="pt", padding=True).to(DEV)
        _txt[t] = F.normalize(clip.text_projection(clip.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output), dim=-1)
    return _txt[t]


def crop(image, box, pad=0.1):
    W, H = image.size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0 = max(0, x0 - pad * w); y0 = max(0, y0 - pad * h)
    x1 = min(W, x1 + pad * w); y1 = min(H, y1 + pad * h)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return image
    return image.crop((x0, y0, x1, y1))


truth = _amber_truth()
items = json.load(open(paths.AMBER_Q_ATTRIBUTE, encoding="utf-8"))
random.Random(0).shuffle(items)

y = []
whole_attr_noun, crop_attr_noun, crop_attr, crop_contrast = [], [], [], []
n_box = 0
for it in items:
    gt = truth.get(int(it["id"])); p = parse_amber_attr(it["query"])
    img = paths.AMBER_IMAGES / it["image"]
    if gt not in ("yes", "no") or not p or not img.exists():
        continue
    noun, attr = p
    image = Image.open(img).convert("RGB")
    box = detect_box(image, noun)
    region = crop(image, box) if box else image
    if box:
        n_box += 1
    wi = clip_img_emb(image); ci = clip_img_emb(region)
    an = clip_txt_emb(f"a photo of a {attr} {noun}")
    nn = clip_txt_emb(f"a photo of a {noun}")
    at = clip_txt_emb(f"{attr}")
    whole_attr_noun.append(float((wi @ an.T).item()))
    crop_attr_noun.append(float((ci @ an.T).item()))
    crop_attr.append(float((ci @ at.T).item()))
    crop_contrast.append(float((ci @ an.T).item() - (ci @ nn.T).item()))
    y.append(1 if gt == "yes" else 0)
    if len(y) >= 1000:
        break

print(f"n={len(y)} pos={sum(y)/len(y):.3f}  boxes_found={n_box}/{len(y)}")
for name, s in [("whole-image  sim(img,'attr noun')", whole_attr_noun),
                ("region-crop  sim(crop,'attr noun')", crop_attr_noun),
                ("region-crop  sim(crop,'attr')", crop_attr),
                ("region-crop  contrast(attr noun - noun)", crop_contrast)]:
    print(f"  AUC {name:42s} = {roc_auc_score(y, s):.4f}")
