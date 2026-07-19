"""Does the healthy PSAS encode OBJECT PRESENCE? Test raw phrase<->image similarity
on POPE (present vs absent), several formulations, report AUC. Decides whether the
detector needs only a better head (info present) or per-element training (info absent)."""
import sys, json, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_hhd import PSASEncoder, parse_pope, _load_jsonl

cfg = load_config()
enc = PSASEncoder(cfg)
N = 1500
items = []
for sub, qf in paths.POPE_SUBSETS.items():
    items += _load_jsonl(qf)
random.Random(0).shuffle(items)

cos_glob, cos_maxp, kl_glob, lbl = [], [], [], []
raw_clip_cos = []
done = 0
for it in items:
    obj = parse_pope(it["text"])
    img = paths.POPE_IMAGE_DIR / it["image"]
    if obj is None or not img.exists():
        continue
    v_mu, v_lv = enc.visual(img)                 # [576, D]
    l_mu, l_lv = enc.phrase(obj)                 # [D]
    vg = v_mu.mean(0)
    cg = F.cosine_similarity(vg[None], l_mu[None]).item()
    cp = F.cosine_similarity(v_mu, l_mu[None]).max().item()
    # KL-based existence proxy (neg mean-dist of nearest patch)
    nd = ((v_mu - l_mu[None]) ** 2).sum(-1).min().item()
    cos_glob.append(cg); cos_maxp.append(cp); kl_glob.append(-nd)
    lbl.append(1 if it["label"] == "yes" else 0)
    done += 1
    if done >= N:
        break

y = lbl
print(f"n={len(y)} pos_rate={sum(y)/len(y):.3f}")
for name, s in [("PSAS cos(global img, phrase)", cos_glob),
                ("PSAS max_patch cos(patch, phrase)", cos_maxp),
                ("PSAS -min_patch_dist", kl_glob)]:
    try:
        print(f"  AUC {name:38s} = {roc_auc_score(y, s):.4f}")
    except Exception as e:
        print(f"  AUC {name}: {e}")

# ---- Reference: raw CLIP text-image zero-shot (CLIP's own text encoder) ----
print("\n[reference] raw CLIP zero-shot object presence (CLIP text vs image, no PSAS):")
from transformers import CLIPModel, CLIPProcessor
import os
cdir = str(paths.MODELS_ROOT / cfg.models.__dict__['clip-vit-l14-336'].local_dir)
try:
    clipm = CLIPModel.from_pretrained(cdir, torch_dtype=torch.float16).to("cuda").eval()
    cproc = CLIPProcessor.from_pretrained(cdir)
    from PIL import Image
    # reuse a subset
    raw_y, raw_s = [], []
    seen_imgs = {}
    cnt = 0
    for it in items:
        obj = parse_pope(it["text"]); img = paths.POPE_IMAGE_DIR / it["image"]
        if obj is None or not img.exists(): continue
        key = it["image"]
        if key not in seen_imgs:
            im = Image.open(img).convert("RGB")
            pin = cproc(images=im, return_tensors="pt").to("cuda", torch.float16)
            with torch.no_grad():
                seen_imgs[key] = F.normalize(clipm.get_image_features(**pin), dim=-1)
        tin = cproc(text=[f"a photo of a {obj}"], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            tf = F.normalize(clipm.get_text_features(**tin), dim=-1)
        raw_s.append((seen_imgs[key] @ tf.T).item()); raw_y.append(1 if it["label"]=="yes" else 0)
        cnt += 1
        if cnt >= N: break
    print(f"  AUC raw CLIP a-photo-of-a-X = {roc_auc_score(raw_y, raw_s):.4f}  (n={len(raw_y)})")
except Exception as e:
    print("  raw CLIP ref failed:", e)
