"""Generative PGD on CHAIR: during LLaVA captioning, suppress the first token of
any COCO-object word that CLIP grounding says is ABSENT from the image. Report
CHAIR-i/s AND recall+length (to show real de-hallucination, not muted captions).
"""
import sys, random, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, CLIPModel, CLIPProcessor, LogitsProcessorList
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_chair import _SYNONYMS, _extract_objects, _load_gt_objects
from cmpsa.eval.eval_hhd import parse_pope, _load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--lam", type=float, default=8.0)
ap.add_argument("--margin", type=float, default=0.0, help="absent if sim < tau - margin")
args = ap.parse_args()

cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"
random.seed(0)
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
tok = lproc.tokenizer
clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))

# ---- COCO categories + surfaces + first-token ids ----
CATS = sorted(set(_SYNONYMS.values()))
cat_surfaces = defaultdict(set)
for surf, cat in _SYNONYMS.items():
    cat_surfaces[cat].add(surf)
cat_firsttok = {}
for cat, surfs in cat_surfaces.items():
    ids = set()
    for s in surfs:
        for pre in (" " + s, s):
            t = tok(pre, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
    cat_firsttok[cat] = ids

@torch.no_grad()
def clip_img(im):
    pin = cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
    return F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)
@torch.no_grad()
def clip_txt(t):
    tin = cproc(text=[t], return_tensors="pt", padding=True).to(DEV)
    return F.normalize(clip.text_projection(clip.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output), dim=-1)
CAT_TXT = torch.cat([clip_txt(f"a photo of a {c}") for c in CATS], 0)   # [80, D]

def grounding(im):
    v = clip_img(im)                      # [1,D]
    sims = (v @ CAT_TXT.T)[0]             # [80]
    return {CATS[i]: float(sims[i]) for i in range(len(CATS))}

# ---- calibrate absent-threshold tau on POPE ----
pope = []
for _, qf in paths.POPE_SUBSETS.items():
    pope += _load_jsonl(qf)
random.shuffle(pope)
sc, yy = [], []
seen = {}
for it in pope[:1200]:
    obj = parse_pope(it["text"]); img = paths.POPE_IMAGE_DIR / it["image"]
    if not obj or not img.exists():
        continue
    k = it["image"]
    if k not in seen:
        seen[k] = clip_img(Image.open(img).convert("RGB"))
    sc.append(float((seen[k] @ clip_txt(f"a photo of a {obj}").T).item()))
    yy.append(1 if it["label"] == "yes" else 0)
sc = np.array(sc); yy = np.array(yy)
best = (0, 0.2)
for t in np.quantile(sc, np.linspace(0.05, 0.95, 37)):
    acc = ((sc >= t).astype(int) == yy).mean()
    if acc > best[0]:
        best = (acc, t)
TAU = best[1]
print(f"calibrated TAU={TAU:.4f} (POPE acc {best[0]:.3f}); absent if sim < TAU-{args.margin}", flush=True)


class Suppress:
    def __init__(self, ids, lam):
        self.ids = torch.tensor(sorted(ids), device=DEV, dtype=torch.long) if ids else None
        self.lam = lam
    def __call__(self, input_ids, scores):
        if self.ids is not None and self.ids.numel():
            scores[:, self.ids] -= self.lam
        return scores

@torch.no_grad()
def caption(im, processor=None):
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    inp = lproc(images=im, text=prompt, return_tensors="pt").to(DEV, torch.float16)
    lp = LogitsProcessorList([processor]) if processor else None
    out = llava.generate(**inp, max_new_tokens=64, do_sample=False, num_beams=1, logits_processor=lp)
    txt = lproc.batch_decode(out, skip_special_tokens=True)[0]
    return txt.split("ASSISTANT:")[-1].strip()

# ---- CHAIR loop ----
gt_objects, image_ids = _load_gt_objects()
image_ids = image_ids[:args.n]

def chair_stats(rows):
    hm = tm = hc = 0; tp = 0; lens = 0
    for m_, gold in rows:
        hal = [o for o in m_ if o not in gold]
        tm += len(m_); hm += len(hal); hc += 1 if hal else 0
        tp += len(set(m_) & gold)
    n = len(rows)
    return dict(chair_i=hm/tm if tm else 0, chair_s=hc/n, obj_per_cap=tm/n,
                true_per_cap=tp/n)

van, pgd = [], []
vlen, plen = 0, 0
for k, iid in enumerate(image_ids):
    ip = paths.coco_val2014_image(iid)
    if not ip.exists():
        continue
    im = Image.open(ip).convert("RGB")
    g = grounding(im)
    absent = [c for c in CATS if g[c] < TAU - args.margin]
    ids = set()
    for c in absent:
        ids |= cat_firsttok[c]
    cap_v = caption(im, None)
    cap_p = caption(im, Suppress(ids, args.lam))
    gold = gt_objects.get(iid, set())
    van.append((_extract_objects(cap_v), gold)); pgd.append((_extract_objects(cap_p), gold))
    vlen += len(cap_v.split()); plen += len(cap_p.split())
    if (k+1) % 50 == 0:
        print(f"  {k+1}/{len(image_ids)}", flush=True)

sv, sp = chair_stats(van), chair_stats(pgd)
n = len(van)
print(f"\nN={n}  lambda={args.lam}")
print(f"           CHAIR-i   CHAIR-s   obj/cap  true-obj/cap  len")
print(f"vanilla    {sv['chair_i']:.4f}   {sv['chair_s']:.4f}    {sv['obj_per_cap']:.2f}     {sv['true_per_cap']:.2f}      {vlen/n:.0f}")
print(f"PGD        {sp['chair_i']:.4f}   {sp['chair_s']:.4f}    {sp['obj_per_cap']:.2f}     {sp['true_per_cap']:.2f}      {plen/n:.0f}")
print(f"delta      {sp['chair_i']-sv['chair_i']:+.4f}   {sp['chair_s']-sv['chair_s']:+.4f}    "
      f"{sp['obj_per_cap']-sv['obj_per_cap']:+.2f}     {sp['true_per_cap']-sv['true_per_cap']:+.2f}")
