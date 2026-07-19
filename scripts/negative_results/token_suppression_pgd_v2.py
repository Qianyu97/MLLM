"""Improved generative PGD on CHAIR: PER-CATEGORY calibrated CLIP grounding
(logistic sim->P(present) fit on COCO GT), SOFT penalty proportional to absence
confidence, only when P(present) < thr (protects common objects like person)."""
import sys, random, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import LlavaForConditionalGeneration, AutoProcessor, CLIPModel, CLIPProcessor, LogitsProcessorList
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import load_json
from cmpsa.eval.eval_chair import _SYNONYMS, _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--ncal", type=int, default=1500)
ap.add_argument("--lam", type=float, default=6.0)
ap.add_argument("--thr", type=float, default=0.35, help="suppress only if P(present) < thr")
args = ap.parse_args()

cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"; random.seed(0); np.random.seed(0)
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b")); tok = lproc.tokenizer
clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))

CATS = sorted(set(_SYNONYMS.values()))
cat_surf = defaultdict(set)
for s, c in _SYNONYMS.items(): cat_surf[c].add(s)
cat_ft = {}
for c, ss in cat_surf.items():
    ids = set()
    for s in ss:
        for pre in (" "+s, s):
            t = tok(pre, add_special_tokens=False).input_ids
            if t: ids.add(t[0])
    cat_ft[c] = ids

@torch.no_grad()
def cimg(im):
    pin = cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
    return F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)
@torch.no_grad()
def ctxt(t):
    tin = cproc(text=[t], return_tensors="pt", padding=True).to(DEV)
    return F.normalize(clip.text_projection(clip.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output), dim=-1)
CAT_TXT = torch.cat([ctxt(f"a photo of a {c}") for c in CATS], 0)

def sims_of(im):
    return (cimg(im) @ CAT_TXT.T)[0].float().cpu().numpy()   # [80]

# ---- per-category calibration on COCO instances GT ----
inst = load_json(paths.COCO_INSTANCES_VAL2014)
id2name = {c["id"]: c["name"] for c in inst["categories"]}
img_cats = defaultdict(set)
for a in inst["annotations"]:
    nm = id2name.get(a["category_id"])
    if nm: img_cats[a["image_id"]].add(nm)
cal_ids = list(img_cats.keys()); random.shuffle(cal_ids); cal_ids = cal_ids[:args.ncal]
print(f"calibrating per-category on {len(cal_ids)} images...", flush=True)
S = []; present = []
for i, iid in enumerate(cal_ids):
    ip = paths.coco_val2014_image(iid)
    if not ip.exists(): continue
    S.append(sims_of(Image.open(ip).convert("RGB")))
    present.append([1 if CATS[j] in img_cats[iid] else 0 for j in range(len(CATS))])
    if (i+1) % 300 == 0: print(f"   cal {i+1}/{len(cal_ids)}", flush=True)
S = np.array(S); present = np.array(present)   # [Ncal,80]

models = {}
GLOBAL = LogisticRegression().fit(S.reshape(-1,1), present.reshape(-1))
for j, c in enumerate(CATS):
    y = present[:, j]
    if y.sum() >= 15 and (1-y).sum() >= 15:
        models[c] = LogisticRegression(class_weight="balanced").fit(S[:, j:j+1], y)
    else:
        models[c] = GLOBAL
def p_present(sims):
    return {CATS[j]: float(models[CATS[j]].predict_proba([[sims[j]]])[0,1]) for j in range(len(CATS))}

class Suppress:
    def __init__(self, tokpen):
        if tokpen:
            self.ids = torch.tensor(list(tokpen.keys()), device=DEV)
            self.w = torch.tensor(list(tokpen.values()), device=DEV, dtype=torch.float16)
        else:
            self.ids = None
    def __call__(self, input_ids, scores):
        if self.ids is not None:
            scores[:, self.ids] -= self.w
        return scores

@torch.no_grad()
def caption(im, proc=None):
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    inp = lproc(images=im, text=prompt, return_tensors="pt").to(DEV, torch.float16)
    lp = LogitsProcessorList([proc]) if proc else None
    out = llava.generate(**inp, max_new_tokens=64, do_sample=False, num_beams=1, logits_processor=lp)
    return lproc.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

gt_objects, image_ids = _load_gt_objects()
test_ids = image_ids[2000:2000+args.n]

def stats(rows):
    hm=tm=hc=tp=0
    for m_, gold in rows:
        hal=[o for o in m_ if o not in gold]; tm+=len(m_); hm+=len(hal); hc+= 1 if hal else 0; tp+=len(set(m_)&gold)
    n=len(rows); return dict(ci=hm/tm if tm else 0, cs=hc/n, opc=tm/n, tpc=tp/n)

van, pgd = [], []; vlen=plen=0; nsupp=0
for k, iid in enumerate(test_ids):
    ip = paths.coco_val2014_image(iid)
    if not ip.exists(): continue
    im = Image.open(ip).convert("RGB")
    pp = p_present(sims_of(im))
    tokpen = {}
    for c in CATS:
        if pp[c] < args.thr:
            pen = args.lam * (0.5 - pp[c])            # soft, proportional to absence confidence
            for tid in cat_ft[c]:
                tokpen[tid] = max(tokpen.get(tid, 0.0), pen)
    nsupp += sum(1 for c in CATS if pp[c] < args.thr)
    cv = caption(im, None); cp = caption(im, Suppress(tokpen))
    gold = gt_objects.get(iid, set())
    van.append((_extract_objects(cv), gold)); pgd.append((_extract_objects(cp), gold))
    vlen += len(cv.split()); plen += len(cp.split())
    if (k+1) % 50 == 0: print(f"  {k+1}/{len(test_ids)}", flush=True)

sv, sp = stats(van), stats(pgd); n = len(van)
print(f"\nN={n} lam={args.lam} thr={args.thr}  avg suppressed cats/img={nsupp/n:.1f}")
print(f"          CHAIR-i  CHAIR-s  obj/cap  true-obj/cap  len")
print(f"vanilla   {sv['ci']:.4f}  {sv['cs']:.4f}   {sv['opc']:.2f}     {sv['tpc']:.2f}      {vlen/n:.0f}")
print(f"PGD       {sp['ci']:.4f}  {sp['cs']:.4f}   {sp['opc']:.2f}     {sp['tpc']:.2f}      {plen/n:.0f}")
print(f"delta     {sp['ci']-sv['ci']:+.4f}  {sp['cs']-sv['cs']:+.4f}   {sp['opc']-sv['opc']:+.2f}     {sp['tpc']-sv['tpc']:+.2f}")
