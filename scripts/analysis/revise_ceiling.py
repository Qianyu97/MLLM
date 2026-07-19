"""DECISIVE novelty test: can grounding catch caption hallucinations?
For vanilla LLaVA captions, split MENTIONED COCO objects into hallucinated (not in
GT) vs true (in GT). Measure the grounding P(present) of each. Then the CEILING of
a detect-then-revise method: remove mentioned objects with P(present)<thr, recompute
CHAIR + recall retained. If hallucinations get LOW P -> revise works (novel path);
if hallucinations get HIGH P too -> correlated bias total -> grounding mitigation dead."""
import sys, random, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import LlavaForConditionalGeneration, AutoProcessor, CLIPModel, CLIPProcessor
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import load_json
from cmpsa.eval.eval_chair import _SYNONYMS, _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300); ap.add_argument("--ncal", type=int, default=800)
args = ap.parse_args()
cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"; random.seed(0); np.random.seed(0)
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))
CATS = sorted(set(_SYNONYMS.values()))

@torch.no_grad()
def cimg(im):
    pin = cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
    return F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)
@torch.no_grad()
def ctxt(t):
    tin = cproc(text=[t], return_tensors="pt", padding=True).to(DEV)
    return F.normalize(clip.text_projection(clip.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output), dim=-1)
CAT_TXT = torch.cat([ctxt(f"a photo of a {c}") for c in CATS], 0)
def sims_of(im): return (cimg(im) @ CAT_TXT.T)[0].float().cpu().numpy()

# per-category calibration
inst = load_json(paths.COCO_INSTANCES_VAL2014); id2name = {c["id"]: c["name"] for c in inst["categories"]}
img_cats = defaultdict(set)
for a in inst["annotations"]:
    nm = id2name.get(a["category_id"])
    if nm: img_cats[a["image_id"]].add(nm)
cal_ids = list(img_cats.keys()); random.shuffle(cal_ids); cal_ids = cal_ids[:args.ncal]
print(f"calibrating on {len(cal_ids)} imgs...", flush=True)
S, P = [], []
for iid in cal_ids:
    ip = paths.coco_val2014_image(iid)
    if not ip.exists(): continue
    S.append(sims_of(Image.open(ip).convert("RGB"))); P.append([1 if CATS[j] in img_cats[iid] else 0 for j in range(len(CATS))])
S = np.array(S); P = np.array(P)
GLOB = LogisticRegression().fit(S.reshape(-1,1), P.reshape(-1)); models = {}
for j, c in enumerate(CATS):
    y = P[:, j]; models[c] = LogisticRegression(class_weight="balanced").fit(S[:, j:j+1], y) if (y.sum()>=15 and (1-y).sum()>=15) else GLOB
def pp_map(im):
    s = sims_of(im); return {CATS[j]: float(models[CATS[j]].predict_proba([[s[j]]])[0,1]) for j in range(len(CATS))}

@torch.no_grad()
def caption(im):
    inp = lproc(images=im, text="USER: <image>\nDescribe this image in detail. ASSISTANT:", return_tensors="pt").to(DEV, torch.float16)
    out = llava.generate(**inp, max_new_tokens=64, do_sample=False, num_beams=1)
    return lproc.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

gt_objects, image_ids = _load_gt_objects(); test_ids = image_ids[2000:2000+args.n]
recs = []   # (cat, P(present), is_hallucinated, image_idx)
per_img = []  # list of (mentioned:set, gold:set, pp:dict)
for k, iid in enumerate(test_ids):
    ip = paths.coco_val2014_image(iid)
    if not ip.exists(): continue
    im = Image.open(ip).convert("RGB"); cap = caption(im); pp = pp_map(im)
    mentioned = _extract_objects(cap); gold = gt_objects.get(iid, set())
    per_img.append((mentioned, gold, pp))
    for c in mentioned:
        recs.append((c, pp.get(c, 0.5), 0 if c in gold else 1))
    if (k+1) % 60 == 0: print(f"  {k+1}/{len(test_ids)}", flush=True)

pv = np.array([r[1] for r in recs]); hal = np.array([r[2] for r in recs])
print(f"\n=== DIAGNOSTIC (mentioned objects: {len(recs)}, hallucinated {hal.mean():.2%}) ===")
print(f"mean P(present): TRUE objs={pv[hal==0].mean():.3f}  HALLUCINATED objs={pv[hal==1].mean():.3f}")
if len(set(hal)) == 2:
    print(f"AUC of (low P(present)) catching hallucinated among mentioned = {roc_auc_score(hal, -pv):.3f}")

print("\n=== detect-then-revise CEILING (remove mentioned objs with P(present)<thr) ===")
print("thr    CHAIR-i  CHAIR-s  true-kept%  hall-removed%")
for thr in [0.2, 0.3, 0.4, 0.5]:
    hm=tm=hc=tp_keep=tp_tot=hal_rm=hal_tot=0
    for mentioned, gold, pp in per_img:
        kept = [c for c in mentioned if pp.get(c, 0.5) >= thr]
        hal_c = [c for c in mentioned if c not in gold]; true_c = [c for c in mentioned if c in gold]
        khal = [c for c in kept if c not in gold]
        tm += len(kept); hm += len(khal); hc += 1 if khal else 0
        tp_keep += len([c for c in kept if c in gold]); tp_tot += len(true_c)
        hal_rm += len(hal_c) - len(khal); hal_tot += len(hal_c)
    n = len(per_img)
    print(f"{thr:.1f}   {hm/tm if tm else 0:.4f}  {hc/n:.4f}   {tp_keep/tp_tot*100 if tp_tot else 0:.1f}       {hal_rm/hal_tot*100 if hal_tot else 0:.1f}")
# baseline (no removal)
hm=tm=hc=0
for mentioned, gold, pp in per_img:
    hal_c=[c for c in mentioned if c not in gold]; tm+=len(mentioned); hm+=len(hal_c); hc+=1 if hal_c else 0
print(f"none  {hm/tm if tm else 0:.4f}  {hc/len(per_img):.4f}   100.0       0.0")
