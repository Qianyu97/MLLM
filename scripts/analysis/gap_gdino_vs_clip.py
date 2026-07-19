"""detect-then-revise with GROUNDING-DINO per-object detection (more precise than
global CLIP). For each object MENTIONED in a vanilla LLaVA caption, run G-DINO
'detect {obj}' and take the max box score as the presence grounding. Diagnostic AUC
+ revise ceiling. If G-DINO separates hallucinated/true better than CLIP (0.73), the
detect-then-revise tradeoff becomes clean -> a real novel CMPSA mitigation."""
import sys, random, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np, torch
from collections import defaultdict
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import LlavaForConditionalGeneration, AutoProcessor, GroundingDinoForObjectDetection
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_chair import _SYNONYMS, _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); args = ap.parse_args()
cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"; random.seed(0); np.random.seed(0)
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
gdir = str(M.parent/"tools"/"grounding_dino")
gdproc = AutoProcessor.from_pretrained(gdir)
gdino = GroundingDinoForObjectDetection.from_pretrained(gdir).to(DEV).eval()

@torch.no_grad()
def gdino_score(image, obj):
    inp = gdproc(images=image, text=obj.lower().strip()+".", return_tensors="pt").to(DEV)
    out = gdino(**inp)
    res = None
    for kw in ("threshold", "box_threshold"):
        try:
            res = gdproc.post_process_grounded_object_detection(out, inp["input_ids"], **{kw: 0.05}, text_threshold=0.05, target_sizes=[image.size[::-1]])[0]; break
        except TypeError: continue
    return float(res["scores"].max()) if (res is not None and len(res["scores"])) else 0.0

@torch.no_grad()
def caption(im):
    inp = lproc(images=im, text="USER: <image>\nDescribe this image in detail. ASSISTANT:", return_tensors="pt").to(DEV, torch.float16)
    out = llava.generate(**inp, max_new_tokens=64, do_sample=False, num_beams=1)
    return lproc.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

# object -> canonical surface for G-DINO query (use the category name itself)
gt_objects, image_ids = _load_gt_objects(); test_ids = image_ids[2000:2000+args.n]
recs = []; per_img = []
for k, iid in enumerate(test_ids):
    try:
        ip = paths.coco_val2014_image(iid)
        if not ip.exists(): continue
        im = Image.open(ip).convert("RGB"); cap = caption(im); gold = gt_objects.get(iid, set())
        mentioned = _extract_objects(cap)
        sc = {c: gdino_score(im, c) for c in mentioned}
        per_img.append((mentioned, gold, sc))
        for c in mentioned:
            recs.append((c, sc[c], 0 if c in gold else 1))
    except Exception as e:
        print(f"  skip {iid}: {type(e).__name__} {str(e)[:80]}", flush=True)
        torch.cuda.empty_cache()
    if (k+1) % 60 == 0:
        torch.cuda.empty_cache(); print(f"  {k+1}/{len(test_ids)}", flush=True)

pv = np.array([r[1] for r in recs]); hal = np.array([r[2] for r in recs])
print(f"\n=== G-DINO DIAGNOSTIC (mentioned {len(recs)}, hallucinated {hal.mean():.2%}) ===")
print(f"mean G-DINO score: TRUE={pv[hal==0].mean():.3f}  HALLUCINATED={pv[hal==1].mean():.3f}")
if len(set(hal)) == 2:
    print(f"AUC (low score catches hallucinated) = {roc_auc_score(hal, -pv):.3f}   [CLIP-global was 0.733]")
print(f"frac mentioned with NO G-DINO box (score=0): true={ (pv[hal==0]==0).mean():.2%}  hall={(pv[hal==1]==0).mean():.2%}")

print("\n=== detect-then-revise CEILING (remove mentioned objs with G-DINO score<thr) ===")
print("thr    CHAIR-i  CHAIR-s  true-kept%  hall-removed%")
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    hm=tm=hc=tk=tt=hr=ht=0
    for mentioned, gold, sc in per_img:
        kept = [c for c in mentioned if sc[c] >= thr]
        hal_c = [c for c in mentioned if c not in gold]; khal = [c for c in kept if c not in gold]
        tm += len(kept); hm += len(khal); hc += 1 if khal else 0
        tk += len([c for c in kept if c in gold]); tt += len([c for c in mentioned if c in gold])
        hr += len(hal_c)-len(khal); ht += len(hal_c)
    n=len(per_img)
    print(f"{thr:.2f}  {hm/tm if tm else 0:.4f}  {hc/n:.4f}   {tk/tt*100 if tt else 0:.1f}       {hr/ht*100 if ht else 0:.1f}")
hm=tm=hc=0
for mentioned, gold, sc in per_img:
    hal_c=[c for c in mentioned if c not in gold]; tm+=len(mentioned); hm+=len(hal_c); hc+=1 if hal_c else 0
print(f"none  {hm/tm if tm else 0:.4f}  {hc/len(per_img):.4f}   100.0       0.0")
