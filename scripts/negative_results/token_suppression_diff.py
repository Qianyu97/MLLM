"""Differentiated PGD = contrastive decoding + grounding PROTECTION.
Compares: vanilla greedy | VCD (uniform contrast) | VCD+protect (reduce over-
suppression of object tokens the specialist grounding confirms PRESENT).
Goal: keep VCD's hallucination drop with less true-object loss => beats plain VCD."""
import sys, random, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import LlavaForConditionalGeneration, AutoProcessor, CLIPModel, CLIPProcessor
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.utils import load_json
from cmpsa.eval.eval_chair import _SYNONYMS, _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100); ap.add_argument("--ncal", type=int, default=800)
ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--noise", type=float, default=0.6); ap.add_argument("--protect", type=float, default=3.0)
ap.add_argument("--ppthr", type=float, default=0.6, help="protect object tokens with P(present)>=ppthr")
ap.add_argument("--maxnew", type=int, default=64)
args = ap.parse_args()

cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"; random.seed(0); np.random.seed(0); torch.manual_seed(0)
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b")); tok = lproc.tokenizer; EOS = tok.eos_token_id
clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))
PROMPT = "USER: <image>\nDescribe this image in detail. ASSISTANT:"

CATS = sorted(set(_SYNONYMS.values())); cat_surf = defaultdict(set)
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
def sims_of(im): return (cimg(im) @ CAT_TXT.T)[0].float().cpu().numpy()

# ---- per-category calibration ----
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
    y = P[:, j]
    models[c] = LogisticRegression(class_weight="balanced").fit(S[:, j:j+1], y) if (y.sum()>=15 and (1-y).sum()>=15) else GLOB
def present_boost(im):
    sims = sims_of(im); b = {}
    for j, c in enumerate(CATS):
        pp = float(models[c].predict_proba([[sims[j]]])[0,1])
        if pp >= args.ppthr:
            for tid in cat_ft[c]: b[tid] = max(b.get(tid, 0.0), args.protect * (pp - 0.5) * 2)
    if not b: return None, None
    return torch.tensor(list(b.keys()), device=DEV), torch.tensor(list(b.values()), device=DEV, dtype=torch.float32)

@torch.no_grad()
def plain(im):
    inp = lproc(images=im, text=PROMPT, return_tensors="pt").to(DEV, torch.float16)
    out = llava.generate(**inp, max_new_tokens=args.maxnew, do_sample=False, num_beams=1)
    return lproc.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

@torch.no_grad()
def contrastive(im, boost_ids=None, boost_w=None):
    a = args.alpha
    inp = lproc(images=im, text=PROMPT, return_tensors="pt").to(DEV, torch.float16)
    pv = inp["pixel_values"]; pvn = (pv + args.noise*torch.randn_like(pv)).to(torch.float16)
    ids, attn = inp["input_ids"], inp["attention_mask"]
    o1 = llava(input_ids=ids, attention_mask=attn, pixel_values=pv, use_cache=True)
    o2 = llava(input_ids=ids, attention_mask=attn, pixel_values=pvn, use_cache=True)
    kv1, kv2 = o1.past_key_values, o2.past_key_values
    l1 = o1.logits[:, -1, :].float(); l2 = o2.logits[:, -1, :].float(); gen = []
    for _ in range(args.maxnew):
        cutoff = l1.max() + torch.log(torch.tensor(args.beta, device=DEV))
        cd = (1+a)*l1 - a*l2
        if boost_ids is not None:
            cd[0, boost_ids] += boost_w            # protect grounded-present objects
        cd = cd.masked_fill(l1 < cutoff, float("-inf"))
        nt = int(cd.argmax(-1))
        if nt == EOS: break
        gen.append(nt); ntt = torch.tensor([[nt]], device=DEV)
        o1 = llava(input_ids=ntt, past_key_values=kv1, use_cache=True); o2 = llava(input_ids=ntt, past_key_values=kv2, use_cache=True)
        kv1, kv2 = o1.past_key_values, o2.past_key_values
        l1 = o1.logits[:, -1, :].float(); l2 = o2.logits[:, -1, :].float()
    return tok.decode(gen, skip_special_tokens=True).strip()

gt_objects, image_ids = _load_gt_objects(); test_ids = image_ids[2000:2000+args.n]
def stats(rows):
    hm=tm=hc=tp=0
    for m_, gold in rows:
        hal=[o for o in m_ if o not in gold]; tm+=len(m_); hm+=len(hal); hc+=1 if hal else 0; tp+=len(set(m_)&gold)
    n=len(rows); return dict(ci=hm/tm if tm else 0, cs=hc/n, opc=tm/n, tpc=tp/n)

R = {"vanilla": [], "VCD": [], "VCD+protect": []}; L = defaultdict(int)
for k, iid in enumerate(test_ids):
    ip = paths.coco_val2014_image(iid)
    if not ip.exists(): continue
    im = Image.open(ip).convert("RGB"); gold = gt_objects.get(iid, set())
    bi, bw = present_boost(im)
    caps = {"vanilla": plain(im), "VCD": contrastive(im), "VCD+protect": contrastive(im, bi, bw)}
    for kk, cap in caps.items():
        R[kk].append((_extract_objects(cap), gold)); L[kk] += len(cap.split())
    if (k+1) % 40 == 0: print(f"  {k+1}/{len(test_ids)}", flush=True)

n = len(R["vanilla"])
print(f"\nN={n} alpha={args.alpha} protect={args.protect} ppthr={args.ppthr}")
print(f"              CHAIR-i  CHAIR-s  obj/cap  true-obj/cap  len")
for kk in ("vanilla", "VCD", "VCD+protect"):
    s = stats(R[kk]); print(f"{kk:13s} {s['ci']:.4f}  {s['cs']:.4f}   {s['opc']:.2f}     {s['tpc']:.2f}      {L[kk]/n:.0f}")
