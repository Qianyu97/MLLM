"""OBJECT-LAYER GATE: does CLIP-OLD grounding evidence improve LLaVA's POPE answers
at MATCHED yes-ratio (i.e. genuine correction, not a yes->no threshold shift)?

Protocol: run vanilla LLaVA (first-token Yes/No) + CLIP grounding on POPE; split
cal/test; on cal pick fusion weight lambda and a decision threshold so the fused
yes-ratio == vanilla yes-ratio; apply to test; compare Acc/F1 at fixed yes-ratio.
"""
import sys, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import (LlavaForConditionalGeneration, AutoProcessor,
                          CLIPModel, CLIPProcessor)
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_hhd import parse_pope, _load_jsonl

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--n", type=int, default=2000)
N = _ap.parse_args().n
cfg = load_config()
M = paths.MODELS_ROOT
DEV = "cuda"
random.seed(0); np.random.seed(0)

# ---- load models ----
llava = LlavaForConditionalGeneration.from_pretrained(
    str(M / "llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M / "llava-1.5-7b"))
clip = CLIPModel.from_pretrained(str(M / "clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M / "clip-vit-l14-336"))

tok = lproc.tokenizer
def ids_for(words):
    out = []
    for w in words:
        for t in tok(w, add_special_tokens=False).input_ids:
            out.append(t)
    return list(set(out))
YES = ids_for(["Yes", "yes", " Yes", " yes"])
NO = ids_for(["No", "no", " No", " no"])

@torch.no_grad()
def llava_pyes(img, question):
    prompt = f"USER: <image>\n{question} Please answer with only Yes or No. ASSISTANT:"
    inp = lproc(images=img, text=prompt, return_tensors="pt").to(DEV, torch.float16)
    out = llava.generate(**inp, max_new_tokens=1, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    logits = out.scores[0][0].float()
    ly = logits[YES].max(); ln = logits[NO].max()
    p = torch.softmax(torch.stack([ly, ln]), 0)
    return float(p[0])

_ic = {}
@torch.no_grad()
def clip_g(img_path, obj):
    k = str(img_path)
    if k not in _ic:
        im = Image.open(img_path).convert("RGB")
        pin = cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
        _ic[k] = F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)
    tin = cproc(text=[f"a photo of a {obj}"], return_tensors="pt", padding=True).to(DEV)
    tf = F.normalize(clip.text_projection(clip.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output), dim=-1)
    return float((_ic[k] @ tf.T).item())

# ---- collect ----
items = []
for sub, qf in paths.POPE_SUBSETS.items():
    items += _load_jsonl(qf)
random.shuffle(items)
rows = []
for it in items:
    obj = parse_pope(it["text"]); img = paths.POPE_IMAGE_DIR / it["image"]
    if obj is None or not img.exists():
        continue
    im = Image.open(img).convert("RGB")
    rows.append({"p_yes": llava_pyes(im, it["text"]), "g": clip_g(img, obj),
                 "y": 1 if it["label"] == "yes" else 0})
    if len(rows) >= N:
        break

p = np.array([r["p_yes"] for r in rows]); g = np.array([r["g"] for r in rows]); y = np.array([r["y"] for r in rows])
n = len(y); idx = np.random.permutation(n); cal, test = idx[:n//2], idx[n//2:]

def stats(pred, yy):
    acc = (pred == yy).mean()
    tp = ((pred==1)&(yy==1)).sum(); fp=((pred==1)&(yy==0)).sum(); fn=((pred==0)&(yy==1)).sum()
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
    return acc, f1, pred.mean()

# vanilla on test
pred_v = (p[test] >= 0.5).astype(int)
acc_v, f1_v, yr_v = stats(pred_v, y[test])
yr_vanilla_test = pred_v.mean()

# standardize using cal stats
def z(a, ref): return (a - ref.mean())/(ref.std()+1e-8)
pz = z(p, p[cal]); gz = z(g, g[cal])

print(f"n={n}  vanilla(test): Acc={acc_v:.4f} F1={f1_v:.4f} yes_ratio={yr_v:.4f}")
print(f"AUC(test): llava_p_yes={roc_auc_score(y[test],p[test]):.4f}  clip_g={roc_auc_score(y[test],g[test]):.4f}")

# fuse; pick lambda on cal by best cal-Acc at threshold matching vanilla cal yes-ratio
yr_target = (p[cal]>=0.5).mean()
best = None
for lam in np.linspace(0, 3, 31):
    fz = pz + lam*gz
    thr = np.quantile(fz[cal], 1 - yr_target)      # threshold so cal yes-ratio == vanilla
    predc = (fz[cal] >= thr).astype(int)
    accc = (predc == y[cal]).mean()
    if best is None or accc > best[0]:
        best = (accc, lam, thr)
_, lam, _ = best
fz = pz + lam*gz
# on test: threshold to match vanilla TEST yes-ratio exactly (fair, fixed yes-ratio)
thr_t = np.quantile(fz[test], 1 - yr_vanilla_test)
pred_f = (fz[test] >= thr_t).astype(int)
acc_f, f1_f, yr_f = stats(pred_f, y[test])
print(f"FUSED (lambda={lam:.2f}, matched yes-ratio): Acc={acc_f:.4f} F1={f1_f:.4f} yes_ratio={yr_f:.4f}")
print(f"AUC(test) fused = {roc_auc_score(y[test], fz[test]):.4f}")
print(f"\n=> Acc gain at MATCHED yes-ratio: {acc_f-acc_v:+.4f}  (F1 {f1_f-f1_v:+.4f})")
