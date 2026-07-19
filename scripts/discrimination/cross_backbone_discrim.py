"""CROSS-BACKBONE OBJECT DISCRIMINATIVE GATE.

Does CLIP-OLD grounding evidence improve each backbone's POPE yes/no answers at a
MATCHED yes-ratio (genuine correction, not a yes->no threshold shift)? Repeats the
object_gate.py protocol for LLaVA-1.5, LLaVA-1.6, InstructBLIP, Qwen-VL-Chat.

CLIP grounding g is backbone-agnostic (same CLIP for all); only p_yes changes.
Run one backbone per process (avoids the reload native crash):
  python cross_backbone_discrim.py --backbone llava16 --n 2000
"""
import sys, os, json, argparse, random, gc
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from cmpsa import paths
from cmpsa.eval.eval_hhd import parse_pope, _load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True,
                choices=["llava15", "llava16", "instructblip", "qwenvl"])
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()
M = paths.MODELS_ROOT; DEV = "cuda"
random.seed(0); np.random.seed(0)
Q_SUFFIX = " Please answer with only Yes or No."


def load_fp16(cls, path, **extra):
    kw = dict(low_cpu_mem_usage=True, device_map={"": 0}, **extra)
    try:
        return cls.from_pretrained(path, dtype=torch.float16, **kw).eval()
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=torch.float16, **kw).eval()


def yesno_ids(tok):
    def ids_for(words):
        out = []
        for w in words:
            out += tok(w, add_special_tokens=False).input_ids
        return list(set(out))
    return ids_for(["Yes", "yes", " Yes", " yes"]), ids_for(["No", "no", " No", " no"])


# ------------------------------------------------------------------ adapters
class LLaVA15:
    dirname = "llava-1.5-7b"; path_based = False
    tmpl = "USER: <image>\n{q}{s} ASSISTANT:"
    def __init__(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(str(M/self.dirname))
        self.model = load_fp16(LlavaForConditionalGeneration, str(M/self.dirname))
        self.tok = self.proc.tokenizer
        self.YES, self.NO = yesno_ids(self.tok)
    @torch.no_grad()
    def pyes(self, image, question):
        prompt = self.tmpl.format(q=question, s=Q_SUFFIX)
        inp = self.proc(images=image, text=prompt, return_tensors="pt").to(DEV)
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        out = self.model.generate(**inp, max_new_tokens=1, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        lg = out.scores[0][0].float()
        ly = lg[self.YES].max(); ln = lg[self.NO].max()
        return float(torch.softmax(torch.stack([ly, ln]), 0)[0])


class LLaVA16(LLaVA15):
    dirname = "llava-1.6-vicuna-7b"
    def __init__(self):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        self.proc = LlavaNextProcessor.from_pretrained(str(M/self.dirname))
        self.model = load_fp16(LlavaNextForConditionalGeneration, str(M/self.dirname))
        self.tok = self.proc.tokenizer
        self.YES, self.NO = yesno_ids(self.tok)


class InstructBLIP:
    dirname = "instructblip-7b"; path_based = False
    def __init__(self):
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        self.proc = InstructBlipProcessor.from_pretrained(str(M/self.dirname))
        self.model = load_fp16(InstructBlipForConditionalGeneration, str(M/self.dirname))
        self.tok = self.proc.tokenizer          # vicuna LLM tokenizer
        self.YES, self.NO = yesno_ids(self.tok)
    @torch.no_grad()
    def pyes(self, image, question):
        text = question + Q_SUFFIX
        inp = self.proc(images=image, text=text, return_tensors="pt").to(DEV)
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        out = self.model.generate(**inp, max_new_tokens=1, do_sample=False, num_beams=1,
                                  output_scores=True, return_dict_in_generate=True)
        lg = out.scores[0][0].float()
        ly = lg[self.YES].max(); ln = lg[self.NO].max()
        return float(torch.softmax(torch.stack([ly, ln]), 0)[0])


class QwenVL:
    dirname = "qwen-vl-chat"; path_based = True
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(M/self.dirname), trust_remote_code=True)
        self.model = load_fp16(AutoModelForCausalLM, str(M/self.dirname),
                               trust_remote_code=True, fp16=True)
        # Yes/No ids on the Qwen tokenizer
        self.YES, self.NO = yesno_ids(self.tok)
    @torch.no_grad()
    def pyes(self, image_path, question):
        # Build the Qwen-VL-Chat prompt manually and take last-token logits.
        q = self.tok.from_list_format([{"image": str(image_path)},
                                       {"text": question + Q_SUFFIX}])
        raw = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
               f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
        ids = self.tok(raw, return_tensors="pt").input_ids.to(DEV)
        lg = self.model(input_ids=ids).logits[0, -1].float()
        ly = lg[self.YES].max(); ln = lg[self.NO].max()
        return float(torch.softmax(torch.stack([ly, ln]), 0)[0])


ADAPTERS = {"llava15": LLaVA15, "llava16": LLaVA16,
            "instructblip": InstructBLIP, "qwenvl": QwenVL}

# ------------------------------------------------------------------ CLIP grounding (shared)
# Loaded lazily (after the backbone) so a large backbone like InstructBLIP does not
# hit the Windows pagefile ceiling (OSError 1455) while both memory-map at once.
from transformers import CLIPModel, CLIPProcessor
clip = None; cproc = None; _ic = {}
def _ensure_clip():
    global clip, cproc
    if clip is None:
        clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
        cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))
@torch.no_grad()
def clip_g(img_path, obj):
    _ensure_clip()
    k = str(img_path)
    if k not in _ic:
        im = Image.open(img_path).convert("RGB")
        pin = cproc(images=im, return_tensors="pt").to(DEV)
        pin["pixel_values"] = pin["pixel_values"].to(torch.float16)
        _ic[k] = F.normalize(clip.visual_projection(
            clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)
    tin = cproc(text=[f"a photo of a {obj}"], return_tensors="pt", padding=True).to(DEV)
    tf = F.normalize(clip.text_projection(clip.text_model(
        input_ids=tin["input_ids"], attention_mask=tin["attention_mask"]).pooler_output), dim=-1)
    return float((_ic[k] @ tf.T).item())

# ------------------------------------------------------------------ collect POPE
bk = ADAPTERS[args.backbone]()
items = []
for sub, qf in paths.POPE_SUBSETS.items():
    items += _load_jsonl(qf)
random.shuffle(items)
rows = []
for it in items:
    obj = parse_pope(it["text"]); img = paths.POPE_IMAGE_DIR / it["image"]
    if obj is None or not img.exists():
        continue
    try:
        im = Image.open(img).convert("RGB")
        pv = bk.pyes(str(img) if bk.path_based else im, it["text"])
        gv = clip_g(img, obj)
    except Exception as e:
        print(f"  skip: {type(e).__name__} {str(e)[:70]}", flush=True); continue
    rows.append({"p_yes": pv, "g": gv, "y": 1 if it["label"] == "yes" else 0})
    if len(rows) % 200 == 0:
        print(f"  {len(rows)}/{args.n}", flush=True)
    if len(rows) >= args.n:
        break

p = np.array([r["p_yes"] for r in rows]); g = np.array([r["g"] for r in rows])
y = np.array([r["y"] for r in rows]); n = len(y)
idx = np.random.permutation(n); cal, test = idx[:n//2], idx[n//2:]

def stats(pred, yy):
    acc = (pred == yy).mean()
    tp = ((pred==1)&(yy==1)).sum(); fp=((pred==1)&(yy==0)).sum(); fn=((pred==0)&(yy==1)).sum()
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
    return float(acc), float(f1), float(pred.mean())

pred_v = (p[test] >= 0.5).astype(int)
acc_v, f1_v, yr_v = stats(pred_v, y[test])
yr_vanilla_test = pred_v.mean()

def z(a, ref): return (a - ref.mean())/(ref.std()+1e-8)
pz = z(p, p[cal]); gz = z(g, g[cal])

yr_target = (p[cal] >= 0.5).mean()
best = None
for lam in np.linspace(0, 3, 31):
    fz = pz + lam*gz
    thr = np.quantile(fz[cal], 1 - yr_target)
    predc = (fz[cal] >= thr).astype(int)
    accc = (predc == y[cal]).mean()
    if best is None or accc > best[0]:
        best = (accc, lam)
lam = best[1]
fz = pz + lam*gz
thr_t = np.quantile(fz[test], 1 - yr_vanilla_test)
pred_f = (fz[test] >= thr_t).astype(int)
acc_f, f1_f, yr_f = stats(pred_f, y[test])
auc_v = float(roc_auc_score(y[test], p[test]))
auc_g = float(roc_auc_score(y[test], g[test]))
auc_f = float(roc_auc_score(y[test], fz[test]))

res = dict(backbone=args.backbone, n=n, lam=float(lam),
           acc_vanilla=acc_v, f1_vanilla=f1_v, yr_vanilla=yr_v,
           acc_fused=acc_f, f1_fused=f1_f, yr_fused=yr_f,
           d_acc=acc_f-acc_v, d_f1=f1_f-f1_v,
           auc_pyes=auc_v, auc_clip=auc_g, auc_fused=auc_f)
outp = paths.RESULTS_ROOT / "metrics" / f"discrim_{args.backbone}.json"
outp.parent.mkdir(parents=True, exist_ok=True)
with open(outp, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)

print(f"\n=== {args.backbone}  N={n}  lambda={lam:.2f} ===")
print(f"vanilla : Acc={acc_v:.4f} F1={f1_v:.4f} yr={yr_v:.4f}  (AUC p_yes={auc_v:.3f}, clip={auc_g:.3f})")
print(f"+ground : Acc={acc_f:.4f} F1={f1_f:.4f} yr={yr_f:.4f}  (AUC fused={auc_f:.3f})")
print(f"=> dAcc={acc_f-acc_v:+.4f}  dF1={f1_f-f1_v:+.4f}  (matched yes-ratio)")
print(f"saved -> {outp}")
