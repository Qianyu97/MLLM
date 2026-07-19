"""Cross-backbone detect-then-revise on CHAIR (3-phase, memory-safe):
  A) backbone captions all images -> B) unload, G-DINO flags absent objects
  C) reload backbone, rewrite flagged captions. Reports CHAIR vanilla vs revise.

Backbones: llava16 (LlavaNext), instructblip (InstructBLIP-vicuna-7b),
           qwenvl (Qwen-VL-Chat, trust_remote_code).
Run: python cross_backbone_revise.py --backbone llava16 --n 500 --thr 0.30
"""
import sys, os, json, argparse, gc
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch
from PIL import Image
from cmpsa import paths
from cmpsa.eval.eval_chair import _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True, choices=["llava15", "llava16", "instructblip", "qwenvl"])
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--thr", type=float, default=0.30)
ap.add_argument("--maxnew", type=int, default=80)
args = ap.parse_args()
M = paths.MODELS_ROOT; DEV = "cuda"
CAP_PROMPT = "Describe this image in detail."


def load_fp16(cls, path, **extra):
    """fp16 + stream-to-GPU loading, robust to the torch_dtype->dtype rename."""
    kw = dict(low_cpu_mem_usage=True, device_map={"": 0}, **extra)
    try:
        return cls.from_pretrained(path, dtype=torch.float16, **kw).eval()
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=torch.float16, **kw).eval()


# ----------------------------------------------------------------- adapters
class LLaVA16:
    dirname = "llava-1.6-vicuna-7b"
    def __init__(self):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        self.proc = LlavaNextProcessor.from_pretrained(str(M/self.dirname))
        self.model = load_fp16(LlavaNextForConditionalGeneration, str(M/self.dirname))
    @torch.no_grad()
    def generate(self, image, text):
        prompt = f"USER: <image>\n{text} ASSISTANT:"
        inp = self.proc(images=image, text=prompt, return_tensors="pt").to(DEV, torch.float16)
        out = self.model.generate(**inp, max_new_tokens=args.maxnew, do_sample=False)
        s = self.proc.batch_decode(out, skip_special_tokens=True)[0]
        return s.split("ASSISTANT:")[-1].strip()
    def caption(self, image):
        return self.generate(image, CAP_PROMPT)
    def rewrite(self, image, cap, flagged):
        t = (f"{cap}\n\nThe following are NOT actually in the image: {', '.join(flagged)}. "
             f"Rewrite the description to remove any mention of them, keeping everything else accurate and fluent.")
        return self.generate(image, t)


class InstructBLIP:
    dirname = "instructblip-7b"
    def __init__(self):
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        self.proc = InstructBlipProcessor.from_pretrained(str(M/self.dirname))
        self.model = load_fp16(InstructBlipForConditionalGeneration, str(M/self.dirname))
    @torch.no_grad()
    def generate(self, image, text):
        inp = self.proc(images=image, text=text, return_tensors="pt").to(DEV, torch.float16)
        out = self.model.generate(**inp, max_new_tokens=args.maxnew, do_sample=False,
                                  num_beams=1, min_length=8)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    def caption(self, image):
        return self.generate(image, CAP_PROMPT)
    def rewrite(self, image, cap, flagged):
        t = (f"This description of the image is partly wrong: \"{cap}\" "
             f"The following are NOT in the image: {', '.join(flagged)}. "
             f"Write a corrected detailed description without them.")
        return self.generate(image, t)


class QwenVL:
    dirname = "qwen-vl-chat"
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(M/self.dirname), trust_remote_code=True)
        self.model = load_fp16(AutoModelForCausalLM, str(M/self.dirname),
                               trust_remote_code=True, fp16=True)
        self.model.generation_config.do_sample = False
        self.model.generation_config.max_new_tokens = args.maxnew
    @torch.no_grad()
    def _chat(self, image_path, text):
        q = self.tok.from_list_format([{"image": str(image_path)}, {"text": text}])
        out, _ = self.model.chat(self.tok, query=q, history=None)
        return out.strip()
    def caption(self, image_path):
        return self._chat(image_path, CAP_PROMPT)
    def rewrite(self, image_path, cap, flagged):
        t = (f"{cap}\n\nThe following are NOT actually in the image: {', '.join(flagged)}. "
             f"Rewrite the description in English to remove any mention of them, keeping everything else accurate.")
        return self._chat(image_path, t)


class LLaVA15(LLaVA16):
    dirname = "llava-1.5-7b"
    def __init__(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(str(M/self.dirname))
        self.model = load_fp16(LlavaForConditionalGeneration, str(M/self.dirname))


ADAPTERS = {"llava15": LLaVA15, "llava16": LLaVA16, "instructblip": InstructBLIP, "qwenvl": QwenVL}
PATH_BASED = {"qwenvl"}          # qwen takes image paths, others PIL images


import re as _re
def sentence_remove(cap, flagged):
    """Backbone-agnostic revise: drop sentences that mention a flagged object."""
    if not flagged:
        return cap
    fset = set(flagged)
    parts = _re.split(r'(?<=[.!?])\s+', cap.strip())
    kept = [s for s in parts if not (set(_extract_objects(s)) & fset)]
    return (" ".join(kept).strip()) or cap


def free(x):
    del x; gc.collect(); torch.cuda.empty_cache()


def load_image(bk, p):
    return str(p) if bk in PATH_BASED else Image.open(p).convert("RGB")


# ----------------------------------------------------------------- pipeline (single-pass)
gt_objects, image_ids = _load_gt_objects()
test_ids = [i for i in image_ids[2000:2000+args.n] if paths.coco_val2014_image(i).exists()]
print(f"backbone={args.backbone}  images={len(test_ids)}", flush=True)

# load backbone + G-DINO co-resident (both fit in 24 GB); no unload/reload
bk = ADAPTERS[args.backbone]()
from transformers import AutoProcessor, GroundingDinoForObjectDetection
gdir = str(M.parent/"tools"/"grounding_dino")
gdproc = AutoProcessor.from_pretrained(gdir)
gdino = GroundingDinoForObjectDetection.from_pretrained(gdir).to(DEV).eval()

@torch.no_grad()
def gscore(image, obj):
    inp = gdproc(images=image, text=obj.lower()+".", return_tensors="pt").to(DEV)
    out = gdino(**inp)
    for kw in ("threshold", "box_threshold"):
        try:
            res = gdproc.post_process_grounded_object_detection(
                out, inp["input_ids"], **{kw: 0.05}, text_threshold=0.05,
                target_sizes=[image.size[::-1]])[0]
            break
        except TypeError:
            continue
    return float(res["scores"].max()) if len(res["scores"]) else 0.0

caps, flags, rev = {}, {}, {}
for k, iid in enumerate(test_ids):
    ip = paths.coco_val2014_image(iid)
    try:
        cap = bk.caption(load_image(args.backbone, ip))
        caps[iid] = cap
        im = Image.open(ip).convert("RGB")
        fl = [o for o in _extract_objects(cap) if gscore(im, o) < args.thr]
        flags[iid] = fl
        if fl:
            rev[iid] = bk.rewrite(load_image(args.backbone, ip), cap, fl)
    except Exception as e:
        print(f"  skip {iid}: {type(e).__name__} {str(e)[:70]}", flush=True)
        torch.cuda.empty_cache()
    if (k+1) % 25 == 0:
        torch.cuda.empty_cache(); print(f"  {k+1}/{len(test_ids)}", flush=True)
nflag = sum(1 for v in flags.values() if v)
print(f"done: {len(caps)} captions, {nflag} flagged, {len(rev)} revised", flush=True)
free(gdino); free(bk)

# --- score ---
def stats(getcap):
    hm = tm = hc = tp = 0; ln = 0
    for iid, cap in caps.items():
        c = getcap(iid, cap); gold = gt_objects.get(iid, set())
        m_ = _extract_objects(c); hal = [o for o in m_ if o not in gold]
        tm += len(m_); hm += len(hal); hc += 1 if hal else 0
        tp += len(set(m_) & gold); ln += len(c.split())
    n = len(caps)
    return dict(ci=hm/tm if tm else 0, cs=hc/n, tpc=tp/n, ln=ln/n)

sv = stats(lambda i, c: c)
sr = stats(lambda i, c: rev.get(i, c))                               # self-rewrite
ss = stats(lambda i, c: sentence_remove(c, flags.get(i, [])))        # sentence-removal
outp = paths.PRED_DIR / f"xbb_{args.backbone}.jsonl"
with open(outp, "w", encoding="utf-8") as f:
    for iid, cap in caps.items():
        f.write(json.dumps({"image_id": int(iid), "vanilla": cap,
                            "flags": flags.get(iid, []), "revised": rev.get(iid),
                            "sent_removed": sentence_remove(cap, flags.get(iid, []))}) + "\n")

print(f"\n=== {args.backbone}  N={len(caps)} flagged={nflag} revised={len(rev)} ===")
print(f"                 CHAIR-i  CHAIR-s  true-obj/cap  len")
print(f"vanilla          {sv['ci']:.4f}  {sv['cs']:.4f}   {sv['tpc']:.2f}      {sv['ln']:.0f}")
print(f"self-rewrite     {sr['ci']:.4f}  {sr['cs']:.4f}   {sr['tpc']:.2f}      {sr['ln']:.0f}   "
      f"(dCHAIR-i {sr['ci']-sv['ci']:+.4f})")
print(f"sentence-remove  {ss['ci']:.4f}  {ss['cs']:.4f}   {ss['tpc']:.2f}      {ss['ln']:.0f}   "
      f"(dCHAIR-i {ss['ci']-sv['ci']:+.4f})")
print(f"saved -> {outp}")
