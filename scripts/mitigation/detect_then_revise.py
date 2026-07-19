"""REAL detect-then-revise: generate caption -> flag hallucinated objects via
G-DINO per-object detection -> LLaVA self-rewrite removing flagged objects.
Measures REAL CHAIR on the revised text + recall/length (not the set-removal ceiling)."""
import sys, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, GroundingDinoForObjectDetection
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_chair import _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--thr", type=float, default=0.30)
ap.add_argument("--save", default=None, help="jsonl path to save {image_id, vanilla, revised}")
args = ap.parse_args()
import json as _json
_caps = []
cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
gdir = str(M.parent/"tools"/"grounding_dino"); gdproc = AutoProcessor.from_pretrained(gdir)
gdino = GroundingDinoForObjectDetection.from_pretrained(gdir).to(DEV).eval()

@torch.no_grad()
def gen(im, prompt, mx=80):
    inp = lproc(images=im, text=prompt, return_tensors="pt").to(DEV, torch.float16)
    out = llava.generate(**inp, max_new_tokens=mx, do_sample=False, num_beams=1)
    return lproc.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

@torch.no_grad()
def gscore(image, obj):
    inp = gdproc(images=image, text=obj.lower()+".", return_tensors="pt").to(DEV)
    out = gdino(**inp)
    for kw in ("threshold", "box_threshold"):
        try:
            res = gdproc.post_process_grounded_object_detection(out, inp["input_ids"], **{kw: 0.05}, text_threshold=0.05, target_sizes=[image.size[::-1]])[0]; break
        except TypeError: continue
    return float(res["scores"].max()) if len(res["scores"]) else 0.0

CAP_PROMPT = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
gt_objects, image_ids = _load_gt_objects(); test_ids = image_ids[2000:2000+args.n]

van, rev = [], []; vlen=rlen=0; nflag=0; nrevised=0
for k, iid in enumerate(test_ids):
    try:
        ip = paths.coco_val2014_image(iid)
        if not ip.exists(): continue
        im = Image.open(ip).convert("RGB"); gold = gt_objects.get(iid, set())
        cap = gen(im, CAP_PROMPT)
        mentioned = _extract_objects(cap)
        flagged = [c for c in mentioned if gscore(im, c) < args.thr]
        if flagged:
            nflag += len(flagged); nrevised += 1
            rp = (f"USER: <image>\n{cap}\n\nThe following are NOT actually in the image: "
                  f"{', '.join(flagged)}. Rewrite the description to remove any mention of them, "
                  f"keeping everything else accurate and fluent. ASSISTANT:")
            rcap = gen(im, rp)
        else:
            rcap = cap
        van.append((_extract_objects(cap), gold)); rev.append((_extract_objects(rcap), gold))
        vlen += len(cap.split()); rlen += len(rcap.split())
        if args.save: _caps.append({"image_id": int(iid), "vanilla": cap, "revised": rcap})
    except Exception as e:
        print(f"  skip {iid}: {str(e)[:70]}", flush=True); torch.cuda.empty_cache()
    if (k+1) % 60 == 0: torch.cuda.empty_cache(); print(f"  {k+1}/{len(test_ids)}", flush=True)

def stats(rows):
    hm=tm=hc=tp=0
    for m_, gold in rows:
        hal=[o for o in m_ if o not in gold]; tm+=len(m_); hm+=len(hal); hc+=1 if hal else 0; tp+=len(set(m_)&gold)
    n=len(rows); return dict(ci=hm/tm if tm else 0, cs=hc/n, opc=tm/n, tpc=tp/n)
sv, sr = stats(van), stats(rev); n=len(van)
print(f"\nN={n} thr={args.thr}  revised {nrevised}/{n} imgs, {nflag} objects flagged")
print(f"              CHAIR-i  CHAIR-s  obj/cap  true-obj/cap  len")
print(f"vanilla       {sv['ci']:.4f}  {sv['cs']:.4f}   {sv['opc']:.2f}     {sv['tpc']:.2f}      {vlen/n:.0f}")
print(f"detect-revise {sr['ci']:.4f}  {sr['cs']:.4f}   {sr['opc']:.2f}     {sr['tpc']:.2f}      {rlen/n:.0f}")
print(f"delta         {sr['ci']-sv['ci']:+.4f}  {sr['cs']-sv['cs']:+.4f}   {sr['opc']-sv['opc']:+.2f}     {sr['tpc']-sv['tpc']:+.2f}")
if args.save:
    with open(args.save, "w", encoding="utf-8") as f:
        for r in _caps: f.write(_json.dumps(r) + "\n")
    print(f"saved {len(_caps)} caption pairs -> {args.save}", flush=True)
