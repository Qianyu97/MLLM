"""Generative PGD via CONTRASTIVE decoding (VCD-style) on CHAIR.
Two-branch greedy decode: logits_final = (1+a)*logits(image) - a*logits(noised_image),
with adaptive-plausibility masking. Subtracting the noised-image branch removes the
language/co-occurrence prior that drives hallucination (the correlated-bias failure
mode that defeated external CLIP grounding). Training-free, image-conditioned.
"""
import sys, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, torch.nn.functional as F
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_chair import _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--beta", type=float, default=0.1, help="adaptive plausibility cutoff")
ap.add_argument("--noise", type=float, default=0.6, help="gaussian noise std on pixel_values")
ap.add_argument("--maxnew", type=int, default=64)
args = ap.parse_args()

cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"
torch.manual_seed(0)
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
tok = lproc.tokenizer
EOS = tok.eos_token_id
PROMPT = "USER: <image>\nDescribe this image in detail. ASSISTANT:"


@torch.no_grad()
def greedy_plain(img):
    inp = lproc(images=img, text=PROMPT, return_tensors="pt").to(DEV, torch.float16)
    out = llava.generate(**inp, max_new_tokens=args.maxnew, do_sample=False, num_beams=1)
    return lproc.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()


@torch.no_grad()
def greedy_vcd(img):
    a = args.alpha
    inp = lproc(images=img, text=PROMPT, return_tensors="pt").to(DEV, torch.float16)
    # noised image branch: same everything, gaussian-noised pixel_values
    pv = inp["pixel_values"]
    pv_noise = (pv + args.noise * torch.randn_like(pv)).to(torch.float16)
    ids = inp["input_ids"]; attn = inp["attention_mask"]
    o1 = llava(input_ids=ids, attention_mask=attn, pixel_values=pv, use_cache=True)
    o2 = llava(input_ids=ids, attention_mask=attn, pixel_values=pv_noise, use_cache=True)
    kv1, kv2 = o1.past_key_values, o2.past_key_values
    l1 = o1.logits[:, -1, :].float(); l2 = o2.logits[:, -1, :].float()
    gen = []
    for _ in range(args.maxnew):
        # adaptive plausibility: keep only tokens within beta of the image-branch max
        cutoff = l1.max() + torch.log(torch.tensor(args.beta, device=DEV))
        mask = l1 < cutoff
        cd = (1 + a) * l1 - a * l2
        cd = cd.masked_fill(mask, float("-inf"))
        nt = int(cd.argmax(-1))
        if nt == EOS:
            break
        gen.append(nt)
        ntt = torch.tensor([[nt]], device=DEV)
        o1 = llava(input_ids=ntt, past_key_values=kv1, use_cache=True)
        o2 = llava(input_ids=ntt, past_key_values=kv2, use_cache=True)
        kv1, kv2 = o1.past_key_values, o2.past_key_values
        l1 = o1.logits[:, -1, :].float(); l2 = o2.logits[:, -1, :].float()
    return tok.decode(gen, skip_special_tokens=True).strip()


gt_objects, image_ids = _load_gt_objects()
test_ids = image_ids[2000:2000+args.n]

# sanity: show 2 captions
for iid in test_ids[:2]:
    ip = paths.coco_val2014_image(iid)
    if ip.exists():
        im = Image.open(ip).convert("RGB")
        print("VANILLA:", greedy_plain(im)[:160], flush=True)
        print("VCD    :", greedy_vcd(im)[:160], flush=True)
        print("---", flush=True)

def stats(rows):
    hm=tm=hc=tp=0
    for m_, gold in rows:
        hal=[o for o in m_ if o not in gold]; tm+=len(m_); hm+=len(hal); hc+=1 if hal else 0; tp+=len(set(m_)&gold)
    n=len(rows); return dict(ci=hm/tm if tm else 0, cs=hc/n, opc=tm/n, tpc=tp/n)

van, vcd = [], []; vlen=cvlen=0
for k, iid in enumerate(test_ids):
    ip = paths.coco_val2014_image(iid)
    if not ip.exists(): continue
    im = Image.open(ip).convert("RGB")
    cv = greedy_plain(im); cc = greedy_vcd(im)
    gold = gt_objects.get(iid, set())
    van.append((_extract_objects(cv), gold)); vcd.append((_extract_objects(cc), gold))
    vlen += len(cv.split()); cvlen += len(cc.split())
    if (k+1) % 50 == 0: print(f"  {k+1}/{len(test_ids)}", flush=True)

sv, sc = stats(van), stats(vcd); n = len(van)
print(f"\nN={n} alpha={args.alpha} beta={args.beta} noise={args.noise}")
print(f"          CHAIR-i  CHAIR-s  obj/cap  true-obj/cap  len")
print(f"vanilla   {sv['ci']:.4f}  {sv['cs']:.4f}   {sv['opc']:.2f}     {sv['tpc']:.2f}      {vlen/n:.0f}")
print(f"VCD       {sc['ci']:.4f}  {sc['cs']:.4f}   {sc['opc']:.2f}     {sc['tpc']:.2f}      {cvlen/n:.0f}")
print(f"delta     {sc['ci']-sv['ci']:+.4f}  {sc['cs']-sv['cs']:+.4f}   {sc['opc']-sv['opc']:+.2f}     {sc['tpc']-sv['tpc']:+.2f}")
