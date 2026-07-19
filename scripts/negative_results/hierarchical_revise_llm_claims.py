"""Unified hierarchical detect-then-revise with LLM-BASED CLAIM EXTRACTION.
LLaVA decomposes its own caption into atomic OBJ/ATTR/REL claims (replacing the
error-prone vocab parser), each grounded by the specialist detector, unsupported
ones flagged, then LLaVA revises. Reports object CHAIR + attr/rel flags + examples."""
import sys, argparse, re, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, torch.nn.functional as F
from PIL import Image
from transformers import (LlavaForConditionalGeneration, AutoProcessor,
                          CLIPModel, CLIPProcessor, GroundingDinoForObjectDetection)
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_chair import _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300); ap.add_argument("--sanity", type=int, default=0)
ap.add_argument("--obj_thr", type=float, default=0.30); ap.add_argument("--attr_thr", type=float, default=-0.03)
ap.add_argument("--rel_thr", type=float, default=0.02)
args = ap.parse_args()
cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))
gdir = str(M.parent/"tools"/"grounding_dino"); gdproc = AutoProcessor.from_pretrained(gdir)
gdino = GroundingDinoForObjectDetection.from_pretrained(gdir).to(DEV).eval()

_CONTACT = {"on","holding","riding","wearing","sitting on","standing on","on top of","carrying","eating"}
_VERT = {"above": 1, "over": 1, "below": -1, "under": -1, "beneath": -1}   # sign of dy=(objcy-subjcy)
_HORZ = {"left of": 1, "to the left of": 1, "right of": -1, "to the right of": -1}  # sign of dx

@torch.no_grad()
def gen(im, prompt, mx=96):
    inp = lproc(images=im, text=prompt, return_tensors="pt").to(DEV, torch.float16)
    out = llava.generate(**inp, max_new_tokens=mx, do_sample=False, num_beams=1)
    return lproc.batch_decode(out, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

@torch.no_grad()
def gbox(image, phrase):
    inp = gdproc(images=image, text=phrase.lower()+".", return_tensors="pt").to(DEV)
    out = gdino(**inp)
    for kw in ("threshold","box_threshold"):
        try:
            res = gdproc.post_process_grounded_object_detection(out, inp["input_ids"], **{kw:0.05}, text_threshold=0.05, target_sizes=[image.size[::-1]])[0]; break
        except TypeError: continue
    if len(res["scores"])==0: return None, 0.0
    i=int(res["scores"].argmax()); return [float(v) for v in res["boxes"][i]], float(res["scores"][i])
@torch.no_grad()
def crop_emb(im):
    pin=cproc(images=im,return_tensors="pt").to(DEV,torch.float16)
    return F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output),dim=-1)
_TE={}
@torch.no_grad()
def txt(t):
    if t not in _TE:
        tin=cproc(text=[t],return_tensors="pt",padding=True).to(DEV)
        _TE[t]=F.normalize(clip.text_projection(clip.text_model(input_ids=tin["input_ids"],attention_mask=tin["attention_mask"]).pooler_output),dim=-1)
    return _TE[t]
def crop(image,b,pad=0.1):
    W,H=image.size;x0,y0,x1,y1=b;w,h=x1-x0,y1-y0
    x0=max(0,x0-pad*w);y0=max(0,y0-pad*h);x1=min(W,x1+pad*w);y1=min(H,y1+pad*h)
    return image if (x1-x0<5 or y1-y0<5) else image.crop((x0,y0,x1,y1))
def overlap(b1,b2):
    ax0,ay0,ax1,ay1=b1;bx0,by0,bx1,by1=b2
    iw=max(0,min(ax1,bx1)-max(ax0,bx0));ih=max(0,min(ay1,by1)-max(ay0,by0))
    a1=(ax1-ax0)*(ay1-ay0);a2=(bx1-bx0)*(by1-by0);return iw*ih/(min(a1,a2)+1e-6)
def cen(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)

EXTRACT = ('USER: <image>\nHere is a description of the image: "{cap}"\n\n'
           'Decompose the description into atomic claims, one per line, NO numbering, using EXACTLY these formats '
           '(note the | separators):\n'
           'OBJ: dog\nATTR: dog | brown\nREL: dog | on | sofa\n\n'
           'List every object, every attribute (as "object | attribute"), and every relation '
           '(as "subject | relation | object") stated in the description. ASSISTANT:')

def extract_claims(im, cap):
    raw = gen(im, EXTRACT.format(cap=cap.replace('"',"'")), mx=200)
    objs, attrs, rels = [], [], []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()   # strip "1. " / "2) " numbering
        up = line.upper()
        if up.startswith("OBJ:"):
            o = line[4:].strip().lower()
            if o: objs.append(o)
        elif up.startswith("ATTR:"):
            body = line[5:].strip()
            if "|" in body:
                p = [x.strip().lower() for x in body.split("|") if x.strip()]
                if len(p) >= 2 and p[0] and p[-1]: attrs.append((p[0], p[-1]))  # obj | ... | attribute
        elif up.startswith("REL:"):
            body = line[4:].strip()
            if body.count("|") >= 2:
                p = [x.strip().lower() for x in body.split("|")]
                if p[0] and p[1] and p[2]: rels.append((p[0], p[1], p[2]))
    # dedup + cap (avoid degenerate repetition loops)
    objs = list(dict.fromkeys(objs))[:15]
    attrs = list(dict.fromkeys(attrs))[:15]
    rels = list(dict.fromkeys(rels))[:15]
    return objs, attrs, rels, raw

def flag(image, objs, attrs, rels):
    fo=[]; box={};
    def getbox(n):
        if n not in box: box[n]=gbox(image,n)
        return box[n]
    for o in objs:
        _,sc=getbox(o)
        if sc<args.obj_thr: fo.append(o)
    fa=[]
    for n,a in attrs:
        b,sc=getbox(n)
        if sc<args.obj_thr or not b: continue
        reg=crop(image,b); ci=crop_emb(reg)
        if float((ci@txt(f"a photo of a {a} {n}").T).item()-(ci@txt(f"a photo of a {n}").T).item())<args.attr_thr: fa.append((n,a))
    fr=[]
    for s,r,o in rels:
        bs,ss=getbox(s); bo,so=getbox(o)
        if ss<args.obj_thr or so<args.obj_thr or not bs or not bo: continue
        rr=r.strip()
        if any(c in rr for c in _CONTACT):
            if overlap(bs,bo)<args.rel_thr: fr.append((s,r,o))
        else:
            key=next((k for k in list(_VERT)+list(_HORZ) if k in rr), None)
            if key:
                (scx,scy),(ocx,ocy)=cen(bs),cen(bo)
                if key in _VERT:
                    val=ocy-scy; exp=_VERT[key]
                    if (val>0)!=(exp>0) and abs(val)>0.05*image.size[1]: fr.append((s,r,o))
                else:
                    val=ocx-scx; exp=_HORZ[key]
                    if (val>0)!=(exp>0) and abs(val)>0.05*image.size[0]: fr.append((s,r,o))
    return fo,fa,fr

def revise(image, cap, fo,fa,fr):
    parts=[]
    if fo: parts.append("objects not present: "+", ".join(fo))
    if fa: parts.append("wrong attributes: "+", ".join(f"{n} is {a}" for n,a in fa))
    if fr: parts.append("wrong relations: "+", ".join(f"{s} {r} {o}" for s,r,o in fr))
    instr=("USER: <image>\n"+cap+"\n\nThe following are inaccurate: "+"; ".join(parts)+
           ". Rewrite the description to fix/remove these while keeping everything else accurate and fluent. ASSISTANT:")
    return gen(image, instr)

gt_objects, image_ids=_load_gt_objects(); test_ids=image_ids[2000:2000+args.n]
CAP="USER: <image>\nDescribe this image in detail. ASSISTANT:"

if args.sanity:
    for iid in test_ids[:args.sanity]:
        ip=paths.coco_val2014_image(iid)
        if not ip.exists(): continue
        im=Image.open(ip).convert("RGB"); cap=gen(im,CAP)
        objs,attrs,rels,raw=extract_claims(im,cap)
        print("CAP:",cap[:140]); print("RAW EXTRACTION:\n",raw); print("  parsed objs",objs,"attrs",attrs,"rels",rels); print("---",flush=True)
    sys.exit(0)

van,rev=[],[]; no=na=nr=nrev=0; examples=[]
for k,iid in enumerate(test_ids):
    try:
        ip=paths.coco_val2014_image(iid)
        if not ip.exists(): continue
        im=Image.open(ip).convert("RGB"); gold=gt_objects.get(iid,set()); cap=gen(im,CAP)
        objs,attrs,rels,_=extract_claims(im,cap); fo,fa,fr=flag(im,objs,attrs,rels)
        if fo or fa or fr:
            rc=revise(im,cap,fo,fa,fr); nrev+=1; no+=len(fo);na+=len(fa);nr+=len(fr)
            if (fa or fr) and len(examples)<8:
                examples.append({"attr":fa,"rel":fr,"obj":fo,"vanilla":cap[:180],"revised":rc[:180]})
        else: rc=cap
        van.append((_extract_objects(cap),gold)); rev.append((_extract_objects(rc),gold))
    except Exception as e:
        print(f"  skip {iid}: {str(e)[:60]}",flush=True); torch.cuda.empty_cache()
    if (k+1)%50==0: torch.cuda.empty_cache(); print(f"  {k+1}/{len(test_ids)}",flush=True)

def stats(rows):
    hm=tm=hc=tp=0
    for m_,gold in rows:
        hal=[o for o in m_ if o not in gold];tm+=len(m_);hm+=len(hal);hc+=1 if hal else 0;tp+=len(set(m_)&gold)
    n=len(rows);return dict(ci=hm/tm if tm else 0,cs=hc/n,tpc=tp/n)
sv,sr=stats(van),stats(rev);n=len(van)
print(f"\nN={n}  revised {nrev}/{n} | flagged obj={no} attr={na} rel={nr}")
print(f"              CHAIR-i  CHAIR-s  true-obj/cap")
print(f"vanilla       {sv['ci']:.4f}  {sv['cs']:.4f}   {sv['tpc']:.2f}")
print(f"hier-revise   {sr['ci']:.4f}  {sr['cs']:.4f}   {sr['tpc']:.2f}")
print(f"delta         {sr['ci']-sv['ci']:+.4f}  {sr['cs']-sv['cs']:+.4f}   {sr['tpc']-sv['tpc']:+.2f}")
print("\n=== attr/rel revision examples ===")
for e in examples: print(json.dumps(e,ensure_ascii=False))
