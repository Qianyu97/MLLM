"""UNIFIED HIERARCHICAL detect-then-revise: flag hallucinated OBJECTS, ATTRIBUTES,
and RELATIONS in a caption via the 3 specialist groundings, then LLaVA self-rewrite.
Objects measured quantitatively on CHAIR; attr/rel flagging demonstrated (counts +
saved qualitative examples). Differentiates from object-centric Woodpecker/LURE."""
import sys, argparse, re, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, torch.nn.functional as F
from PIL import Image
from transformers import (LlavaForConditionalGeneration, AutoProcessor,
                          CLIPModel, CLIPProcessor, GroundingDinoForObjectDetection)
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_chair import _SYNONYMS, _extract_objects, _load_gt_objects

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--obj_thr", type=float, default=0.30)
ap.add_argument("--attr_thr", type=float, default=-0.02)
ap.add_argument("--rel_thr", type=float, default=0.03)
args = ap.parse_args()
cfg = load_config(); M = paths.MODELS_ROOT; DEV = "cuda"
llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))
gdir = str(M.parent/"tools"/"grounding_dino"); gdproc = AutoProcessor.from_pretrained(gdir)
gdino = GroundingDinoForObjectDetection.from_pretrained(gdir).to(DEV).eval()

_ATTR = {"red","blue","green","yellow","black","white","orange","purple","pink","brown","gray","grey",
         "large","small","big","tiny","tall","short","long","round","square","old","new","young","wooden",
         "metal","plastic","bright","dark","shiny","wet","dry","open","closed","empty","full","striped"}
_REL = {"on","under","above","below","behind","near","beside","inside","holding","riding","wearing",
        "sitting on","standing on","next to","in front of","on top of"}
_REL_CONTACT = {"on","holding","riding","wearing","sitting on","standing on","on top of","next to"}
# object surfaces -> canonical (single-word surfaces for simple adjacency parsing)
_OBJ_SURF = {k: v for k, v in _SYNONYMS.items() if " " not in k}

@torch.no_grad()
def gbox(image, phrase):
    inp = gdproc(images=image, text=phrase.lower()+".", return_tensors="pt").to(DEV)
    out = gdino(**inp)
    for kw in ("threshold","box_threshold"):
        try:
            res = gdproc.post_process_grounded_object_detection(out, inp["input_ids"], **{kw:0.05}, text_threshold=0.05, target_sizes=[image.size[::-1]])[0]; break
        except TypeError: continue
    if len(res["scores"])==0: return None, 0.0
    i = int(res["scores"].argmax())
    return [float(v) for v in res["boxes"][i]], float(res["scores"][i])

@torch.no_grad()
def crop_emb(im):
    pin = cproc(images=im, return_tensors="pt").to(DEV, torch.float16)
    return F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output), dim=-1)
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

def parse(caption):
    toks = re.sub(r"[^a-z0-9 ]+"," ",caption.lower()).split()
    attrs=[]; rels=[]
    for i,t in enumerate(toks):
        if t in _ATTR and i+1<len(toks) and toks[i+1] in _OBJ_SURF:
            attrs.append((t, toks[i+1]))
    # relations: nearest object before/after a relation word (single or two-word)
    joined=" "+ " ".join(toks) +" "
    for rel in sorted(_REL,key=len,reverse=True):
        for m in re.finditer(rf" {re.escape(rel)} ", joined):
            left=joined[:m.start()].split(); right=joined[m.end():].split()
            s=next((w for w in reversed(left) if w in _OBJ_SURF), None)
            o=next((w for w in right if w in _OBJ_SURF), None)
            if s and o and s!=o: rels.append((s,rel,o))
    return attrs, rels

@torch.no_grad()
def gen(im, prompt, mx=80):
    inp=lproc(images=im,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    out=llava.generate(**inp,max_new_tokens=mx,do_sample=False,num_beams=1)
    return lproc.batch_decode(out,skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

def flag_all(image, caption):
    objs=_extract_objects(caption); attrs,rels=parse(caption)
    f_obj=[];
    for o in objs:
        _,sc=gbox(image,o)
        if sc<args.obj_thr: f_obj.append(o)
    f_attr=[]
    for a,n in attrs:
        b,sc=gbox(image,n)
        if sc<args.obj_thr: continue  # noun itself absent -> object layer handles it
        reg=crop(image,b) if b else image; ci=crop_emb(reg)
        contrast=float((ci@txt(f"a photo of a {a} {n}").T).item()-(ci@txt(f"a photo of a {n}").T).item())
        if contrast<args.attr_thr: f_attr.append((a,n))
    f_rel=[]
    for s,r,o in rels:
        bs,ss=gbox(image,s); bo,so=gbox(image,o)
        if ss<args.obj_thr or so<args.obj_thr or not bs or not bo: continue
        if r in _REL_CONTACT and overlap(bs,bo)<args.rel_thr: f_rel.append((s,r,o))
    return f_obj,f_attr,f_rel

def revise(image, caption, f_obj, f_attr, f_rel):
    parts=[]
    if f_obj: parts.append("objects not present: "+", ".join(f_obj))
    if f_attr: parts.append("wrong attributes: "+", ".join(f"{a} {n}" for a,n in f_attr))
    if f_rel: parts.append("wrong relations: "+", ".join(f"{s} {r} {o}" for s,r,o in f_rel))
    instr=("USER: <image>\n"+caption+"\n\nThe following are inaccurate: "+"; ".join(parts)+
           ". Rewrite the description to fix/remove these while keeping everything else accurate and fluent. ASSISTANT:")
    return gen(image, instr)

gt_objects, image_ids=_load_gt_objects(); test_ids=image_ids[2000:2000+args.n]
van, rev=[],[]; no=na=nr=nrev=0; examples=[]
for k,iid in enumerate(test_ids):
    try:
        ip=paths.coco_val2014_image(iid)
        if not ip.exists(): continue
        im=Image.open(ip).convert("RGB"); gold=gt_objects.get(iid,set())
        cap=gen(im,"USER: <image>\nDescribe this image in detail. ASSISTANT:")
        fo,fa,fr=flag_all(im,cap)
        if fo or fa or fr:
            rc=revise(im,cap,fo,fa,fr); nrev+=1; no+=len(fo);na+=len(fa);nr+=len(fr)
            if (fa or fr) and len(examples)<6:
                examples.append({"flagged_attr":fa,"flagged_rel":fr,"flagged_obj":fo,"vanilla":cap[:200],"revised":rc[:200]})
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
print(f"\nN={n}  revised {nrev}/{n} imgs | flagged objects={no} attributes={na} relations={nr}")
print(f"              CHAIR-i  CHAIR-s  true-obj/cap")
print(f"vanilla       {sv['ci']:.4f}  {sv['cs']:.4f}   {sv['tpc']:.2f}")
print(f"hier-revise   {sr['ci']:.4f}  {sr['cs']:.4f}   {sr['tpc']:.2f}")
print(f"delta         {sr['ci']-sv['ci']:+.4f}  {sr['cs']-sv['cs']:+.4f}   {sr['tpc']-sv['tpc']:+.2f}")
print("\n=== qualitative attr/rel revision examples ===")
for e in examples:
    print(json.dumps(e,ensure_ascii=False))
