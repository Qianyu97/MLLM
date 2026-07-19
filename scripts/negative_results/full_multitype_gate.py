"""FULL MULTI-TYPE GATE: for object/attribute/relation, fuse vanilla LLaVA yes/no
with the specialist grounding, at MATCHED yes-ratio (cal/test split). Prove Acc/F1
gains for all three types => genuine type-specific de-hallucination, not a shift."""
import sys, json, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import (LlavaForConditionalGeneration, AutoProcessor,
                          CLIPModel, CLIPProcessor, GroundingDinoForObjectDetection)
from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.eval.eval_hhd import (parse_pope, parse_amber_attr, parse_amber_rel,
                                 _load_jsonl, _amber_truth)

cfg = load_config(); M = paths.MODELS_ROOT; TOOLS = M.parent / "tools"; DEV = "cuda"
random.seed(0); np.random.seed(0)

llava = LlavaForConditionalGeneration.from_pretrained(str(M/"llava-1.5-7b"), torch_dtype=torch.float16).to(DEV).eval()
lproc = AutoProcessor.from_pretrained(str(M/"llava-1.5-7b"))
clip = CLIPModel.from_pretrained(str(M/"clip-vit-l14-336"), torch_dtype=torch.float16).to(DEV).eval()
cproc = CLIPProcessor.from_pretrained(str(M/"clip-vit-l14-336"))
gdproc = AutoProcessor.from_pretrained(str(TOOLS/"grounding_dino"))
gdino = GroundingDinoForObjectDetection.from_pretrained(str(TOOLS/"grounding_dino")).to(DEV).eval()

tok = lproc.tokenizer
def _ids(ws):
    o=[]
    for w in ws:
        o += tok(w, add_special_tokens=False).input_ids
    return list(set(o))
YES=_ids(["Yes","yes"," Yes"," yes"]); NO=_ids(["No","no"," No"," no"])

@torch.no_grad()
def llava_pyes(im, q):
    prompt=f"USER: <image>\n{q} Please answer with only Yes or No. ASSISTANT:"
    inp=lproc(images=im,text=prompt,return_tensors="pt").to(DEV,torch.float16)
    out=llava.generate(**inp,max_new_tokens=1,do_sample=False,output_scores=True,return_dict_in_generate=True)
    lg=out.scores[0][0].float(); p=torch.softmax(torch.stack([lg[YES].max(),lg[NO].max()]),0)
    return float(p[0])

_ie={}
@torch.no_grad()
def cimg(path):
    k=str(path)
    if k not in _ie:
        im=Image.open(path).convert("RGB"); pin=cproc(images=im,return_tensors="pt").to(DEV,torch.float16)
        _ie[k]=F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output),dim=-1)
    return _ie[k]
@torch.no_grad()
def cimg_crop(im):
    pin=cproc(images=im,return_tensors="pt").to(DEV,torch.float16)
    return F.normalize(clip.visual_projection(clip.vision_model(pixel_values=pin["pixel_values"]).pooler_output),dim=-1)
_te={}
@torch.no_grad()
def ctxt(t):
    if t not in _te:
        tin=cproc(text=[t],return_tensors="pt",padding=True).to(DEV)
        _te[t]=F.normalize(clip.text_projection(clip.text_model(input_ids=tin["input_ids"],attention_mask=tin["attention_mask"]).pooler_output),dim=-1)
    return _te[t]

@torch.no_grad()
def gbox(image, phrase):
    inp=gdproc(images=image,text=phrase.lower().strip()+".",return_tensors="pt").to(DEV)
    out=gdino(**inp)
    for kw in ("threshold","box_threshold"):
        try:
            res=gdproc.post_process_grounded_object_detection(out,inp["input_ids"],**{kw:0.15},text_threshold=0.15,target_sizes=[image.size[::-1]])[0]; break
        except TypeError: continue
    if len(res["scores"])==0: return None
    return [float(v) for v in res["boxes"][int(res["scores"].argmax())]]

def crop(image,b,pad=0.1):
    W,H=image.size; x0,y0,x1,y1=b; w,h=x1-x0,y1-y0
    x0=max(0,x0-pad*w);y0=max(0,y0-pad*h);x1=min(W,x1+pad*w);y1=min(H,y1+pad*h)
    return image if (x1-x0<5 or y1-y0<5) else image.crop((x0,y0,x1,y1))

def overlap(b1,b2):
    ax0,ay0,ax1,ay1=b1;bx0,by0,bx1,by1=b2
    iw=max(0,min(ax1,bx1)-max(ax0,bx0));ih=max(0,min(ay1,by1)-max(ay0,by0));inter=iw*ih
    a1=(ax1-ax0)*(ay1-ay0);a2=(bx1-bx0)*(by1-by0)
    return inter/(min(a1,a2)+1e-6)

# ---- collect rows per bed ----
def collect_pope(n):
    items=[]
    for _,qf in paths.POPE_SUBSETS.items(): items+=_load_jsonl(qf)
    random.shuffle(items); rows=[]
    for it in items:
        obj=parse_pope(it["text"]); img=paths.POPE_IMAGE_DIR/it["image"]
        if not obj or not img.exists(): continue
        im=Image.open(img).convert("RGB")
        g=float((cimg(img)@ctxt(f"a photo of a {obj}").T).item())
        rows.append((llava_pyes(im,it["text"]),g,1 if it["label"]=="yes" else 0))
        if len(rows)>=n: break
    return rows

def collect_attr(n):
    truth=_amber_truth(); items=json.load(open(paths.AMBER_Q_ATTRIBUTE,encoding="utf-8")); random.shuffle(items); rows=[]
    for it in items:
        gt=truth.get(int(it["id"])); p=parse_amber_attr(it["query"]); img=paths.AMBER_IMAGES/it["image"]
        if gt not in ("yes","no") or not p or not img.exists(): continue
        noun,attr=p; im=Image.open(img).convert("RGB"); b=gbox(im,noun); region=crop(im,b) if b else im
        ci=cimg_crop(region); g=float((ci@ctxt(f"a photo of a {attr} {noun}").T).item()-(ci@ctxt(f"a photo of a {noun}").T).item())
        rows.append((llava_pyes(im,it["query"]),g,1 if gt=="yes" else 0))
        if len(rows)>=n: break
    return rows

def collect_rel(n):
    truth=_amber_truth(); items=json.load(open(paths.AMBER_Q_RELATION,encoding="utf-8")); random.shuffle(items); rows=[]
    for it in items:
        gt=truth.get(int(it["id"])); p=parse_amber_rel(it["query"]); img=paths.AMBER_IMAGES/it["image"]
        if gt not in ("yes","no") or not p or not img.exists(): continue
        s_,_,o_=p; im=Image.open(img).convert("RGB"); bs=gbox(im,s_); bo=gbox(im,o_)
        g=overlap(bs,bo) if (bs and bo) else 0.0
        rows.append((llava_pyes(im,it["query"]),g,1 if gt=="yes" else 0))
        if len(rows)>=n: break
    return rows

def gate(name, rows):
    p=np.array([r[0] for r in rows]);g=np.array([r[1] for r in rows]);y=np.array([r[2] for r in rows])
    n=len(y);idx=np.random.permutation(n);cal,te=idx[:n//2],idx[n//2:]
    def st(pred,yy):
        acc=(pred==yy).mean();tp=((pred==1)&(yy==1)).sum();fp=((pred==1)&(yy==0)).sum();fn=((pred==0)&(yy==1)).sum()
        f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0;return acc,f1
    predv=(p[te]>=0.5).astype(int);accv,f1v=st(predv,y[te]);yrv=predv.mean()
    pz=(p-p[cal].mean())/(p[cal].std()+1e-8);gz=(g-g[cal].mean())/(g[cal].std()+1e-8)
    yr_t=(p[cal]>=0.5).mean();best=None
    for lam in np.linspace(0,3,31):
        fz=pz+lam*gz;thr=np.quantile(fz[cal],1-yr_t);pc=(fz[cal]>=thr).astype(int);ac=(pc==y[cal]).mean()
        if best is None or ac>best[0]: best=(ac,lam)
    lam=best[1];fz=pz+lam*gz;thr=np.quantile(fz[te],1-yrv);predf=(fz[te]>=thr).astype(int);accf,f1f=st(predf,y[te])
    print(f"[{name}] n={n} | vanilla Acc={accv:.4f} F1={f1v:.4f} yr={yrv:.3f} | grounding AUC={roc_auc_score(y[te],g[te]):.3f} "
          f"| FUSED(lam={lam:.2f}) Acc={accf:.4f} F1={f1f:.4f} yr={predf.mean():.3f} | dAcc={accf-accv:+.4f} dF1={f1f-f1v:+.4f}",flush=True)

print("collecting POPE (object)...",flush=True); gate("OLD object  POPE", collect_pope(2000))
print("collecting AMBER-attr...",flush=True); gate("ALD attribute AMBER", collect_attr(1400))
print("collecting AMBER-rel...",flush=True); gate("RLD relation  AMBER", collect_rel(1000))
