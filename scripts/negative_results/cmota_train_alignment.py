"""Scaled CM-OTA redesign: extract paired pooled features (once) -> train PVE/PLE
heads with the validated collapse-resistant loss -> save cmota.pt/pretrain_proj.pt
-> verify no-collapse + cross-modal retrieval on a held-out image split.

Images are read E:-or-G: (train2017); features cache to E:. Uses COCO train2017
captions (self-contained; avoids val2014 leakage into POPE/CHAIR).

Run (after weights downloaded):
  python train_align.py extract --n-pairs 40000
  python train_align.py train  --steps 3000 --batch 1024
"""
import os, sys, json, random, argparse, glob
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))  # repo-relative
import torch, torch.nn.functional as F

from cmpsa import paths
from cmpsa.config import load_config
from cmpsa.models.pve_ple import PVEHead, PLEHead
from cmpsa.models.cm_ota import cmota_align_loss

CACHE = os.path.expandvars(r"${CMPSA_DATA_ROOT}\cache\align_pairs.pt")
CAPS_TRAIN2017 = os.path.expandvars(r"${CMPSA_DATA_ROOT}\basic\coco\annotations\captions_train2017.json")
TRAIN_DIRS = [os.path.expandvars(r"${CMPSA_DATA_ROOT}\basic\coco\images\train2017"),
              r"G:\cmpsa_data\basic\coco\images\train2017"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def resolve(img_id):
    name = f"{int(img_id):012d}.jpg"
    for d in TRAIN_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _models_dir(cfg, key):
    return str(paths.MODELS_ROOT / getattr(cfg.models, key).local_dir)


# ------------------------------------------------------------------ extract
def extract(n_pairs):
    from PIL import Image
    from transformers import (CLIPImageProcessor, CLIPVisionModel,
                              AutoTokenizer, AutoModel)
    cfg = load_config()
    with open(CAPS_TRAIN2017, encoding="utf-8") as f:
        anns = json.load(f)["annotations"]
    random.Random(42).shuffle(anns)
    rows = []
    for a in anns:
        p = resolve(a["image_id"])
        cap = str(a.get("caption", "")).strip()
        if p and cap:
            rows.append((int(a["image_id"]), p, cap))
        if len(rows) >= n_pairs:
            break
    print(f"assembled {len(rows)} (image,caption) pairs; unique images "
          f"{len({r[0] for r in rows})}", flush=True)

    # ---- CLIP pass ----
    cdir = _models_dir(cfg, cfg.visual_backbone.key)
    print("loading CLIP from", cdir, flush=True)
    cproc = CLIPImageProcessor.from_pretrained(cdir)
    cmodel = CLIPVisionModel.from_pretrained(cdir, torch_dtype=torch.float16).to(DEV).eval()
    vfeat = torch.zeros(len(rows), cfg.visual_backbone.feature_dim, dtype=torch.float16)
    BS = 32
    with torch.no_grad():
        for i in range(0, len(rows), BS):
            imgs = [Image.open(r[1]).convert("RGB") for r in rows[i:i + BS]]
            inp = cproc(images=imgs, return_tensors="pt").to(DEV, torch.float16)
            out = cmodel(**inp).last_hidden_state[:, 1:, :].mean(1)   # [b,1024]
            vfeat[i:i + BS] = out.cpu()
            if i % (BS * 50) == 0:
                print(f"  clip {i}/{len(rows)}", flush=True)
    del cmodel; torch.cuda.empty_cache()

    # ---- Llama pass ----
    tdir = _models_dir(cfg, cfg.text_backbone.key)
    print("loading text backbone from", tdir, flush=True)
    tok = AutoTokenizer.from_pretrained(tdir)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tmodel = AutoModel.from_pretrained(tdir, torch_dtype=torch.float16,
                                       output_hidden_states=True).to(DEV).eval()
    lfeat = torch.zeros(len(rows), cfg.text_backbone.feature_dim, dtype=torch.float16)
    BS = 32
    with torch.no_grad():
        for i in range(0, len(rows), BS):
            caps = [r[2] for r in rows[i:i + BS]]
            enc = tok(caps, return_tensors="pt", padding=True, truncation=True,
                      max_length=64).to(DEV)
            hs = tmodel(**enc).hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).half()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            lfeat[i:i + BS] = pooled.cpu()
            if i % (BS * 50) == 0:
                print(f"  llama {i}/{len(rows)}", flush=True)
    del tmodel; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save({"v": vfeat, "l": lfeat,
                "img": torch.tensor([r[0] for r in rows])}, CACHE)
    print("saved cache ->", CACHE, flush=True)


# ------------------------------------------------------------------ train
def make_heads(cfg):
    pj = cfg.projection
    pve = PVEHead(cfg.visual_backbone.feature_dim, pj.psas_dim, pj.hidden_dim,
                  pj.min_logvar, pj.max_logvar).to(DEV)
    ple = PLEHead(cfg.text_backbone.feature_dim, pj.psas_dim, pj.hidden_dim,
                  pj.min_logvar, pj.max_logvar).to(DEV)
    return pve, ple


@torch.no_grad()
def evaluate(pve, ple, V, L, img):
    pve.eval(); ple.eval()
    uniq = torch.unique(img)
    # one image feature row per unique image
    first = {int(im): (img == im).nonzero()[0].item() for im in uniq}
    Vi = V[[first[int(im)] for im in uniq]].to(DEV)
    vm, _ = pve(Vi)
    lm, _ = ple(L.to(DEV))
    vn = F.normalize(vm, dim=-1); ln = F.normalize(lm, dim=-1)
    n = vn.shape[0]
    pc = float((vn @ vn.t()).sum() - n) / (n * (n - 1))
    id2row = {int(im): k for k, im in enumerate(uniq)}
    gt = torch.tensor([id2row[int(i)] for i in img], device=DEV)
    S = ln @ vn.t()
    rank = S.argsort(1, descending=True)
    r1 = float((rank[:, 0] == gt).float().mean())
    r5 = float((rank[:, :5] == gt[:, None]).any(1).float().mean())
    return dict(pairwise_cos=round(pc, 4), R1=round(r1, 3), R5=round(r5, 3),
                n_img=n, n_cap=len(gt))


def train(steps, batch, lr):
    cfg = load_config()
    data = torch.load(CACHE, weights_only=False)
    V, L, img = data["v"].float(), data["l"].float(), data["img"]
    uniq = torch.unique(img).tolist(); random.Random(0).shuffle(uniq)
    n_val = max(200, int(0.1 * len(uniq)))
    val_imgs = set(uniq[:n_val])
    tr = torch.tensor([i for i, im in enumerate(img.tolist()) if im not in val_imgs])
    va = torch.tensor([i for i, im in enumerate(img.tolist()) if im in val_imgs])
    print(f"cache: {len(img)} pairs, {len(uniq)} imgs | train {len(tr)} val {len(va)} "
          f"(val imgs {len(val_imgs)}) dev={DEV}", flush=True)

    pve, ple = make_heads(cfg)
    opt = torch.optim.AdamW(list(pve.parameters()) + list(ple.parameters()), lr=lr)
    Vtr, Ltr, Itr = V[tr].to(DEV), L[tr].to(DEV), img[tr].to(DEV)
    pve.train(); ple.train()
    N = len(tr)
    for s in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), device=DEV)
        vm, vl = pve(Vtr[idx]); lm, ll = ple(Ltr[idx])
        out = cmota_align_loss(vm, vl, lm, ll, cfg, n_pos=len(idx), img_ids=Itr[idx])
        opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()
        if s % 250 == 0 or s == steps - 1:
            print(f"  step{s:5d} loss={float(out['loss']):.3f} "
                  f"nce={float(out['l_nce']):.3f} pull={float(out['l_pull']):.3f} "
                  f"sd={float(out['l_sd']):.3f} vic={float(out['l_vic']):.4f}", flush=True)

    m = evaluate(pve, ple, V[va], L[va], img[va])
    print("HELD-OUT:", m, "  chance R@1 =", round(1 / m["n_img"], 4), flush=True)

    paths.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"pve": pve.state_dict(), "ple": ple.state_dict(),
            "config": {"psas_dim": cfg.projection.psas_dim,
                       "hidden_dim": cfg.projection.hidden_dim,
                       "visual_in": cfg.visual_backbone.feature_dim,
                       "text_in": cfg.text_backbone.feature_dim}}
    torch.save({**meta, "stage": "B_cmota_align", "objective": "align",
                "steps": steps, "val": m}, paths.CKPT_DIR / "cmota.pt")
    torch.save({**meta, "stage": "A_pretrain_proj", "steps": steps},
               paths.CKPT_DIR / "pretrain_proj.pt")
    print("saved cmota.pt + pretrain_proj.pt ->", paths.CKPT_DIR, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--n-pairs", type=int, default=40000)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.n_pairs)
    else:
        train(a.steps, a.batch, a.lr)
