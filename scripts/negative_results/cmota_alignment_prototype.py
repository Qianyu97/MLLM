"""Prototype + validate the REDESIGNED CM-OTA objective on cached features only.

Head-to-head vs the OLD objective (uniform-marginal <P,C>). Metrics on a held-out
image split: (1) collapse = avg pairwise cosine of pooled PSAS means (low=good),
(2) caption->image retrieval R@1/R@5 (>> chance = alignment works).
No model weights needed: uses G: cached CLIP (per-image) + LLaMA (per-caption) feats.
"""
import os, glob, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = r"G:\cmpsa_data\cache"
ANN = os.path.expandvars(r"${CMPSA_DATA_ROOT}\basic\coco\annotations\captions_val2017.json")
SPLIT = "val2017"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PSAS, HID = 256, 2048
torch.manual_seed(0); random.seed(0)


# ---------- heads ----------
class GaussianHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, HID), nn.GELU(),
                                   nn.Linear(HID, HID), nn.GELU())
        self.mu = nn.Linear(HID, PSAS)
        self.lv = nn.Linear(HID, PSAS)

    def forward(self, x):
        h = self.trunk(x)
        return self.mu(h), torch.clamp(self.lv(h), -8.0, 4.0)


# ---------- OT primitives (from cm_ota.py) ----------
def gaussian_w2(mu1, lv1, mu2, lv2):
    s1, s2 = torch.exp(0.5 * lv1), torch.exp(0.5 * lv2)
    mean = ((mu1[:, None, :] - mu2[None, :, :]) ** 2).sum(-1)
    cov = ((s1[:, None, :] - s2[None, :, :]) ** 2).sum(-1)
    return mean + cov


def sinkhorn(cost, eps=0.05, iters=50):
    n, m = cost.shape
    log_a = torch.full((n,), -torch.log(torch.tensor(float(n))), device=cost.device)
    log_b = torch.full((m,), -torch.log(torch.tensor(float(m))), device=cost.device)
    K = -cost / eps
    f = torch.zeros(n, device=cost.device); g = torch.zeros(m, device=cost.device)
    for _ in range(iters):
        f = log_a - torch.logsumexp(K + g[None, :], dim=1)
        g = log_b - torch.logsumexp(K + f[:, None], dim=0)
    return torch.exp(f[:, None] + K + g[None, :])


def ot_cost(vm, vl, lm, ll, eps=0.05, iters=50):
    c = gaussian_w2(vm, vl, lm, ll)
    p = sinkhorn(c, eps, iters)
    return (p * c).sum()


# ---------- losses ----------
def sinkhorn_divergence(vm, vl, lm, ll):
    return ot_cost(vm, vl, lm, ll) - 0.5 * ot_cost(vm, vl, vm, vl) - 0.5 * ot_cost(lm, ll, lm, ll)


def vicreg_std_hinge(mu, gamma=1.0):
    std = torch.sqrt(mu.var(dim=0) + 1e-6)         # per-dim std across batch
    return F.relu(gamma - std).mean()


def new_loss(vm, vl, lm, ll, img_ids, temp=0.07,
             w_nce=1.0, w_pull=0.05, w_sd=0.1, w_vic=1.0, w_kl=0.01):
    # InfoNCE on cosine(mu), masking same-image false negatives
    vn = F.normalize(vm, dim=-1); ln = F.normalize(lm, dim=-1)
    sim = (vn @ ln.t()) / temp                     # [B,B]
    B = sim.shape[0]
    same = (img_ids[:, None] == img_ids[None, :])
    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    mask = same & (~eye)
    sim_i2t = sim.masked_fill(mask, float("-inf"))
    sim_t2i = sim.t().masked_fill(mask.t(), float("-inf"))
    tgt = torch.arange(B, device=sim.device)
    l_nce = 0.5 * (F.cross_entropy(sim_i2t, tgt) + F.cross_entropy(sim_t2i, tgt))
    # matched-pair W2 pull (diagonal of pairwise W2)
    l_pull = torch.diagonal(gaussian_w2(vm, vl, lm, ll)).mean()
    # debiased Sinkhorn divergence between the two Gaussian sets
    l_sd = sinkhorn_divergence(vm, vl, lm, ll)
    # VICReg variance hinge on the MEANS (anti-collapse) + light logvar reg
    l_vic = vicreg_std_hinge(vm) + vicreg_std_hinge(lm)
    l_kl = 0.5 * (vl.pow(2).mean() + ll.pow(2).mean())
    total = w_nce * l_nce + w_pull * l_pull + w_sd * l_sd + w_vic * l_vic + w_kl * l_kl
    return total, dict(nce=float(l_nce), pull=float(l_pull), sd=float(l_sd),
                       vic=float(l_vic), kl=float(l_kl))


def old_loss(vm, vl, lm, ll, img_ids, lambda_ot=1.0, lambda_kl=0.1):
    # the ORIGINAL cmota_loss: uniform-marginal <P,C> + cov-KL-to-unit (means free)
    cost = gaussian_w2(vm, vl, lm, ll)
    plan = sinkhorn(cost, 0.05, 50)
    l_ot = (plan * cost).sum()
    var_v = torch.exp(vl); var_l = torch.exp(ll)
    l_kl = 0.5 * (var_v - 1 - vl).sum(-1).mean() + 0.5 * (var_l - 1 - ll).sum(-1).mean()
    return lambda_ot * l_ot + lambda_kl * l_kl, dict(ot=float(l_ot), kl=float(l_kl))


# ---------- data ----------
def load_pairs():
    clip_dir = os.path.join(CACHE, "clip_features", SPLIT)
    llama_dir = os.path.join(CACHE, "llama_features", SPLIT)
    clip_ids = {int(os.path.splitext(os.path.basename(f))[0]): f
                for f in glob.glob(os.path.join(clip_dir, "*.pt"))
                if os.path.splitext(os.path.basename(f))[0].isdigit()}
    with open(ANN, encoding="utf-8") as f:
        data = json.load(f)
    cap2img = {str(a["id"]): int(a["image_id"]) for a in data["annotations"]}
    pairs = []
    img_feat_cache = {}
    for lf in glob.glob(os.path.join(llama_dir, "*.pt")):
        cid = os.path.splitext(os.path.basename(lf))[0]
        img = cap2img.get(cid)
        if img is None or img not in clip_ids:
            continue
        if img not in img_feat_cache:
            vt = torch.load(clip_ids[img], map_location="cpu", weights_only=False).float()
            img_feat_cache[img] = vt[1:].mean(0)              # drop CLS, pool -> [1024]
        lt = torch.load(lf, map_location="cpu", weights_only=False).float()
        pairs.append((img, img_feat_cache[img], lt.mean(0)))  # (imgid, vfeat[1024], lfeat[4096])
    return pairs


def evaluate(pve, ple, pairs):
    pve.eval(); ple.eval()
    imgs = sorted({p[0] for p in pairs})
    idx = {im: k for k, im in enumerate(imgs)}
    with torch.no_grad():
        V = torch.stack([next(p[1] for p in pairs if p[0] == im) for im in imgs]).to(DEV)
        vm, _ = pve(V)
        Lmu = []
        gt = []
        for im, vf, lf in pairs:
            lm, _ = ple(lf[None].to(DEV))
            Lmu.append(lm[0]); gt.append(idx[im])
        Lmu = torch.stack(Lmu); gt = torch.tensor(gt, device=DEV)
        # collapse metric
        vn = F.normalize(vm, dim=-1)
        cos = (vn @ vn.t())
        n = cos.shape[0]
        pair_cos = float((cos.sum() - cos.diag().sum()) / (n * (n - 1)))
        # caption->image retrieval by cosine
        ln = F.normalize(Lmu, dim=-1)
        S = ln @ vn.t()                                   # [Ncap, Nimg]
        rank = S.argsort(dim=1, descending=True)
        r1 = float((rank[:, 0] == gt).float().mean())
        r5 = float((rank[:, :5] == gt[:, None]).any(1).float().mean())
    return dict(pair_cos=pair_cos, R1=r1, R5=r5, n_img=len(imgs), n_cap=len(gt))


def run(loss_fn, name, tr, va, steps=500, lr=1e-3):
    pve = GaussianHead(1024).to(DEV); ple = GaussianHead(4096).to(DEV)
    opt = torch.optim.AdamW(list(pve.parameters()) + list(ple.parameters()), lr=lr)
    V = torch.stack([p[1] for p in tr]).to(DEV)
    L = torch.stack([p[2] for p in tr]).to(DEV)
    ids = torch.tensor([p[0] for p in tr], device=DEV)
    pve.train(); ple.train()
    for s in range(steps):
        vm, vl = pve(V); lm, ll = ple(L)
        loss, parts = loss_fn(vm, vl, lm, ll, ids)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 100 == 0 or s == steps - 1:
            print(f"  [{name}] step{s:4d} loss={float(loss):.4f} {parts}")
    m = evaluate(pve, ple, va)
    print(f"  [{name}] VAL: pairwise_cos={m['pair_cos']:.4f}  R@1={m['R1']:.3f}  "
          f"R@5={m['R5']:.3f}  (chance R@1={1/m['n_img']:.3f}, {m['n_img']} imgs / {m['n_cap']} caps)")
    return m


def main():
    pairs = load_pairs()
    imgs = sorted({p[0] for p in pairs})
    random.shuffle(imgs)
    n_val = max(20, int(0.25 * len(imgs)))
    val_imgs = set(imgs[:n_val])
    tr = [p for p in pairs if p[0] not in val_imgs]
    va = [p for p in pairs if p[0] in val_imgs]
    print(f"pairs={len(pairs)} images={len(imgs)} | train pairs={len(tr)} val pairs={len(va)} "
          f"(val imgs={len(val_imgs)}) dev={DEV}")
    print("\n=== OLD objective (uniform-marginal <P,C>, means free) ===")
    run(old_loss, "OLD", tr, va)
    print("\n=== NEW objective (InfoNCE + W2-pull + Sinkhorn-div + VICReg) ===")
    run(new_loss, "NEW", tr, va)


if __name__ == "__main__":
    main()
