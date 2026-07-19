"""Empirically confirm (or refute) CM-OTA representational collapse.

Loads the trained PVE/PLE heads (Stage A pretrain_proj.pt, Stage B cmota.pt with
LoRA merged) plus a random head, pushes G:'s cached CLIP/LLaMA features through
them, and measures how much the pooled PSAS mean vector VARIES ACROSS INPUTS.

Collapse signature: near-zero cross-input dispersion / average pairwise cosine ~1.0
(every image/caption maps to essentially the same point).
"""
import glob, os, sys, math
import torch
import torch.nn as nn

CKPT = r"G:\cmpsa_data\cmpsa_project\results\checkpoints"
CLIP_DIR = r"G:\cmpsa_data\cache\clip_features\val2017"
LLAMA_DIR = r"G:\cmpsa_data\cache\llama_features\val2017"
N = 300
PSAS, HID = 256, 2048


class GaussianHead(nn.Module):
    def __init__(self, in_dim, psas=PSAS, hid=HID):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(),
                                   nn.Linear(hid, hid), nn.GELU())
        self.mu_head = nn.Linear(hid, psas)
        self.logvar_head = nn.Linear(hid, psas)

    def forward(self, x):
        h = self.trunk(x)
        return self.mu_head(h), torch.clamp(self.logvar_head(h), -8.0, 4.0)


def merge_peft(state, rank=16, scale=2.0):
    if not any(k.startswith("base_model.model.") for k in state):
        return state
    out = {}
    for mod in ("trunk.0", "trunk.2", "mu_head", "logvar_head"):
        base = f"base_model.model.{mod}"
        w = state.get(f"{base}.base_layer.weight")
        if w is None:
            continue
        w = w.clone().float()
        a = state.get(f"{base}.lora_A.default.weight")
        b = state.get(f"{base}.lora_B.default.weight")
        if a is not None and b is not None:
            upd = (b.float() @ a.float()) * scale
            if upd.shape == w.shape:
                w = w + upd
        out[f"{mod}.weight"] = w
        bkey = f"{base}.base_layer.bias"
        if bkey in state:
            out[f"{mod}.bias"] = state[bkey].float()
    return out


def load_head(in_dim, ckpt_path, side):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sub = st.get(side, st)
    rank = int(st.get("lora_rank", 16) or 16)
    merged = merge_peft(sub, rank)
    head = GaussianHead(in_dim)
    miss, unexp = head.load_state_dict(merged, strict=False)
    return head.eval(), len(miss), len(unexp)


def as_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("feat", "features", "last_hidden_state", "hidden", "x"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k].float()
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v.float()
    raise TypeError(f"cannot extract tensor from {type(obj)}")


def pooled_mu(head, files, drop_cls):
    mus = []
    logvars = []
    with torch.no_grad():
        for f in files:
            t = as_tensor(torch.load(f, map_location="cpu", weights_only=False))
            if t.dim() == 3:
                t = t[0]
            if drop_cls and t.shape[0] > 1:
                t = t[1:]                      # drop CLS like pgd_decode does
            mu, logvar = head(t)               # [N, PSAS]
            mus.append(mu.mean(0))             # pool over tokens/patches
            logvars.append(logvar.mean(0))
    return torch.stack(mus), torch.stack(logvars)


def report(name, M, LV):
    # cross-input dispersion of the pooled mean vector
    center = M.mean(0)
    per_dim_std = M.std(0)                      # [PSAS]
    disp = float(per_dim_std.mean())            # avg per-dim std across inputs
    rel = float((M - center).norm(dim=1).mean() / (center.norm() + 1e-9))
    # average pairwise cosine of pooled mu (collapse -> ~1.0)
    Mn = M / (M.norm(dim=1, keepdim=True) + 1e-9)
    C = Mn @ Mn.t()
    n = C.shape[0]
    off = (C.sum() - C.diag().sum()) / (n * (n - 1))
    sigma = float(torch.exp(0.5 * LV).mean())
    print(f"  {name:16s} | cross-input mu-std={disp:.4e} | rel-disp={rel:.4e} "
          f"| avg pairwise cos={float(off):.4f} | mean sigma={sigma:.3f}")
    return disp


def main():
    clip_files = sorted(glob.glob(os.path.join(CLIP_DIR, "*.pt")))[:N]
    llama_files = sorted(glob.glob(os.path.join(LLAMA_DIR, "*.pt")))[:N]
    print(f"clip files={len(clip_files)}  llama files={len(llama_files)}")
    if clip_files:
        s = as_tensor(torch.load(clip_files[0], map_location="cpu", weights_only=False))
        print(f"sample clip feat shape={tuple(s.shape)}")
    if llama_files:
        s = as_tensor(torch.load(llama_files[0], map_location="cpu", weights_only=False))
        print(f"sample llama feat shape={tuple(s.shape)}")

    torch.manual_seed(0)
    heads_v = {
        "random": (GaussianHead(1024).eval(), 0, 0),
        "stageA_pretrain": load_head(1024, os.path.join(CKPT, "pretrain_proj.pt"), "pve"),
        "stageB_cmota": load_head(1024, os.path.join(CKPT, "cmota.pt"), "pve"),
    }
    print("\n=== PVE (visual, CLIP-L/14-336 val2017 patches) ===")
    for name, (h, m, u) in heads_v.items():
        M, LV = pooled_mu(h, clip_files, drop_cls=True)
        d = report(f"{name}(miss{m})", M, LV)

    heads_t = {
        "random": (GaussianHead(4096).eval(), 0, 0),
        "stageA_pretrain": load_head(4096, os.path.join(CKPT, "pretrain_proj.pt"), "ple"),
        "stageB_cmota": load_head(4096, os.path.join(CKPT, "cmota.pt"), "ple"),
    }
    print("\n=== PLE (text, LLaMA-2-7B val2017 captions) ===")
    for name, (h, m, u) in heads_t.items():
        M, LV = pooled_mu(h, llama_files, drop_cls=False)
        report(f"{name}(miss{m})", M, LV)

    print("\n=== hhd.pt contents (confirm prior-predictor calibrator) ===")
    hhd = torch.load(os.path.join(CKPT, "hhd.pt"), map_location="cpu", weights_only=False)
    print("  keys:", list(hhd.keys()))
    print("  stage:", hhd.get("stage"), " steps:", hhd.get("steps"))
    print("  thresholds:", hhd.get("thresholds"))
    cal = hhd.get("calibrator")
    if cal:
        print("  calibrator scale:", cal.get("scale"))
        print("  calibrator bias :", cal.get("bias"))


if __name__ == "__main__":
    main()
