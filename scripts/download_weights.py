# -*- coding: utf-8 -*-
"""Download every model weight this project needs (~97 GB total).

Weights are NOT redistributed in this repository: they are large and each carries
its own upstream licence. This script pulls them from the Hugging Face Hub (or the
hf-mirror.com mirror, which is much faster in mainland China) into $CMPSA_MODELS_ROOT.

Usage
-----
    # everything (~97 GB)
    python scripts/download_weights.py --all

    # only what a given experiment needs
    python scripts/download_weights.py --group core          # CLIP + G-DINO + LLaVA-1.5 (~22 GB)
    python scripts/download_weights.py --group backbones     # the 3 extra backbones (~61 GB)
    python scripts/download_weights.py --only llava-1.5-7b

    # mirror + custom destination
    set HF_ENDPOINT=https://hf-mirror.com
    python scripts/download_weights.py --all --dest /path/to/models
"""
import argparse
import os
import sys

# Use the mirror by default unless the user already set an endpoint.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install -U huggingface_hub")

# name -> (repo_id, group, approx_size_gb)
MODELS = {
    "clip-vit-l14-336":     ("openai/clip-vit-large-patch14-336",   "core",      1.6),
    "grounding_dino":       ("IDEA-Research/grounding-dino-base",   "core",      1.0),
    "llava-1.5-7b":         ("llava-hf/llava-1.5-7b-hf",            "core",     13.2),
    "llava-1.6-vicuna-7b":  ("llava-hf/llava-v1.6-vicuna-7b-hf",    "backbones",13.2),
    "instructblip-7b":      ("Salesforce/instructblip-vicuna-7b",   "backbones",29.5),
    "qwen-vl-chat":         ("Qwen/Qwen-VL-Chat",                   "backbones",18.0),
}


def default_dest():
    root = os.environ.get("CMPSA_MODELS_ROOT")
    if root:
        return root
    data = os.environ.get("CMPSA_DATA_ROOT")
    if data:
        return os.path.join(data, "models")
    return os.path.join(os.getcwd(), "models")


def fetch(name, repo_id, dest):
    target = os.path.join(dest, name)
    os.makedirs(target, exist_ok=True)
    print(f"\n=== {name}  <-  {repo_id}\n    -> {target}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        resume_download=True,          # safe to re-run after an interruption
        max_workers=4,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],   # skip non-PyTorch formats
    )
    print(f"=== done: {name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download every model")
    ap.add_argument("--group", choices=["core", "backbones"], help="download one group")
    ap.add_argument("--only", choices=list(MODELS), help="download a single model")
    ap.add_argument("--dest", default=None, help="destination (default: $CMPSA_MODELS_ROOT)")
    args = ap.parse_args()

    if args.only:
        picks = [args.only]
    elif args.group:
        picks = [k for k, v in MODELS.items() if v[1] == args.group]
    elif args.all:
        picks = list(MODELS)
    else:
        ap.print_help()
        print("\nAvailable models:")
        for k, (r, g, s) in MODELS.items():
            print(f"  {k:<22} {s:>5.1f} GB  [{g}]  {r}")
        return

    dest = args.dest or default_dest()
    total = sum(MODELS[p][2] for p in picks)
    print(f"endpoint : {os.environ['HF_ENDPOINT']}")
    print(f"dest     : {dest}")
    print(f"models   : {', '.join(picks)}  (~{total:.1f} GB)")

    for p in picks:
        repo_id = MODELS[p][0]
        try:
            fetch(p, repo_id, dest)
        except Exception as e:
            print(f"!!! FAILED {p}: {type(e).__name__}: {e}", flush=True)

    print("\nAll requested downloads finished.")
    print("NOTE: InstructBLIP also needs a BERT tokenizer under "
          "<instructblip-7b>/qformer_tokenizer/ — see README (Troubleshooting).")


if __name__ == "__main__":
    main()
