# Hierarchical Grounding Fusion for MLLM Hallucination

Code and data for the paper **“Hierarchical Grounding Fusion for Detecting and
Mitigating Object, Attribute, and Relation Hallucinations in Multimodal Large
Language Models.”**

A **training-free, plug-and-play** framework that fuses off-the-shelf grounding
evidence to (1) **detect** object / attribute / relation hallucinations, (2)
**mitigate** object hallucination by detect-then-revise, and (3) **improve
yes/no answering** by calibrated decision fusion — on **four different backbones**,
with **no fine-tuning of any model**.

It deploys as a **post-hoc wrapper** around an unmodified backbone, at three access
levels: **L0** (detection + sentence removal — needs only the image and the output
text, so it can wrap even API-only models), **L1** (self-rewrite — needs re-prompting
an instruction-following backbone), **L2** (decision fusion — needs the token
probability `p_yes`). The levels describe what was measured: L0 transferred to all
four backbones, L1 failed exactly on the weak instruction-follower, L2 was always run
with logit access.

---

## Headline results

**Detection** (full-scale, public benchmarks). P/R are at the best-F1 threshold.
The last column is a learned probabilistic alignment we trained first — it sits at
**chance**, which is exactly why the final method fuses frozen off-the-shelf signals
instead of a learned one.

| Layer | Benchmark | n | Pos. | AUC | AP | Best-F1 | P@F1 | R@F1 | Learned align. AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OLD (object) | POPE | 9000 | 50% | **0.804** | 0.821 | 0.739 | 0.662 | 0.836 | 0.507 |
| ALD (attribute) | AMBER-attr | 4764 | 50% | **0.729** | 0.706 | 0.717 | 0.601 | 0.888 | 0.543 |
| RLD (relation, contact) | AMBER-rel | 1664 | 58.6% | **0.706** | 0.726 | 0.796 | 0.683 | 0.952 | 0.473 |
| RLD (relation, direction) | VG-Rel | 2000 | 50% | **0.622** | 0.577 | 0.679 | 0.541 | 0.913 | 0.500 |

> Two things worth reading off this table. (1) At the best-F1 threshold the detectors
> are **high-recall / low-precision** (object: R=0.836 at P=0.662). That is fine for
> flagging and filtering, but it is the *wrong* operating point for editing text
> automatically — at P=0.66 a third of the removals would delete true content. This is
> the first symptom of the detection–mitigation gap, and why the revision stage
> switches to a precise, region-level source at a high threshold.
> (2) AMBER-relation's class prior is **0.586**, so its high F1 flatters it; compare
> each row against its own prior, not across rows.

**Mitigation** — CHAIR-i (%) on CHAIR-500, lower is better

| Backbone | Vanilla | Self-rewrite | Sentence removal | rel. |
|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | 16.9 | 14.1 | **11.7** | −30.9% |
| LLaVA-1.6-7B | 7.5 | 7.1 | **6.0** | −19.9% |
| InstructBLIP-7B | 19.2 | 19.3 | **11.4** | −40.7% |
| Qwen-VL-Chat | 15.9 | 13.8 | **10.6** | −33.3% |

**Discrimination** — POPE accuracy (%) at a **matched yes-ratio**

| Backbone | Vanilla | +grounding | Δ |
|---|---:|---:|---:|
| LLaVA-1.5-7B | 87.0 | 88.3 | +1.3 |
| LLaVA-1.6-7B | 87.9 | 87.9 | +0.0 |
| InstructBLIP-7B | 84.6 | **86.6** | **+2.0** |
| Qwen-VL-Chat | 85.1 | 86.5 | +1.4 |

> The LLaVA-1.6 **+0.0** is a genuine null result and is reported as such: that
> backbone’s own answers are already the strongest (answer AUC 0.951), so there is
> little headroom for an external signal. Grounding helps most where the backbone
> leaves room to help.

**Key finding — the detection–mitigation gap.** Strong detection does *not*
automatically give strong mitigation. Global grounding shares the MLLM’s
co-occurrence bias, so token-by-token decoder guidance fails; a *precise*
(region-level) detector used **post hoc** succeeds. Detector precision — not the
mechanism’s name — is the deciding factor.

**Why, formally — the residual view.** Write the grounding score and the model’s
log-evidence as loadings on a shared co-occurrence bias `b`:

```
g = α·b + ε_g ,   ℓ_p = β·b + ε_p ,   ρ = corr(g, ℓ_p)
g⊥ = g − (Cov(g, ℓ_p) / Var(ℓ_p)) · ℓ_p        # the part the model does NOT already know
Var(g⊥) = (1 − ρ²) · Var(g)
```

Only `g⊥` can flip a decision the model gets wrong; the collinear part `α·b` merely
restates what the model already believes. So the achievable gain scales with
`Var(g⊥)` and vanishes as `ρ → 1` **no matter how accurate the source is** — a source
can be simultaneously accurate and useless. This one quantity explains three
otherwise-unrelated results: decoder guidance fails (global CLIP is collinear with
the language prior exactly on the objects that get hallucinated), region-level
post-hoc revision works (large residual), and grounding adds **+0.0** on LLaVA-1.6
(its own evidence is already strong, so there is no headroom). The design rule for
fusion: maximize source precision *on the model's error region*, not marginal
source accuracy.

---

## What is / is not in this repo

| | |
|---|---|
| ✅ **Included** | All method + experiment code, the **HalluProbe-VL** probe set we built, the per-item detection dumps needed to reproduce every figure, the figure/paper build scripts, and the honest **negative-result** code. |
| ❌ **Not included: model weights** | ~**97 GB** across 6 models. GitHub caps files at 100 MB, and LLaVA / Qwen-VL / InstructBLIP / CLIP / Grounding-DINO each carry their own upstream licence, so we do not redistribute them. Get them with one command: `python scripts/download_weights.py --all` (see below). |
| ❌ **Not included: third-party corpora** | COCO, Visual Genome, POPE, AMBER are public datasets with their own terms — download them from the original sources (see *Data*). |

---

## Install

```bash
git clone <your-repo-url> && cd <repo>
conda create -n hgf python=3.10 -y && conda activate hgf
pip install -e .            # installs the `hgfusion` plugin + the `cmpsa` library
```
(`pip install -r requirements.txt` also works if you prefer not to install the package.)

## Use it as a plugin — 3 lines (L0)

`hgfusion` wraps **any** captioning model — including API-only ones — because L0
consumes nothing but the image and the model's output text:

```python
from hgfusion import HGFWrapper

w = HGFWrapper(models_root="/path/to/models")        # CLIP + Grounding-DINO
caption = my_model_or_api(image)                     # ANY backbone, even closed
clean   = w.revise_caption(image, caption).text      # hallucinated sentences dropped
```

Inspect instead of edit, plug in your own rewrite (L1), or fuse with your model's
answer probability (L2):

```python
report = w.verify_caption(image, caption)            # L0: {object: score}, flagged list
rev    = w.revise_caption(image, caption,            # L1: your backbone rewrites
                          strategy="rewrite", rewrite_fn=my_rewrite)
w.calibrate(records)                                 # L2: fit on (p_yes, g, label) triples
ans    = w.answer(image, "Is there a dog?", p_yes=0.62)
```

Run the end-to-end demo (verified on a real image; a planted "cat" sentence gets
caught at s=0.29 < τ=0.30 and removed while dog/table/cake are kept):

```bash
python examples/wrap_demo.py --image img.jpg \
    --caption "A dog sits at a table. A cat sleeps on the floor."
```

> **Important:** pin `transformers==4.49.0`. Version 5.x uses a threaded checkpoint
> loader that segfaults on Windows for InstructBLIP / LLaVA-1.6 / Qwen-VL.

## Get the weights (~97 GB)

```bash
# uses https://hf-mirror.com by default; override with HF_ENDPOINT
python scripts/download_weights.py --group core        # CLIP + G-DINO + LLaVA-1.5 (~22 GB)
python scripts/download_weights.py --group backbones   # LLaVA-1.6 + InstructBLIP + Qwen-VL (~61 GB)
python scripts/download_weights.py --all               # everything
```

## Configure paths

The code resolves every path through `src/cmpsa/paths.py`, driven by three
environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CMPSA_DATA_ROOT` | root holding `basic/`, `benchmarks/`, `derived/` | parent of the repo |
| `CMPSA_MODELS_ROOT` | where the weights live | `$CMPSA_DATA_ROOT/models` |
| `CMPSA_RESULTS_ROOT` | where runs write artefacts | `<repo>/results` |

```bash
export CMPSA_DATA_ROOT=/path/to/cmpsa_data      # Windows: set CMPSA_DATA_ROOT=E:\...
export CMPSA_MODELS_ROOT=$CMPSA_DATA_ROOT/models
```

## Data

| Dataset | Where to get it | Used for |
|---|---|---|
| COCO 2014 val (+ annotations) | https://cocodataset.org | POPE, CHAIR |
| COCO 2017 val | https://cocodataset.org | HalluProbe-VL object probes |
| Visual Genome | https://homes.cs.washington.edu/~ranjay/visualgenome/ | VG-Rel / VG-Attr probes |
| POPE | https://github.com/RUCAIBox/POPE | object discrimination |
| AMBER | https://github.com/junyangwang0410/AMBER | attribute / relation |

Expected layout under `$CMPSA_DATA_ROOT` (see `src/cmpsa/paths.py` for the full map):

```
cmpsa_data/
├── basic/coco/images/{val2014,val2017}/ , basic/coco/annotations/
├── basic/visual_genome/{images,objects.json,relationships.json,attributes.json}
├── benchmarks/{pope,amber}/
└── models/            <- download_weights.py writes here
```

---

## HalluProbe-VL

A **leak-free** diagnostic for three-kind hallucination detection, released in
`data/halluprobe_vl/`:

| Split | Released n | Evaluated n | AUC | Source | Note |
|---|---:|---:|---:|---|---|
| `probes/object.jsonl` | 4594 | 4594 | **0.828** | COCO **2017** val | balanced, high-co-occurrence hard negatives; disjoint from POPE/CHAIR (COCO 2014) |
| `probes/attribute.jsonl` | 1996 | 555 | **0.685** | Visual Genome | |
| `probes/relation.jsonl` | 2000 | 634 | **0.651** | Visual Genome | |

Ground truth comes from the **source annotations, not from tool predictions**, so
the grounding tools remain the *evaluated* detectors rather than the labellers.
`image` fields are **relative** paths (e.g. `val2017/000000001353.jpg`); join them
against your COCO/VG copy.

> **Released n vs. evaluated n.** Attribute and relation probes are only scorable
> when the referenced object can be localized to a region, so their AUC is computed
> on the **evaluable subsets** (555 / 634), not on all released probes. The object
> layer has no such restriction and is scored on all 4594. We state this explicitly
> rather than quoting the larger numbers.

The object AUC (0.83) matches its POPE score (0.80) even though the probes are
disjoint from POPE's images and adversarially chosen against co-occurrence —
evidence the detector generalizes rather than over-fitting one benchmark.

---

## Reproduce the paper

```bash
# Detection (Table 2, Fig. 2)
python scripts/detectors/old_clip_zeroshot.py
python scripts/detectors/ald_region_crop.py
python scripts/detectors/rld_box_geometry.py

# The detection–mitigation gap (Fig. 3)
python scripts/analysis/gap_gdino_vs_clip.py
python scripts/analysis/revise_ceiling.py

# Object mitigation (Table 3, Fig. 4)
python scripts/mitigation/detect_then_revise.py --n 500 --thr 0.30
python scripts/mitigation/vcd_baseline.py            # VCD comparison
python scripts/mitigation/caption_quality.py         # BLEU / ROUGE / coverage

# Cross-backbone (Table 4/5, Fig. 5/6)  -- one process per backbone
for b in llava15 llava16 instructblip qwenvl; do
  python scripts/mitigation/cross_backbone_revise.py   --backbone $b --n 500 --thr 0.30
  python scripts/discrimination/cross_backbone_discrim.py --backbone $b --n 2000
done

# HalluProbe-VL (Fig. 7)
python scripts/halluprobe/build_halluprobe_obj.py
python scripts/halluprobe/eval_halluprobe.py

# Single-backbone discrimination (Appendix A, Table 6, Fig. 8)
python scripts/discrimination/object_gate.py --n 9000

# Figures + the Word manuscript
python paper/figures.py
python paper/build_docx.py
```

## Repository layout

```
├── src/hgfusion/           # the plug-and-play wrapper (pip-installable; L0/L1/L2 API)
├── src/cmpsa/              # library: paths, config, eval harness, model wrappers
├── examples/wrap_demo.py   # end-to-end plugin demo (verified on a real image)
├── pyproject.toml          # `pip install -e .`
├── scripts/
│   ├── download_weights.py
│   ├── detectors/          # OLD / ALD / RLD grounding detectors
│   ├── analysis/           # the detection–mitigation gap study
│   ├── mitigation/         # detect-then-revise, VCD baseline, caption quality
│   ├── discrimination/     # matched-yes-ratio decision fusion
│   ├── halluprobe/         # build + evaluate HalluProbe-VL
│   └── negative_results/   # things that did NOT work (see below)
├── data/halluprobe_vl/     # our released probe set
├── results/                # metrics + per-item detection dumps (figure inputs)
└── paper/                  # figures.py, build_docx.py, figs/
```

## Negative results (`scripts/negative_results/`)

We publish what failed, because it is the evidence behind the paper’s central claim:

- **Token-suppression / probability-guided decoding** — suppressing absent-object
  tokens during generation, naive and per-class calibrated. Did **not** lower CHAIR
  (slightly raised CHAIR-i): hallucinated objects pass the *shared* grounding bias.
- **Hierarchical revise** (vocab-parser and LLM-claim-extraction variants) — editing
  attributes/relations as well as objects gave **no** measured object-CHAIR gain and
  diluted the clean object-only benefit.
- **CM-OTA learned probabilistic alignment** — a learned cross-modal alignment we
  tried first. It collapsed representationally and scored ~0.50 AUC; it is **not**
  part of the final method. `cmota_confirm_collapse.py` documents the diagnosis.

## Known limitations

- Mitigation is validated for **objects** (CHAIR and AMBER’s caption task score
  objects only). Attribute/relation are contributed at the **detection** level.
- **Self-rewrite** needs an instruction-following backbone; on InstructBLIP only the
  deterministic **sentence-removal** variant works. Sentence removal is coarser and
  can drop a true object sharing a sentence with a hallucinated one.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Native crash / segfault loading InstructBLIP, LLaVA-1.6, Qwen-VL on Windows | `pip install transformers==4.49.0` (5.x threaded loader is the cause) |
| `BatchEncoding.to() takes 2 positional arguments but 3 were given` | transformers 4.49 removed the dtype positional: use `.to(device)` then cast `pixel_values` to fp16 |
| InstructBLIP: `Can't load tokenizer` | it needs a BERT tokenizer at `<instructblip-7b>/qformer_tokenizer/` (vocab.txt, tokenizer.json, tokenizer_config.json, special_tokens_map.json) — copy from any `bert-base-uncased` |
| `OSError ... (os error 1455)` on Windows | pagefile exhausted: load the big backbone **before** other models, or raise the Windows pagefile |
| Qwen-VL import errors | needs `transformers_stream_generator`, `tiktoken`, `einops` |

## Licence

Code in this repository: **MIT** (see `LICENSE`).
HalluProbe-VL annotations derive from COCO and Visual Genome and remain subject to
those datasets’ original terms. Model weights are **not** distributed here and remain
under their respective upstream licences.

## Citation

See `CITATION.cff`.
