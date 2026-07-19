# GPT drawing prompt — Fig. 1 (overall research framework / technical route map)

Target file: `figs/fig1_framework.png`
(The current file is a functional matplotlib stand-in. Replace it with the drawing
produced from the prompt below, keep the same filename, then re-run
`python build_docx.py`.)

> Division of labour — the two figures must NOT look alike:
> **Fig. 1 = the RESEARCH architecture**: what the whole study does — problem →
> method → validation → findings & deliverables. Four stacked layers read
> TOP-TO-BOTTOM. No formulas.
> **Fig. 2 = the ALGORITHM's structural flow** (`FIG2_PROMPT.md`): how the method
> runs at inference — input → decompose → score → decide → output, LEFT-TO-RIGHT,
> with the equations. The two different reading directions + different content
> guarantee the two figures cannot be confused.

---

## What this figure must make a reader understand

One glance must answer: **what problem, what method, how validated, and what the
research finally delivers** (a finding + a usable plug-in + a dataset).

Rules:
1. **Four full-width horizontal LAYERS, read top-to-bottom**, each a rounded band
   with a bold left-side label. One thick downward arrow between consecutive
   layers on the centreline. This is a research route map, NOT a runtime pipeline —
   no image/caption flowing through it.
2. **No mathematical formulas anywhere** (they live in Fig. 2). Numbers are allowed
   only inside the Validation layer and the deliverable chips.
3. **The bottom layer is the payoff.** Findings and deliverables must be visually
   the strongest band (slightly darker fill or thicker border): the reader must see
   that the study ends in concrete outputs — a principle, a plug-in, a dataset.
4. Keep the section anchors (§1, §3 …) on the layer labels.

**Text tip:** proof-read all numbers (0.80/0.73/0.71/0.62 · −20~−41% · +2.0 ·
4,594/1,996/2,000) after generating; if small text garbles, request SVG code
instead of a raster image.

---

## PROMPT

Create a flat vector technical diagram for an academic journal paper (Elsevier /
Information Fusion style): a four-layer RESEARCH ROUTE MAP read top-to-bottom.
Landscape, aspect ratio 16:9, high resolution, white background. NO gradients, NO
3D, NO shadows, NO clip-art, NO photographs, NO mathematical formulas. Thin 1px
outlines, rounded corners, Arial/Helvetica, generous white space.

PALETTE: fills — very light blue #EDF3FB, light green #EAF7F1, light peach #FDEEE6,
light gray #F2F2F2, white. Strokes — blue #0072B2, green #009E73, vermilion
#D55E00, purple #CC79A7, orange #E69F00, dark gray #444444, red #C1121F (used once,
for the negative finding). Text near-black #111111.

STRUCTURE: four full-width rounded horizontal bands stacked top to bottom, joined
by ONE thick downward arrow between each pair, on the centreline. Each band has a
bold header on its left edge. Inside each band, small rounded chips sit in one row
(wrap to two rows only if needed).

LAYER 1 — header "Problem (§1)" — light gray band.
  Three chips, one per hallucination kind, each with a five-word example:
    blue chip      "Object — 'a dog' that is not there"
    green chip     "Attribute — 'a red car' that is blue"
    vermilion chip "Relation — 'cup on the table' … it is under"
  And a fourth, white chip with a dark border, set slightly apart:
    "Open question: does strong detection give strong mitigation?"

LAYER 2 — header "Method — hierarchical grounding fusion (§3)" — light blue band.
  Left group, three chips titled "matched detectors (all models frozen)":
    "OLD · CLIP zero-shot presence"
    "ALD · region-crop CLIP contrast"
    "RLD · bounding-box geometry"
  A small "fuse" connector, then a right group of three chips titled "three uses":
    purple "Detect — flag unsupported claims"
    orange "Revise — drop or rewrite them"
    blue   "Fuse — combine with the model's own answer"
  A thin footer line inside the band, gray italic:
    "training-free · wraps any backbone (even API-only) at levels L0 / L1 / L2"

LAYER 3 — header "Validation (§4–5)" — light green band.
  Five compact result chips, plain numbers, no formulas:
    "Detection AUC 0.80 / 0.73 / 0.71 / 0.62  (POPE · AMBER · VG-Rel)"
    "Leak-free HalluProbe-VL: object AUC 0.83 — generalizes"
    "Mitigation: CHAIR-i −20% ~ −41% on 4 backbones, quality unchanged"
    "Discrimination: up to +2.0 accuracy at matched yes-ratio"
    red-bordered chip: "Decoder steering with the same evidence FAILS (§5.4)"

LAYER 4 — header "Findings & deliverables (§6–7)" — slightly darker band, the
visual anchor of the figure. Two labelled sub-groups:
  "Findings:" two white chips with dark text —
    "The detection–mitigation gap: detection strength does not transfer"
    "The residual principle: a source helps only where the model is wrong"
  "Deliverables:" three strong chips —
    green  "hgfusion plug-in — pip-installable, 3-line integration (L0)"
    blue   "HalluProbe-VL diagnostic — 4,594 / 1,996 / 2,000 probes"
    gray   "code, adapters for 4 backbones, and honest negative results"

ONE-LINE STORY at the very bottom, centred, small gray italic:
   "From three kinds of hallucination to a training-free plug-in —
    validated on four backbones, explained by one principle."

TYPOGRAPHY: all text horizontal; the four layer headers and the three downward
arrows dominate the visual hierarchy; chips stay small and evenly spaced. Spell
all text exactly as given.
