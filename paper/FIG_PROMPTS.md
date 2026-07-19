# Figure drawing prompts — index

The GPT drawing prompts for the two schematic figures live in one file each, so
there is a single source of truth per figure.

| Figure | Prompt file | Target image | Role |
|---|---|---|---|
| **Fig. 1** | [`FIG1_PROMPT.md`](FIG1_PROMPT.md) | `figs/fig1_framework.png` | **overall research framework** (technical route map) |
| **Fig. 2** | [`FIG2_PROMPT.md`](FIG2_PROMPT.md) | `figs/fig2_principle.png` | **the algorithm's structural flow** (flowchart with the equations) |

## The two figures are deliberately different in kind

They must **not** look like the same diagram — different content AND different
reading direction:

- **Fig. 1 — the research architecture (top-to-bottom, no formulas).** Four stacked
  layers: Problem (three hallucination kinds + the open question) → Method (three
  matched detectors, three uses, wrapper deployment) → Validation (the full-scale
  results incl. the negative decoder-steering finding) → **Findings & deliverables**
  (the gap, the residual principle, the hgfusion plug-in, HalluProbe-VL). It answers
  *what the study does and what it delivers*.

- **Fig. 2 — the algorithm flow (left-to-right, carries the math).** A true
  flowchart for one (image, output) pair: input → decompose (Eq. 1) → three parallel
  scorers (Eqs. 4–7) → threshold test (Eq. 3) + revise diamond (Eqs. 8–9) + decision
  fusion (Eqs. 12–13) → three outputs (report / verified caption / calibrated
  answer). It answers *what exactly is computed, in what order, and what comes out*.

Shared elements appear exactly once: research context, results, and deliverables
are in Fig. 1 only; equations and the runtime branching are in Fig. 2 only.

## Usage

Open the relevant file, copy the block under its `## PROMPT` heading into an image
model, save the result over the target filename in `figs/`, then re-run
`python build_docx.py`. Each file also carries a "what this figure must make a
reader understand" preamble and the wiring/proof-reading rules; read it before
generating.
