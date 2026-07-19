# GPT drawing prompt — Fig. 2 (algorithm structural flow)

Target file: `figs/fig2_principle.png`
(The current file is a functional matplotlib stand-in. Replace it with the drawing
produced from the prompt below, keep the same filename, then re-run
`python build_docx.py`.)

> Division of labour — the two figures must NOT look alike:
> **Fig. 1** (`FIG1_PROMPT.md`) = the RESEARCH architecture: four stacked layers,
> top-to-bottom, no formulas.
> **Fig. 2 = the ALGORITHM's structural flow**: how one (image, output) pair is
> processed at inference — LEFT-TO-RIGHT flowchart with the actual equations,
> a decision diamond, and three concrete outputs. This figure carries the math.

---

## What this figure must make a reader understand

One glance must answer: **given an image and a model's output, what exactly does
the algorithm compute, in what order, and what comes out.**

Rules:
1. **A true flowchart, read LEFT-TO-RIGHT**: input → decompose → score (three
   parallel scorers, each with its formula) → threshold/decide → act → output.
   Use flowchart conventions: rounded rectangles for process steps, ONE diamond
   for the branching decision, parallelogram-ish cards for inputs/outputs.
2. **The equations appear here, next to the step that computes them**, with their
   numbers from the paper: Eq. 1 decompose, Eqs. 4–7 scores, Eq. 3 threshold,
   Eq. 8 flag set, Eq. 9 sentence removal, Eqs. 12–13 fusion. Typeset all
   sub/superscripts cleanly.
3. **Honest wiring:** all three scorers feed the threshold test / Report path;
   ONLY the object score s_obj feeds the revise diamond and the fusion step.
4. Do NOT redraw the research context (problem, benchmarks, deliverables) — that
   is Fig. 1's job. No layer bands, no result numbers, no plug-in framing.

**Text tip:** proof-read every formula after generating (especially subscripts in
g_obj, g_attr, τ_o, p̂_yes, Q_{1−r₀}); if the math garbles, request SVG code
instead of a raster image.

---

## PROMPT

Create a flat vector ALGORITHM FLOWCHART for an academic journal paper (Elsevier /
Information Fusion style), read strictly left to right. Landscape, aspect ratio
16:8, high resolution, white background. NO gradients, NO 3D, NO shadows, NO
clip-art. Thin 1px outlines, rounded corners, Arial/Helvetica; render every
formula as clean typeset math with correct subscripts and superscripts.

PALETTE: fills — very light blue #EDF3FB, light green #EAF7F1, light peach #FDEEE6,
light gray #F2F2F2, white. Strokes — blue #0072B2, green #009E73, vermilion
#D55E00, purple #CC79A7, orange #E69F00, dark gray #444444. Arrows dark gray;
object-evidence arrows BLUE #0072B2.

COLUMN 1 — INPUT (far left):
  A slanted-corner input card: "INPUT   image I  +  model output y
  (caption or yes–no answer)".

COLUMN 2 — DECOMPOSE:
  Process box: "decompose y into claims" with the equation
     M(y) = M_obj ∪ M_attr ∪ M_rel        (Eq. 1)
  One arrow in from the input card; three arrows out, fanning to the three scorers.

COLUMN 3 — SCORE (three parallel scorer boxes, stacked; each carries its formula
and a tiny thumbnail):
  blue box "OLD — object presence"
     g_obj = cos( φ_v(I) , φ_t('a photo of a o') )        (Eq. 4)
     thumbnail: a whole photo.
  green box "ALD — attribute contrast"
     g_attr = cos(R_o, 'a red car') − cos(R_o, 'a car')   (Eq. 5)
     thumbnail: a bounding box with its cropped patch.
  vermilion box "RLD — relation geometry"
     contact: ov = |b_s ∩ b_o| / min(|b_s|,|b_o|)          (Eq. 6)
     direction: ⟨c_s − c_o, u_r⟩ / ‖c_s − c_o‖             (Eq. 7)
     thumbnail: two rectangles with centroids and a dashed arrow.

COLUMN 4 — TEST & DECIDE:
  Top: process box "threshold test  ĥ(c) = 1[ g(c) < τ ]   (Eq. 3)" fed by ALL
  THREE scorers (three gray arrows). Its output arrow goes to Output card A
  ("evidence report").
  Below it: ONE diamond, fed ONLY by the OLD scorer (blue arrow), labelled:
     "F = { o : s_obj(I,o) < τ_o }   (Eq. 8)      F = ∅ ?"
  Two labelled branches leave the diamond:
     green italic "yes → keep y unchanged"
     orange italic "no → revise"
  The orange branch enters a process box "revise" with two stacked options:
     "sentence removal:  y′ = ⊕ sentences with no flagged object   (Eq. 9)"
     "self-rewrite: re-prompt the backbone to remove F"
  Bottom: process box "decision fusion (yes/no questions)" fed by the OLD scorer
  (blue arrow) AND by a thin dashed gray arrow entering from below labelled
  "p_yes from the backbone":
     z = p̂_yes + λ·ĝ        (Eq. 12)
     answer 'yes' iff z ≥ τ* = Q_{1−r₀}(z)     (Eq. 13, matched yes-ratio)

COLUMN 5 — OUTPUT (far right), three slanted-corner output cards stacked:
  purple card A "evidence report — per-claim scores + flags"
  orange card B "verified caption y′"
  blue   card C "calibrated yes/no answer"
  Arrows: threshold test → A;  "keep"/"revise" branches → B;  fusion → C.

Label the two blue arrows once, in small blue italic: "object evidence only".

ONE-LINE STORY at the very bottom, centred, small gray italic:
   "One pass: decompose the output, score each claim with its matched signal,
    then report, repair, or fuse."

TYPOGRAPHY: all text horizontal; the left-to-right flow must be obvious at a
glance; equations sit inside or directly beneath their step boxes, never floating
free. Spell all text and equation numbers exactly as given.
