# -*- coding: utf-8 -*-
"""
Publication-grade figures for the Information Fusion submission
"Hierarchical Grounding Fusion for Detecting and Mitigating Object, Attribute,
 and Relation Hallucinations in Multimodal Large Language Models".

Every tunable parameter (sizes in mm, colours, fonts, DPI, and every plotted
number) is collected in the CONFIG block so it is easy to adjust for rebuttal.
Figures are saved as PDF (vector) and PNG (600 DPI) into ./figs.

Figure map (matches the manuscript):
  fig1_framework      §1   190 mm   pipeline overview      (GPT prompt available)
  fig2_principle      §3   190 mm   model principle        (GPT prompt available)
  fig3_detection      §5.2 190 mm   ROC + PR, 2 panels
  fig4_halluprobe     §5.3  85 mm   HalluProbe-VL AUC
  fig5_gap            §5.4 190 mm   detection-mitigation gap, 3 panels
  fig6_mitigation     §5.5 140 mm   object mitigation, 2 panels
  fig7_crossbackbone  §5.6 140 mm   cross-backbone mitigation
  fig8_discrim_xbb    §5.7 140 mm   cross-backbone discrimination
  fig9_discriminative App  85 mm    single-backbone discrimination

Run:
    python figures.py               # all
    python figures.py detection gap # only the named figures
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================================
# CONFIG  --  edit here
# =====================================================================================
CONFIG = {
    "outdir": os.path.join(HERE, "figs"),
    "dpi": 600,
    "formats": ["pdf", "png"],

    # Elsevier widths: single ~90 mm, 1.5-col 140 mm, double 190 mm.
    "width_mm": {"large": 190.0, "medium": 140.0, "small": 85.0},

    "font_family": "Arial",
    "font_size": {"base": 8, "label": 8, "tick": 7, "legend": 7, "title": 9, "annot": 7},

    # 42 = embed TrueType -> text stays EDITABLE/selectable in Illustrator.
    # (matplotlib's default 3 = Type 3, which converts glyphs to outlines.)
    "pdf_fonttype": 42,
    # Math font. "stixsans" is a sans math face that matches Arial visually and has
    # every symbol we use. Set to "custom" (with the mathtext.* keys below) to force
    # Arial into math too -- but Arial lacks some glyphs we need, e.g. the circled
    # plus and angle brackets, which would then fall back or render as boxes.
    "mathtext_fontset": "stixsans",

    # Okabe-Ito colour-blind-safe palette
    "colors": {
        "blue":   "#0072B2", "orange": "#E69F00", "green": "#009E73",
        "vermil": "#D55E00", "purple": "#CC79A7", "sky":   "#56B4E9",
        "yellow": "#F0E442", "gray":   "#8C8C8C", "dark":  "#333333",
        "light":  "#EAEAEA",
    },
    "grid_alpha": 0.25,
    "linewidth": 1.8,

    # ---- detection: per-item dumps (ROC) and precomputed PR curves ----
    "roc_dir": os.path.join(HERE, "..", "results", "predictions", "hhd_detection_specialist"),
    "pr_json": os.path.join(HERE, "pr_curves.json"),
    "roc_beds": [
        ("pope",       "OLD (object) - POPE",               "blue"),
        ("amber_attr", "ALD (attribute) - AMBER",           "green"),
        ("amber_rel",  "RLD (relation, contact) - AMBER",   "vermil"),
        ("vg_rel",     "RLD (relation, direction) - VG-Rel","purple"),
    ],

    # ---- HalluProbe-VL detection (n = evaluable probes, see manuscript) ----
    "detect_halluprobe": {  # label: (AUC, F1, n_evaluated)
        "Object":    (0.828, 0.749, 4594),
        "Attribute": (0.685, 0.689, 555),
        "Relation":  (0.651, 0.695, 634),
    },

    # ---- detection-mitigation gap ----
    "gap_scores": {
        "CLIP (global)":   {"true": 0.552, "hall": 0.491, "auc": 0.733},
        "G-DINO (region)": {"true": 0.629, "hall": 0.382, "auc": 0.852},
    },
    "ceiling": {
        "thr":       [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "chair_i":   [0.1174, 0.1174, 0.1174, 0.1130, 0.1130, 0.0973, 0.0809],
        "true_kept": [100.0, 100.0, 100.0, 99.8, 99.8, 99.8, 97.8],
        "hall_rm":   [0.0, 0.0, 0.0, 4.4, 4.4, 19.1, 35.3],
    },
    "strategy": {  # name: (delta_chair_i_pp, delta_true_obj)
        "Token suppression\n(CLIP, naive)":      (+1.7, -0.39),
        "Token suppression\n(CLIP, calibrated)": (+1.6, -0.00),
        "Contrastive decode\n(VCD)":             (+0.7, -0.11),
        "Detect-then-revise\n(ours)":            (-3.95, -0.02),
    },

    # ---- object mitigation CHAIR-500 (matched 80-token, LLaVA-1.5, tf-4.49) ----
    "mitig": {  # method: (chair_i, chair_s, true_obj)
        "Vanilla":           (0.1694, 0.2860, 1.83),
        "Self-\nrewrite":    (0.1408, 0.2420, 1.82),
        "Sentence-\nremove": (0.1171, 0.1900, 1.73),
    },
    "quality": {  # metric: (vanilla, ours)
        "BLEU-4":    (0.0677, 0.0691),
        "ROUGE-L":   (0.2415, 0.2461),
        "Obj. cov.": (0.6152, 0.6072),
    },

    # ---- single-backbone discriminative POPE-9000 (matched yes-ratio) ----
    "discrim": {  # method: (acc, f1, yes_ratio, auc)
        "Vanilla":            (0.8669, 0.8612, 0.4598, 0.9361),
        "+grounding\n(ours)": (0.8749, 0.8696, 0.4598, 0.9411),
    },

    # ---- cross-backbone CHAIR-i (%) (vanilla, self-rewrite, sentence-remove) ----
    "xbackbone": {
        "LLaVA-1.5":    (16.94, 14.08, 11.71),
        "LLaVA-1.6":    (7.48, 7.10, 5.99),
        "InstructBLIP": (19.24, 19.32, 11.40),
        "Qwen-VL":      (15.93, 13.82, 10.63),
    },

    # ---- cross-backbone discriminative (acc_vanilla, acc_fused, f1_vanilla, f1_fused) ----
    "discrim_xbb": {
        "LLaVA-1.5":    (0.8700, 0.8830, 0.8649, 0.8785),
        "LLaVA-1.6":    (0.8790, 0.8790, 0.8683, 0.8683),
        "InstructBLIP": (0.8460, 0.8660, 0.8406, 0.8613),
        "Qwen-VL":      (0.8510, 0.8650, 0.8331, 0.8488),
    },
}

C = CONFIG["colors"]


# =====================================================================================
# style / helpers
# =====================================================================================
def _register_arial():
    for fn in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        p = os.path.join(r"C:\Windows\Fonts", fn)
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass


def setup_style():
    _register_arial()
    fs = CONFIG["font_size"]
    plt.rcParams.update({
        # --- keep text as real, editable text ---------------------------------
        # matplotlib's PDF/PS default is Type 3, which bakes glyphs into drawing
        # procedures: Illustrator then sees outlines, not text. Type 42 embeds the
        # TrueType font instead, so every label stays selectable and editable and
        # keeps its Arial identity. svg.fonttype="none" leaves SVG text as text.
        "pdf.fonttype": CONFIG["pdf_fonttype"],
        "ps.fonttype": CONFIG["pdf_fonttype"],
        "svg.fonttype": "none",
        "pdf.compression": 6,

        "font.family": CONFIG["font_family"],
        "font.sans-serif": [CONFIG["font_family"], "DejaVu Sans"],
        "mathtext.fontset": CONFIG["mathtext_fontset"],
        "font.size": fs["base"], "axes.labelsize": fs["label"],
        "axes.titlesize": fs["title"], "xtick.labelsize": fs["tick"],
        "ytick.labelsize": fs["tick"], "legend.fontsize": fs["legend"],
        "axes.linewidth": 0.8, "axes.edgecolor": "#4d4d4d",
        "axes.grid": True, "grid.color": "#c9c9c9", "grid.linewidth": 0.5,
        "grid.alpha": CONFIG["grid_alpha"],
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": "#bbbbbb", "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def figsize(width_key, height_mm):
    return (CONFIG["width_mm"][width_key] / 25.4, height_mm / 25.4)


def save(fig, name):
    os.makedirs(CONFIG["outdir"], exist_ok=True)
    for ext in CONFIG["formats"]:
        fig.savefig(os.path.join(CONFIG["outdir"], f"{name}.{ext}"), dpi=CONFIG["dpi"])
    plt.close(fig)
    print("saved", name)


def _panel_tag(ax, tag, dx=-0.16, dy=1.02):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=CONFIG["font_size"]["title"],
            fontweight="bold", va="bottom", ha="left")


def _box(ax, x, y, w, h, text, fc, ec, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                linewidth=1.0, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111111")


def _arrow(ax, x0, y0, x1, y1, color="#555555", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
                                 linewidth=1.0, color=color, shrinkA=1, shrinkB=1,
                                 linestyle=ls))


# =====================================================================================
# Fig. 1 : framework overview (stand-in; polished version via GPT prompt)
# =====================================================================================
def fig_framework():
    """Overall research route map: 4 stacked layers, Problem -> Method -> Validation
    -> Findings & deliverables (matches FIG1_PROMPT.md)."""
    fig, ax = plt.subplots(figsize=figsize("large", 92))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")

    def band(y, h, fc, ec, header):
        ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    linewidth=1.1, facecolor=fc, edgecolor=ec))
        ax.text(2.6, y + h - 1.9, header, ha="left", fontsize=6.8, fontweight="bold",
                color="#222222")

    # ---------- Layer 1: Problem ----------
    band(51, 13, "#f5f5f5", "#999999", "Problem  (§1)")
    _box(ax, 3, 52.5, 21, 6, "Object hallucination\n'a dog' that is not there", "#eaf3fb", C["blue"], 5.4)
    _box(ax, 26, 52.5, 21, 6, "Attribute hallucination\n'a red car' that is blue", "#eaf7f1", C["green"], 5.4)
    _box(ax, 49, 52.5, 21, 6, "Relation hallucination\n'cup on the table' ... it is under", "#fdeee6", C["vermil"], 5.4)
    _box(ax, 73, 52.5, 24, 6, "Open question:\ndoes strong detection give strong mitigation?",
         "white", "#444444", 5.4)
    _arrow(ax, 50, 51, 50, 48.6)

    # ---------- Layer 2: Method ----------
    band(35, 13, "#f4f8fc", C["blue"], "Method — hierarchical grounding fusion  (§3)")
    _box(ax, 3, 39, 14.5, 5, "OLD\nCLIP zero-shot presence", "#eaf3fb", C["blue"], 5.0)
    _box(ax, 19, 39, 14.5, 5, "ALD\nregion-crop CLIP contrast", "#eaf7f1", C["green"], 5.0)
    _box(ax, 35, 39, 14.5, 5, "RLD\nbounding-box geometry", "#fdeee6", C["vermil"], 5.0)
    ax.text(52.2, 41.5, "fuse", ha="center", fontsize=5.6, style="italic", color="#555555")
    _arrow(ax, 50, 41.5, 54.5, 41.5)
    _box(ax, 55.5, 39, 13, 5, "Detect\nflag unsupported claims", "#f3f0f8", C["purple"], 5.0)
    _box(ax, 70, 39, 13, 5, "Revise\ndrop or rewrite them", "#fbf3ea", C["orange"], 5.0)
    _box(ax, 84.5, 39, 13, 5, "Fuse\nwith the model's answer", "#eaf3fb", C["blue"], 5.0)
    ax.text(50, 36.9, "training-free  ·  all models frozen  ·  wraps any backbone "
            "(even API-only) at levels L0 / L1 / L2", ha="center", fontsize=5.3,
            style="italic", color="#555555")
    _arrow(ax, 50, 35, 50, 32.6)

    # ---------- Layer 3: Validation ----------
    band(19, 13, "#f2faf6", C["green"], "Validation  (§4–5)")
    _box(ax, 3, 21, 18, 6.5, "Detection AUC\n0.80 / 0.73 / 0.71 / 0.62\n(POPE · AMBER · VG-Rel)",
         "white", C["blue"], 5.0)
    _box(ax, 23, 21, 18, 6.5, "Leak-free HalluProbe-VL\nobject AUC 0.83\n(generalizes)",
         "white", C["green"], 5.0)
    _box(ax, 43, 21, 18, 6.5, "Mitigation\nCHAIR-i  -20% ~ -41%\non 4 backbones", "white", C["orange"], 5.0)
    _box(ax, 63, 21, 16, 6.5, "Discrimination\nup to +2.0 accuracy\n(matched yes-ratio)",
         "white", C["purple"], 5.0)
    _box(ax, 81, 21, 16, 6.5, "Decoder steering\nwith the same evidence\nFAILS  (§5.4)",
         "white", "#C1121F", 5.0)
    _arrow(ax, 50, 19, 50, 16.6)

    # ---------- Layer 4: Findings & deliverables ----------
    band(3, 13, "#ececf4", "#444444", "Findings & deliverables  (§6–7)")
    ax.text(3, 11.2, "Findings:", fontsize=5.6, fontweight="bold", color="#333333")
    _box(ax, 12, 9.3, 25, 4.2, "the detection–mitigation gap:\ndetection strength does not transfer",
         "white", "#444444", 4.9)
    _box(ax, 39.5, 9.3, 27, 4.2, "the residual principle:\na source helps only where the model is wrong",
         "white", "#444444", 4.9)
    ax.text(3, 5.9, "Deliverables:", fontsize=5.6, fontweight="bold", color="#333333")
    _box(ax, 13, 4.0, 25, 4.0, "hgfusion plug-in — pip-installable, 3-line integration (L0)",
         "#eaf7f1", C["green"], 4.9)
    _box(ax, 40.5, 4.0, 26, 4.0, "HalluProbe-VL diagnostic — 4,594 / 1,996 / 2,000 probes",
         "#eaf3fb", C["blue"], 4.9)
    _box(ax, 69, 4.0, 28, 4.0, "code, adapters for 4 backbones, honest negative results",
         "#f2f2f2", "#888888", 4.9)

    ax.text(50, 0.6, "From three kinds of hallucination to a training-free plug-in — "
            "validated on four backbones, explained by one principle.",
            ha="center", fontsize=5.8, style="italic", color="#666666")
    fig.tight_layout()
    save(fig, "fig1_framework")


# =====================================================================================
# Fig. 2 : algorithm structural flow (left-to-right flowchart with the equations).
#          Stand-in; polished version via FIG2_PROMPT.md.
# =====================================================================================
def fig_principle():
    fig, ax = plt.subplots(figsize=figsize("large", 88))
    ax.set_xlim(0, 104); ax.set_ylim(0, 54); ax.axis("off")
    blue, green, verm = C["blue"], C["green"], C["vermil"]

    # ---- column 1: input ----
    _box(ax, 1, 22, 12, 10, "INPUT\nimage $I$  +\noutput $y$", C["light"], "#888888", 5.8)

    # ---- column 2: decompose ----
    _box(ax, 16, 22, 16, 10,
         "decompose\n$M(y)=M_{obj}\\cup M_{attr}\\cup M_{rel}$\n(Eq. 1)",
         "#f7f7f7", "#666666", 5.4)
    _arrow(ax, 13, 27, 16, 27)

    # ---- column 3: three scorers ----
    _box(ax, 36, 40, 24, 8.5,
         "OLD — object\n$g_{obj}=\\cos(\\phi_v(I),\\phi_t(\\pi_o))$\n(Eq. 4)",
         "#eaf3fb", blue, 5.4)
    _box(ax, 36, 27, 24, 8.5,
         "ALD — attribute\n$g_{attr}=\\cos(R_o,\\pi_{a,o})-\\cos(R_o,\\pi_o)$\n(Eq. 5)",
         "#eaf7f1", green, 5.4)
    _box(ax, 36, 14, 24, 8.5,
         "RLD — relation\n$\\mathrm{ov}(b_s,b_o)$ ;  $\\langle c_s-c_o,\\,u_r\\rangle$\n(Eqs. 6–7)",
         "#fdeee6", verm, 5.4)
    _arrow(ax, 32, 29, 36, 44)
    _arrow(ax, 32, 27, 36, 31)
    _arrow(ax, 32, 25, 36, 18)

    # ---- column 4: test & decide ----
    _box(ax, 66, 43, 20, 7.5,
         "threshold test\n$\\hat{h}(c)=\\mathbf{1}[\\,g(c)<\\tau\\,]$  (Eq. 3)",
         "#f3f0f8", C["purple"], 5.4)
    _arrow(ax, 60, 45.5, 66, 46.5)
    _arrow(ax, 60, 31.5, 66, 44.8)
    _arrow(ax, 60, 18.5, 66, 43.6)

    # decision diamond (object path only)
    dia = plt.Polygon([(66, 30), (73, 34.5), (80, 30), (73, 25.5)],
                      closed=True, facecolor="white", edgecolor="#444444", linewidth=1.1)
    ax.add_patch(dia)
    ax.text(73, 30.6, "$F=\\{o:\\,s_{obj}<\\tau_o\\}$", ha="center", fontsize=5.2)
    ax.text(73, 28.4, "$F=\\emptyset$ ?  (Eq. 8)", ha="center", fontsize=5.2)
    _arrow(ax, 60, 43.5, 66, 31.5, color=blue)          # OLD -> diamond
    ax.text(60.5, 38.6, "object\nevidence only", fontsize=4.9, style="italic",
            color=blue, ha="center")

    ax.text(80.8, 32.6, "yes → keep $y$", fontsize=5.0, style="italic", color=green)
    ax.text(69.2, 22.9, "no → revise", fontsize=5.0, style="italic", color=C["orange"])
    _arrow(ax, 73, 25.5, 73, 21.5)
    _box(ax, 63, 13, 20, 8,
         "revise\nsentence removal (Eq. 9)\nor self-rewrite (re-prompt)",
         "#fbf3ea", C["orange"], 5.2)

    # decision fusion (bottom)
    _box(ax, 63, 2, 20, 8.5,
         "decision fusion\n$z=\\hat{p}_{yes}+\\lambda\\hat{g}$  (Eq. 12)\n"
         "$\\tau^\\star=Q_{1-r_0}(z)$  (Eq. 13)",
         "#eaf3fb", blue, 5.2)
    _arrow(ax, 58, 40, 63, 7.5, color=blue)             # OLD -> fusion
    _arrow(ax, 52, 2.8, 63, 4.2, color="#888888", ls="--")
    ax.text(51.5, 2.4, "$p_{yes}$ from the backbone", ha="right", fontsize=5.0,
            color="#777777", style="italic")

    # ---- column 5: outputs ----
    _box(ax, 90, 43, 13, 7.5, "evidence report\nscores + flags", "#f3f0f8", C["purple"], 5.2)
    _box(ax, 90, 24, 13, 7.5, "verified\ncaption  $y'$", "#fbf3ea", C["orange"], 5.2)
    _box(ax, 90, 4, 13, 7.5, "calibrated\nyes/no answer", "#eaf3fb", blue, 5.2)
    _arrow(ax, 86, 46.7, 90, 46.7)                       # test -> report
    _arrow(ax, 80, 30, 90, 28.6, color=green)            # keep -> caption
    _arrow(ax, 83, 17, 90, 26.4, color=C["orange"])      # revise -> caption
    _arrow(ax, 83, 6.2, 90, 7.2)                         # fusion -> answer

    ax.text(52, 0.4, "One pass: decompose the output, score each claim with its "
            "matched signal, then report, repair, or fuse.",
            ha="center", fontsize=5.8, style="italic", color="#666666")
    fig.tight_layout()
    save(fig, "fig2_principle")


# =====================================================================================
# Fig. 3 : detection curves -- (a) ROC, (b) precision-recall
# =====================================================================================
def _load_roc(bed):
    from sklearn.metrics import roc_curve, roc_auc_score
    fp = os.path.join(CONFIG["roc_dir"], f"{bed}__full.jsonl")
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    y = np.array([r["label"] for r in rows]); s = np.array([r["score"] for r in rows], float)
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, roc_auc_score(y, s)


def fig_detection():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=figsize("large", 78))

    # (a) ROC
    a0.plot([0, 1], [0, 1], "--", color=C["gray"], lw=1.0, label="chance (0.50)")
    for bed, name, ck in CONFIG["roc_beds"]:
        try:
            fpr, tpr, auc = _load_roc(bed)
        except Exception as e:
            print("  skip roc", bed, e); continue
        a0.plot(fpr, tpr, color=C[ck], lw=CONFIG["linewidth"], label=f"{name}  ({auc:.3f})")
    a0.set_xlabel("False positive rate"); a0.set_ylabel("True positive rate")
    a0.set_xlim(0, 1); a0.set_ylim(0, 1.01)
    a0.legend(loc="lower right", title="AUC", title_fontsize=6.5)
    _panel_tag(a0, "(a)", dx=-0.13)

    # (b) Precision-Recall
    try:
        pr = json.load(open(CONFIG["pr_json"], encoding="utf-8"))
    except Exception as e:
        print("  no pr_curves.json:", e); pr = {}
    for bed, name, ck in CONFIG["roc_beds"]:
        if bed not in pr:
            continue
        d = pr[bed]
        a1.plot(d["recall"], d["precision"], color=C[ck], lw=CONFIG["linewidth"],
                label=f"{name}  ({d['ap']:.3f})")
        a1.axhline(d["pos_rate"], ls=":", lw=0.7, color=C[ck], alpha=0.55)
    a1.set_xlabel("Recall"); a1.set_ylabel("Precision")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.01)
    a1.legend(loc="lower left", title="AP", title_fontsize=6.5)
    a1.text(0.985, 0.985, "dotted = class prior (chance)", ha="right", va="top",
            fontsize=6.0, color="#777777", transform=a1.transAxes)
    _panel_tag(a1, "(b)", dx=-0.13)

    fig.tight_layout(w_pad=2.4)
    save(fig, "fig3_detection")


# =====================================================================================
# Fig. 4 : HalluProbe-VL per-type detection
# =====================================================================================
def fig_halluprobe():
    fig, ax = plt.subplots(figsize=figsize("small", 68))
    labels = list(CONFIG["detect_halluprobe"].keys())
    auc = [CONFIG["detect_halluprobe"][k][0] for k in labels]
    ns = [CONFIG["detect_halluprobe"][k][2] for k in labels]
    x = np.arange(len(labels)); cols = [C["blue"], C["green"], C["vermil"]]
    ax.bar(x, auc, 0.6, color=cols, edgecolor="white", linewidth=0.6)
    ax.axhline(0.5, ls="--", color=C["gray"], lw=1.0)
    ax.text(len(labels) - 0.5, 0.515, "chance", fontsize=6.3, color=C["gray"], ha="right")
    for i, (a, n) in enumerate(zip(auc, ns)):
        ax.text(i, a + 0.012, f"{a:.3f}", ha="center", fontsize=CONFIG["font_size"]["annot"])
        ax.text(i, 0.03, f"n={n}", ha="center", fontsize=6.0, color="#666666")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("detection AUC"); ax.set_ylim(0, 0.92)
    fig.tight_layout()
    save(fig, "fig4_halluprobe")


# =====================================================================================
# Fig. 5 : detection-mitigation gap (3 panels)
# =====================================================================================
def fig_gap():
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=figsize("large", 62))

    keys = list(CONFIG["gap_scores"].keys())
    x = np.arange(len(keys)); w = 0.36
    tv = [CONFIG["gap_scores"][k]["true"] for k in keys]
    hv = [CONFIG["gap_scores"][k]["hall"] for k in keys]
    a0.bar(x - w/2, tv, w, color=C["green"], label="true objects", edgecolor="white", linewidth=0.6)
    a0.bar(x + w/2, hv, w, color=C["vermil"], label="hallucinated", edgecolor="white", linewidth=0.6)
    a0.set_xticks(x)
    a0.set_xticklabels([f"{k}\n(AUC {CONFIG['gap_scores'][k]['auc']:.2f})" for k in keys])
    a0.set_ylabel("mean presence score"); a0.set_ylim(0, 0.92)
    a0.legend(loc="upper right"); _panel_tag(a0, "(a)")

    thr = CONFIG["ceiling"]["thr"]
    a1b = a1.twinx()
    l1, = a1.plot(thr, [100*c for c in CONFIG["ceiling"]["chair_i"]], "-o", color=C["blue"],
                  lw=CONFIG["linewidth"], ms=3.5)
    l2, = a1b.plot(thr, CONFIG["ceiling"]["true_kept"], "-s", color=C["green"],
                   lw=CONFIG["linewidth"], ms=3.5)
    a1.set_xlabel("presence threshold"); a1.set_ylabel("CHAIR-i (%)", color=C["blue"])
    a1b.set_ylabel("true objects kept (%)", color=C["green"]); a1b.set_ylim(90, 101)
    a1b.grid(False)
    a1.tick_params(axis="y", labelcolor=C["blue"]); a1b.tick_params(axis="y", labelcolor=C["green"])
    a1.legend([l1, l2], ["CHAIR-i (%)", "true kept (%)"], loc="lower left")
    _panel_tag(a1, "(b)")

    names = list(CONFIG["strategy"].keys())
    dch = [CONFIG["strategy"][n][0] for n in names]
    cols = [C["gray"], C["gray"], C["orange"], C["blue"]]
    y = np.arange(len(names))[::-1]
    a2.barh(y, dch, color=cols, edgecolor="white", linewidth=0.6)
    a2.axvline(0, color="#666666", lw=0.8)
    for yi, d in zip(y, dch):
        a2.text(d + (0.15 if d >= 0 else -0.15), yi, f"{d:+.1f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=CONFIG["font_size"]["annot"])
    a2.set_yticks(y); a2.set_yticklabels(names)
    a2.set_xlabel(r"$\Delta$CHAIR-i (pp)  $\downarrow$ better")
    a2.set_xlim(-5.4, 3.2); _panel_tag(a2, "(c)")
    fig.tight_layout(w_pad=2.6)
    save(fig, "fig5_gap")


# =====================================================================================
# Fig. 6 : object mitigation (2 panels)
# =====================================================================================
def fig_mitigation():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=figsize("medium", 66))
    methods = list(CONFIG["mitig"].keys())
    ci = [CONFIG["mitig"][m][0]*100 for m in methods]
    cs = [CONFIG["mitig"][m][1]*100 for m in methods]
    x = np.arange(len(methods)); w = 0.36
    a0.bar(x - w/2, ci, w, color=C["blue"], label="CHAIR-i", edgecolor="white", linewidth=0.6)
    a0.bar(x + w/2, cs, w, color=C["sky"], label="CHAIR-s", edgecolor="white", linewidth=0.6)
    for i in range(len(methods)):
        a0.text(i - w/2, ci[i] + 0.4, f"{ci[i]:.1f}", ha="center", fontsize=CONFIG["font_size"]["annot"])
        a0.text(i + w/2, cs[i] + 0.4, f"{cs[i]:.1f}", ha="center", fontsize=CONFIG["font_size"]["annot"])
    a0.set_xticks(x); a0.set_xticklabels([m.replace("\n", " ") for m in methods], fontsize=6.6)
    a0.set_ylabel("hallucination rate (%)"); a0.set_ylim(0, 34)
    a0.legend(loc="upper right"); _panel_tag(a0, "(a)")

    mets = list(CONFIG["quality"].keys())
    van = [CONFIG["quality"][m][0] for m in mets]
    our = [CONFIG["quality"][m][1] for m in mets]
    x = np.arange(len(mets)); w = 0.36
    a1.bar(x - w/2, van, w, color=C["gray"], label="vanilla", edgecolor="white", linewidth=0.6)
    a1.bar(x + w/2, our, w, color=C["orange"], label="ours (revise)", edgecolor="white", linewidth=0.6)
    for i in range(len(mets)):
        a1.text(i + w/2, our[i] + 0.008, f"{our[i]-van[i]:+.3f}", ha="center", fontsize=6.3, color="#333333")
    a1.set_xticks(x); a1.set_xticklabels(mets)
    a1.set_ylabel("score"); a1.set_ylim(0, 0.72)
    a1.legend(loc="upper center"); _panel_tag(a1, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_mitigation")


# =====================================================================================
# Fig. 7 : cross-backbone mitigation
# =====================================================================================
def fig_crossbackbone():
    fig, ax = plt.subplots(figsize=figsize("medium", 74))
    bks = list(CONFIG["xbackbone"].keys())
    van = [CONFIG["xbackbone"][b][0] for b in bks]
    slf = [CONFIG["xbackbone"][b][1] for b in bks]
    snt = [CONFIG["xbackbone"][b][2] for b in bks]
    x = np.arange(len(bks)); w = 0.26
    ax.bar(x - w, van, w, color=C["gray"], label="Vanilla", edgecolor="white", linewidth=0.6)
    ax.bar(x, slf, w, color=C["sky"], label="Self-rewrite", edgecolor="white", linewidth=0.6)
    ax.bar(x + w, snt, w, color=C["vermil"], label="Sentence-remove", edgecolor="white", linewidth=0.6)
    for i in range(len(bks)):
        ax.text(i + w, snt[i] + 0.3, f"{snt[i]:.1f}", ha="center", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(bks)
    ax.set_ylabel("CHAIR-i (%)  $\\downarrow$"); ax.set_ylim(0, 22)
    ax.legend(loc="upper right"); ax.set_title("Cross-backbone object de-hallucination")
    fig.tight_layout()
    save(fig, "fig7_crossbackbone")


# =====================================================================================
# Fig. 8 : cross-backbone discriminative gain
# =====================================================================================
def fig_discrim_xbb():
    fig, ax = plt.subplots(figsize=figsize("medium", 72))
    bks = list(CONFIG["discrim_xbb"].keys())
    van = [CONFIG["discrim_xbb"][b][0]*100 for b in bks]
    fus = [CONFIG["discrim_xbb"][b][1]*100 for b in bks]
    x = np.arange(len(bks)); w = 0.36
    ax.bar(x - w/2, van, w, color=C["gray"], label="Vanilla", edgecolor="white", linewidth=0.6)
    ax.bar(x + w/2, fus, w, color=C["blue"], label="+grounding (ours)", edgecolor="white", linewidth=0.6)
    for i, b in enumerate(bks):
        d = (CONFIG["discrim_xbb"][b][1] - CONFIG["discrim_xbb"][b][0]) * 100
        ax.text(i + w/2, fus[i] + 0.15, f"{d:+.1f}", ha="center", fontsize=6.4,
                color=(C["green"] if d > 0.05 else "#777777"))
    ax.set_xticks(x); ax.set_xticklabels(bks)
    ax.set_ylabel("POPE accuracy (%)"); ax.set_ylim(82, 90)
    ax.legend(loc="upper right"); ax.set_title("Discriminative gain at matched yes-ratio")
    fig.tight_layout()
    save(fig, "fig8_discrim_xbb")


# =====================================================================================
# Fig. 9 (appendix) : single-backbone discriminative POPE fusion
# =====================================================================================
def fig_discriminative():
    fig, ax = plt.subplots(figsize=figsize("small", 66))
    methods = list(CONFIG["discrim"].keys())
    acc = [CONFIG["discrim"][m][0]*100 for m in methods]
    f1 = [CONFIG["discrim"][m][1]*100 for m in methods]
    x = np.arange(len(methods)); w = 0.36
    ax.bar(x - w/2, acc, w, color=C["blue"], label="Accuracy", edgecolor="white", linewidth=0.6)
    ax.bar(x + w/2, f1, w, color=C["green"], label="F1", edgecolor="white", linewidth=0.6)
    for i in range(len(methods)):
        ax.text(i - w/2, acc[i] + 0.15, f"{acc[i]:.1f}", ha="center", fontsize=6.4)
        ax.text(i + w/2, f1[i] + 0.15, f"{f1[i]:.1f}", ha="center", fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("\n", " ") for m in methods], fontsize=7)
    ax.set_ylabel("POPE score (%)"); ax.set_ylim(84, 89)
    ax.legend(loc="upper left")
    ax.text(0.5, 84.4, "matched yes-ratio = 0.460", transform=ax.transData,
            fontsize=6.3, color="#555555")
    fig.tight_layout()
    save(fig, "fig9_discriminative")


FIGURES = {
    "framework":      fig_framework,
    "principle":      fig_principle,
    "detection":      fig_detection,
    "halluprobe":     fig_halluprobe,
    "gap":            fig_gap,
    "mitigation":     fig_mitigation,
    "crossbackbone":  fig_crossbackbone,
    "discrimxbb":     fig_discrim_xbb,
    "discriminative": fig_discriminative,
}


def main():
    setup_style()
    for name in (sys.argv[1:] or list(FIGURES)):
        if name in FIGURES:
            FIGURES[name]()
        else:
            print("unknown figure:", name, "| available:", list(FIGURES))


if __name__ == "__main__":
    main()
