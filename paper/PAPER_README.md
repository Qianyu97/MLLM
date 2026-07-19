# Information Fusion submission — build package

**Paper:** *Hierarchical Grounding Fusion for Detecting and Mitigating Object,
Attribute, and Relation Hallucinations in Multimodal Large Language Models*

## Files
| File | What it is |
|---|---|
| `CMPSA_InformationFusion.docx` | The full paper (Word), Arial, embedded figures + rendered equations |
| `figures.py` | Central figure script. Edit the `CONFIG` block, then `python figures.py` |
| `build_docx.py` | Builds the .docx. All text/tables/refs live in the `CONTENT` block at the bottom |
| `FIG_PROMPTS.md` | GPT drawing prompts for Fig. 1 and Fig. 2 (polished replacements) |
| `pr_curves.json` | Precomputed precision–recall curves (from the real per-item dumps) |
| `figs/` | fig1–fig9, each as `.pdf` (vector) + `.png` (600 dpi) |
| `eqs/` | 15 equation images (600 dpi) rendered by `build_docx.py` |

## Rebuild
```
python figures.py        # regenerate all figures (190/140/85 mm, 600 dpi, Arial)
python build_docx.py     # regenerate CMPSA_InformationFusion.docx
```

## Structure
1. Introduction · 2. Related work · **3. Method** (3.1 formulation, 3.2 hypothesis
test, 3.3 grounding signals, 3.4 detect-then-revise, 3.5 decision fusion,
**3.6 residual view**, **3.7 deployment as a plug-in wrapper (access levels
L0/L1/L2) & cost**) · **4. Data** (4.1 corpora, 4.2 benchmarks,
4.3 HalluProbe-VL) · 5. Experiments (5.1–5.7) · 6. Discussion (incl. Practicality) ·
7. Conclusion · Appendix A · Appendix B · References (26, real, recency-weighted).

**Plug-in framing (validated, not aspirational):** L0 = detection + sentence removal,
needs only image + output text (wraps even API-only backbones; the level that
transferred to all four backbones) · L1 = self-rewrite, needs re-prompting an
instruction-following backbone · L2 = decision fusion, needs token probability p_yes.

## Figure map and placement
| Fig | Where | Width | Content |
|---|---|---|---|
| 1 | §1 | 190 mm | framework overview *(GPT prompt available)* |
| 2 | §3 | 190 mm | model principle / signal flow *(GPT prompt available)* |
| 3 | §5.2 | 190 mm | detection ROC + precision–recall (2 panels) |
| 4 | §5.3 | 85 mm | HalluProbe-VL AUC per kind |
| 5 | §5.4 | 190 mm | detection–mitigation gap (3 panels) |
| 6 | §5.5 | 140 mm | object mitigation + caption quality |
| 7 | §5.6 | 140 mm | cross-backbone mitigation |
| 8 | §5.7 | 140 mm | cross-backbone discrimination |
| 9 | **App. A** | 85 mm | single-backbone discrimination |

**Main text:** Fig. 1–8, Table 1–7.  **Appendix A:** Fig. 9 + Table 8.
**Appendix B:** negative results (text only).

| Table | Where | Content |
|---|---|---|
| 1 | §2 | positioning vs. related work |
| 2 | §4.2 | datasets: task, source, n, class balance |
| 3 | §5.2 | detection: AUC/AP/F1 + **P@F1, R@F1, τ\*** + **learned-alignment baseline** |
| 4 | §5.3 | HalluProbe-VL detection (released n vs. **evaluated n**) |
| 5 | §5.5 | object mitigation (CHAIR-500) |
| 6 | §5.6 | cross-backbone mitigation |
| 7 | §5.7 | cross-backbone discrimination |
| 8 | App. A | single-backbone discrimination |

## Notes on integrity
- All 26 references are real; method/related-work citations are weighted to
  2023–2024, with foundational tools (CLIP, COCO, VG, CHAIR, BLEU, ROUGE)
  necessarily older.
- Every number comes from the project's re-checked full-scale runs
  (`results/metrics`, `results/predictions`). Table 3's P/R/τ\* were recomputed
  directly from the per-item dumps.
- Reported honestly and on purpose: the LLaVA-1.6 discriminative result is a
  genuine **+0.0** null; the learned alignment is at **chance** (0.47–0.54);
  HalluProbe-VL attribute/relation AUC is computed on the **evaluable subsets**
  (555 / 634) rather than all released probes (1,996 / 2,000).
