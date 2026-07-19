"""CMPSA paper visualization: tables (make_tables) and figures (make_figures).

This subpackage is *plotting only*. It must run on a CPU-only / data-prep box,
so it imports **pandas / matplotlib / seaborn / numpy / json** exclusively and
NEVER imports torch or transformers.

Modules
-------
- :mod:`cmpsa.viz.make_tables`   aggregate RESULTS/metrics/** into paper tables
  (Table5 main, Table6 HalluProbe, Table7 ablation, Table8 cross-backbone,
  Table9 efficiency) and write them as both CSV and LaTeX to ``paths.TABLES_DIR``.
- :mod:`cmpsa.viz.make_figures`  render the paper figures (fig12 dataset stats,
  fig13 ablation, fig14 robustness, fig_psas_tsne, fig_main_compare) as PNG+PDF
  to ``paths.FIGURES_DIR``.

Both modules expose a ``--demo`` flag to synthesize placeholder data so the
plotting / table pipeline can be exercised end-to-end before real results exist.
Every demo artefact is watermarked ``DEMO(占位,非真实结果)``.
"""

__all__ = ["make_tables", "make_figures"]
