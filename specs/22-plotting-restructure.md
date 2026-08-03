# 22 — Plotting Code Restructure

**Status**: Implemented (Phase 1 + Phase 2; Phase 3 skipped — paper-frozen, optional)
**Scope**: `src/simplegnn/framework/utils/evaluation.py`, `src/simplegnn/utils/utils.py`,
`src/simplegnn/datasets/utils/graph_drawing.py`, `experiments/base_paper/src/*.py`,
`experiments/base_paper/regression/substructure_counting/plot_substructure_counting.py`

## Findings (verified 2026-08-03)

### 1. Dead plotting/evaluation code in the core package

Grep over `src/`, `examples/`, `experiments/`, `tests/` (py + sh) found **zero callers** for:

| Function | Location | Size |
|---|---|---|
| `epoch_accuracy` | `evaluation.py:10` | ~195 lines |
| `evaluateGraphLearningNN` | `evaluation.py:206` | ~200 lines |
| `model_selection_evaluation_mae` | `evaluation.py:787` | ~210 lines |
| `live_plotter` | `utils.py:410` | |
| `plot_init` | `utils.py:435` | |
| `plot_learning_data` | `utils.py:462` | |
| `add_values` | `utils.py:491` | |
| `live_plotter_lines` | `utils.py:499` | |

The **only live function** in `evaluation.py` is `model_selection_evaluation`
(imported by `core.py:57`), and it does **not plot at all**. The
`from matplotlib import pyplot as plt` at `evaluation.py:5` and the mid-file
`import matplotlib.pyplot as plt` at `utils.py:285` serve only dead code.

Consequences of the dead code:
- The hardcoded `Results/{db}/Plots/...` paths and `plt.show()` calls (the
  worst structural smells) live entirely in dead functions.
- The `Results/` (capital, dead code) vs `results/` (lowercase, live code)
  path inconsistency disappears once dead code is removed.
- `model_selection_evaluation_mae` is documented as the regression variant,
  but regression experiments (ZINC/QM9/OGB) go through
  `core.evaluate_results()` → `model_selection_evaluation`, which handles
  them. The mae variant is a superseded legacy path.

### 2. Verbatim duplication in paper-figure scripts

- `CustomColorMap` exists **three times**, byte-identical:
  `graph_drawing.py:108` (canonical, tested), `plot_zinc.py:28`,
  `plot_substructure_counting.py:26`.
- `parameter_update()` (pgf/LaTeX rcParams setup) is duplicated verbatim in
  `plot_zinc.py:13` and `plot_substructure_counting.py:12`, and the same
  rcParams block is inlined at least twice more inside
  `latex_plots.py` (lines ~14, ~125).
- The position-cache pattern (`.../Plots/Positions/` + `resolve_positions`)
  is repeated across `latex_plots.py:277` and `plot_zinc.py:77`.

### 3. What is already well-structured (keep as-is)

- `graph_drawing.py`: real module boundary (`GraphDrawing`, `CustomColorMap`,
  `TabColorMap`, `compute_positions`, `resolve_positions`), covered by
  `tests/test_layer_drawing.py`.
- ShareGNN `draw(ax, ...)` methods (`inv_based_message_passing.py:1786`,
  `inv_based_pooling.py:335`): correctly take an `ax`, delegate to
  `GraphDrawing`, never save/show. This is the pattern the rest should follow.

## Plan

### Phase 1 — Delete dead code (core package becomes plot-free)

1. `evaluation.py`: delete `epoch_accuracy`, `evaluateGraphLearningNN`,
   `model_selection_evaluation_mae`; drop the matplotlib import; fix the
   dangling `See Also` reference to the mae variant in
   `model_selection_evaluation`'s docstring. File shrinks ~996 → ~380 lines
   and contains only the live model-selection logic.
2. `utils.py`: delete the five dead plotting functions and the mid-file
   matplotlib import (lines 285, 410–507). Also fixes the import-order
   violation (import buried mid-file).
3. Result: matplotlib usage inside `src/simplegnn/` is confined to
   `datasets/utils/graph_drawing.py`, `graph_functions.py` (debug `plot=`
   flags), `custom_benchmarks/snowflakes.py` (debug flags), and the ShareGNN
   `draw()` methods — all visualization-by-design.

*Decision point*: deletion is recommended (git history preserves the code;
`evaluateGraphLearningNN`'s selection logic is superseded by
`model_selection_evaluation`). Alternative: move to
`experiments/base_paper/src/legacy_evaluation.py` if interactive use is
still expected.

### Phase 2 — Deduplicate paper-figure helpers

1. New module `experiments/base_paper/src/plot_common.py`:
   - `setup_pgf(font_size=30)` — the shared rcParams/pgf block, replacing
     both `parameter_update()` copies and the inline blocks in
     `latex_plots.py`.
   - `save_latex_figure(fig, path)` — `mkdir -p` + `savefig(...,
     bbox_inches='tight', backend='pgf')` boilerplate.
2. Delete local `CustomColorMap` copies in `plot_zinc.py` and
   `plot_substructure_counting.py`; import from
   `simplegnn.datasets.utils.graph_drawing` (already the tested canonical
   copy, already imported by these scripts).
3. `latex_plots.py`, `plot_zinc.py`, `plot_substructure_counting.py` import
   from `plot_common` (scripts already `sys.path`-hack or run from repo root;
   keep whatever import mechanism they currently use).

### Phase 3 — Optional polish (separate commit, can be skipped)

- Extract the repeated position-cache snippet into
  `resolve_positions`-adjacent helper or `plot_common`.
- `latex_plots.py` (653 lines) could split into `ablation_plots.py` /
  `network_visualization.py` / `weight_analysis.py`, but the functions are
  independent and paper-frozen — only worth doing if these figures will be
  regenerated for a revision.

## Non-goals

- No new `framework/utils/plotting.py`: the training-curve plotting it would
  host is exactly the dead code being deleted. If live training-curve plots
  are wanted later, write them fresh against the current CSV format,
  returning `Figure` objects (no `show()`, no hardcoded paths).
- No changes to `graph_drawing.py` internals or ShareGNN `draw()` APIs.

## Verification

1. `pytest tests -q` (includes `test_layer_drawing.py` and import smoke tests).
2. `grep -rn "matplotlib" src/simplegnn/framework src/simplegnn/utils` → empty.
3. Import smoke: `python -c "from simplegnn.framework.core import FrameworkMain"`.
4. Paper scripts still import: `python -c "import sys; sys.path.insert(0,'.');
   import experiments.base_paper.src.latex_plots"` (or run their `main()`
   guards if data present).

## Risk assessment

- **Low**: all deleted functions have zero static callers; the only consumer
  of `evaluation.py` is `core.py:57` (untouched function).
- **Residual risk**: interactive/notebook use of the deleted evaluation
  helpers outside the repo. Mitigation: plain deletion in a dedicated commit
  ("Remove dead plotting/evaluation code") so it is trivially revertable.
