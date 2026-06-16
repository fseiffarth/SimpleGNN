# Add Errica-style "fair" per-fold model selection / evaluation

## Context

Today SimpleGNN does **global** model selection: `evaluate_results()` averages each
hyperparameter config's validation accuracy *across all folds*, picks **one** global
best config, and `run_best_configuration()` re-trains that single config on every test
fold (`src/simplegnn/framework/core.py`, `model_selection_evaluation` in
`src/simplegnn/framework/utils/evaluation.py`).

The user wants the **fair comparison protocol** from Errica et al., *"A Fair Comparison
of Graph Neural Networks for Graph Classification"* (ICLR 2020, arXiv:1912.09893),
confirmed from the paper:

1. **10-fold outer CV** for model assessment.
2. **Inner holdout** (90/10 train/val) for model selection **inside each outer fold**.
3. The selected best hyperparameter config **may differ per fold** ("we obtain different
   best hyper-parameter configurations").
4. **3 final training runs** per fold; the fold's test score is their mean.
5. Final metric = **mean ± std across the 10 folds**.

This is a perfect fit for the existing data: each split fold already carries its own
`train`/`validation`/`test` (`core.py:721-724`), so the per-fold `validation` is the
inner-holdout selection signal and the per-fold `test` is the outer assessment. The
grid search (`run_configurations`) already trains every `(config, fold, run)` and logs
`ValidationAccuracy` **and** `TestAccuracy` per epoch — so the fair selection is a new
*aggregation*, plus an optional per-fold re-run mirroring the existing best-model flow.

The goal: **add** a fair evaluation path (do not change the current global one), giving
up to 10 different per-fold models and an Errica-faithful mean±std test estimate.

## Current mechanics (verified)

- Per-run CSV (`model_configuration.py:1197`) columns: `Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss(..);EpochAccuracy;...;ValidationLoss;ValidationAccuracy;...;TestLoss;TestAccuracy;...`. `ValidationNumber` = fold.
- File name: `{db}_{config}_Results_run_id_{run}_validation_step_{fold}.csv`; `model_selection_evaluation` parses `ConfigurationId` from the `Configuration_XXXXXX` token and reads files matching `run_id`+`validation_step`; "best model" files are matched by the substring `Best_Configuration`.
- Best-epoch-within-a-run convention (reuse verbatim): max `ValidationAccuracy`, ties → min `ValidationLoss` (and `evaluation_type='loss'` inverts this) — see `evaluation.py:599-624`.
- `num_runs` (grid, default 1) and `evaluation_run_number` (re-run, default 3) control seeds.

## Changes

### 1. `src/simplegnn/framework/utils/evaluation.py` — new `fair_model_selection_evaluation(...)`

New function modeled on `model_selection_evaluation` (reuse its file-loading and
best-epoch selection so behavior matches the global path):

- Load the grid result CSVs (or, when `evaluate_best_model=True`, the fair re-run files
  tagged `Best_Configuration_Fair`).
- For each `(ConfigurationId, RunNumber, ValidationNumber)` pick the best epoch (same
  criterion as the global path), then **group by `(ValidationNumber, ConfigurationId)`**
  and average over `RunNumber` (size-weighted, same as existing aggregation) → per
  `(fold, config)` validation/test mean & std.
- **Per fold**, select the `ConfigurationId` with best mean `ValidationAccuracy` (tie →
  min `ValidationLoss`; honor `evaluation_type`). This is the per-fold model selection.
- Write `summary_fair.csv`: one row per fold — `ValidationNumber, ConfigurationId,
  N_runs, Validation Accuracy Mean/Std, Validation Loss Mean/Std, Test Accuracy
  Mean/Std, Test Loss Mean/Std, Epoch`.
- Write `summary_fair_mean.csv`: the headline Errica number — `Test Accuracy Mean` =
  mean **across folds** of the per-fold selected test accuracy, `Test Accuracy Std` =
  std across folds; plus mean validation accuracy and the list of selected config ids.
- `get_best_per_fold=True` → return `{validation_id: config_id}` (used by the re-run).

### 2. `src/simplegnn/framework/core.py` — two new methods (no change to existing ones)

- `evaluate_results_fair(self, evaluate_best_model=False)` — mirror of `evaluate_results`
  (same per-dataset loop and skip-if-exists guard, but on `summary_fair*.csv`); delegates
  to `fair_model_selection_evaluation`. With `evaluate_best_model=False` it computes the
  fair estimate **directly from the grid results** (uses grid `num_runs` as the per-fold
  runs); with `True` it aggregates the per-fold re-runs.
- `run_best_configuration_fair(self, num_threads=-1)` — mirror of
  `run_best_configuration`: get `{fold: config}` via `get_best_per_fold=True`, then for
  each `(fold, run in range(evaluation_run_number))` re-train **that fold's** selected
  config on that fold via the existing `self.run_configuration(...)`, labeling output
  `config_id = f'Best_Configuration_Fair_{best_per_fold[fold]:06d}'` (per-fold files stay
  distinct via the `validation_step_{fold}` suffix already in the filename). Parallelized
  over `(fold, run)` exactly like the existing method.

### Resulting fair pipeline (added alongside the current 5-step flow)

```
preprocessing(1)
run_configurations(-1)                              # existing grid (set num_runs=3 to match Errica's 3 runs in the lightweight path)
evaluate_results_fair()                             # per-fold selection -> summary_fair.csv (+ _mean)
run_best_configuration_fair(-1)                     # re-train each fold's selected config 3x  (fully faithful)
evaluate_results_fair(evaluate_best_model=True)     # final mean±std across folds -> summary_fair*.csv
```

The lightweight path (stop after `evaluate_results_fair()`) already yields the "10
different models, per-fold test" result the user described; the re-run step adds Errica's
3-runs-per-fold robustness.

### 3. (Optional wiring) experiment scripts

Leave existing scripts untouched. Optionally add a `--fair/--no-fair` flag (or a sibling
runner) to `experiments/base_paper/classification/tu/experiments_fair_real_world.py` that
swaps the last three calls for the fair pipeline, writing under a `Fair/` results subdir.
Decide during implementation; not required for the capability.

## Out of scope / follow-ups

- Regression/MAE fair selection (there is a separate `model_selection_evaluation_mae`);
  the first cut targets classification `accuracy`/`loss`. DHFR (the motivating case) is
  classification.
- No change to training, splits, or the global evaluation path.

## Verification

1. **Unit test** `tests/test_fair_model_selection_unit.py` (no torch needed —
   `evaluation.py` only uses pandas/numpy): synthesize a tiny `Results/` dir with
   hand-made per-run CSVs (e.g. 3 configs × 10 folds × 2 runs) where a *different* config
   is best per fold; assert `fair_model_selection_evaluation` (a) selects the intended
   config per fold in `summary_fair.csv`, (b) `summary_fair_mean.csv` test mean/std equal
   the hand-computed values, (c) `get_best_per_fold` returns the right map, (d) the global
   `model_selection_evaluation` output is unchanged (regression guard).
2. **End-to-end on DHFR** using the recovered interpreter (the venv packages are intact;
   drive them with `/opt/blender-5.1.2-linux-x64/5.1/python/bin/python3.13` and
   `PYTHONPATH=src:venv/lib/python3.13/site-packages`, or repoint `venv/bin/python3.13`):
   a small DHFR-only main config (reduced epochs, a 2-config grid) →
   `run_configurations` → `evaluate_results_fair()` → `run_best_configuration_fair()` →
   `evaluate_results_fair(evaluate_best_model=True)`; confirm `summary_fair.csv` lists one
   selected config per fold and `summary_fair_mean.csv` holds the cross-fold mean±std.
3. Run the existing unit suite (`pytest tests -q`) to confirm no regression once the
   interpreter is restored.

## Critical files

- `src/simplegnn/framework/utils/evaluation.py` — add `fair_model_selection_evaluation` (reuse `model_selection_evaluation` logic at lines 513-700).
- `src/simplegnn/framework/core.py` — add `evaluate_results_fair` (model on `evaluate_results`, 303-387) and `run_best_configuration_fair` (model on `run_best_configuration`, 389-478).
- `tests/test_fair_model_selection_unit.py` — new unit test.
