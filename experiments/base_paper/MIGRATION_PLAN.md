# Migrate `experiments/base_paper/` to the current FrameworkMain API and config schema

## Context

`experiments/base_paper/` is a snapshot of the **old** repository layout (formerly `paper_experiments/`), predating the `simplegnn` package refactor. Nothing in it runs against the current code:

1. **Dead API** — every entry point uses `from src.Experiment.ExperimentMain import ExperimentMain`; the current API is `from simplegnn.framework.core import FrameworkMain` (`src/simplegnn/framework/core.py:66`).
2. **Wrong path prefix** — scripts/configs reference `paper_experiments/...`; actual location is `experiments/base_paper/...`. Shell scripts also compute the wrong `ROOT_DIR` (one level up instead of repo root).
3. **Old config schema** — monolithic main configs (`network_config_file`, inline hyperparameters, `networks:` key) vs. the current three-tier schema (`paths: {data, results, splits, models, hyperparameters}` + separate models yml + parameters yml). 106 YAML files affected; 69 are generated ablation-threshold variants.
4. **Missing split sources** — scripts copy splits from `Data/Splits[Simple]/` (gone); current splits live in `src/simplegnn/datasets/splits/{fair,standard,fixed,substructure_counting}/` and are referenced directly. Synthetic-dataset splits don't exist anywhere and must be generated (the framework now **requires** the split JSON to exist at `FrameworkMain` construction — `configuration_checks.py:68-76`).
5. **Removed dependencies** — `experiments/base_paper/src/classification_baselines.py` and `get_gnn_comparison_data.py` import `src.Competitors.Kernels.*` / `src.Preprocessing.GraphData.*`, which were deleted from the repo.
6. **QM9.sh** targets `regression/QM9/main_QM.py`, which doesn't exist anywhere.

**User decisions:** full migration of all scripts + configs; helpers: fix what's fixable, mark the rest deprecated (don't delete); QM9.sh: keep, mark broken.

## Verified schema deltas

### Main config → three-tier split
| Old | New |
|---|---|
| dataset `type:` | `source:` (values `TUDataset`, `ZINC`, `gnn_benchmark`, `SubstructureBenchmark`, `generate_from_function` all still valid) |
| dataset `network_config_file:` | `paths.models:` |
| dataset `with_splits`, `split_function`, `split_function_args` | **drop** — split JSON path is mandatory in `paths.splits` |
| dataset `validation_folds` | **keep** (still read at `core.py:715`); set to the fold count of the splits file (10 TU/synthetic, 5 CSL, 1 ZINC/substructure) |
| dataset `generate_function`, `generate_function_args` | keep verbatim (still consumed, `preprocessing.py:103`) |
| top-level `paths:` block | per-dataset `paths: {data, labels, properties, results, models, hyperparameters, splits}` — labels/properties are required whenever the model has an `invariant_based_convolution` layer (all base_paper models) |
| all inline hyperparameters (`optimizer`, `loss`, `batch_size`, `learning_rate`, `epochs`, `input_features`, `device`, `precision`, `mode`, `early_stopping`, `scheduler`, `training_data_sampling`, `weight_initialization`, `rule_occurrence_threshold`, `rule_occurrence_upper_threshold`, `best_model`, `convolution_grad`, `aggregation_grad`) | move to a new `parameters_*.yml` — **all keys verified still consumed** (configuration_checks.py:160-208, model_configuration.py, inv_based_pooling.py:83-108, inv_based_message_passing.py:271-272) |
| top-level `bias: True` (substructure main), `use_feature_transformation` (commented) | drop — no consumer |

### Model config renames
| Old | New | Where enforced |
|---|---|---|
| top-level `networks:` | `models:` | configuration_checks.py:130 (22 files) |
| linear `output_features:` | `out_features:` | layer_loader.py:165 |
| `activation: 'tanh'` (classification configs, ~836 occurrences) | `'torch.nn.Tanh()'` | framework_layer.py:281-296 (exact-string ACTIVATION_MAP, no aliases) |
| full-form conv heads without `num:` | add `num: 1` per head (or collapse duplicates) | layer_loader.py:224-225 |
| `invariant_based_aggregation` layer-level `out_dim: D` | remove; `num: D` per head | cf. examples/zinc/models_ShareGNN.yml |
| short-form layers (`heads: [1]`, list-style labels/properties — all classification configs) | unchanged, still supported | layer_loader.py:45-153 |

Full-form fixes needed only in: `regression/ZINC/configs/network_ZINC{,_test,_full}.yml` and `regression/substructure_counting/configs/network_substructure_counting{,_multi}.yml`.

### Splits mapping (existence verified)
| Experiment | `source` | splits |
|---|---|---|
| ZINC / ZINC test | `ZINC` | `src/simplegnn/datasets/splits/fixed/ZINC_splits.json` |
| ZINC full (name `ZINC-full`) | `ZINC` | `.../fixed/ZINC-full_splits.json` |
| Substructure counting (7 datasets) | `SubstructureBenchmark` | `.../substructure_counting/<name>_splits.json` (all 7 exist) |
| TU fair + both ablations (6 datasets) | `TUDataset` | `.../fair/<name>_splits.json` (all 6 exist) |
| TU standard/sota (4 datasets) | `TUDataset` | `.../standard/<name>_splits.json` (all 4 exist) |
| Synthetic: CSL | `gnn_benchmark` | **must generate** → `experiments/base_paper/splits/synthetic/CSL_splits.json` (5 folds) |
| Synthetic: EvenOddRings2_16, EvenOddRingsCount16, LongRings100, Snowflakes | `generate_from_function` (functions verified in `custom_datasets.py`) | **must generate** → `experiments/base_paper/splits/synthetic/<name>_splits.json` (10 folds) |

Data paths: `data/ZINC/`, `data/TUDatasets/`, `data/SubstructureBenchmark/`, `data/Synthetic/` (+ `labels/`, `properties/` subdirs). Results: `results/base_paper/<regression|classification>/...`. All auto-created by preprocessing.

## Migration status — ALL PHASES COMPLETE (Phases 0–8)

Every phase below has been implemented. Validation performed (the venv was broken —
its base interpreter `/usr/bin/python3.13` is missing — so torch-dependent steps
could not be run; see caveat):

- **Config-load equivalent**: a torch-free validator (replicating
  `configuration_checks` + the real `layer_loader`) was run on **all 84 migrated
  main configs** (3 ZINC, 1 substructure, 4 synthetic, 6 TU fair/sota, 1 distance,
  69 threshold) — all pass (required keys, paths, split-file existence, source
  whitelist, generate_function existence, no leftover `networks:`/`output_features`/
  `out_dim`/`'tanh'`, valid full/short-form layers with `num`/`out_features`).
- **All 17 base_paper `.py` files** `py_compile` cleanly; both deprecated scripts
  exit non-zero with their notice; `QM9.sh` exits 1 with its notice; all 8 shell
  scripts pass `bash -n`.
- **Could NOT run** (need a working torch env): `preprocessing()` smoke tests,
  the ZINC micro end-to-end, and `pytest tests -q`. Rebuild the venv
  (`./install.sh`) to complete these.

## Work plan (phased, each smoke-tested before the next)

### Phase 0 — Shared infrastructure  ✅ DONE
1. **Synthetic split JSONs** — *deviation from original plan (user decision): reuse the
   existing paper splits instead of regenerating.* The 5 split files now live at
   `experiments/base_paper/splits/synthetic/{CSL,EvenOddRings2_16,EvenOddRingsCount16,LongRings100,Snowflakes}_splits.json`,
   recovered verbatim from the predecessor repo `RuleGNN` (the old `paper_experiments/`
   layout this folder snapshots). The 4 ring/snowflake files come from `RuleGNN/Data/Splits/`;
   **CSL** uses `RuleGNN/paper_experiments/Data/Splits/CSL_splits.json` because every other
   CSL copy had validation overlapping test (rejected by `load_splits` as non-disjoint).
   All 5 validated: correct fold counts (CSL 5, rest 10), pairwise-disjoint, and union ==
   `0..N-1` for N = 150/1200/1200/1200/1000. Provenance recorded in
   `experiments/base_paper/splits/synthetic/README.md`. **No `tools/generate_synthetic_splits.py`
   was created** — drop it from "Critical files" expectations.
2. **All 8 shell scripts** ✅ migrated to the `examples/zinc/run_zinc.sh` pattern:
   `ROOT_DIR` = repo root (two levels up), `PYTHONPATH=$ROOT_DIR/src`, `cd $ROOT_DIR`, run
   `python experiments/base_paper/<...>.py --num_threads $NUM_THREADS`. All pass `bash -n`.
   `QM9.sh` boilerplate migrated; its broken-experiment notice is still deferred to Phase 8.

### Phase 1 — ZINC (template: `examples/zinc/`)
- 3 main configs → new schema; **new** `configs/parameters_ZINC*.yml` (content ≈ `examples/zinc/parameters.yml`).
- 3 network ymls: `networks:`→`models:`, `output_features`→`out_features`, conv heads `num: 1`, aggregation `out_dim: 10` → per-head `num: 10`.
- `main_ZINC{,_test,_full}.py`: `FrameworkMain`, method names per examples pattern, config paths → `experiments/base_paper/...`.
- Smoke test with the test variant (data/ZINC already on disk).

### Phase 2 — Substructure counting
- Main config → 7 dataset entries (`SubstructureBenchmark`, per-dataset model path, shared new parameters yml; drop top-level `bias`).
- 2 network ymls full-form fixes; `main_substructure_counting.py` → FrameworkMain (`evaluate_model("multi", best=False)` is API-compatible).

### Phase 3 — Synthetic classification (needs Phase 0 splits)
- 4 main configs → 5 dataset entries each (CSL `gnn_benchmark`; 4 × `generate_from_function` keeping generate args verbatim) + 4 new parameters ymls (differ only in `input_features` / grad flags / results path).
- 5 model configs: `networks:`→`models:` + tanh fix only (short-form).
- `experiments_synthetic.py`: FrameworkMain, delete split-copy helper, fix 4 config paths.

### Phase 4 — TU fair + standard
- 4 fair main configs (splits/fair/) + 2 sota main configs (splits/standard/) + parameters ymls (encoder: `aggregation_grad: False`; decoder: `convolution_grad: False`).
- `config_molecule.yml`, `config_social.yml`: models/tanh fix.
- `experiments_fair_real_world.py`, `experiments_standard_real_world.py`: FrameworkMain (`evaluate_results(evaluate_validation_only=True)` still supported, core.py:304), delete split copying.

### Phase 5 — Ablation distances
- Main config + parameters yml; 4 `config_ablation_distances_*.yml` (models/tanh); `experiments_ablation_distance.py`.

### Phase 6 — Ablation threshold (69 configs via script)
- **New** `experiments/base_paper/tools/migrate_threshold_configs.py`: for each `{lower,lower_upper,upper}/main_config_ablation_threshold_{1..20,30,40,50}.yml`, extract `rule_occurrence_threshold` (+ optional `rule_occurrence_upper_threshold`) and results subpath; emit new-style main yml (6 TU datasets, splits/fair/) + per-threshold parameters yml, overwriting in place to preserve the naming `experiments_ablation_threshold.py` constructs.
- Hand-migrate the 6 shared `config_ablation_threshold_*.yml`.
- `experiments_ablation_threshold.py`: FrameworkMain, drop split copying; note it iterates `range(1, 21)` while 30/40/50 configs also exist — extend the list to cover all 23.

### Phase 7 — Helpers in `experiments/base_paper/src/`
- **Fixable**: `latex.py` (FrameworkMain; ShareGNN layer imports → `simplegnn.models.ShareGNN.layers.*`; `get_run_configs` → `simplegnn.framework.run_configuration`; results paths), `latex_plots.py` (graph-drawing imports → `simplegnn.datasets.utils.graph_drawing`), `plot_zinc.py` + `regression/substructure_counting/plot_substructure_counting.py` (`Load_Splits(path, db)` → `load_splits(json_path)`; adapt from positional-tuple to dict `{'test','train','validation'}` return), `classification_sharegnn.py` (rewrite `paper_experiments.*` imports to call the migrated sibling scripts).
- **Deprecated** (dependencies removed from repo): `classification_baselines.py`, `get_gnn_comparison_data.py` — prominent `DEPRECATED` header + main-guard that prints the notice and exits non-zero.

### Phase 8 — QM9.sh
- Keep; replace run line with a stderr notice ("QM9 experiment files were lost in the repo migration; splits preserved at `src/simplegnn/datasets/splits/tu_splits/QM9_splits.json`") + `exit 1`.

## Verification

1. **Config-load check** for every migrated main yml (cheap, no data): construct `FrameworkMain(path)` and expand all run configs via `get_run_configs` — catches missing `num`, leftover `output_features`/`networks:`/`'tanh'`, and missing split files. Run for all 69 threshold configs too.
2. **Preprocessing smoke test** (`preprocessing(num_threads=1)`) on the cheapest config per family: ZINC test, `triangle` (substructure), `LongRings100` (synthetic), `IMDB-BINARY` (fair + standard), threshold `lower/1`, distances.
3. One micro end-to-end: `run_configurations(1)` on the ZINC test variant only.
4. `bash -n` on all shell scripts; run `ZINC_test.sh` until preprocessing starts; `QM9.sh` must exit non-zero with the notice.
5. `pytest tests -q` stays green (no `src/simplegnn` code is touched).

## Critical files
- `src/simplegnn/framework/utils/configuration_checks.py` — the contract every new config must satisfy
- `src/simplegnn/models/layers/utils/layer_loader.py` — full-form vs short-form layer rules (`num`, `out_features`)
- `examples/zinc/{main.yml,models_ShareGNN.yml,parameters.yml,run_zinc.sh,main.py}` — working migration template
- `src/simplegnn/utils/tu_splits.py` (`_build_splits`) — reused by the synthetic split generator
- `experiments/base_paper/regression/ZINC/configs/network_ZINC.yml` — hardest full-form model migration, pattern for the rest
