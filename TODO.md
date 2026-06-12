# TODO — Code Review Findings

A prioritized backlog from a code review of `src/simplegnn/`. Each item lists a `file:line`
reference, a short description, and a suggested action. This is a review backlog — items here
are **not yet implemented**.

---

## Bugs / Correctness

### [High] Loaded splits are lost when `preprocessing(num_threads>1)`

`FrameworkMain.preprocessing()` (`src/simplegnn/framework/core.py:642`) dispatches one
`Preprocessing(...)` per dataset through `joblib.Parallel(n_jobs=num_threads)`.
`Preprocessing.__init__` mutates the config dict **in place**
(`src/simplegnn/framework/utils/preprocessing.py:393`/`395`:
`self.experiment_configuration['splits'] = load_splits(...)`), and
`self.experiment_configuration` *is* the same dict object as `self.network_configurations[key]`.

- With `num_threads == 1`, joblib runs in-process, so the mutation lands on the parent's dict
  and splits are available downstream.
- With `num_threads > 1`, joblib's loky/multiprocessing backend pickles a **copy** of the dict
  into each worker. `'splits'` is set on the copy and discarded when the worker exits; the
  `Preprocessing` return value is ignored, so nothing flows back to the parent.

Downstream consumers then fail:
- `run_configurations()` (`core.py:293`) → `len(configuration.get('splits')['train'])` →
  `configuration.get('splits')` is `None` → `TypeError: 'NoneType' object is not subscriptable`.
- `run_best_configuration()` (`core.py:447`) → `len(configuration.get('splits', {}).get('validation', None))`
  → `len(None)` → `TypeError`.

Disk side-effects (folders, generated data, ShareGNN labels/properties) survive because they are
written inside the worker; only the in-memory `'splits'` value is process-local and lost. This is
why every `examples/*/main.py` calls `preprocessing(1)`.

**Fix options (preferred → minimal):**
1. Make `run_configurations()` / `run_best_configuration()` load splits explicitly from the
   configured path via the existing `load_splits()` helper (already imported at `core.py:60`;
   pattern already used in `evaluate_model()` at `core.py:909`). Removes the cross-process coupling.
2. Have `preprocessing()` collect worker results and merge returned splits back into
   `self.network_configurations` instead of relying on argument mutation.
3. At minimum: document the `num_threads=1` constraint and raise a clear error when `'splits'`
   is missing instead of a bare `TypeError`.

### `copy_experiment_config` never copies the config

`core.py:1074` — `if not results_path.joinpath(f"{graph_db_name}/config.yml"):`. A `Path` is always
truthy, so `not Path(...)` is always `False` and the body never runs; the experiment config is
never archived to the results directory. Should test `.exists()` (and the intent looks inverted —
copy when it does *not* already exist).

### `collect_paths` discards the merged paths it builds

`core.py:990-1016` — the function deep-copies main-config paths and applies model-config overrides
into a local `paths` dict, then overwrites that variable with
`paths = dataset_configuration.get('paths', None)` (`core.py:1016`). The override work is silently
thrown away, so model-config path overrides have no effect.

### Self-reference typo in `generate_data`

`preprocessing.py:268` and `preprocessing.py:317` reference `self.dataset_configuration`, which is
never assigned (the attribute is `self.experiment_configuration`). These fallback branches raise
`AttributeError` if reached.

### Mutable default arguments

`core.py:792` (`graph_ids=[]`) and `src/simplegnn/utils/utils.py:286` (`graph_labels=[]`) use
mutable default arguments. Replace with `None` sentinels to avoid shared-state surprises.

---

## Robustness / Error Handling

### Pervasive bare `except:`

Many `except:` clauses swallow all exceptions (including `KeyboardInterrupt`) and hide tracebacks,
especially around data generation and label/property loading. Replace with targeted exceptions and
logging; at minimum re-raise after printing. Sites include:

- `framework/utils/preprocessing.py:270, 308, 319`
- `framework/utils/parameters.py:429`
- `framework/core.py:192`
- `framework/model_configuration.py:1128-1181` (multiple)
- `datasets/graph_dataset.py:1303`
- `datasets/utils/node_labeling.py` (~15 sites)
- `datasets/utils/edge_labeling.py:95, 101`
- `models/ShareGNN/preprocessing/properties.py:144, 190`
- `utils/utils.py:167-195`

### `FrameworkMain.__init__` masks config errors

`core.py:189-193` wraps config load + validation in a bare `except` that re-raises a generic
`ValueError(f"Config file {...} could not be loaded")`, hiding the real cause (e.g. a YAML syntax
error or a failed validation check). Narrow the `except` and chain the original exception (`raise ... from e`).

---

## Code Quality / Maintenance

### In-code TODOs to triage

- `core.py:747, 753, 991, 1002`
- `framework/run_configuration.py:130`
- `framework/model_configuration.py:404, 1285`
- `framework/utils/configuration_checks.py:157, 162`
- `datasets/graph_dataset.py:244, 1255`
- `datasets/graph_dataset_preprocessing.py:141, 147, 154`
- `models/model.py:437`
- `models/ShareGNN/layers/inv_based_message_passing.py:236, 358, 408`
- `models/ShareGNN/preprocessing/preprocessing.py:205`

### Undeclared and drifting dependencies

- `networkx` (imported in ~17 source files) and `scikit-learn` (~2 files) are used in `src/simplegnn/`
  but are **not declared** in `pyproject.toml`. They are currently relied on transitively (e.g. via
  `torch-geometric`), which is fragile. Declare them as direct dependencies. `numpy` is likewise
  used pervasively but only pulled in transitively.
- `requirements.txt` uses `~=` pins while `pyproject.toml` uses `>=,<` ranges for the same packages.
  Reconcile so the two sources don't drift.

### `num_threads` reassignment across loop iterations

In `run_configurations` (`core.py:271-294`) and `run_best_configuration` (`core.py:456-461`),
`num_threads` is repeatedly reassigned via `min(...)` inside the per-dataset/per-config loop. Once
clamped down (e.g. by `num_workers` or `len(run_loops)`), it cannot grow back, so later datasets may
run with fewer workers than requested. Use a separate local variable per iteration.

---

## Enhancements

- Add an integration test that runs `preprocessing(num_threads=2)` and asserts the splits are
  available to `run_configurations()` — guards the headline bug once fixed.
- Consider a real CLI entry point: `click` is a declared dependency but no console script is defined
  in `pyproject.toml`.
