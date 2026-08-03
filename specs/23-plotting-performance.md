# 23 — Plotting Performance

**Status**: Implemented (Phases 1–4; optional Phase 3.2 in-process memo and
Phase 4.4 multiprocessing skipped — revisit only if regeneration is still slow
on a machine that has the full result sets). Implementation notes:
`get_model` routes through `load_ordinary_model`, fixing the scripts' calls to
the removed `FrameworkMain.load_model` (same fix applied to `latex.py:666`);
`ablation_threshold`'s dead `mean_epoch`/`mean_epoch_time` computations (read
every per-epoch CSV, never plotted) were dropped; the unused
`evaluate_model_on_graphs` call in `plot_substructure_counting.py` (second
full dataset load per task, outputs only fed commented-out code) was removed.
Verified: full pytest suite green, new equivalence test
`test_get_all_param_indices_matches_per_graph_rows` green, all scripts
import-clean; end-to-end figure regeneration not runnable locally (base_paper
classification models/results absent).
**Scope**: `experiments/base_paper/src/latex_plots.py`, `plot_zinc.py`, `plot_common.py`,
`experiments/base_paper/regression/substructure_counting/plot_substructure_counting.py`,
`src/simplegnn/models/ShareGNN/layers/inv_based_message_passing.py` (one new accessor),
`src/simplegnn/datasets/utils/graph_drawing.py` (optional memo)
**Builds on**: specs/22-plotting-restructure.md (dedup done; this spec is speed only)

## Bottleneck analysis (verified 2026-08-03, by reading the code paths)

Ordered by expected wall-clock impact:

### B1. Full dataset reload per figure call (dominant)

Every `plot_network` / `plot_shared_weights` / `plot_specific_graphs_from_db` call
constructs a fresh `FrameworkMain(config)` and calls `experiment.load_model(...)`,
which runs `preprocess_graph_data()` (`load_model.py:74/145` → `core.py:1103`):
the whole `GraphDataset` plus all ShareGNN node labels and pairwise properties
(distance tensors — the large artifacts) are re-read from disk and moved to
device **per call**.

`latex_plots.main()` does this ~26 times, including exact repeats of the same
`(config, db_name)` pair — e.g. `plot_network(plot_network_path, 'NCI1', ...)`
and `plot_shared_weights(plot_network_path, 'NCI1')` each load NCI1 from
scratch; DHFR is loaded 4×, NCI1/NCI109/Mutagenicity 3–4× each. For
NCI1-scale datasets with distance properties this is seconds-to-minutes per
load, so redundant loading alone likely dominates total runtime.

### B2. Per-graph Python loop over the whole dataset in `rules_vs_occurences`

`latex_plots.py:387`:

```python
weights = torch.cat([layer.get_graph_weights(g)[:, 3]
                     for g in range(len(layer.graph_data))]).numpy()
```

`get_graph_weights(g)` calls `_assemble_rows([g])`
(`inv_based_message_passing.py:1766/720`), which is explicitly batched — the
loop makes N single-graph calls (N = 4110 for NCI1) where **one** call
`_assemble_rows(range(N))` does the same work with a handful of vectorized
gathers. Runs once per `plot_shared_weights` call → 16× in `main()`.

### B3. `np.vectorize` param→property lookup, computed twice

`latex_plots.py:415-418` and again at `:454-457`:

```python
f = lambda x: np.max(np.where(x >= np.array(layer.weight_offset)))
property_indices = np.vectorize(f)(sort_indices)
```

One Python lambda invocation (allocating two temp arrays) **per weight**
(~10⁴–10⁵ weights), and the identical result is recomputed in
`rules_vs_weights`. Equivalent vectorized form:

```python
offsets = np.asarray(layer.weight_offset)
property_indices = np.searchsorted(offsets, sort_indices, side='right') - 1
```

(verify equivalence on one dataset; `>=` + `side='right'` off-by-one is the
thing to test). Then pass `property_indices` from `rules_vs_occurences` into
`rules_vs_weights` instead of recomputing.

### B4. Kamada–Kawai layout recomputed per subplot

`compute_positions` with `draw_type='kawai'` is O(n³)-ish. The `pos_path`
file cache exists (`resolve_positions`, graph_drawing.py:75) but several call
sites don't use it, so the same graph's layout is recomputed for every head/
column drawn:

- `plot_zinc.py:56-61` — 3 per-head `draw()` calls without `pos_path` (same
  graph as the first call, which does pass it).
- `plot_substructure_counting.py:65-72` — 5 per-head `draw()` calls without
  `pos_path`, × 7 datasets.
- `latex_plots.py:362` (`plot_specific_graphs_from_db`) — no `pos_path`.

### B5. pgf/lualatex rendering is serial

Every `save_latex_figure` spawns a full lualatex compile (seconds per
figure); `latex_plots.main()` produces ~50 figures strictly sequentially.
The compiles are independent and embarrassingly parallel. Note
`bbox_inches='tight'` forces an extra layout pass per save — acceptable, but
parallelism is the lever here.

### B6. Blocking `plt.show()` and missing skip-cache in the two standalone scripts

- `plot_zinc.py:66` and `plot_substructure_counting.py:98` call `plt.show()`
  after saving — with a GUI backend this **blocks until the window is closed**
  (7 windows in the substructure script), which reads as "plotting is very
  slow" even though it's waiting on the user.
- Neither script has the `if not Path(...).exists()` skip guard that
  `latex_plots.py` functions have, so they always redo everything.

### B7. Minor: figures never closed; quadruple directory scan

- No `plt.close(fig)` after saving anywhere → ~50 live figures accumulate in
  pyplot state (memory, slight slowdown).
- `ablation_threshold` walks the same 23 ablation result dirs in 4 separate
  loops and `pd.concat`s inside a loop — measurable only with many files, but
  trivially merged into one loop.

## Plan

### Phase 1 — Cache dataset/model loading (B1) — biggest win

1. Add to `plot_common.py`:
   - `get_experiment(config_path)` — memoized `FrameworkMain` per config path
     (plain module-level `dict`; paths are strings, no hashing issues).
   - `get_model(config_path, db_name, **kwargs)` — memoized
     `experiment.load_model(...)` per `(config_path, db_name, config_id,
     run_id, validation_id, best)` key.
2. Replace every `FrameworkMain(...)`/`load_model(...)` pair in
   `latex_plots.py` (3 sites), `plot_zinc.py`, `plot_substructure_counting.py`
   with the cached accessors.
3. Memory guard: the cache can hold several large datasets at once. Add
   `clear_model_cache()` and call it in `latex_plots.main()` between dataset
   groups if RAM becomes a problem (keep `main()`'s call order grouped by
   dataset so at most one or two datasets are hot at a time — reorder the
   call list accordingly; the current order interleaves datasets).

Deeper option (only if still slow): `preprocess_graph_data` itself could take
an in-process cache, but keeping the cache in the plotting layer avoids
touching framework code used by training.

### Phase 2 — Vectorize `rules_vs_occurences` / `rules_vs_weights` (B2, B3)

1. Add a small public accessor on `InvariantBasedMessagePassingLayer`
   (next to `get_graph_weights`, ~line 1766):

   ```python
   def get_all_param_indices(self):
       """param_idx column of the assembled rows for the whole dataset."""
       _, _, _, params, _ = self._assemble_rows(range(len(self.graph_data)))
       return params
   ```

2. `rules_vs_occurences`: use it instead of the per-graph `torch.cat` loop;
   drop the stray unused `weight_array = np.zeros(...)` line while there.
3. Replace both `np.vectorize` blocks with the `np.searchsorted` form;
   compute `property_indices` once in `rules_vs_occurences` and return it /
   pass it to `rules_vs_weights` (its signature already receives
   `sort_indices, steps` — add `property_indices`).
4. Equivalence check: assert old-vs-new `property_indices` match on one small
   dataset before deleting the old code path.

### Phase 3 — Position caching at every draw site (B4)

1. Pass the existing `pos_path` argument in the per-head `draw()` calls in
   `plot_zinc.py` and `plot_substructure_counting.py`, and add a
   position-cache path in `plot_specific_graphs_from_db` (same
   `Plots/Positions/{db}_{graph_id}_pos.txt` pattern used elsewhere).
2. Optional belt-and-braces: in-process memo in `resolve_positions`
   (dict keyed by `str(pos_path)` when non-empty) so repeated draws of the
   same graph don't even re-read the file. Keep the file cache as-is —
   it also pins layouts across runs for reproducible figures.

### Phase 4 — Throughput and hygiene (B5–B7)

1. `plot_zinc.py` / `plot_substructure_counting.py`: delete `plt.show()`;
   add the same output-exists skip guard `latex_plots.py` uses.
2. Add `plt.close(fig)` inside `save_latex_figure` (every caller saves as its
   last step; closing there fixes all sites at once).
3. Merge `ablation_threshold`'s four scan loops into one.
4. Parallelize `latex_plots.main()` (optional, after Phases 1–3 are
   measured): group figure tasks by dataset, run groups in a
   `multiprocessing` pool (`joblib` is already a dependency). Each worker
   process gets its own matplotlib/lualatex — this parallelizes the pgf
   compiles too. Do **not** combine naively with the Phase-1 cache: the cache
   is per-process, so grouping by dataset per worker is what keeps loads at
   one per dataset. Skip this phase if Phases 1–3 already make runtime
   acceptable — it adds the most complexity for the least certain gain.
5. Explicitly *not* changing `pgf.texsystem` (lualatex) — output fidelity for
   the paper outweighs compile speed; parallelism attacks the same cost
   safely.

## Measurement

Before/after timing, using whatever result sets exist locally (note: only
`results/base_paper/classification/{RealWorld,Sota}` are present right now —
the Ablation/Distance figures will skip; benchmark on the network/
shared-weight figures which do have data):

1. Delete a representative subset of output PDFs (they gate on `exists()`),
   e.g. NCI1 + DHFR visualization and shared-weight figures.
2. `time python experiments/base_paper/src/latex_plots.py` before and after
   each phase; record per-figure times with a simple `time.perf_counter()`
   wrapper in `main()` if finer attribution is needed.
3. Expected: Phase 1 removes ~15 redundant dataset loads; Phase 2 turns 16
   O(N)-Python-loop passes into single batched calls; Phase 3 removes
   ~40 redundant Kamada–Kawai layouts.

## Verification

1. `pytest tests -q` (layer accessor touches `inv_based_message_passing.py`).
2. Regenerate one figure of each kind and visually diff against the previous
   PDF/PNG (positions are pinned by the `Positions/` cache files, so figures
   should be pixel-stable except for intentional no-ops).
3. Phase-2 equivalence assert (old vs new `property_indices`) on one dataset.

## Risks

- **Low**: all changes are in paper scripts except the one additive layer
  accessor; no training/eval code paths touched.
- Cache memory growth (Phase 1) — mitigated by dataset-grouped ordering and
  `clear_model_cache()`.
- `np.searchsorted` off-by-one vs the old `>=`-based lookup — covered by the
  explicit equivalence check before removal.
- Parallel pgf (Phase 4.4) can interleave lualatex temp files — matplotlib
  handles per-process temp dirs, but keep this phase optional and last.
