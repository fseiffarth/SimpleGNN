# OGB graph-property-prediction experiments

ShareGNN on the single-task `ogbg-mol*` datasets from the Open Graph Benchmark,
using OGB's official scaffold splits.

## Datasets

| Dataset | Graphs | Task | OGB metric | Config |
|---|---|---|---|---|
| `ogbg-molhiv` | 41,127 | binary classification | ROC-AUC | `configs/main_molhiv.yml` |
| `ogbg-molbace` | 1,513 | binary classification | ROC-AUC | `configs/main_molbace.yml` |
| `ogbg-molbbbp` | 2,039 | binary classification | ROC-AUC | `configs/main_molbbbp.yml` |
| `ogbg-molesol` | 1,128 | regression (logS) | RMSE | `configs/main_molesol.yml` |
| `ogbg-molfreesolv` | 642 | regression (kcal/mol) | RMSE | `configs/main_molfreesolv.yml` |
| `ogbg-mollipo` | 4,200 | regression (logD) | RMSE | `configs/main_mollipo.yml` |

Classification runs share `models_ogb_classification.yml` /
`parameters_ogb_classification.yml`; regression runs share the `_regression`
pair. Splits live in `src/simplegnn/datasets/splits/fixed/ogbg-*_splits.json`.

### Not included: the multi-task datasets

`ogbg-moltox21` (12 tasks), `molsider` (27), `molclintox` (2), `moltoxcast`
(617), `molmuv` (17), `molpcba` (128) and `molchembl` (1310) are **not** covered.
Their targets are NaN-masked matrices — each molecule is labelled for only some
of the assays — which needs a masked multi-label loss and per-task metric
averaging. The framework's classification head takes one target per graph
(`graph_dataset.py:1349-1374`), so supporting them is a framework change, not a
config change.

The pre-existing `configs/main.yml` targets `ogbg-moltox21` and predates this
setup: it uses the old model schema (`networks:`/`convolution:`, see
`models_ShareGNN.yml`) and has no `splits:` path, so it fails the main-config
check. It is left in place untouched; use the `main_<dataset>.yml` configs.

## Running

```bash
./experiments/ogb/run_ogb.sh              # molhiv (default)
./experiments/ogb/run_ogb.sh molbace      # a single dataset
./experiments/ogb/run_ogb.sh --all        # all six, smallest first
NUM_THREADS=4 ./experiments/ogb/run_ogb.sh molhiv
```

The script generates any missing split file, then runs the standard pipeline
(grid search → model selection → rerun best → test evaluation). To run without
the wrapper:

```bash
python -m simplegnn.utils.ogb_splits --all       # once, generates all split files
python experiments/ogb/main_ogb.py --dataset molhiv --num_threads 8
```

Results land in `results/ogb/<dataset>/`; `summary_best_mean.csv` holds the
final test numbers.

## Metric handling

Classification uses `evaluation_metric: roc_auc` (per-epoch metric and
best-epoch checkpointing) **and** `evaluation_type: roc_auc` (cross-config model
selection). Both are needed — the first writes the `EpochAUC`/`ValidationAUC`/
`TestAUC` columns, the second makes `evaluate_results()` select on them instead
of on accuracy. Accuracy is not a useful selector here: `ogbg-molhiv` is ~3.5%
positive, so an all-negative predictor scores 96.5%.

`ogbg-molhiv`'s class imbalance is handled with `weighted_loss: True` (per-batch
class weights) rather than undersampling, which would discard most of the
training set.

## The `ogbg-molhiv` overfitting sweep

The baseline `main_molhiv.yml` run saturates train AUC and peaks validation AUC
inside the first epoch. Part of that is inherent: 32,901 scaffold-split training
molecules at ~3.5% positive is ~1,150 positive graphs, one epoch at
`batch_size: 64` is already 515 optimizer steps, and published GIN baselines
peak just as early (test ROC-AUC ~0.75–0.78). The rest is the config — no weight
decay, no dropout, and `rule_occurrence_threshold: 1`, which gives a weight to
every label-pair/property combination that occurs even once across 41k
molecules.

Two arms isolate capacity from optimization:

| Arm | Config | `rule_occurrence_threshold` |
|---|---|---|
| A | `configs/main_molhiv_t1.yml` | 1 (baseline vocabulary) |
| B | `configs/main_molhiv_t50.yml` | 50 (rare-rule tail dropped) |

```bash
./experiments/ogb/run_ogb.sh molhiv-t1
./experiments/ogb/run_ogb.sh molhiv-t50
```

The threshold is read as a scalar inside the layer
(`inv_based_message_passing.py:1145`), not in the grid loop, so it cannot be a
list — hence one parameter file per value. The two files differ only in that
line. Each arm writes to its own results root (`results/ogb_molhiv_t1/`,
`results/ogb_molhiv_t50/`); a shared root would let the second run overwrite the
first, since configuration ids restart at 0 per run.

Both arms use `models_ogb_molhiv.yml` and share these changes against the
baseline classification configs:

- `AdamW` with a `weight_decay: [0.0, 0.01]` grid (2 configs per arm). AdamW's
  decoupled decay makes this a real regularization axis; 0.01 is the torch
  default, 0.1 is the next step up if the gap does not move. The optimizer is
  given as a plain string on purpose — a `weight_decay` inside an optimizer dict
  would override the grid (`model_configuration.py:1039-1042`).
- `learning_rate: 0.0001` (was 1e-3). Per-epoch checkpointing cannot capture an
  optimum reached inside epoch 1; at 1e-4 the same trajectory spans ~10 epochs.
- `epochs: 40`, early-stopping patience 10 (was 200/30) — the extra epochs were
  never used.
- Dropout 0.5 on the graph-level readout, as an explicit
  `{layer_type: dropout, p: 0.5}` layer. **The top-level `dropout:`
  hyperparameter does nothing here**: `model.py:213` reads `para.dropout` and
  never uses it, and the invariant and linear layers do not apply
  `layer_args['dropout']` either (only gcn/gin/glu do).
- `weighted_loss: False`. At a 3.5% positive rate the per-batch weights are
  ~0.965 vs ~0.035 (`model_configuration.py:605-611`), a ~27× upweight of the ~2
  positives in a batch of 64, which for a rank metric mostly speeds up
  memorizing them. Re-enable as a separate A/B once the threshold picture is
  clear.

The architecture is otherwise byte-identical to `models_ogb_classification.yml`
so the arms differ only in vocabulary size. Trimming the `induced_cycles` heads'
`distances: [1..10]` and dropping the depth-1 `wl_labeled_edges` head is the
follow-up if arm B wins.

Memory note: each parallel config holds its own rule tables for a 41k-graph
dataset, so run these with a modest `NUM_THREADS` (2–4) rather than `-1`.
