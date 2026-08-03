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
