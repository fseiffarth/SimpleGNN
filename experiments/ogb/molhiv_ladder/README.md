# ogbg-molhiv capacity ladder

A cumulative sequence of ShareGNN experiments on `ogbg-molhiv`. Each stage adds
one change on top of the previous stage. `main_ogb.py` runs all stages in order:

```bash
NUM_THREADS=4 ./experiments/ogb/run_ogb.sh molhiv-ladder
# resume at a later stage (0-based index), e.g. after a crash in s2:
FROM_STAGE=2 NUM_THREADS=4 ./experiments/ogb/run_ogb.sh molhiv-ladder
# without the wrapper
python experiments/ogb/main_ogb.py --dataset molhiv-ladder --num_threads 4 --from_stage 0
```

## Why this ladder

In the walks + L1 baseline run (`results/ogb_molhiv_walks_l1_t2/`), train AUC
rose from 0.62 to 0.88 within 1.5 epochs. Validation and test moved in opposite
directions across seeds: validation 0.854 gave test 0.733, while validation
0.818 gave test 0.767. The model memorizes training scaffolds. The ladder
therefore removes capacity first and changes the training objective last.

The graph-only target band is CIN, GSN and P-WL at test ROC-AUC 0.80–0.81 on
the OGB leaderboard. Entries at 0.82 and above use molecular fingerprints and
are outside this ladder.

## Stages

| # | Stage | Change against the previous stage | Files |
|---|---|---|---|
| 0 | `s0_protocol` | Baseline architecture and parameters (frozen copies of `models_ogb_classification.yml` and `parameters_ogb_classification.yml`). Ten seeds (`evaluation_run_number: 10`), epoch-end checkpoint selection (no `step_validation`), and saved best-checkpoint predictions | `main_s0_protocol.yml`, `parameters_s0_protocol.yml`, `models_baseline.yml` |
| 1 | `s1_threshold` | `rule_occurrence_threshold` 2 → 25 | `main_s1_threshold.yml`, `parameters_s1_threshold.yml` |
| 2 | `s2_trimmed` | Remove the `wl_labeled_edges depth: 1` convolution head. Reduce induced-cycle convolution distances from 1..10 to 1..4. Reduce pooling from 5×8 to 3×4 heads | `main_s2_trimmed.yml`, `models_trimmed.yml` |
| 3 | `s3_l1` | Proximal L1 for convolution and aggregation 0.01 → 0.05. Encoder L1 stays 0.01 | `main_s3_l1.yml`, `parameters_s3_l1.yml` |
| 4 | `s4_auc_loss` | Training loss CE → CE + AUC-margin (`CEAUCMarginLoss`). Validation and test loss stay CE | `main_s4_auc_loss.yml`, `parameters_s4_auc_loss.yml` |

The stages are cumulative and use a fixed order. A stage does not select a
winner from the previous stage automatically: molhiv validation AUC
anti-correlates with test AUC once the model overfits, so automatic selection
would follow noise. Read `ladder_summary.csv` after the run. If a stage hurts,
revert that change in the subsequent stage files and rerun from that stage.

Each stage has one hyperparameter configuration. `run_configurations` trains
it once, and `run_best_configuration` trains it again with ten seeds. Each
stage writes to its own results root under `results/ogb_molhiv_ladder/<stage>/`,
because configuration ids restart at 0 in every run.

## Seed ensemble (item 4)

The ensemble needs no additional training stage. `save_best_predictions: True`
writes `model_<config>_run_<r>_val_step_<k>.predictions.npz` next to each
best checkpoint. The file contains the validation and test graph ids, the
positive-class scores and the labels at the selected checkpoint. After each
stage, `simplegnn.framework.utils.seed_ensemble` averages the ten
`Best_Configuration` runs. It writes `seed_ensemble.json`, which contains:

- mean ± std test and validation AUC of the single seeds,
- test and validation AUC of the probability-averaged ensemble,
- test and validation AUC of the rank-averaged ensemble.

## AUC-margin loss (item 5)

`CEAUCMarginLoss` adds `auc_weight · mean((margin − (s⁺ − s⁻))₊²)` to the
cross-entropy. The sum runs over all positive/negative pairs in the batch.
`s` is the softmax positive-class probability. With the LibAUC default margin
1.0 and scores in [0, 1], the hinge is always active, so the term equals the
pairwise square AUC surrogate behind LibAUC's AUCM loss. This implementation
uses batch pairs, not LibAUC's min-max reformulation and PESG optimizer, so it
works with AdamW unchanged. A batch without positives contributes only the
cross-entropy term. At batch size 64 and a 3.5 % positive rate, a batch has
about two positives, or about 120 pairs.

Configure the term with `auc_margin_loss: {weight: 1.0, margin: 1.0}` in the
parameter file.

## Output

`results/ogb_molhiv_ladder/ladder_summary.csv` contains one row per completed
stage. Columns are validation and test AUC mean ± std from
`summary_best_mean.csv`, and ensemble test AUC from `seed_ensemble.json`. The
file is rewritten after every stage, so a partial ladder still has a summary.

## Results (2026-10-22)

From `results/ogb_molhiv_ladder/ladder_summary.csv`, test ROC-AUC with ten
seeds:

| Stage | Test AUC | Ensemble test |
|---|---|---|
| s0_protocol | 0.762 ± 0.015 | 0.771 |
| s1_threshold | 0.758 ± 0.008 | 0.764 |
| s2_trimmed | 0.764 ± 0.008 | 0.774 |
| s3_l1 | 0.765 ± 0.008 | 0.776 |
| s4_auc_loss | 0.762 ± 0.011 | 0.779 |

No stage moved the single-seed test AUC beyond noise. The test optimum came
at epoch 0–1 in every stage, and L1 and weight decay were numerically inert at
lr 1e-4. Ladder 2 (`../molhiv_ladder2/README.md`) contains the full analysis
and the follow-up experiments.

## Resources

Ten seeds per stage run as parallel processes. As with the other molhiv
configs, each process holds its own rule tables, so the main configs cap the
process count with `num_workers: 4`.
