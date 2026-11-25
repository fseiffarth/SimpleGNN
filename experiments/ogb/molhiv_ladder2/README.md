# ogbg-molhiv ladder 2

Six ShareGNN experiments on `ogbg-molhiv`, run in sequence by `main_ogb.py`:

```bash
NUM_THREADS=4 ./experiments/ogb/run_ogb.sh molhiv-ladder2
FROM_STAGE=3 NUM_THREADS=4 ./experiments/ogb/run_ogb.sh molhiv-ladder2   # resume at t3
```

Results go to `results/ogb_molhiv_ladder2/<stage>/`, and the summary goes to
`results/ogb_molhiv_ladder2/ladder_summary.csv`. That file has the same
columns as the ladder-1 summary.

## What ladder 1 showed

From `results/ogb_molhiv_ladder/`, with ten seeds per stage:

| Stage | Params | Test AUC | Ensemble test | Best epochs (10 seeds) |
|---|---|---|---|---|
| s0_protocol | 1,169k | 0.762 ± 0.015 | 0.771 | 0–1 |
| s1_threshold | 356k | 0.758 ± 0.008 | 0.764 | 1–3 |
| s2_trimmed | 161k | 0.764 ± 0.008 | 0.774 | 0–1 |
| s3_l1 | 161k | 0.765 ± 0.008 | 0.776 | 0–1 |
| s4_auc_loss | 161k | 0.762 ± 0.011 | **0.779** | 0–5 |

1. **No stage moved single-seed test AUC.** All stage means lie within
   0.758–0.765. The standard error of a 10-seed mean is 0.003–0.005, so the
   differences are noise.
2. **Capacity is not the bottleneck.** Parameters fell 7× from s0 to s2. The
   test AUC and the epoch of the best checkpoint did not change.
3. **The test optimum comes early in every stage.** The best test AUC over all
   epochs (an oracle, about 0.77) is reached at epoch 0–1, or at epoch 5 or
   earlier with the AUC loss. Train AUC then rises to 0.93–0.96 while test
   falls. Checkpoint selection costs only about 0.006 against that oracle.
4. **L1 and weight decay were inert at lr 1e-4.** The proximal L1 shrink is
   `lr·λ` per step: 5e-6 at λ = 0.05. After s3 and s4, 94–95 % of the
   rule weights were still nonzero. AdamW shrinks by `lr·wd` = 1e-6 per
   step. Neither regularizer acted within the first two epochs, where the
   optimum lies.
5. **Seed ensembling always helped,** by +0.006 to +0.018. The gain was
   largest with the AUC-margin loss (s4).
6. **Validation does not rank seeds.** The Spearman correlation between
   validation and test AUC across the ten seeds was −0.32 to 0.18.
7. **About 80 % of training was wasted.** Each run trained 31–36 epochs
   because early-stopping patience was 30, but every run's best epoch was 5
   or earlier.

## Design

t0 is ladder-1's s4 configuration with a shorter budget: 20 epochs, patience 8.
Because ladder 1's changes did not stack into a gain, stages t1–t5 are
**single-change arms off t0**, not cumulative steps. Compare every arm to t0.
With the same seeds, t0 should reproduce s4 (single seeds 0.762, ensemble
0.779). If it does not, the gap is the run-to-run noise floor of the
10-seed mean.

| # | Stage | Change against t0 | Hypothesis tested |
|---|---|---|---|
| 0 | `t0_reference` | Budget only: 20 epochs, patience 8 | Same results as s4 at about 40 % of the compute |
| 1 | `t1_weight_decay` | `weight_decay` 0.01 → 1.0 (`lr·wd` = 1e-4 per step, about 5 % shrink per epoch) | Decay that acts inside the first epochs delays memorization |
| 2 | `t2_l1` | L1 for convolution and aggregation 0.05 → 0.5 (`lr·λ` = 5e-5 per step) | Real rule selection, which s3 never reached |
| 3 | `t3_node_dropout` | Dropout 0.3 on node features before pooling (`models_node_dropout.yml`) | Noise on node representations, as in GIN's per-layer dropout on molhiv |
| 4 | `t4_batch256` | `batch_size` 64 → 256 | About 9 positives per batch, giving the AUC-margin term about 2,200 pairs instead of about 120. 129 steps per epoch, so the epoch-end checkpoints are 4× closer in optimizer steps |
| 5 | `t5_pool_only` | No convolution. Pooling over primary, WL-with-edges depth 0 and 1, and rings 5–6 (`models_pool_only.yml`) | Whether message passing adds anything beyond a learned substructure bag. RF on ECFP fingerprints reaches 0.80–0.82 |

`t3` and `t5` change only the model file and reuse
`parameters_t0_reference.yml`. `t0`–`t2` and `t4` reuse ladder-1's
`models_trimmed.yml`.

## Interpreting the outcome

- An arm counts as an improvement only if its mean beats t0 by more than
  about 0.01. That is twice the standard error of a 10-seed mean. Ensemble
  AUCs are single numbers without a spread, so give them less weight.
- If t1 or t2 delays the test optimum (best epochs later than 0–1) without
  raising it, the ceiling lies in the representation, not in optimization.
- If t5 matches t0, the convolution contributes nothing on scaffold splits.
  The next ladder should then build on pooled substructure vocabularies
  (WL depth 2, more ring types) rather than on message passing.
- If no arm beats t0, the remaining graph-only levers need framework
  changes. Examples are FLAG-style adversarial feature augmentation and
  checkpoint averaging (SWA) across the early epochs.

## Framework fix made for t5

`check_model_configuration_file` used to mark a model as invariant only when
it contained an `invariant_based_convolution` layer. Otherwise it cleared the
label and property paths. An encoder → pooling model therefore failed at
label loading. Pooling and positional-encoding layers now count as invariant
layers too (see `tests/test_configuration_checks_unit.py`).

## Resources

The ten seed reruns of each stage run one after another (`run_pipeline` in
`main_ogb.py`). The first t4 attempt ran four in parallel and ran out of GPU
memory: one batch-256 worker held up to 37 GiB, and four of them exceeded the
95 GiB card. On the GPU an epoch takes about 4 s, so a sequential seed run of
10–13 epochs takes about one minute. t4 and t5 resume with:

```bash
FROM_STAGE=4 ./experiments/ogb/run_ogb.sh molhiv-ladder2
```
