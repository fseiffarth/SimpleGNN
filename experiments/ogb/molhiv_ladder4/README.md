# ogbg-molhiv ladder 4: residual convolutions and a longer budget

Follow-up to ladder 3. Its best stage, v3 (two stacked distance-1
convolution blocks), scored 0.786 ± 0.007 with a seed ensemble of 0.801.
A third block (v4) was worse. `main_ogb.py` runs the stages in sequence and
reruns the seeds one after another:

```bash
./experiments/ogb/run_ogb.sh molhiv-ladder4
FROM_STAGE=1 ./experiments/ogb/run_ogb.sh molhiv-ladder4   # resume at w2
```

Results are written to `results/ogb_molhiv_ladder4/<stage>/`, and the summary
to `results/ogb_molhiv_ladder4/ladder_summary.csv`.

## Reference

**Ladder-3 `v3_local1_two_conv`** (0.786 ± 0.007, ensemble 0.801) is the
reference. Every stage uses the same seeds (42–51), so the comparison is
paired by seed. w3 is also compared with ladder-3 v4 (0.780 ± 0.007), its
version without residuals.

## Stages

| # | Stage | Model / parameters | Change against v3 |
|---|---|---|---|
| 0 | `w1_two_conv_residual` | `models_two_conv_residual.yml` | `residual: true` on both convolutions |
| 1 | `w2_two_conv_30_epochs` | `parameters_w2_30_epochs.yml` | 30 epochs instead of 20 (patience stays 8) |
| 2 | `w3_three_conv_residual` | `models_three_conv_residual.yml` | v4 (three blocks) with `residual: true` on all three convolutions |

w1 and w3 keep `../molhiv_ladder2/parameters_t0_reference.yml`. w2 keeps v3's
model, `../molhiv_ladder3/models_local1_two_conv.yml`.

**Why the residual.** The convolution's `residual: true` (the existing
option from specs/15, used on ZINC) adds the layer input to every head's
output: `out = conv(x) + x`. Distance-1 convolutions exclude distance 0, so
in v3 a node's own features never reach its own output; only its label does,
through the bias. A residual restores the self term, as GIN's `(1 + ε)·x`
does. It also gives the gradient a direct path through the stack. That
matters for w3: if the residual is what three blocks were missing, w3 should
beat v4 and close the gap to w1.

**What w2 changes.** The plan assumed w2 would replay v3's first 20 epochs.
It does not: the batch-shuffle seed is `epoch + run_id · n_epochs + seed`
(`model_configuration.py`, `get_train_batches`), so the epoch count enters
every run except run 0. Seed 42 reproduced v3 exactly; seeds 43–51 kept
v3's initialization but drew a different batch order. That makes w2 a
batch-order replicate of v3, which turned out to be useful (see Results).
Comparisons inside ladders 2–4 are unaffected, since all their other stages
use 20 epochs.

## Reading the outcome

- w1 > v3 by more than about 0.01 (paired, most seeds) → the self term
  matters. w1 becomes the new reference and the reported number.
- w3 > v4 → the residual removes the depth saturation. If w3 also reaches
  w1, more depth is worth a further ladder; if not, two blocks is the depth.
- Validation did not rank the ladder-3 stages, so the winner is again a
  test-set choice. Report it as an ablation over fixed seeds (10-seed mean ±
  std plus seed ensemble), not as a validation-selected number. A single
  model's test AUC has a bootstrap standard error of about 0.027.

## Results

Ten seeds (42–51), test ROC-AUC at the validation-selected epoch. The paired
differences are per seed against ladder-3 v3.

| Stage | Val | Test | Paired Δ vs v3 | Seeds better | p | Train AUC @ best | Ensemble |
|---|---|---|---|---|---|---|---|
| v3 (reference) | 0.820 | 0.786 ± 0.007 | – | – | – | 0.857 | 0.801 |
| w1_two_conv_residual | 0.819 | 0.740 ± 0.023 | −0.046 | 0/10 | 0.0003 | 0.804 | 0.773 |
| w2_two_conv_30_epochs | 0.822 | 0.785 ± 0.007 | −0.001 | 3/10 | 0.57 | 0.846 | 0.809 |
| w3_three_conv_residual | 0.824 | 0.747 ± 0.021 | −0.040 | 0/10 | 0.0005 | 0.806 | 0.766 |

w3 is also 0.033 below v4, its version without residuals (0/10 seeds,
p = 0.003).

- **Residual convolutions hurt badly, and they underfit rather than
  overfit.** Train AUC at the selected epoch falls from 0.86 to 0.80, and
  epoch-0 validation falls from 0.77 to 0.75. The convolution weights are
  not dead: their sparsity and magnitude match v3's. The likely cause is
  scale. The residual adds the LayerNorm'd input (per-feature scale ≈ 1) to
  each of the 12 heads, while a distance-1 message is a sum over 2–3
  neighbours with rule weights of |w| ≈ 0.01. The messages are then a few
  percent of the block output, and the model behaves close to a node-only
  model. This is a hypothesis; the activation ratio has not been measured.
  A residual would need `pre_layer_norm`, a learned residual scale or a
  larger convolution init to be a fair test. As configured here, the
  residual does not help.
- **A longer budget gives nothing.** w2 matches v3 (−0.001, p = 0.57).
  Its best epochs stay in 1–14.
- **Batch order alone moves results by about as much as the effects we are
  measuring.** For seeds 43–51, w2 and v3 differ only in batch order. Their
  per-seed test AUCs differ with sd 0.008, and their ensembles score 0.801
  and 0.809. A single paired seed difference below about 0.01, or an
  ensemble difference below about 0.01, is within this noise.

**Molhiv numbers to report:** v3 stays the best architecture. Pooling v3
and w2, which share the model and differ only in batch order and budget,
gives 0.786 ± 0.007 over 20 runs and a 20-model ensemble of 0.807. The
10-seed v3 number is 0.786 ± 0.007, with an ensemble of 0.801.
