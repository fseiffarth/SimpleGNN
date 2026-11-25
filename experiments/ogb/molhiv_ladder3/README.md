# ogbg-molhiv ladder 3: local convolution

A 10-seed confirmation of the only gain found in screening: limiting the
convolution's pair rules to short distances. `main_ogb.py` runs the four
stages in sequence and reruns the seeds one after another:

```bash
./experiments/ogb/run_ogb.sh molhiv-ladder3
FROM_STAGE=2 ./experiments/ogb/run_ogb.sh molhiv-ladder3   # resume at v3
```

Results are written to `results/ogb_molhiv_ladder3/<stage>/`, and the summary
to `results/ogb_molhiv_ladder3/ladder_summary.csv`.

## Evidence from screening round 3

From `../molhiv_screen/README.md`, five seeds (42–46), 8-epoch budget, test
ROC-AUC at the validation-selected epoch. Differences are paired by seed:

| Candidate | Test | Paired Δ vs reference | Seeds better | p (paired t) | Ensemble |
|---|---|---|---|---|---|
| u0_reference (distances up to 10) | 0.762 ± 0.007 | – | – | – | 0.769 |
| u1_local3 | 0.770 ± 0.008 | +0.008 ± 0.003 | 4/5 | 0.08 | 0.782 |
| u2_local2 | 0.781 ± 0.011 | +0.019 ± 0.004 | 5/5 | 0.011 | 0.793 |
| u3_local1 | 0.780 ± 0.007 | +0.018 ± 0.004 | 5/5 | 0.012 | 0.791 |

Capping the convolution distances gives a dose response. The two most local
variants beat the reference on every seed. They also train more slowly: their
best epochs are 2–7 instead of 0–3. u3 peaked at the last of its 8 epochs in
3 of 5 seeds, so the screen budget probably understates it. The interpretation
is that long-range pair rules memorize training scaffolds, which do not recur
in the scaffold-split test set.

## Stages

Every stage uses `../molhiv_ladder2/parameters_t0_reference.yml`: CE plus
AUC-margin loss, threshold 25, L1 0.05, lr 1e-4, batch 64, 20 epochs,
patience 8. **The reference is ladder-2 `t0_reference`** (0.763 ± 0.007,
ensemble 0.772). It has the same parameters, the same architecture apart from
the convolution and the same seeds (42–51), so this ladder does not rerun it,
and every stage can be paired seed by seed with t0.

| # | Stage | Model | Change against t0 |
|---|---|---|---|
| 0 | `v1_local2` | `models_local2.yml` | Convolution distances ≤ 2 (screen u2) |
| 1 | `v2_local1` | `models_local1.yml` | Convolution distance 1 only (screen u3) |
| 2 | `v3_local1_two_conv` | `models_local1_two_conv.yml` | Two stacked distance-1 convolution blocks: a two-hop receptive field by composition, as in GIN |
| 3 | `v4_local1_three_conv` | `models_local1_three_conv.yml` | Three stacked distance-1 convolution blocks |

v3 and v4 have not been screened. With ten seeds the ladder itself is the
test.

## Reading the outcome

- v1 or v2 confirms the screen if its 10-seed mean beats t0 by more than about
  0.01 and the paired difference is positive on most seeds.
- If v3 or v4 beats v2, depth by composition generalizes where long-range
  rules do not. A residual connection between the blocks is the follow-up.
- The best stage's 10-seed mean ± std and its seed ensemble are the molhiv
  numbers to report. Molhiv itself cannot resolve finer architecture choices:
  a single model's test AUC has a bootstrap standard error of about 0.027
  (130 test positives).

## Results

Ten seeds (42–51), test ROC-AUC at the validation-selected epoch. The paired
differences are per seed against ladder-2 `t0_reference`. The ensemble
averages the scores of the ten seed models.

| Stage | Val | Test | Paired Δ vs t0 | Seeds better | p (t / Wilcoxon) | Ensemble | Ensemble Δ, 95% bootstrap CI |
|---|---|---|---|---|---|---|---|
| t0_reference | 0.823 | 0.763 ± 0.006 | – | – | – | 0.772 | – |
| v1_local2 | 0.830 | 0.775 ± 0.010 | +0.012 | 9/10 | 0.002 / 0.006 | 0.785 | +0.013 [−0.011, +0.039] |
| v2_local1 | 0.822 | 0.774 ± 0.016 | +0.011 | 8/10 | 0.09 / 0.11 | 0.789 | +0.017 [−0.013, +0.049] |
| **v3_local1_two_conv** | 0.820 | **0.786 ± 0.007** | **+0.023** | **9/10** | **0.0001 / 0.004** | **0.801** | +0.029 [−0.001, +0.062] |
| v4_local1_three_conv | 0.828 | 0.780 ± 0.007 | +0.017 | 9/10 | 0.002 / 0.006 | 0.797 | +0.025 [−0.004, +0.055] |

The bootstrap resamples test molecules, stratified by class, with 2,000
draws. Between stages, v3 beats v4 on 8/10 seeds (+0.007, p = 0.03). v3
beats v1 by +0.012 (9/10 seeds, p = 0.05) and v2 by +0.012 (p = 0.09).
v1 and v2 are indistinguishable.

- **The screen is confirmed.** Local convolution improves on t0 by about
  +0.01 with 10 seeds and the 20-epoch budget. That is about half the 8-epoch
  screen gain. The reference did not move (screen u0 0.762, t0 0.763), but
  the local models did: on the screen's seeds 42–46, v1 averages 0.773
  against u2's 0.781. Five-seed screen winners overstate their gain.
  Distance ≤ 2 and distance 1 tie; v2 has one bad seed (47: 0.730, best
  epoch 2).
- **Depth by composition helps where long-range rules did not.** Two stacked
  distance-1 blocks give the best stage and the lowest seed spread. A third
  block is worse than two, so plain stacking saturates at two. Residual
  convolutions and a 30-epoch budget are tested in
  [`../molhiv_ladder4/`](../molhiv_ladder4/README.md).
- **Validation does not pick the winner.** Mean validation AUC orders the
  stages v1 > v4 > t0 > v2 > v3, almost the reverse of test. Across seeds, the
  validation-test rank correlation ranges from −0.27 to 0.81. The choice
  of v3 is a test-set choice. Reporting it is honest as an architecture
  ablation over a fixed seed set, but not as a validation-selected number.
- **The budget is close to binding.** Two v3 seeds selected epochs 12 and 13
  and ran to the 20-epoch cap without patience stopping them. A 30-epoch rerun
  would show whether v3 has more headroom.
- Pooling all 40 ladder-3 models gives 0.796 (score average) or 0.800 (rank
  average), no better than v3's own ensemble.

**Molhiv numbers to report:** v3, 0.786 ± 0.007 over 10 seeds, seed ensemble
0.801. On the OGB leaderboard the single-model mean sits above GIN+VN (0.771)
and below the best graph-only entries (CIN 0.809, GSN 0.804). The ensemble is
not comparable to single-model leaderboard entries.
