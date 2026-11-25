# molhiv architecture screening

A fast screen of candidate ShareGNN architectures on `ogbg-molhiv`. It
runs the grid search only, with three seeds per candidate. Promising
candidates then get a 10-seed ladder stage.

```bash
# both rounds, one after another (round 1, then round 2)
PYTHONPATH=src venv/bin/python experiments/ogb/molhiv_screen/screen.py \
    --config experiments/ogb/molhiv_screen/main_round1.yml \
    --config experiments/ogb/molhiv_screen/main_round2.yml --num_threads 4
# summarize existing results again
PYTHONPATH=src venv/bin/python experiments/ogb/molhiv_screen/screen.py \
    --config experiments/ogb/molhiv_screen/main_round1.yml \
    --config experiments/ogb/molhiv_screen/main_round2.yml --analyze_only
```

The per-candidate table of each round is printed when the round finishes and written to
`results/ogb_molhiv_screen/<round>/screen_summary.csv`.

## Why screen architectures

Ladders 1 and 2 changed capacity and regularization. They tested the rule
threshold, a trimmed architecture, L1 at 0.05 and 0.5, weight decay at 0.01
and 1.0, node dropout and an AUC-margin loss. None of them moved single-seed
test AUC off 0.762–0.765. L1 0.5 made it worse (0.708 ± 0.047, underfitting).
An ensemble of all 80 completed seed models reaches only 0.777, and the
models' test rankings are highly correlated (mean pairwise Spearman 0.69).
Combining more models of this family therefore cannot raise the ceiling.
The candidates here change what the model represents.

## Budget and noise

Shared parameters (`parameters_screen.yml`) are the ladder-2 t0 parameters with
at most 8 epochs and patience 3. On the GPU an epoch takes about 4 s; four
parallel workers fit at batch 64 (`num_workers: 4`). With three seeds, the standard
error of a candidate mean is about 0.005 (seed std 0.007–0.010). Only
differences above about 0.015 against `r0_reference` count as signal. The
table also reports:

- **Oracle Test:** the best test AUC over all epochs. It separates a higher
  ceiling from a better checkpoint choice.
- **Ensemble Test:** the AUC of the probability-averaged 3-seed ensemble.

## Round 1 candidates (`models_round1.yml`)

| Id | Candidate | Change against the t0 architecture | Question |
|---|---|---|---|
| 0 | r0_reference | none | Does the 8-epoch screen reproduce t0 (0.763)? |
| 1 | r1_local_conv | Every convolution distance head capped at 3 (the primary head had 1..10) | Do long-range pair rules memorize scaffolds? |
| 2 | r2_two_conv | r1 plus a second convolution with distances ≤ 2 | Does depth by composition help, as in GIN and CIN? |
| 3 | r3_no_encoder | No closed-walk encoder | Does the encoder add signal or noise? |
| 4 | r4_pool_wl1 | Pooling adds a WL-with-edges depth-1 head | Does an ECFP-like substructure readout help? |
| 5 | r5_optimal_arch | `../models_ogb_optimal_classification.yml` (residual pre-LN convolution, simple cycles, `aggr_channels`) | The prepared, never-run design |
| 6 | r6_mean_pool | r0 with `reduce: mean` pooling (node-count normalized, as in OGB's GIN baselines) | Does size-invariant pooling generalize better across scaffolds? |

## Round 2 candidates (`models_round2.yml`)

Round 2 does not depend on round 1's results. It changes the axes that round
1 leaves alone: embedding width, readout structure, encoder labels and ring
types. It uses the same parameters and seeds as round 1, so round 1's
`r0_reference` is the reference for both rounds.

| Id | Candidate | Change against the t0 architecture | Question |
|---|---|---|---|
| 0 | q0_wide_embedding | Input linear 16 → 64 features per node | Is the 16-dim node embedding a bottleneck? |
| 1 | q1_channelwise_readout | Unflattened pooling (`flatten: False`) with one 64 → 16 matrix per head (`mode: channel_wise`) | Does a structured readout generalize better than the dense flattened one? |
| 2 | q2_attention_readout | Unflattened pooling with gated attention over the heads (`attention_readout`) | Does adaptive head weighting help? |
| 3 | q3_bond_walks | Encoder walks weighted by bond type (`closed_walks_weighted`, specs/24) | Does the encoder need bond types? |
| 4 | q4_simple_cycles | Convolution and pooling heads for simple cycles of length 5–6 | Do fused ring systems (simple vs induced cycles) help? |
| 5 | q5_ring_sizes_3_8 | Pooling ring head covers cycle lengths 3–8 instead of 5–6 | Do small and large rings help? |

## Checked

All 13 candidates of both rounds built, trained for one epoch, saved their
predictions and appeared in the summary table in a CPU smoke run on
`ogbg-molbace` (one seed).

## Results of rounds 1 and 2 (GPU, 2026-10-22)

Test ROC-AUC at the validation-selected epoch, three seeds, compared with
`r0_reference` (0.767 ± 0.003). Ladder-2 t0 reached 0.763 with ten seeds, so
the 8-epoch screen is calibrated. "Oracle" is the best test AUC at any epoch.

| Candidate | Test | Δ | Oracle | Ensemble |
|---|---|---|---|---|
| r1_local_conv | 0.774 ± 0.005 | +0.007 | 0.779 | 0.785 |
| r2_two_conv | 0.769 ± 0.005 | +0.002 | 0.777 | 0.776 |
| q5_ring_sizes_3_8 | 0.767 ± 0.008 | 0.000 | 0.770 | 0.781 |
| r3_no_encoder | 0.762 ± 0.007 | −0.005 | 0.768 | 0.765 |
| q3_bond_walks | 0.762 ± 0.007 | −0.005 | 0.768 | 0.773 |
| q1_channelwise_readout | 0.762 ± 0.006 | −0.006 | 0.767 | 0.774 |
| q2_attention_readout | 0.760 ± 0.018 | −0.008 | 0.777 | 0.771 |
| q0_wide_embedding | 0.758 ± 0.014 | −0.009 | 0.767 | 0.768 |
| q4_simple_cycles | 0.755 ± 0.005 | −0.012 | 0.771 | 0.760 |
| r6_mean_pool | 0.749 ± 0.011 | −0.019 | 0.765 | 0.752 |
| r4_pool_wl1 | 0.743 ± 0.008 | −0.024 | 0.749 | 0.748 |
| r5_optimal_arch | 0.734 ± 0.011 | −0.033 | 0.739 | 0.747 |

- **More specific vocabulary hurts.** The radius-1 WL pooling head (r4),
  simple-cycle heads (q4) and the head-rich optimal design (r5) are the
  clearest results in both rounds.
- **Mean pooling hurts** (r6). Molecule size carries signal on molhiv.
- **Neutral:** readout structure (q1, q2), embedding width (q0), bond-weighted
  walks (q3) and dropping the encoder (r3).
- **Local convolution is the only gain.** Capping convolution distances at
  3 (r1) has the best mean, oracle and ensemble. The ensemble's 0.785 is the
  highest of any model so far. Train AUC at the selected epoch drops from
  0.85 to 0.78, so r1 memorizes less. The +0.007 is about one standard error
  with three seeds, so it is not established yet. The direction does match
  the other results: long-range and fine-grained rules memorize scaffolds.
  Ladder 1's capacity cuts never touched the primary distance head (1..10),
  which explains why they changed nothing.

## Round 3 candidates (`models_round3.yml`)

Round 3 follows the locality and coarseness direction with five seeds
(`num_runs: 5`). u0 and u1 repeat round-1 r0 and r1 as anchors. Seeds are
deterministic (42 + run id), so their first three seeds reproduce round 1,
and seeds 45 and 46 are new.

```bash
PYTHONPATH=src venv/bin/python experiments/ogb/molhiv_screen/screen.py \
    --config experiments/ogb/molhiv_screen/main_round3.yml --num_threads 4
```

| Id | Candidate | Change | Question |
|---|---|---|---|
| 0 | u0_reference | Round-1 r0 | Anchor with five seeds |
| 1 | u1_local3 | Round-1 r1: convolution distances ≤ 3 | Does the +0.007 hold with five seeds? |
| 2 | u2_local2 | Convolution distances ≤ 2 | Dose response: more locality |
| 3 | u3_local1 | Convolution distance 1 only (bond neighbours) | Dose response: GIN-like locality |
| 4 | u4_local3_no_wl_conv | u1 without the WL-labeled convolution heads | Coarser convolution vocabulary |
| 5 | u5_local3_coarse_pool | u1 pooling only on primary and ring labels (no WL-with-edges head) | Coarser pooling vocabulary (r4 in reverse) |
| 6 | u6_local3_rings_3_8 | u1 with pooling ring lengths 3..8 | q5's ensemble gain on top of u1 |

A candidate that beats u0 by more than about 0.012 (about two standard errors
of a five-seed difference) goes into a 10-seed ladder stage. That stage
includes the seed ensemble, which is where the local models gained most.

## Results of round 3 (GPU, 2026-10-22)

Five seeds; differences are paired by seed against `u0_reference`.

| Candidate | Test | Paired Δ | Seeds better | p | Ensemble |
|---|---|---|---|---|---|
| u0_reference | 0.762 ± 0.007 | – | – | – | 0.769 |
| u1_local3 | 0.770 ± 0.008 | +0.008 | 4/5 | 0.08 | 0.782 |
| u2_local2 | 0.781 ± 0.011 | +0.019 | 5/5 | 0.011 | 0.793 |
| u3_local1 | 0.780 ± 0.007 | +0.018 | 5/5 | 0.012 | 0.791 |
| u4_local3_no_wl_conv | 0.768 ± 0.014 | +0.006 | 4/5 | 0.52 | 0.780 |
| u6_local3_rings_3_8 | 0.770 ± 0.010 | +0.008 | 4/5 | 0.24 | 0.784 |
| u5_local3_coarse_pool | 0.759 ± 0.010 | −0.003 | 1/5 | 0.67 | 0.771 |

Locality has a dose response: distances ≤ 2 and distance 1 beat the reference
on every seed. u3 peaked at the last of its 8 epochs in 3 of 5 seeds, so the
screen budget probably understates it. The 10-seed confirmation is
`../molhiv_ladder3/`.
