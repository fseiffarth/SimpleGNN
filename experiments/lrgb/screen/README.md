# LRGB screening: what works for ShareGNN on Peptides-struct

A fast architecture screen on Peptides-struct from the Long Range Graph
Benchmark. Every candidate trains with three seeds; there is no
best-configuration rerun. Winners go into a longer confirmation run.

```bash
PYTHONPATH=src venv/bin/python experiments/lrgb/screen/screen.py \
    --config experiments/lrgb/screen/main_round1.yml --num_threads 3
# summary only, from existing results:
PYTHONPATH=src venv/bin/python experiments/lrgb/screen/screen.py \
    --config experiments/lrgb/screen/main_round1.yml --analyze_only
```

Results are written to `results/lrgb_screen/round1/`, and the table to
`results/lrgb_screen/round1/screen_summary.csv`.

## Why Peptides-struct, and why first

- **The framework reports the official metric.** Peptides-struct is an
  11-target graph regression scored by MAE. The pooled MAE the framework
  reports over all graph × target entries equals LRGB's mean of the
  per-target MAEs. The targets are already standardized in the dataset (mean
  0, std 1), so the numbers are in LRGB units. Peptides-func needs Average
  Precision, which the framework does not compute (specs/20); it follows
  once the screen shows what works.
- **It can resolve architecture choices, unlike molhiv.** The test set has
  2,331 graphs and 11 continuous targets. Published seed standard deviations
  are about 0.001–0.002 MAE, against molhiv's bootstrap SE of 0.027 AUC.
- **It tests ShareGNN's distinctive mechanism.** A ShareGNN convolution can
  give every (label pair, shortest-path distance) its own weight, so one
  layer reaches as far as the distance property goes. An MPNN needs one
  layer per hop.

## Dataset facts (measured on the local copy)

| | |
|---|---|
| Graphs | 15,535 (official split 10,873 / 2,331 / 2,331) |
| Nodes per graph | mean 151, median 136, max 444 |
| Diameter | mean 56, median 52, max 158 |
| Atom types (primary label) | 6; bond labels: 3 |

Ordered node pairs within distance d, over the whole dataset, measured on a
2,000-graph sample. Each convolution head materializes this many rows on the
GPU at about 9 bytes per row:

| d ≤ | 1 | 3 | 8 | 15 | 30 | all |
|---|---|---|---|---|---|---|
| pairs | 4.7 M | 18 M | 65 M | 139 M | 279 M | 449 M |
| pairs per node | 2 | 8 | 28 | 59 | 119 | 192 |

There are only 6 atom types, so a primary-label head has about 36 label
pairs per distance, roughly 1,100 rules even at d ≤ 30. Long range costs
memory and compute, not parameters, which keeps the 500k-parameter LRGB
budget out of play.

## Reference points (Peptides-struct test MAE, lower is better)

These are taken from the literature as summarized in `../OPTIMAL_CONFIG.md`;
verify them before citing.

| Model | MAE | Source |
|---|---|---|
| GCN / GINE / GatedGCN, original LRGB settings | 0.350 / 0.355 / 0.342 | Dwivedi et al. 2022 |
| GraphGPS | 0.250 | Rampášek et al. 2022 |
| GCN, tuned (6 layers, 3-layer MLP head, 250 epochs, cosine) | 0.246 | Tönshoff et al. 2023 |

The screen trains for 60 epochs instead of 250, so its numbers rank the
candidates. They are not leaderboard entries.

## Round 1: distance range and depth

All candidates share one recipe (`parameters_screen.yml`): AdamW, lr 1e-3,
no weight decay, cosine schedule with 5 warmup epochs, 60 epochs, no early
stopping, batch 128, MAE loss, float precision, rule threshold 5, and no L1.
The epoch with the best validation MAE is selected.

The reference block is: linear 64 + LayerNorm. Then each convolution block
is a ShareGNN convolution with three heads, followed by linear 64 +
LayerNorm. The three heads are:

- primary × primary over distances D (2 heads)
- WL-0 × WL-0 over distances D (2 heads)
- primary × primary over bond-labelled distance 1 (2 heads)

The readout is label-stratified pooling (primary, WL-0), followed by
LayerNorm and a 3-layer MLP head (64 → 64 → 11), the head Tönshoff et al.
found decisive.

| Id | Candidate | Change against p0 | Rows on GPU (approx.) |
|---|---|---|---|
| 0 | `p0_reference` | two blocks, D = 1..15 | 1.1 B (10 GB) |
| 1 | `p1_local` | D = {1}: an MPNN-like baseline | 56 M (0.5 GB) |
| 2 | `p2_d3` | D = 1..3 | 160 M (1.5 GB) |
| 3 | `p3_d8` | D = 1..8 | 540 M (5 GB) |
| 4 | `p4_d30` | D = 1..30 (cutoff-30 distance file) | 2.2 B (20 GB) |
| 5 | `p5_one_block` | one block instead of two | 0.57 B (5 GB) |
| 6 | `p6_row_norm` | `degree_normalization: row` on both convolutions | 1.1 B (10 GB) |
| 7 | `p7_walk_pe` | closed-walk encoder (k = 2..8, RWSE analogue) before the input linear | 1.1 B (10 GB) |

- **p1–p4 against p0 is the main question:** does the error keep falling as
  the distance range grows? That is the long-range claim. With two blocks,
  the receptive field is 2·max(D).
- **p5** asks whether depth still matters once one layer reaches distance 15.
  On molhiv two local blocks beat one.
- **p6** controls for the neighbourhood size. A node sums over about 2
  partners at D = 1 and about 119 at D ≤ 30. Mean aggregation separates
  "more range" from "bigger sums".
- **p7** is the positional-encoding gain the LRGB literature reports, in
  ShareGNN form.

`num_workers: 3` keeps the heaviest three-run combinations within the 95 GiB
card. Preprocessing generates two missing property files on the first run:
distances with cutoff 30 (for p4) and closed-walk labels (for p7).

## Reading the outcome

- Treat a candidate as different from p0 when its 3-seed mean MAE differs
  by more than about 0.005 and the seeds agree. This is a guess, to be
  replaced by the observed seed spread.
- **MAE falls steadily up to D = 15/30:** long-range rules are the lever.
  Round 2 widens the range (cutoff 60 or all pairs, ~4 GB per head) and tests
  which heads need it.
- **MAE flattens at D = 3–8:** local structure is enough. Round 2 is about
  depth (more stacked local blocks) and the head recipe.
- **The `EpochTime` column** of the result CSVs gives the cost per epoch for
  each distance range, for planning the 250-epoch confirmation run.
