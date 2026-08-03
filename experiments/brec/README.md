# BREC — GNN expressiveness benchmark

[BREC](https://arxiv.org/abs/2304.07702) (Wang & Zhang, *An Empirical Study of
Realized GNN Expressiveness*, ICML 2024) measures how many non-isomorphic graph
pairs a model can actually tell apart. It supersedes EXP / CSL / SR25 as the
standard expressiveness benchmark: 400 pairs spanning 1-WL-hard to 4-WL-hard
instances, with a statistical test that controls for false positives.

Data: `brec_v3.npy` from the MIT-licensed [GraphPKU/BREC](https://github.com/GraphPKU/BREC)
repository (`BREC_data_all.zip`, downloaded automatically into `data/BREC/raw/`).

Implementation notes and design decisions: `specs/21-brec-expressiveness-benchmark.md`.

## Categories

| Category | pair ids | pairs | nodes/graph | diameter |
|---|---|---|---|---|
| Basic | 0–59 | 60 | 10 | 2–4 |
| Regular | 60–159 | 100 | 7–35 | 2–5 |
| Extension | 160–259 | 100 | 10–19 | 2–5 |
| CFI | 260–359 | 100 | 18–198 | 4–17 |
| 4-Vertex_Condition | 360–379 | 20 | 63 | 2 |
| Distance_Regular | 380–399 | 20 | 30–63 | 3 |

The paper's results table groups these into four columns and folds
4-Vertex_Condition and Distance_Regular into "Regular (140)". This runner
reports the six categories of the reference implementation, so sum those three
before comparing against the paper.

## The RPC protocol

BREC is not a train/validation/test task, so it does not go through
`FrameworkMain.run_configurations()`. For each pair independently:

1. Take the pair's 32 relabelings of both graphs (64 graphs).
2. Train a fresh model for 50 epochs to push the two graphs' 16-dimensional
   embeddings apart (`CosineEmbeddingLoss`, target `-1`, Adam 1e-3/wd 1e-5).
3. Compute a Hotelling T² statistic over the 32 embedding differences.
4. Repeat 1–3 on a *known-isomorphic* control pair.
5. The pair counts as **distinguished** iff `T² > 5.0` and `T²` differs from the
   control's — so a model that emits a large statistic for everything, including
   isomorphic graphs, scores zero.

`main_brec.py` implements this and reuses the framework only for the data
pipeline (dataset construction, invariant labels and properties) and
`GraphModel`. All protocol constants are transcribed from
`GraphPKU/BREC@Release:DropGNN/test_BREC.py`.

## Running

```bash
# fastest end-to-end check: the GIN control needs no invariant labels, so only
# the dataset is built (~0.15 s/pair afterwards)
MODEL=gin PARTS=Basic PAIRS=3 EPOCHS=3 ./experiments/brec/run_brec.sh

# same, exercising the ShareGNN label/property path on the reduced dataset
MODEL=smoke PARTS=Basic PAIRS=2 EPOCHS=3 ./experiments/brec/run_brec.sh

# GIN negative control, full protocol (expected ~0/400 — see below)
MODEL=gin ./experiments/brec/run_brec.sh

# ShareGNN, full protocol — see "Degenerate pairs" below for why EPSILON is set
EPSILON=1e-7 ./experiments/brec/run_brec.sh
```

Pairs are evaluated independently, so a full ShareGNN run can be sharded across
processes by category (`PARTS=CFI`, `PARTS=Regular`, …); each shard writes its
own CSV.

Or directly:

```bash
python experiments/brec/main_brec.py --dataset BREC \
    --config experiments/brec/configs/main_config_brec.yml
```

Useful flags: `--parts Basic,CFI`, `--pairs N` (first N pairs per category),
`--epochs N`, `--epsilon-matrix X`, `--skip-preprocessing`, `--config_id N`
(index into the model config's grid). Per-pair statistics (both T² values, loss,
verdicts, seconds) are written to
`<results>/<dataset>/brec_rpc_<config_id>[_<parts>][_firstN].csv` — the
selection is part of the name so shards do not overwrite each other.

## Configs

| File | Dataset | Model | Purpose |
|---|---|---|---|
| `configs/main_config_brec.yml` | `BREC` | `models_brec.yml` | ShareGNN, official protocol |
| `configs/main_config_brec_gin.yml` | `BREC` | `models_brec_gin.yml` | GIN negative control |
| `configs/main_config_brec_smoke.yml` | `BREC-r4` | `models_brec.yml` | cheap smoke runs |

`BREC-r<k>` keeps only *k* of each graph's 32 relabelings (`BREC-r4` is 6,400
graphs instead of 51,200), which cuts preprocessing cost proportionally. **Runs
on a reduced dataset are not citable scores**: with 4 samples for a
16-dimensional difference vector the covariance is rank-deficient, and the
pseudo-inverse makes the test far more permissive than the protocol's 32
relabelings.

## Reference scores

From Table 2 of the paper (pairs distinguished / 400):

| Model | Basic (60) | Regular (140) | Extension (100) | CFI (100) | Total (400) |
|---|---|---|---|---|---|
| 3-WL | 60 | 50 | 100 | 60 | **270** |
| SPD-WL | 16 | 14 | 41 | 12 | 83 |
| PPGN | 60 | 50 | 100 | 23 | 233 |
| GSN | 60 | 99 | 95 | 0 | 254 |
| I²-GNN | 60 | 100 | 100 | 21 | 281 |
| KP-GNN | 60 | 106 | 98 | 11 | 275 |
| SUN | 60 | 50 | 100 | 13 | 223 |
| NGNN | 59 | 48 | 59 | 0 | 166 |
| δ-k-LGNN | 60 | 50 | 100 | 6 | 216 |
| Graphormer | 16 | 12 | 41 | 10 | 79 |

Plain message passing is absent from the table because **every BREC pair is
1-WL-indistinguishable by construction** — GIN's true score is 0.

## Measured here: the GIN control

Full protocol (400 pairs × 50 epochs, 32 relabelings), `models_brec_gin.yml`:
see `results/brec/gin/BREC/brec_rpc_0.csv`. The expected result is 0/400, and
what the run produces is a small number of *marginal* false positives — pairs
whose T² lands just above the 5.0 threshold. That is inherent to RPC, not a bug:
the threshold is a fixed significance level, so over 400 independent tests a few
crossings are expected (the paper introduces RAPC precisely to "provide an upper
bound for false positive rates"). Anything beyond a handful, or any pair with a
large T², would indicate a harness problem.

## Caveats

- **Constant node features are mandatory.** `parameters_brec.yml` sets
  `input_features: {name: constant, in_dimensions: 1, value: 1.0}`. Adding
  `random_variation:` (as the `share_gnn_basic` example does) would let any
  model separate any pair, reporting 400/400 while also breaking the
  reliability control.
- **Degenerate pairs — this affects ShareGNN in practice.** The T² statistic
  asks whether the mean embedding difference is large *relative to its variance
  across relabelings*. The reference implementation relies on its models being
  permutation-sensitive in floating point, so that variance is small but
  non-zero. ShareGNN's invariant layers are exactly permutation-invariant, so
  the variance is numerically zero and the statistic is meaningless: `pinv` of
  an exactly-zero covariance gives `T² = 0` (false negative), and of a
  near-zero one gives absurd values (measured: ~10³¹). The runner flags these
  pairs `[DEGENERATE]` and counts them.

  For such a model, run with `--epsilon-matrix 1e-7`. With `S ≪ εI` the
  statistic becomes `T² ≈ ‖d̄‖²/ε`, i.e. the decision reduces to "the mean
  embedding difference exceeds `√(5ε)`" — exactly the right question for a
  deterministic invariant model, and the reference's own suggested workaround
  (shipped commented out in its `T2_calculation`). Measured on Basic pairs 0–1:
  `T² ≈ 1.6·10⁸` for the pair versus **exactly 0.000** for the isomorphic
  control, which is the clean signature of a model that separates the pair and
  maps isomorphic graphs to identical embeddings.
- **Relabelings must exceed the embedding width.** `cov(D)` is
  `output_dim × output_dim` estimated from `num_relabel` samples, so any variant
  with `num_relabel ≤ 16` has a singular covariance by construction and inflates
  the statistic. The protocol's 32-vs-16 is fine; `BREC-r4` is not (the runner
  warns). `BREC-r20` is the smallest statistically valid reduced variant.
- **Receptive field.** `models_brec.yml` caps the `distances` property at 8,
  while the largest CFI graphs have diameter 17. Node pairs further apart than
  8 hops do not exchange messages, a disclosed limitation.
- **Cycle labels.** The architecture uses one `induced_cycles` (chordless) label
  over lengths 3–6 rather than per-length `simple_cycles` heads. On the
  benchmark's densest graphs (4-Vertex_Condition: 63 nodes, average degree 30)
  simple-cycle enumeration up to length 5 takes ~29 s/graph versus ~1.6 s for
  chordless cycles up to length 6, which is the difference between a feasible
  and an infeasible preprocessing run over 51,200 graphs.
- **Preprocessing is single-threaded here, and slow.** `_adaptive_parallel_map`
  (`simplegnn/datasets/utils/node_labeling.py`) decides whether to parallelize
  label generation by timing the *first five* graphs. BREC's first five are
  10-node Basic graphs, so the projection comes out near zero and the whole
  dataset runs serially — including the 4-Vertex_Condition graphs that dominate
  the cost. Expect ~15 min for `BREC-r4` and on the order of two hours for the
  full 51,200-graph `BREC`, one time (labels and properties are cached under
  `data/BREC/`). Both BREC configs share those paths, so the ShareGNN and GIN
  runs pay it once between them.
- **Threshold.** The paper defines `T² = q · d̄ᵀ S⁻¹ d̄` and derives the
  threshold from an F-distribution quantile; the released code drops the factor
  `q` and hard-codes `5.0`. Published BREC numbers come from the code, so the
  code is what is reproduced here.
