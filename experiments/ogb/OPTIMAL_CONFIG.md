# Optimal ShareGNN configs for the OGB `ogbg-mol*` datasets

Design rationale for the `*_optimal` config arm. No experiments were run to
produce these files; every choice below is justified either by published OGB
results or by tuning evidence already in this repository (the ZINC test
v2→v12 series and the molhiv t1/t50 sweep). Treat the remaining grid axes as
the open questions.

## Files

| File | Role |
|---|---|
| `models_ogb_optimal_classification.yml` | Architecture for molhiv / molbace / molbbbp (2 logits) |
| `models_ogb_optimal_regression.yml` | Same architecture, 1 output unit, readout dropout 0.2 |
| `parameters_ogb_optimal_molhiv.yml` | molhiv: lr 1e-4, threshold 10, wd grid {0.01, 0.1} |
| `parameters_ogb_optimal_classification.yml` | molbace / molbbbp: lr 1e-3, threshold 5, wd grid {0, 0.01} |
| `parameters_ogb_optimal_regression.yml` | molesol / molfreesolv / mollipo: RMSE loss, threshold 3, wd grid {1e-5, 0.01} |
| `configs/main_<dataset>_optimal.yml` (x6) | Main configs; results root `results/ogb_optimal/` |

Run with the existing pipeline (the `<dataset>-optimal` names are registered
in `main_ogb.py` and follow run_ogb.sh's `<dataset>-<variant>` convention):

```bash
NUM_THREADS=4 ./experiments/ogb/run_ogb.sh molhiv-optimal
./experiments/ogb/run_ogb.sh molbace-optimal
```

Each parameters file has exactly one grid axis (`weight_decay`, 2 values), so
every run is 2 configurations. Memory note from the README still applies to
molhiv: run with `NUM_THREADS` 2–4, not `-1`.

## What the literature says (and how it maps to ShareGNN)

1. **Substructure/fingerprint methods dominate ogbg-molhiv.** On the official
   leaderboard, plain message passing plateaus at GIN+VN 0.7707 ± 0.0149,
   while fingerprint-based and hybrid entries occupy the top: MorganFP +
   Random Forest 0.8060, Neural FingerPrints 0.8232, Graphormer+FPs 0.8225,
   HIG 0.8403, Multi-RF Fusion + Multi-GNN 0.8476
   ([leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/)). Morgan
   fingerprints are essentially counted local substructures — which is exactly
   what ShareGNN's invariant vocabulary (WL colors, cycles, closed walks)
   encodes as *learnable weights* rather than a fixed hash. This is the main
   reason to bet on the cycle/walk-heavy architecture rather than mimicking a
   deep GIN: the model family is closer to the winning end of the leaderboard
   than to the MPNN baselines.
2. **Baseline envelope** (Hu et al., *Open Graph Benchmark*, NeurIPS 2020,
   [arXiv:2005.00687](https://arxiv.org/abs/2005.00687)): 5-layer GNNs,
   hidden 300, mean pooling, dropout tuned in {0.0, 0.5}; the official
   `examples/graphproppred/mol` code trains with Adam, lr 1e-3, batch 32,
   100 epochs. More depth/width does not help on these datasets; scaffold
   generalization and regularization are the bottleneck. Hence: batch 32,
   100-epoch budget for molhiv, dropout 0.5 on the classification readout,
   and no attempt to grow the network beyond one conv + one aggregation
   block.
3. **Small models are competitive on molhiv** — a 32K-parameter GIN reaches
   ~0.77–0.78 ([tiny-GIN-for-ogbg-molhiv](https://github.com/willy-b/tiny-GIN-for-ogbg-molhiv)),
   and seed variance spans ~0.74–0.78. Capacity is not the constraint;
   vocabulary control is (see `rule_occurrence_threshold` below).
4. **Training protocol** (Dwivedi et al., *Benchmarking GNNs*,
   [arXiv:2003.00982](https://arxiv.org/abs/2003.00982)): ReduceLROnPlateau
   with factor 0.5 / patience 10 and stopping when the LR reaches its floor.
   Already adopted by the tuned ZINC configs; carried over verbatim
   (`scheduler` + `early_stopping.lr_threshold`).

## What was kept from the existing OGB configs

- The overall schema and layer skeleton of `models_ogb_classification.yml` /
  `models_ogb_regression.yml`: input encoding → invariant convolution →
  channel-aggregating linear → invariant aggregation → MLP readout.
- Distances capped at 10 (covers essentially every mol* diameter; ZINC's 1..23
  range would only add empty rule slots).
- `input_features: {name: all, transformation: one_hot_labels_normalize_features}`
  — OGB's 9 atom features carry real chemistry (charge, chirality, H count,
  aromaticity) and MoleculeNet models depend on them.
- Metric plumbing from the README: `evaluation_metric: roc_auc` **and**
  `evaluation_type: roc_auc` for classification (accuracy is meaningless at
  molhiv's 3.5% positive rate); `evaluation_type: loss` plus standard-score
  output normalization + inversion for regression.
- The molhiv sweep's optimization conclusions (`parameters_ogb_molhiv_t1.yml`):
  AdamW with `weight_decay` as a real grid axis (plain-string optimizer so the
  grid is honored), lr 1e-4 on molhiv because per-epoch checkpointing cannot
  resolve an optimum reached inside epoch 1 at lr 1e-3, `weighted_loss: False`
  (the ~27x positive upweight mostly accelerates memorizing individual
  positives under a rank metric), and readout dropout as an explicit layer —
  the top-level `dropout:` hyperparameter never reaches the model.

## What was changed, and why

### Architecture (both model files)

1. **`invariant_based_positional_encoding` as node encoder** (from ZINC
   v2–v12, specs/15) replacing the plain 16-dim linear input layer. Heads:
   closed-walk counts of lengths 2–8 (× primary label), primary, WL depth 0,
   WL-with-edges depth 0/1; `num: 2` each → 22 learned dims. **Divergence
   from ZINC:** `concatenate_input: true` (ZINC uses `false` and discards
   input features) because OGB atom features cannot be recovered from graph
   invariants. Closed walks are the spectral/ring-density signal that made
   the ZINC encoder work; lengths capped at 8 (not ZINC's 10) to limit the
   molhiv vocabulary.
2. **Pre-norm residual convolution** (`residual: true, pre_layer_norm: true`)
   replacing the free-standing `layer_norm` before the conv — the
   transformer-style block ZINC v2+ converged on; residual also covers the
   no-distance-0 case.
3. **`simple_cycles` heads added next to `induced_cycles`** in both the
   convolution and the aggregation (from ZINC v3+), but only lengths 5 and 6
   — the aromatic/aliphatic rings that dominate scaffold-split molecular
   benchmarks. ZINC uses lengths 5–10; molhiv has 4x the graphs and a larger
   atom vocabulary, so the long-cycle tail is deliberately dropped (it was
   also flagged as first-to-trim in the molhiv sweep notes).
4. **Cycle conv heads at `num: 1`** (ZINC v12 uses 1; the old OGB config used
   2) — halves the fastest-growing weight tables at no observed ZINC cost.
5. **`mode: aggr_channels` on the post-conv linear** (ZINC idiom): mixes the
   conv's output channels per node instead of flattening; out 64 (not ZINC's
   100) to stay in the small-model regime that is competitive on molhiv.
6. **Readout**: single hidden 64 + layer_norm + dropout (0.5 classification /
   0.2 regression) + linear head. Regression gets the lower rate because
   freesolv/esol have only ~0.5–0.9k training graphs and heavy dropout on a
   squared-error target adds noise rather than rank robustness; the OGB paper
   itself tunes dropout per dataset in {0.0, 0.5}.

### Hyperparameters

| Choice | Value | Why |
|---|---|---|
| `precision` | float | Halves time/memory; tuned ZINC runs float with no drift |
| `share_gnn_forward` | `{batched: True}` | ZINC v12 default; faster invariant forward |
| Optimizer | AdamW (plain string) + `weight_decay` grid | Decoupled decay is a real regularization axis; dict form would silently kill the grid |
| `rule_occurrence_threshold` | 10 (molhiv) / 5 (bace, bbbp) / 3 (regression) | Strongest anti-overfitting lever: deletes rare-rule parameters instead of shrinking them. 10 is the ZINC v12 operating point at ~10k train graphs; scaled down with dataset size so 642-graph freesolv is not starved. Scalar (read in the layer), not grid-searchable |
| `l1_regularization` | 0.01 conv/aggr/encoding | ZINC v12 proximal-L1 rule selection; zeroes uninformative rules exactly |
| `weight_initialization` | `mean_aggregation` gain 1.0, ratio 0.5 (+ PE normal 0.1) | ZINC v14-spec init family the v-series settled on; PE entry is mandatory for the encoder layer |
| `batch_size` | 32 | OGB example-code standard; on molhiv also more checkpoint resolution than 64 |
| `learning_rate` | 1e-4 (molhiv) / 1e-3 (rest) | molhiv: epoch-1 peak problem (sweep README). Small sets: ~40 steps/epoch, 1e-3 is the literature standard and resolution is fine |
| `epochs` / patience | 100/20 (molhiv), 200/30 (bace, bbbp), 300/40 (regression) | OGB budget for molhiv; ZINC-protocol budget elsewhere; `lr_threshold` stops runs at the LR floor anyway |
| `scheduler` | ReduceLROnPlateau 0.5/10, min_lr 1e-6 | Benchmarking-GNNs protocol, comparable to literature baselines |
| `loss` (regression) | RMSE | Train on the leaderboard metric; MAE is still logged per epoch |

## Expected trade-offs / open questions

- **`concatenate_input: true` vs `false`** is untested for this family on
  OGB. If the learned PE dominates, the ZINC-style pure-embedding encoder
  (with `input_features: constant`) is the natural ablation.
- **Threshold values are transferred, not tuned.** The molhiv t1/t50 sweep
  brackets the answer for molhiv; if t50 wins decisively there, raise 10 → 25+
  here. For freesolv, 3 may still be too high (only ~510 train graphs).
- **`weighted_loss: False` on molhiv** follows the sweep's reasoning, not a
  completed A/B; re-enabling it is a one-line experiment.
- **Vocabulary trimming vs capacity**: if the optimal arm still saturates
  train AUC in epochs 1–3, the next levers (in order) are: drop the
  `wl_labeled_edges depth 1` conv head, shorten the cycle-head distance
  ranges (1..10 → 1..6), and raise the threshold — all flagged in the sweep
  notes.
- **Reference targets**: beating GIN+VN (0.7707 ± 0.0149 ROC-AUC on molhiv)
  would put ShareGNN above every pure message-passing entry; the
  fingerprint hybrids (≥ 0.82) are the aspirational band and the fairest
  comparison class for an invariant-vocabulary model.

## Sources

- OGB graph property prediction leaderboard — https://ogb.stanford.edu/docs/leader_graphprop/
- Hu et al., *Open Graph Benchmark: Datasets for Machine Learning on Graphs*, NeurIPS 2020 — https://arxiv.org/abs/2005.00687
- Dwivedi et al., *Benchmarking Graph Neural Networks*, JMLR 2023 — https://arxiv.org/abs/2003.00982
- willy-b, *tiny-GIN-for-ogbg-molhiv* (32K-param GIN at ~0.77–0.78, seed-variance data) — https://github.com/willy-b/tiny-GIN-for-ogbg-molhiv
- OGB dataset/task definitions — https://ogb.stanford.edu/docs/graphprop/
- In-repo: `experiments/ogb/README.md` (molhiv t1/t50 sweep analysis),
  `experiments/base_paper/regression/ZINC/configs/*_test_v2..v12*` (ZINC
  tuning series), `specs/15-inv-based-positional-encoding.md`,
  `specs/24-edge-labeled-closed-walks.md`.
