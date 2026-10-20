# QM9 Optimal ShareGNN Configuration — Rationale

Files (all new; nothing existing was modified):

- `experiments/qm9/models_QM9_optimal.yml` — network architecture
- `experiments/qm9/parameters_QM9_optimal.yml` — training hyperparameters
- `experiments/qm9/configs/main_config_QM9_optimal.yml` — main config (target 0 = dipole moment mu)

Run with the existing entry point by pointing `main_QM9.py` (or a copy) at
`configs/main_config_QM9_optimal.yml`. No experiments were run to produce these
files; they encode the best available prior knowledge from (a) the tuned ZINC
configs in this repo and (b) published QM9 practice for 2D/topology-only GNNs.

## 1. Comparison class: 2D / topology-only

ShareGNN consumes graph topology plus node/edge labels. The fair QM9 reference
class is therefore GNNs **without 3D coordinates**: Gilmer et al.'s MPNN
ablations, GNN-FiLM, Nested GNN's 1-GNN base, and the "coordinate-free QM9"
track of Gasteiger et al.'s synthetic-coordinates paper. SchNet/DimeNet(++)
MAEs are 1–2 orders of magnitude lower **because they use 3D geometry** — they
are upper-bound references only, not baselines to match.

A subtle repo-specific trap: `QMGraphDataPreprocessing`
(`src/simplegnn/datasets/graph_dataset_preprocessing.py:323`) concatenates the
3D positions (`dataset.pos`) into `node_attributes`, and the existing
`parameters_QM9.yml` feeds them to the model via
`input_features: {name: all, ...}`. The optimal config avoids this by using the
invariant-based positional encoding with `concatenate_input: false` and
`input_features: {name: constant, ...}` — input node features are discarded, so
the model is provably coordinate-free. Atom types still enter through the
`primary`-label heads of the encoder/convolution/aggregation.

## 2. What was taken from the tuned ZINC configs (closest task, same framework)

Source: `experiments/base_paper/regression/ZINC/configs/{network,parameters}_ZINC_test_v12.yml`
(v2 → v12 ablation chain, specs/09, specs/14, specs/15):

- **Invariant-based positional encoding as node encoder** (`num: 2` per head,
  heads reuse the aggregation label set so no extra label files are generated).
- **Pre-norm residual convolution block** (`residual: true`,
  `pre_layer_norm: true`) instead of a free-standing layer_norm.
- **Label families**: cycles (induced + simple) + `primary` + `wl_labeled`
  depth 0 + `wl_labeled_edges` depth 0/1; combined `[cycles, primary]` labels
  for encoder/aggregation heads.
- **Bond-type head** via `edge_label_distances` (cutoff 1) — QM9's four bond
  types (single/double/triple/aromatic) become distinct message rules.
- **Readout**: `linear aggr_channels(100)` → layer_norm →
  `invariant_based_aggregation` (`num: 10` per head) → reshape → MLP
  100 → 100 → 1 with LeakyReLU + layer_norm.
- **Optimizer/regularization recipe**: AdamW (weight_decay 1e-5), MAE loss,
  `mean_aggregation` weight init (gain 1.0, mean_ratio 0.5),
  proximal L1 0.01 on convolution/aggregation, ReduceLROnPlateau
  (factor 0.5, min_lr 1e-5) with lr-floor early stopping,
  `share_gnn_forward: {batched: True}`.

## 3. QM9-specific adaptations

QM9 graphs include hydrogens: ≤ 29 atoms, ≤ 9 heavy atoms, 130,831 molecules
(~110k train under the fixed 80/10/10 split — ~10x ZINC-subset).

| Knob | ZINC v12 | QM9 optimal | Why |
|---|---|---|---|
| Cycle lengths | 5–10 | induced 3–6, simple 5–8 | rings only span heavy atoms (≤ 9); QM9 is rich in 3–6 rings; induced == simple for short lengths, simple adds fused-ring perimeters at 5–8 |
| Closed-walk lengths | 2–10 | 2–8 | small graphs; walk spectra beyond length 8 add little and blow up the vocabulary |
| Distance values | 1–23 | 1–12 (cycle/primary heads), 1–4 (WL heads), 1–2 (WL-1 edges) | QM9 shortest-path diameters ≤ ~12 (old QM9 config's 0–23 wasted parameters) |
| `rule_occurrence_threshold` | 10 | 20 | 10x more graphs makes an absolute count of 10 a much weaker rarity filter; 20 keeps the effective cutoff comparable (sweep {10, 20, 50}) |
| L1 on encoding | 0.01 | 0.0 | encoder-table pruning is the risky variant (ZINC v10 ablated it off); with 10x data the encoder overfits less |
| Batch size | 32 | 128 | ~860 steps/epoch; large-dataset practice (64–128), amortizes the batched forward |
| Epochs | 500 | 200 | 200 epochs ≈ 170k steps at batch 128, already ≥ ZINC's total step count; lr-floor stop usually ends runs earlier |
| Scheduler patience | 10 | 5 | patience is measured in epochs and QM9 epochs are ~3x longer in steps |
| Precision | float | float | kept; standardized targets are well-conditioned in float32 (old QM9 config's `double` doubles CPU cost) |
| Target handling | none | z-score + inversion | see below |

## 4. What was taken from the literature

- **Per-target training** (`regression_targets: [0]`, one run per property):
  Gilmer et al. found individually trained models consistently beat joint
  training across all 13 targets, by up to ~40%; per-target training remains
  standard QM9 protocol (also used by Nested GNN and the GNN-FiLM/QM9 line of
  work). Joint training (set `regression_targets` to a list and widen the last
  linear layer) is ~13x cheaper but measurably worse — use it only for quick
  screening. Change `regression_targets` + `results` path per target; PyG
  order: 0 mu, 1 alpha, 2 HOMO, 3 LUMO, 4 gap, 5 R2, 6 ZPVE, 7 U0, 8 U, 9 H,
  10 G, 11 Cv.
- **Target standardization to mean 0 / variance 1** during training, inverted
  before metric computation (`output_features` / `invert_outputs:
  {normalization: 'standard'}`): Gilmer et al. normalized all targets; the
  inversion keeps reported MAE in native units, comparable to published tables.
- **MAE (L1) loss**: QM9's metric; training on it directly is standard for QM9
  regressors and matches the repo's ZINC protocol.
- **lr 1e-3 + ReduceLROnPlateau(0.5) to 1e-5 with stop-at-floor**: the
  Dwivedi et al. "Benchmarking GNNs" molecular protocol, already adopted by
  the ZINC configs; kept for cross-benchmark comparability. (Gilmer used a
  hand-tuned exponential decay from ~2e-4 over 540 epochs at batch 20 — the
  plateau schedule is the modern, less brittle equivalent.)
- **Hidden width ~100, deep-ish readout**: within the 64–200 range used across
  2D QM9 baselines (GNN-FiLM and the tf2-gnn implementation use hidden dims in
  this range with 8 propagation steps; ShareGNN trades depth for
  long-distance message rules, so a single conv block with distances 1–12
  covers the same receptive field).

### Sources

- Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017 —
  https://arxiv.org/abs/1704.01212 (PDF: https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf)
  — QM9 splits (10k val / 10k test), target normalization, per-target > joint,
  batch 20 / 540 epochs, lr range 1e-5–5e-4 with decay.
- Brockschmidt, *GNN-FiLM: Graph Neural Networks with Feature-wise Linear
  Modulation*, ICML 2020 — https://arxiv.org/pdf/1906.12192 (impl:
  https://github.com/microsoft/tf2-gnn/blob/master/README.md) — extensively
  tuned 2D QM9 baselines, 8 layers, per-task protocol.
- Dwivedi et al., *Benchmarking Graph Neural Networks*, JMLR 2023 —
  https://arxiv.org/abs/2003.00982 — lr 1e-3, ReduceLROnPlateau factor 0.5,
  min_lr 1e-5, stop at lr floor (adopted via the ZINC configs).
- Zhang & Li, *Nested Graph Neural Networks*, NeurIPS 2021 —
  https://openreview.net/pdf?id=7_eLEvFjCi3 — per-target QM9 protocol for
  2D message-passing baselines (1-GNN etc.).
- Gasteiger et al., *Directional Message Passing on Molecular Graphs via
  Synthetic Coordinates*, NeurIPS 2021 —
  https://proceedings.neurips.cc/paper_files/paper/2021/file/82489c9737cc245530c7a6ebef3753ec-Paper.pdf
  — the "coordinate-free QM9" comparison class; quantifies the 2D-vs-3D gap
  (SchNet/DimeNet as 3D upper bounds only).

## 5. Expected trade-offs / open sweeps

- **Per-target vs joint**: 12 targets x ~1 run each is the accuracy-optimal
  protocol but 12x the compute; the config ships per-target with target 0.
- **`rule_occurrence_threshold` 20**: the highest-leverage single scalar. Too
  low → WL-depth-1 edge vocabularies explode on 130k graphs (memory +
  overfitting); too high → deletes informative rare rules (e.g. length-8
  perimeter cycles). Sweep {10, 20, 50} before committing to the 12-target run.
- **Preprocessing cost**: label/property generation runs once over 130k graphs
  and is the dominant one-off cost; the encoder reusing the aggregation label
  set keeps the label-file count at 12 + the 6 conv-only sets.
- **Batch 128 vs 32**: 128 assumed for throughput at 110k graphs; if the
  plateau scheduler stalls high, batch 64 + lr 1e-3 is the fallback (both can
  be grid-searched by listing them in `batch_size:`).
- **intensive alternative**: `attention_readout` (see
  `network_ZINC_attention.yml`) was tried on ZINC but did not displace the
  invariant aggregation; not carried over.
