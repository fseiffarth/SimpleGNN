# QM9 starting experiments: redesign and screen

Nothing in `experiments/qm9/` has been trained yet: `results/qm9/` is empty
and `OPTIMAL_CONFIG.md` states that its configs were never run. This folder
replaces both existing setups with a staged start: a three-target
architecture screen first, then a regularization sweep, then the full
per-target runs.

**Not yet verified:** none of these files has been loaded by `FrameworkMain`
or run. The first step below is that check.

## Problems with the existing configs

| Config | Problem |
|---|---|
| `models/parameters_QM9.yml` (migrated RuleGNN) | `input_features: all` feeds the 3D positions, so the model is not 2D. `rule_occurrence_threshold: 1` keeps every rare rule (memorization; see ZINC-full). Distances 0-23, but the QM9 diameter is 10. `precision: double`. LR decay factor 0.9 without an LR-floor stop. |
| `models/parameters_QM9_optimal.yml` | `weight_decay: 1e-5` with AdamW at lr 1e-3 shrinks weights by only ~1% over a run, so it has no real effect. Distances 11-12 never occur. 3- and 4-rings appear only as induced cycles in the convolution, not as node invariants. A single 200-epoch, 12-target plan with no screen before it. |

Both are left in place. If the screen confirms this design, they can be deleted.

## Protocol and comparison class

- **Split:** the existing fixed 80/10/10 split (seed 42,
  `splits/fixed/QM9_splits.json`), one model per target, 2 seeds in the
  screen and 3 in the final runs. This matches the k-GNN / Nested GNN /
  I²-GNN protocol (random 80/10/10, per target).
- **Input:** topology plus atom and bond labels only. The constant input
  feature means the 3D positions never reach the model.
- **Caveat on literature numbers:** the per-target 80/10/10 baselines are
  *not* 2D. Nested GNN's `run_qm9.py` always adds a `Distance` transform,
  which puts 3D interatomic distances on the edges. Treat those tables as
  references with geometry, not as 2D baselines. Coordinate-free QM9 results
  exist only under the GNN-FiLM protocol (fixed 110k/10k/10k split, 13
  targets, its own target scaling). Comparing with them would need that
  split generated as a second split file; round 3 can add it.
- **Units:** PyG stores HOMO, LUMO, gap, ZPVE, U0, U, H and G in eV. The
  80/10/10 tables report them in Hartree, converted by dividing by 27.2114.
  `screen.py` adds a `Test (Ha)` column for these targets.

### Standard hyperparameters in the literature

| Source | Optimizer, LR | Scheduler | Batch | Epochs | Loss / targets |
|---|---|---|---|---|---|
| Nested GNN `run_qm9.py` (k-GNN protocol) | Adam 1e-3 | Plateau ×0.7, patience 5, min 1e-5 | 64 | 200 | MSE on z-scored target; test MAE at the best-validation epoch |
| I²-GNN (ICLR 2023), App. F.2 | Adam 1e-3 | Plateau ×0.95, patience 10 | 64 | 400 | per target, 80/10/10 |
| Rethinking FWL (2306.03266)\* | 1e-3 | Plateau ×0.9, patience 5 | 128 | 350 | per target |
| DRew / SPN (GNN-FiLM protocol, 2D) | — | — | — | 300 | ~800k params, 8 layers, 3 runs |
| 3D Infomax, PNA fine-tuning\* | Adam 7e-5 | Plateau ×0.5, patience 25 | 128 | — | quantum targets |

\* From the search summary, not checked against the paper text.

Consensus: lr 1e-3, a ReduceLROnPlateau schedule down to 1e-5, batch 64-128,
200-400 epochs, z-scored targets with MAE reported in target units, and
3 seeds.

### Reference values (80/10/10, with 3D distances, energies in Hartree; I²-GNN Table 4)

| Target | 1-GNN | 1-2-3-GNN | NGNN | I²-GNN |
|---|---|---|---|---|
| μ (D) | 0.493 | 0.476 | 0.428 | 0.428 |
| gap (Ha) | 0.0049 | 0.0048 | 0.0038 | 0.0038 |

The U0 column of that table mixes scales across rows (1-GNN 2.32 vs
1-2-3-GNN 0.0427), so it is left out.

## Architecture (`models_round1.yml`, candidate 0 `base`)

QM9 graphs include hydrogens: ≤ 29 nodes, ≤ 9 heavy atoms, diameter ≤ 10
(`QM9_properties_distances.yml`). Primary node labels are atomic numbers
(H, C, N, O, F). Primary edge labels are the four bond types.

**Node label set N** (encoder `num: 2`, pooling `num: 10`; one set of label files for both):

| Heads | Label | What it encodes on QM9 |
|---|---|---|
| 1 | `primary` | element |
| 1 | `wl_labeled` d0 | element + neighbor elements: hydrogen count / valence |
| 1 | `wl_labeled_edges` d0 | plus bond orders: hybridization, i.e. the chemical atom type |
| 1 | `wl_labeled_edges` d1 | 2-hop environment (functional group) |
| 6 | `[closed_walks, primary]` k = 3-8 | local ring and branching spectrum (k = 2 is the degree, which WL d0 already refines) |
| 6 | `[induced_cycles, primary]` k = 3-8 | ring membership; QM9 is rich in 3- and 4-rings, which ZINC's 5-10 range missed |

**Convolution** (one pre-norm residual block, 16 heads; distances 1-10 cover
every node pair, so one block is enough for the receptive field):

| Heads | Head/tail labels | Distances |
|---|---|---|
| 6 | `induced_cycles` k = 3-8 | 1-10 |
| 5 | `simple_cycles` k = 4-8 (they differ from induced ones on fused and bridged cages) | 1-10 |
| 1 (`num: 2`) | `primary` × `primary`: a learned topological pair potential | 1-10 |
| 1 (`num: 2`) | `primary` × `primary`, bond-typed (`edge_label_distances`) | 1 |
| 1 | `wl_labeled` d0 | 1-4 |
| 1 (`num: 2`) | `wl_labeled_edges` d0 | 1-4 |
| 1 (`num: 2`) | `wl_labeled_edges` d1 | 1-2 |

The readout is the same as ZINC: linear `aggr_channels` 100, layer norm,
pooling, then an MLP 100 → 100 → 1.

**Candidates in round 1:**

| Id | Name | Change | Question |
|---|---|---|---|
| 0 | `base` | — | reference |
| 1 | `edge_walks` | `closed_walks` → `closed_walks_edges` (specs/24) in the encoder and pooling | Does splitting walks by bond type help, e.g. aromatic vs saturated rings? |
| 2 | `mean_pool` | `reduce: mean` in the pooling | Mean should help the intensive targets (gap) and hurt the extensive ones (U0). |

## Hyperparameters (`parameters_screen.yml`)

| Parameter | Screen | Final (round 3) | Why |
|---|---|---|---|
| optimizer | AdamW, wd 1e-3 | winner of round 2 | lr-scaled decay; 1e-5 does nothing (ZINC-full finding) |
| lr | 1e-3 | 1e-3 | literature consensus |
| scheduler | Plateau ×0.5, patience 4 | Plateau ×0.7, patience 5, min 1e-5 | the final run uses the k-GNN/NGNN schedule; the screen is shortened |
| epochs | 80 (stops at the LR floor) | 300 (stops at the LR floor) | equal budgets keep the arms paired by seed |
| batch | 128 | 128 | ~820 steps per epoch |
| loss | MAE | MAE | the metric |
| `rule_occurrence_threshold` | 20 | winner of round 2 | 2 memorizes on ZINC-full; 20 matched its validation MAE |
| L1 | off | off | threshold + decay are the regularizers |
| targets | z-scored, inverted for metrics | same | standard |
| precision / device | float / cuda | same | |

## Rounds

1. **Round 1 (this folder):** 3 candidates × 2 seeds × targets μ (0), gap (4)
   and U0 (7). These cover a geometry-dependent target, an intensive
   orbital energy and an extensive energy. That makes 18 runs.
2. **Round 2:** on the round-1 winner, targets μ and U0:
   `rule_occurrence_threshold` ∈ {5, 20, 50} (separate parameter files,
   since it is not grid-searchable) × weight decay ∈ {1e-4, 1e-3, 1e-2}
   (one grid, listed under `optimizer`).
3. **Round 3:** the final config with the round-3 schedule, all 12 targets
   (`main_QM9.py`-style, `run_best_configuration` with 3 seeds). For gap,
   HOMO and LUMO, use mean pooling if round 1 says so.

## Cost and how to run (not started)

- **Preprocessing (one-off):** new label files are needed: `closed_walks`
  k = 3-8 (plus `closed_walks_edges` for candidate 1), induced cycles
  k = 3-5, simple cycles k = 4-8, and their `primary` combinations. The
  existing QM9 labels took 4-40 s each; the distances are already computed.
- **Training time per epoch on QM9 is unknown.** Measure it from the first
  epoch of round 1 before committing to round 3, which has 12 targets × 3
  seeds × up to 300 epochs.
- **Memory:** `num_workers: 2`. Parallel workers each hold a full copy of
  the data, which has caused OOMs before. Raise the count only while
  watching `free -h`.

```bash
# from the repo root, venv active
python experiments/qm9/screen/screen.py \
    --config experiments/qm9/screen/main_round1_mu.yml \
    --config experiments/qm9/screen/main_round1_gap.yml \
    --config experiments/qm9/screen/main_round1_U0.yml
# summaries only
python experiments/qm9/screen/screen.py --analyze_only --config experiments/qm9/screen/main_round1_mu.yml
```

Each target writes `results/qm9_screen/round1/<target>/screen_summary.csv`.

## Sources

- Zhang & Li, Nested GNN, `run_qm9.py`: https://github.com/muhanzhang/NestedGNN/blob/master/run_qm9.py
- Huang et al., I²-GNN (ICLR 2023), Table 4 and App. F.2: https://arxiv.org/abs/2210.13978
- Gutteridge et al., DRew (ICML 2023), Sec. 5.2 / Table 3: https://arxiv.org/abs/2305.08018
- Abboud et al., Shortest Path Networks: https://arxiv.org/abs/2206.01003
- Brockschmidt, GNN-FiLM (ICML 2020): https://arxiv.org/abs/1906.12192
- Feng et al., Rethinking FWL: https://arxiv.org/abs/2306.03266
- Stärk et al., 3D Infomax: https://arxiv.org/abs/2110.04126
- Gilmer et al., MPNN (ICML 2017): https://arxiv.org/abs/1704.01212
