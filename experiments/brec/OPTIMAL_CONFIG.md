# Optimized BREC configs — rationale

Deliverables (no existing file was modified):

| File | Role |
|---|---|
| `models_brec_optimal.yml` | 3 network variants, selected via `--config_id 0/1/2` |
| `parameters_brec_optimal.yml` | parameters config (all training lists singletons, on purpose) |
| `configs/main_config_brec_optimal.yml` | wiring; shares the `data/BREC/` label/property cache |

```bash
python experiments/brec/main_brec.py --dataset BREC \
    --config experiments/brec/configs/main_config_brec_optimal.yml \
    --config_id 0 --epsilon-matrix 1e-7
```

`--epsilon-matrix 1e-7` remains mandatory for ShareGNN (degenerate-covariance
false negative, see `specs/21-brec-expressiveness-benchmark.md`).

## The one design fact that drives everything

ShareGNN is deterministic and exactly permutation-invariant, so under the
epsilon-ridge T² the RPC decision collapses to *"is the mean embedding
difference non-zero"* — i.e. **a pair is distinguished iff the two graphs'
invariant fingerprints (label histograms + (head-label, tail-label, distance)
rule multisets) differ**. Training hyperparameters are nearly irrelevant; the
BREC paper's own ablation confirms this pattern for feature-based models
(Table 3: GSN scores 254/400 with random weights and 254/400 trained;
I²-GNN 267 random vs 281 trained). Therefore the optimization budget goes
entirely into the **label vocabulary and the distance (receptive-field) range**,
not into lr/epochs/batch grids — which `main_brec.py` hard-codes to the
protocol values anyway (Adam 1e-3 / wd 1e-5 / batch 16 / 50 epochs, transcribed
from `GraphPKU/BREC@Release:DropGNN/test_BREC.py`). Those values sit inside the
paper's searched ranges (lr {1e-3, 1e-4, 1e-5}, wd {1e-3…1e-5}, batch
{8, 16, 32}, epochs 10–100, hidden dims 16–128; Appendix O + Table 10), so no
deviation is warranted.

## What was taken from where

**From the existing repo configs** (`models_brec.yml`, the transfer backbone
`network_transfer_Substructure_pretrain.yml`, ZINC `network_ZINC_test_v12.yml`):

- The whole layer skeleton: invariant positional encoding
  (`concatenate_input: false`) → pre-norm residual invariant convolution →
  `linear aggr_channels` → `layer_norm` → invariant aggregation → MLP →
  **final linear width 16** (the RPC protocol's `OUTPUT_DIM`; must not change).
- No `primary`/`wl_labeled` heads (BREC graphs are unlabeled), `induced_cycles`
  (chordless) 3–6 instead of `simple_cycles` (simple-cycle enumeration is
  ~29 s/graph on the dense 4-Vertex_Condition graphs vs ~1.6 s chordless —
  infeasible over 51,200 graphs), constant input features, `double` precision,
  `rule_occurrence_threshold: 1`, `mean_aggregation` weight init.
- Product labels (`label_type: [a, b]`) and per-length closed-walk heads, both
  production-proven in the ZINC v12 config; stacked invariant convolutions,
  proven in `config_EvenOddRings.yml`.

**From the literature:**

1. **Receptive field is the dominant lever for CFI.** BREC's subgraph-radius
   analysis (Table 5): a radius-6 view can in principle distinguish 385/400,
   radius 8 → 398, radius 10 → 400; radius 6 already beats 3-WL, and the
   best distance-encoding models use radius 8 (Appendix J). The baseline
   `models_brec.yml` caps `distances` at 8 while CFI diameters reach 17 — so
   **all variants widen the WL-head distance range to 1–17**. This is free
   outside CFI: distance property files are generated uncapped (`cutoff: None`
   in `preprocessing.py`), rules are only instantiated for distances that
   actually occur, and every non-CFI category has diameter ≤ 5.
2. **Clique counts (GSN evidence).** Chordless-cycle labels are blind to
   cliques by definition, yet GSN's S₄ substructure counts (cliques included)
   reach 99/140 on Regular graphs and S₃→S₄ jumps 105→243 total (Table 7).
   Variant 0/2 add a `cliques` label (maximal-clique count vector per node) as
   encoder, convolution and aggregation heads — aimed squarely at the strongly
   regular / 4-Vertex_Condition graphs where per-node closed-walk counts are
   provably uninformative (walk-regular graphs with identical SRG parameters).
3. **Edge-anchored refinement (GSN-e evidence).** GSN-e (edge-level counts)
   beats GSN-v node-level counts (Table 7). ShareGNN's convolution is already
   keyed on (head-label, tail-label, distance) pairs — i.e. edge/pair-level —
   and variant 0 sharpens this with an `induced_cycles × wl(depth 1)` product
   label, a strictly finer partition than either factor.
4. **Cycle counting up to length 6 (I²-GNN evidence).** I²-GNN, whose defining
   feature is provable counting of cycles up to length 6, is the strongest GNN
   on BREC (281/400, 100/140 Regular). The cycle vocabulary (lengths 3–6, one
   count-vector label) is kept as the workhorse; walks 7/8 are added in the
   encoder for extra spectral resolution at negligible preprocessing cost
   (matrix powers).
5. **Depth as iteration count (3-WL evidence).** 3-WL needs ~5 iterations for
   its full 270/400 (Table 6), and the paper sets 5–6 layers for k-WL-hierarchy
   models accordingly (Appendix K). A single invariant convolution realizes only
   1-step rules; variant 2 stacks a second block (composition creates 2-step
   invariants) at ~2× per-pair runtime.
6. **Protocol constants and seed handling.** `NUM_RELABEL=32`, `OUTPUT_DIM=16`,
   `THRESHOLD=5.0`, `SEED=2023` from the reference implementation; the paper
   additionally repeats runs over 10 seeds and reports the best reliable run
   (Appendix O). `main_brec.py` seeds per pair (`SEED + pair_id`); for a
   deterministic invariant model, seed variance only matters through the
   near-zero-variance pathway that the epsilon ridge already handles, so no
   multi-seed sweep is encoded in the configs.

## The three variants and their trade-offs

| | Variant 0 (default) | Variant 1 | Variant 2 |
|---|---|---|---|
| New labels to generate | cliques, walks 7/8, cycles×wl product | **none** | cliques, walks 7/8 |
| WL-head distances | 1–17 | 1–17 | 1–17 |
| Conv blocks | 1 | 1 | 2 |
| Cost | preprocessing ↑ | none over baseline | runtime ~2× |
| Targets | Regular/4VC/DR + CFI | CFI only (ablation) | CFI depth |

- **Variant 1** is the zero-risk ablation: byte-identical to `models_brec.yml`
  except for the widened WL distance ranges, so it runs immediately on an
  existing `data/BREC/` cache and isolates the receptive-field effect.
- **Variant 0** is the recommended scoring config. Its main risk is
  preprocessing cost of `cliques` (uncapped `nx.find_cliques`) on the 20
  4-Vertex_Condition pairs (63 nodes, average degree 30). The label is
  deliberately *uncapped* because the generator counts **maximal** cliques and
  *skips* (rather than truncates) cliques above `max_clique_size`, which would
  silently discard exactly the dense-graph signal we want. If clique
  enumeration proves too slow there, fall back to variant 1 or shard
  preprocessing by category.
- **Variant 2** trades runtime for compositional depth; expected to matter
  mainly on the 3-WL-distinguishable CFI subset (60/100 pairs is the 3-WL
  ceiling there; 20 CFI pairs are 4-WL-hard and out of reach for any config
  here).

Expected upper bounds worth keeping in mind when reading results: every
category except CFI is fully coverable in principle with radius ≤ 6
(N₃ distinguishes everything but CFI, Table 5); the 4-WL-hard CFI pairs (20)
and possibly some 3-WL-hard ones will stay undistinguished — the strongest
published GNN totals are I²-GNN 281 and (post-paper) N²-GNN ≈ 284/400.

## Sources

- Wang & Zhang, *An Empirical Study of Realized GNN Expressiveness* (BREC),
  ICML 2024 — https://arxiv.org/abs/2304.07702 (Tables 3, 5–7, 10; Appendices
  J, K, L, O). Primary source for radius/iteration analysis, GSN-e/S₄
  results, searched hyperparameter ranges, seed protocol.
- GraphPKU/BREC reference implementation —
  https://github.com/GraphPKU/BREC (`DropGNN/test_BREC.py`: the protocol
  constants hard-coded in `main_brec.py`).
- Huang, Peng, Ma & Zhang, *Boosting the Cycle Counting Power of Graph Neural
  Networks with I²-GNNs*, ICLR 2023 — https://arxiv.org/abs/2210.13978
  (cycle counting to length 6 ⇒ best-on-BREC; motivates the cycle vocabulary).
- Bouritsas et al., *Improving Graph Neural Network Expressivity via Subgraph
  Isomorphism Counting* (GSN), TPAMI 2022 — https://arxiv.org/abs/2006.09252
  (substructure/clique counts; GSN-e edge-anchored variant).
- Li et al., *Distance Encoding: Design Provably More Powerful Neural Networks
  for Graph Representation Learning*, NeurIPS 2020 —
  https://arxiv.org/abs/2009.00142 (distance features as an expressiveness
  mechanism; BREC's appendix repeatedly credits shortest-path distance).
- Repo-internal: `specs/21-brec-expressiveness-benchmark.md`,
  `experiments/brec/README.md` (protocol, degenerate-covariance analysis,
  preprocessing cost measurements).
