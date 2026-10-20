# Optimal ShareGNN configurations for LRGB (Peptides-func / Peptides-struct)

Literature-informed redesign of the baseline LRGB configs in this folder.
**No experiments have been run with these configs yet** — they encode the best
published LRGB training recipes mapped onto ShareGNN's invariant-based
architecture, plus the strongest tuned idioms from this repo's ZINC work.

## Files

| File | Role |
|---|---|
| `models_peptides_func_optimal.yml` | Network config, Peptides-func (10 multi-label outputs) |
| `models_peptides_struct_optimal.yml` | Network config, Peptides-struct (11 regression outputs); identical except output width |
| `parameters_peptides_func_optimal.yml` | Hyperparameters, func (BCEWithLogitsLoss) |
| `parameters_peptides_struct_optimal.yml` | Hyperparameters, struct (MAE, z-scored targets) |
| `configs/main_config_peptides_func_optimal.yml` | Main config wiring (results go to `results/lrgb_optimal/`) |
| `configs/main_config_peptides_struct_optimal.yml` | Main config wiring |

The existing baseline files (`models_peptides_*.yml`, `parameters_peptides_*.yml`,
`configs/main_config_peptides_*.yml`) are untouched. The func task still uses
the documented `graph_regression` + `BCEWithLogitsLoss` multi-label workaround
(see `configs/main_config_peptides_func.yml` and spec 20): reported
"MAE"/"accuracy" are surrogates, **not** the benchmark's Average Precision.

## Primary literature sources

1. **Tönshoff, Ritzert, Rosenbluth, Grohe — "Where Did the Gap Go? Reassessing
   the Long-Range Graph Benchmark"** (TMLR 2024 / LoG 2023),
   <https://arxiv.org/abs/2309.00367>. The key reference: basic hyperparameter
   tuning lifts a plain GCN to **0.6860 ± 0.0050 AP** on Peptides-func and
   **0.2460 ± 0.0007 MAE** on Peptides-struct (6 layers, hidden 235, ~490k
   params), erasing most of the claimed transformer gap. Their tuned recipe:
   AdamW, lr 1e-3, cosine schedule with linear warmup, dropout 0.1, batch 200,
   250 epochs, **3-layer MLP prediction head** (the single biggest factor,
   especially on struct), RWSE (func) / LapPE (struct) positional encodings.
2. **Dwivedi et al. — "Long Range Graph Benchmark"** (NeurIPS 2022 D&B),
   <https://arxiv.org/abs/2206.08164>. Defines the datasets, official splits,
   the ~500k parameter budget convention, and the metrics (AP / MAE).
   Peptides graphs: ~15.5k graphs, ~150 nodes average, mean diameter ~57.
3. **Rampášek et al. — "Recipe for a General, Powerful, Scalable Graph
   Transformer" (GraphGPS)** (NeurIPS 2022), <https://arxiv.org/abs/2205.12454>.
   Peptides settings: AdamW (weight decay 0), cosine-with-warmup (~5 warmup
   epochs), batch 128, LapPE/RWSE encoders. The framework's `CosineWarmup`
   scheduler is an exact port of this schedule
   (`src/simplegnn/framework/model_configuration.py`).
4. **Ma et al. — GRIT** (ICML 2023), <https://arxiv.org/abs/2305.17589>.
   Relative random-walk probabilities (RRWP) as pairwise encodings — evidence
   that *pairwise walk/distance-indexed* interactions (ShareGNN's native
   mechanism) are what strong LRGB models exploit.
5. **This repo's tuned ZINC configs**
   (`experiments/base_paper/regression/ZINC/configs/network_ZINC_test_v12.yml`,
   `parameters_ZINC_test_v12.yml`, specs/15): closed-walk positional-encoding
   heads, pre-norm residual convolution, `mean_aggregation` init,
   `rule_occurrence_threshold` and proximal L1 as the decisive ShareGNN
   regularizers.

## Architecture rationale (network configs)

### 1. Encoder: closed-walk invariant positional encoding (RWSE analogue)

Tönshoff et al. show RWSE/LapPE are worth several AP points on peptides.
ShareGNN's native equivalent is the `invariant_based_positional_encoding`
layer over `closed_walks` labels — `(A^k)_ii` counts are exactly the
(unnormalized) random-walk-return structure RWSE encodes. Heads use walk
lengths k = 2..8 combined with the primary atom label (ZINC v11/v12 idiom;
k = 1 omitted, always 0 on simple graphs), plus primary / WL-0 / WL-edge
context heads.

Deviation from ZINC v12: `concatenate_input: true` (v12 used `false`).
Peptides' 9-column OGB atom features carry 8 attribute columns beyond the
primary label; discarding them (as v12 does for ZINC's bare atom types) would
lose chemistry information. Concatenation also mirrors how GCN+RWSE appends
the encoding to node features. LapPE (best for struct in Tönshoff et al.) has
no ShareGNN analogue, so both tasks use the walk encoder; their ablations show
both PEs give large, similar gains.

### 2. One pre-norm residual convolution with distance-stratified heads

Transformer-style `residual: true` + `pre_layer_norm: true` on the
`invariant_based_convolution`, as introduced in ZINC v2 and kept through v12.

The long-range mechanism: instead of stacking 6-10 MPNN layers (Tönshoff's
GCN/GINE) or global attention (GPS/GRIT), one ShareGNN convolution
message-passes over **shortest-path distance properties** directly:

| Head | Labels | Distances | Why |
|---|---|---|---|
| 1 | primary × primary | 1..20 | long-range atom-type interaction rules |
| 2 | WL-0 × WL-0 | 1..20 | long-range, degree-refined |
| 3, 4 | induced 5-/6-cycles × primary | 1..15 | aromatic side-chain rings interacting with the backbone |
| 5 | WL-edges-0 pairs | 1..4 | fine local structure |
| 6 | WL-edges-1 pairs | 1..2 | finest labels, shortest range (rule count grows with label-alphabet²) |
| 7 | primary pairs, `edge_label_distances` | 1 | bond-type-aware 1-hop head |

**Cutoff trade-off (the key knob):** the baseline capped `distances` at 15;
these configs raise it to **20**. Peptides' mean diameter is ~57, so full-range
all-pairs distances would be the "true" long-range setting, but storage/compute
scale with (#pairs within cutoff) × 15.5k graphs — spec 20 documents why full
range is intractable here. Tönshoff et al.'s result that a 6-layer GCN
(receptive field 6) reaches 0.686 AP strongly suggests most of the signal is
at short-to-mid range; cutoff 20 already more than triples the effective range
of any published MPNN baseline while staying tractable. **If preprocessing
time/RAM allows, raising the cutoff on heads 1-2 (e.g. to 30) is the first
thing to try**; conversely drop to 15 to reuse the baseline's already-generated
property files.

Coarse labels get long ranges, fine labels short ranges, because the rule
count per head ≈ (#label pairs) × (#distance values) — this is the
ShareGNN version of staying inside LRGB's ~500k parameter budget. The exact
parameter count is data-dependent (rules are instantiated from label pairs
realized in the data and pruned by `rule_occurrence_threshold`), so verify
against the budget on the first preprocessing run.

### 3. Readout and head

`invariant_based_aggregation` (label-stratified global readout, `num: 10` per
head — QM9/baseline idiom) followed by a **3-layer MLP head
(96 → 96 → out) with dropout 0.1 and layer norm**. Tönshoff et al. attribute
"most of the additional performance" to replacing the linear head with an MLP,
particularly on Peptides-struct — the baseline configs here had a 2-layer
head (64) and no dropout anywhere. Hidden width 96 keeps the dense part small
so the parameter budget goes to invariant rules.

## Hyperparameter rationale (parameters configs)

| Setting | Value | Source |
|---|---|---|
| optimizer | AdamW, weight decay 0 | Tönshoff et al. / GraphGPS peptides (decay 0); ShareGNN regularization handled by rule threshold + L1 instead |
| learning_rate | 1e-3 | unanimous across tuned GCN/GINE (Tönshoff) and GRIT on peptides |
| scheduler | `CosineWarmup`, 10 warmup epochs | Tönshoff et al. found cosine+warmup clearly better than ReduceLROnPlateau on LRGB; framework port of GraphGPS/GRIT's `cosine_with_warmup` (GPS uses 5 warmup epochs at 200 epochs; 10 at 250 is proportional) |
| epochs | 250, no early stopping | Tönshoff et al.'s budget; cosine schedule must run to completion; `best_model: True` still selects the best validation epoch |
| batch_size | 128 | between GPS (128) and Tönshoff (200); memory-conscious for ~150-node graphs; drop to 64 on OOM |
| precision | float | ZINC v12 (double is 2x memory for no accuracy need here; baseline used double) |
| `share_gnn_forward: {batched: True}` | on | ZINC v12; required for reasonable throughput at batch 128 |
| rule_occurrence_threshold | 5 | ZINC v12 found 10 best on 12k small graphs; peptides' larger graphs make every rule more frequent, so 5 is the conservative transfer. Sweep {1, 5, 10, 20} manually (not grid-searchable) |
| l1_regularization | conv/aggr 0.01, encoding 0.001 | ZINC v12 values for conv/aggr (learned rule selection — prunes uninformative long-distance rules, complementing the raised cutoff); encoding 10x gentler per v12's warning about pruning input embeddings prematurely |
| weight_initialization | `mean_aggregation` (gain 1.0) + PE normal(0, 0.1) | ZINC v12 |
| dropout | 0.1 (network config) | Tönshoff et al. (both tasks) |
| input_features | `one_hot_labels_normalize_features` | keep full OGB atom features (baseline choice); PE concatenates on top |
| loss (func) | BCEWithLogitsLoss | multi-label workaround, unchanged from baseline (spec 20) |
| loss (struct) | MAE + standard-normalized/inverted targets | benchmark metric; QM9/baseline idiom for comparable units |

## Expected trade-offs / open risks

- **Preprocessing cost**: cutoff 20 distance properties + closed-walk labels
  (k ≤ 8) + WL-edges depth-1 labels over 15.5k × ~150-node graphs is a
  substantial one-off preprocessing job; run `preprocessing(num_threads=1)`
  first and budget disk for `data/LRGB/properties/`.
- **Parameter budget**: invariant-rule counts are data-dependent; if the first
  run reports ≫500k parameters, raise `rule_occurrence_threshold` (5 → 10/20)
  before shrinking the architecture — v12's evidence says pruning beats
  shrinking.
- **Func metric**: results remain BCE-loss surrogates until a real multi-label
  AP evaluation path exists (spec 20's "future work"); struct MAE is directly
  comparable to the published 0.246-0.249 numbers.
- **Memory**: batch 128 × ~150 nodes × long-range rule application is the
  main OOM risk (cf. the repo memory note on parallel-fold OOMs) — first
  fallbacks: batch 64, `num_threads` capped, or cutoff 15.
