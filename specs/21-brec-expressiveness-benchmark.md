# 21 — BREC expressiveness benchmark (RPC evaluation)

## Context

BREC (Wang & Zhang, *An Empirical Study of Realized GNN Expressiveness*,
ICML 2024 — [arXiv:2304.07702](https://arxiv.org/pdf/2304.07702), earlier titled
*Towards Better Evaluation of GNN Expressiveness with BREC Dataset*;
MIT-licensed data at
[GraphPKU/BREC](https://github.com/GraphPKU/BREC)) is the current standard
benchmark for *realized* GNN expressiveness. It supersedes EXP / CSL / SR25:
400 non-isomorphic graph pairs in six difficulty categories, spanning 1-WL up
to 4-WL-hard instances.

| Category | pair ids | pairs | nodes/graph (verified) |
|---|---|---|---|
| Basic | 0–59 | 60 | 10 |
| Regular | 60–159 | 100 | 7–35 |
| Extension | 160–259 | 100 | 10–19 |
| CFI | 260–359 | 100 | 18–198 |
| 4-Vertex_Condition | 360–379 | 20 | 63 |
| Distance_Regular | 380–399 | 20 | 30–63 |

This is the expressiveness half of the gap identified when surveying synthetic
GNN benchmarks; the counting half (Chen et al. 2020 substructure counting) is
already in the repo as `source: SubstructureBenchmark`
(`SubstructureBenchmarkPreprocessing`, `data/SubstructureBenchmark/`).

## Critical finding: BREC cannot use `FrameworkMain.run_configurations()`

BREC is not a train/val/test task, so no `task:` in the framework's schema
fits it. Its official metric is **RPC (Reliable Paired Comparisons)**: for each
pair independently, a *siamese* model is trained to push the two graphs' output
embeddings apart, and the pair counts as distinguished only if a **Hotelling
T² statistic** over 32 relabelings exceeds a threshold *and* a **reliability
control** on a known-isomorphic pair stays below it. Consequences:

1. **400 independent models**, each trained from scratch for 50 epochs on 64
   graphs — not one model over a split.
2. **No labels.** The loss is `CosineEmbeddingLoss(target=-1)` between the two
   halves of the batch; there is no `y` to regress or classify.
3. **The decision rule is a statistical test**, not a metric averaged over a
   test set, and the reliability control means a pair can be *rejected* for
   being too easy to separate spuriously.

`ModelConfiguration.train_configuration()` assumes all three of the opposites.
**Decision:** BREC gets a dedicated runner, `experiments/brec/main_brec.py`,
that reuses the framework's *data* pipeline (`Preprocessing` →
`preprocess_graph_data` → `load_preprocessed_data_and_parameters` → invariant
labels/properties on disk) and `GraphModel`, but implements the RPC loop
itself. `FrameworkMain.preprocessing()` is still what generates the dataset,
labels and properties, so ShareGNN configs work unchanged.

## Protocol (transcribed from the reference implementation)

Constants from `GraphPKU/BREC@Release:DropGNN/test_BREC.py` (identical across
the repo's per-model copies), reproduced in `experiments/brec/main_brec.py`:

```
NUM_RELABEL = 32     SAMPLE_NUM = 400     EPOCH = 50        BATCH_SIZE = 16
OUTPUT_DIM = 16      LEARNING_RATE = 1e-3 WEIGHT_DECAY = 1e-5
MARGIN = 0.0         THRESHOLD = 5.0      EPSILON_CMP = 1e-6   SEED = 2023
```

Per pair id `i ∈ [0, 400)`:

1. `traintest` set = the 64 graphs at dataset positions `[64i, 64i+64)` —
   32 relabelings of `G_A` and `G_B`, interleaved `A₀ B₀ A₁ B₁ …`.
2. `reliability` set = the 64 graphs at `[64(i+400), 64(i+400)+64)` — the
   same construction for a pair that *is* isomorphic (verified: `nx.is_isomorphic`
   is True for id 400's two graphs, False for id 0's).
3. Train a fresh model for 50 epochs, Adam(1e-3, wd 1e-5) +
   `ReduceLROnPlateau`, loss `CosineEmbeddingLoss(margin=0)` with target `-1`
   on `(pred[0::2], pred[1::2])`.
4. `T²(S) = D̄ᵀ · pinv(cov(D)) · D̄` where `D = X − Y`, `X`/`Y` are the
   `16 × 32` matrices of the A- and B-embeddings.
5. **distinguished** iff `T²(traintest) > 5.0` *and*
   `not isclose(T²(traintest), T²(reliability), atol=1e-6)`;
   **reliable** iff `T²(reliability) < 5.0`.

Reported: per-category and total `distinguished / pairs`, plus
`fail_in_reliability`.

Note on the threshold: the paper's Eq. 8 defines
`T² = q · d̄ᵀ S⁻¹ d̄` and derives the threshold from an F-distribution
quantile, while the released code drops the factor `q` and hard-codes
`THRESHOLD = 5.0`. Every published BREC score was produced by the code, so the
code is what is reproduced here.

Reference scores from Table 2 of the paper (pairs distinguished / 400, with the
paper's category grouping — its "Regular (140)" column merges the code's
Regular + 4-Vertex_Condition + Distance_Regular):

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

Plain 1-WL/GIN is not in the table because **all 400 pairs are
1-WL-indistinguishable by construction** — a message-passing baseline's true
score is 0. That makes `models_brec_gin.yml` a *negative* control: pairs it
"wins" are false positives of the T² test. A few marginal ones (T² just above
5.0) are inherent to RPC's fixed significance level over 400 tests — the paper
introduces RAPC specifically to bound that rate — but many, or any with a large
T², would mean the harness is broken.

## Work

1. **Data** — `BRECGraphDataPreprocessing` (`graph_dataset_preprocessing.py`).
   Downloads `BREC_data_all.zip` (7.3 MB) from the `Release` branch into
   `data/BREC/raw/`, reads `brec_v3.npy` (51,200 graph6 byte strings =
   800 ids × 32 relabelings × 2 graphs), and converts to the framework's
   collated format. BREC graphs are **unlabeled**: `x` = ones `[N, 1]`,
   `primary_node_labels` = zeros, no node/edge attributes — same shape of
   contract as `SubstructureBenchmarkPreprocessing`. `y` is a dummy
   zeros column (the framework requires one; the RPC runner never reads it).
   Graph order is preserved exactly, because the runner addresses pairs by
   arithmetic on the index.
2. **Relabeling subsets** — the dataset name carries the count:
   `BREC` = 32 relabelings (official), `BREC-r<k>` = the first *k* of each
   graph's 32 relabelings. Subsetting happens at dataset-construction time so
   that a cheap run does not pay for preprocessing all 51,200 graphs (≈1.78 M
   nodes). `BREC-r4` (6,400 graphs) is the smoke-test variant.
3. **Dispatch** — `elif self.from_existing_data == 'BREC':` in
   `graph_dataset.py`; `'BREC'` added to the `source` allow-list in
   `configuration_checks.py`.
4. **Splits** — `simplegnn/utils/brec_splits.py`. The framework requires a
   split file, the RPC runner ignores it: emit one fold that is a *valid*
   disjoint partition (pair id 0 → validation, pair id 1 → test, the rest →
   train) and document it as unused.
5. **`experiments/brec/`** — `main_brec.py` (RPC runner, `click` CLI with
   `--parts`, `--pairs`, `--epochs`, `--epsilon-matrix`, `--num-threads`),
   three configs (`main_config_brec.yml` for ShareGNN, `main_config_brec_gin.yml`
   for the control, `main_config_brec_smoke.yml` for `BREC-r4`),
   `models_brec.yml` (ShareGNN, adapted from
   the unlabeled-graph backbone in
   `experiments/base_paper/transfer/Substructure_to_TU/configs/network_transfer_Substructure_pretrain.yml`
   — no `primary`/`wl_labeled` heads, see [[sharegnn-transfer-invariant-labels]]
   — with the final linear widened to `OUTPUT_DIM = 16`), `models_brec_gin.yml`
   (1-WL GIN negative control, expected 0/400), `parameters_brec.yml`
   (`input_features: {name: constant, in_dimensions: 1, value: 1.0}` —
   constant features are mandatory: random features would fake expressiveness
   *and* break the reliability control), `run_brec.sh`, `README.md`.
6. **Tests** — `tests/test_brec_unit.py`: T² statistic against a hand-computed
   case, the pair-index layout, `build_brec_splits` disjointness, and the
   decision rule's truth table.

## Discovered during implementation: the degenerate-covariance false negative

`t2_statistic` uses `pinv`, so a *perfectly consistent* separation collapses to
zero: if all 32 relabelings give the same difference vector, `cov(D)` is the
zero matrix, `pinv(0) = 0`, and `T² = 0` no matter how far apart the two graphs'
embeddings are. The reference implementation relies on floating-point noise
across permutations to avoid this (its docstring says as much) and ships the fix
— a ridge `S + 1e-7·I` — commented out.

That assumption is safe for the float32 models on large graphs it was written
for, but not for a deterministic double-precision permutation-equivariant model
on a 10-node graph, where the embeddings come out bit-identical. **Confirmed
empirically**: the ShareGNN config flags every pair it was run on as degenerate,
with the near-singular (not exactly singular) covariance producing T² ≈ 10³¹
instead of 0 — garbage either way.

So the runner (a) keeps the reference behavior by default, (b) detects the case
per pair (`is_degenerate`: zero covariance with non-zero mean difference),
counts it, and prints a warning naming the fix, and (c) exposes
`--epsilon-matrix 1e-7` to apply the reference's ridge. `tests/test_brec_unit.py`
pins all three.

With the ridge and `S ≪ εI`, `T² ≈ ‖d̄‖²/ε`, so the decision reduces to
"the mean embedding difference exceeds `√(5ε)`" — the right question for a
deterministic invariant model. Measured on Basic pairs 0–1: T² ≈ 1.6·10⁸ for the
pair against **exactly 0.000** for the isomorphic control. **ShareGNN runs
should therefore use `--epsilon-matrix 1e-7`**, disclosed as a deviation from
the reference defaults that the reference itself provides for this case.

A related guard: `cov(D)` is `output_dim × output_dim` estimated from
`num_relabel` samples, so `num_relabel ≤ output_dim` is singular by
construction. The protocol's 32-vs-16 is fine, `BREC-r4` is not, and the runner
warns whenever it holds (`BREC-r20` is the smallest valid reduced variant).

## Cost note

`_adaptive_parallel_map` (`datasets/utils/node_labeling.py`) estimates label
cost from the *first five* graphs before deciding to parallelize. BREC's first
five are 10-node Basic graphs, so the projection is ~0 and label generation runs
serially over the whole dataset — including the 63-node/degree-30
4-Vertex_Condition graphs that dominate it. Measured: ~15 min for `BREC-r4`,
extrapolating to ~2 h for the full `BREC`, once (cached under `data/BREC/`, and
shared by the ShareGNN and GIN configs). Left as-is: fixing the heuristic means
changing a shared code path for every dataset, which is out of scope here.

## Verification (results)

1. **Splits** — `brec_splits` writes valid single-fold splits for `BREC` and
   `BREC-r4`; `tests/test_brec_unit.py` checks disjointness, full coverage, and
   that the shipped JSON matches the generator. ✅
2. **Dataset, exactly** — both variants build (6,400 / 51,200 graphs). Rather
   than isomorphism-testing the pairs (VF2 on strongly regular graphs is
   pathologically slow — a first attempt hung), verification compares each
   dataset graph's *edge set* against `nx.from_graph6_bytes` of the
   corresponding raw string: 104 graphs sampled across all six categories,
   0 mismatches. `x` is a constant ones column, `y` a dummy zero,
   `primary_node_labels` all zero. 1-WL hashes are equal within every sampled
   pair, which independently confirms the benchmark's premise. ✅
3. **GIN negative control** — full protocol, 400 pairs × 50 epochs: **0/400
   distinguished, 0 reliability failures, 0 degenerate**, ~3 s/pair. This is the
   harness's false-positive check: 1-WL provably cannot separate any BREC pair,
   and the T² test agrees. ✅
4. **ShareGNN path** — runs end to end on `BREC-r4` (both with and without the
   ridge), exercising the batched invariant forward, per-pair model construction
   and the cached invariant index structures. It surfaced the degenerate-
   covariance issue documented above. A full ShareGNN score is *not* produced
   here: preprocessing the 51,200-graph dataset takes ~2 h and 400 pairs ×
   50 epochs × 64 graphs on CPU is a multi-day job. `experiments/brec/README.md`
   documents the command and the sharding-by-category workaround.
5. `pytest tests -q` green (24 new tests in `tests/test_brec_unit.py`).
