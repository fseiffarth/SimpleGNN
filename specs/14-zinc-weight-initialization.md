# 14 — Better Weight Initialization for ZINC (ShareGNN)

**Status**: Analysis + one empirical result. The zero-mean experiment (a)+(b)
below was tried and **regressed badly on ZINC** — see "Empirical Update" before
acting on any recommendation here. Configs have been reverted to the original
`lower_upper` baseline.
**Scope**: `examples/zinc/parameters.yml`, `src/simplegnn/models/ShareGNN/layers/inv_based_message_passing.py`, `src/simplegnn/models/ShareGNN/layers/inv_based_pooling.py`

---

## Empirical Update (supersedes the recommendations below)

We applied suggestions (a) + (b) — zeroed biases and zero-mean
`normal, std: 0.1` for convolution/aggregation — to both `examples/zinc/parameters.yml`
and `experiments/base_paper/regression/ZINC/configs/parameters_ZINC_test.yml`.
Result: **much worse** than the `lower_upper` baseline. Both files were reverted.

### Why zero-mean was the wrong direction

1. **The ZINC target is not negative-mean.** Measured on the PyG ZINC subset
   train split (`subset=True`, 10k graphs):

   ```
   mean = 0.0153   std = 2.011   min = -42.04   max = 3.80   frac_negative = 0.41
   ```

   Target mean ≈ 0 and 59% of values are ≥ 0. The distribution is left-skewed
   (long negative tail) but centered at zero. So an output offset is not the
   issue — and the network ends in `linear (out_features: 1, bias: true,
   Identity)` whose trainable bias absorbs any constant offset in a few epochs
   anyway.

2. **`lower_upper`'s `randn` "bug" is actually load-bearing.** Because
   `inv_based_message_passing.py:1078-1081` uses `randn` (not `rand`), the old
   init produced weights with a **nonzero mean**:

   ```
   mean = -1/√n ,  std = 2/√n   ⟹   mean/std = -0.5   (independent of n)
   ```

   A nonzero-mean weight applied across a message-passing row makes the
   invariant conv behave like **coherent (negative) mean-aggregation**:

   ```
   Σⱼ wᵢⱼ xⱼ  ≈  mean_w · Σⱼ xⱼ  +  noise
   ```

   i.e. the network starts life close to a sensible mean-pooling GNN. The
   **sign is irrelevant** (the readout can flip it); the **nonzero mean** is
   what matters.

3. **Zero-mean `normal` destroyed that coherent component.** With `mean: 0.0`
   the conv starts as pure zero-mean random projections — no aggregation signal.
   Zeroing the conv/aggregation biases removed the same nonzero-mean effect
   there too, and `std: 0.1` was a blind guess likely mis-scaled vs the old
   `2/√n`. Together these explain the regression.

### Revised takeaway

The improvement axis is **not** "zero-mean fan-in init". Any replacement must
**preserve a nonzero weight mean** (coherent aggregation) at roughly the old
`mean/std = -0.5` ratio, **at the correct (tiny) scale**.

### Measured effective scale (ZINC-test network)

Instantiating the ZINC-test network and reading `Param_W.numel()` per group:

| group       | n (num_weights) | effective `std = 2/√n` | effective `mean = −1/√n` |
|-------------|-----------------|------------------------|--------------------------|
| convolution | 397,593         | **0.00317**            | −0.00159                 |
| aggregation | 142,710         | **0.00529**            | −0.00265                 |

Key consequence: the baseline trains at a **very small** std (~0.003–0.005).
The `std: 0.1` scalar tried in the failed experiment was **~20–30× too large** —
that alone explains the blow-up, independent of the zero-mean issue. Any hand-
picked scalar std is dangerous here; scale must track `2/√n`.

### Implemented: `mean_aggregation` init type

A new init type was added to **both**
`inv_based_message_passing.py` and `inv_based_pooling.py` (`init_weights`):

```python
elif weight_initialization.get('type', None) == 'mean_aggregation':
    gain = weight_initialization.get('gain', 1.0)
    mean_ratio = weight_initialization.get('mean_ratio', -0.5)
    std = gain * 2.0 / np.sqrt(num_weights)
    torch.nn.init.normal_(weights, mean=mean_ratio * std, std=std)
```

- `gain=1.0, mean_ratio=-0.5` **reproduces the old `lower_upper` init exactly** —
  verified: conv `mean=-0.00159 std=0.00317`, aggr `mean=-0.00267 std=0.00529`,
  matching the table above. So switching the config is a zero-risk rename with
  knobs, not a behavior change.
- `gain` sweeps scale (keeps fan-in tracking, `gain=2` → 2× std); `mean_ratio`
  sweeps the aggregation prior (`0.0` = zero-mean, which we showed hurts).
- Bug-free (uses in-place `normal_`, no `randn`/`rand` confusion, respects the
  pooling bias shape).

Both `examples/zinc/parameters.yml` and
`experiments/base_paper/regression/ZINC/configs/parameters_ZINC_test.yml` now
use `mean_aggregation` with the baseline-reproducing defaults.

### Suggested sweeps from here (all now safe — scale stays fan-in-aware)

1. `gain ∈ {0.5, 1.0, 2.0, 4.0}` — is the tiny default scale actually optimal?
2. `mean_ratio ∈ {-1.0, -0.5, -0.25}` — how strong should the aggregation prior
   be? (Avoid `0.0`; that's the regression case.)
3. Only after those: revisit true row-fan-in std (option c) if `2/√n` proves a
   poor proxy for the real per-node fan-in.

The remaining sections are the original pre-experiment analysis; read them
through the lens of this update.

---

## Current State

The ZINC experiment (`examples/zinc/parameters.yml:35-38`) initializes all four
parameter groups with `lower_upper`:

```yaml
weight_initialization: { convolution:      { type: 'lower_upper', value: -0.001 },
                         convolution_bias: { type: 'lower_upper', value: 0.0 },
                         aggregation:      { type: 'lower_upper', value: -0.001 },
                         aggregation_bias: { type: 'lower_upper', value: 0.0 } }
```

`lower_upper` scales by `n = num_weights` — the **total number of distinct
shared parameters** for the layer (label-pair × property combinations summed
over heads).

## Issues Identified

### 1. `lower_upper` is not what its name claims (bug)

`inv_based_message_passing.py:1078-1081`:

```python
lower, upper = -(1.0 / np.sqrt(num_weights)), (1.0 / np.sqrt(num_weights))
weights = nn.Parameter(lower + torch.randn(num_weights, dtype=self.precision) * (upper - lower))
```

This uses `randn` (standard normal), not `rand` (uniform). The result is a
**Gaussian with mean = −1/√n and std = 2/√n**, not a uniform distribution over
[−1/√n, 1/√n]. Consequences:

- ZINC weights start with a systematic **negative mean**.
- With `num_weights` in the thousands (23 distance values × label-pair combos ×
  heads) the mean shift is small per weight, but the distribution is not the
  intended one, and the std is 2× the intended half-range.

The same `lower_upper` branch exists in `inv_based_pooling.py` (`init_weights`,
around line 132) and should be checked for the same `randn`/`rand` confusion.

### 2. Wrong "fan-in" everywhere

Both `lower_upper` and `he` scale by the *number of distinct shared
parameters*, not by the operation's fan-in:

```python
elif weight_initialization.get('type', None) == 'he':
    std = np.sqrt(2.0 / num_weights)          # <-- num_weights, not fan-in
```

In the forward pass, each output node sums over the nonzero entries of its row
in `current_W` (shape `(H, N, N)`). With `properties: distances: [1..23]` in
`examples/zinc/models_ShareGNN.yml`, that row covers essentially **all nodes in
the graph** (~23 for ZINC molecules). Variance-preserving init should scale by
that **row fan-in**, not by the parameter-vocabulary size. As the
label/property vocabulary grows, the current scheme makes weights *smaller*
even though the per-node sum stays the same size → vanishing activations in
early epochs.

Note: this means switching the YAML to `type: 'he'` would **not** give real He
initialization either.

### 3. Random bias init

`convolution_bias` / `aggregation_bias` also use `lower_upper`. Standard
practice is zeros — random biases add label-dependent noise the optimizer has
to unlearn.

### 4. Linear layers not covered by the config

The first `linear` layer and any final readout `linear` use PyTorch's default
Kaiming-uniform; `weight_initialization` only covers convolution/aggregation.
For MAE regression, near-zero init of the *last* linear layer often stabilizes
early training noticeably — worth exposing in the config.

---

## Suggestions (in rough order of expected payoff)

### a) Zero the biases — free, no code change

```yaml
convolution_bias: { type: 'constant', value: 0.0 }
aggregation_bias: { type: 'constant', value: 0.0 }
```

### b) Zero-mean normal with tunable std — immediate baseline, no code change

Avoids both the mean-shift bug and the vocabulary-size scaling; the existing
grid-search machinery (lists in YAML) sweeps the std:

```yaml
convolution: { type: 'normal', mean: 0.0, std: [0.3, 0.1, 0.03] }
aggregation: { type: 'normal', mean: 0.0, std: [0.3, 0.1, 0.03] }
```

For ZINC's fan-in of ~20–25 contributors per node, `1/√23 ≈ 0.2` is the
theoretically motivated starting point.

### c) True effective-fan-in init (new `init_type`)

The layer already precomputes the index structures (`_assemble_rows`), so the
**mean row nnz** D̄ per head is cheaply available at init time. A new
`glorot_row` / `fan_in_row` type with `std = gain / √D̄` would be the
principled analog of Kaiming init for this weight-sharing scheme. It also
adapts automatically when the `distances` list changes (e.g. `[1..3]` vs
`[1..23]` gives very different fan-ins).

### d) Distance-decayed init

Physically, distance-1 neighbors matter most for molecular properties.
Initializing weights for property value *d* with std ∝ `1/d` (or `γ^d`) biases
the model toward local structure at start while keeping long-range terms
learnable. Likely helps regression targets like constrained solubility that
are dominated by local motifs plus a few global corrections.

### e) Data-driven rescaling (LSUV-style)

Given how unusual the weight sharing is, the most robust option: after init,
push one batch through, then rescale `Param_W` per head so each head's output
std ≈ 1. ~10 lines, no theory needed, immune to future architecture changes.

### f) Occurrence-aware scaling (speculative)

`rule_occurrence_threshold: 2` means some (label-pair, distance) parameters
appear in thousands of graphs and others in two. Scaling each parameter's init
std by `1/√count_ij` equalizes each parameter's expected contribution to the
initial output variance across the dataset.

---

## Recommended First Experiment

> **Note:** the original recommendation here — (a) + (b) zero-mean normal — was
> tried and regressed (see Empirical Update at the top). Do **not** start there.

Revised plan:

1. **Baseline restored.** Both configs are back on `lower_upper`. Confirm the
   baseline MAE before experimenting further.
2. **Coherence-preserving normal.** Try `normal, mean: -0.5·std` (e.g.
   `mean: -0.05, std: 0.1`) for convolution/aggregation — reproduces the old
   mean-aggregation prior without the `randn`/`rand` ambiguity. Sweep `std` by
   editing the value across runs (grid search can't expand a list nested inside
   `weight_initialization` — see the grid-search caveat below).
3. **Do NOT rename/"fix" the `randn` branch to `rand`** without re-benchmarking:
   the nonzero mean it produces is load-bearing on ZINC. If renaming for
   clarity, preserve the mean (make it an explicit mean-aggregation init), don't
   switch to a true uniform.
4. **Implement (c) reframed** — fan-in-aware *std* **plus** a nonzero
   mean-aggregation term (not zero-mean). Removes the scale guessing while
   keeping the coherent-aggregation prior.
5. Extend `weight_initialization` to cover `linear` layers (esp. near-zero
   readout init for regression).
6. Optionally prototype (e) LSUV rescaling as a config flag.

### Grid-search caveat

`run_configuration.py:140-146` only expands a fixed set of top-level keys
(`batch_size`, `learning_rate`, `epochs`, `dropout`, `optimizer`,
`weight_decay`, `loss`). `weight_initialization` is passed through verbatim, so
a list nested inside it (e.g. `std: [0.3, 0.1, 0.03]`) is **not** swept — it
would reach `torch.nn.init.normal_(std=[...])` and crash. Sweep init
hyperparameters by editing the scalar across runs, or add
`weight_initialization` handling to the grid loop first.
