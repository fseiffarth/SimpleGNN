# ShareGNN Transfer Learning — Cross-Dataset Pretrain → Finetune

**Created:** 2026-07-17
**Priority:** HIGH
**Effort:** ~4–6 days (phased; a usable Phase 1 lands in ~1 day)
**Status:** Plan only. No implementation in this document.

> **Builds on** [`07-transfer-learning-redesign.md`](07-transfer-learning-redesign.md) and
> [`07a-transfer-learning-config-schema.md`](07a-transfer-learning-config-schema.md). Those specs
> define the generic, YAML-driven `transfer:` block, the `framework/utils/transfer.py` module, the
> fine-tuning strategies (finetune / linear_probe / differential_lr), provenance, and the
> "stop the bleeding" cleanup of the dead `pretraining_finetuning`/`create_splits` calls. **This spec
> does not re-specify those.** It adds the piece 07/07a explicitly deferred: making transfer actually
> work for **ShareGNN's invariant layers**, whose weight tensors are dataset-sized and therefore break
> a plain `state_dict` copy across datasets.

---

## 1. Goal

Train a ShareGNN on one or more source datasets (e.g. **NCI1**, **NCI109**, **Mutagenicity**), then
transfer the trained model to a different target dataset (**DHFR**) or a different task, **replacing
the final (readout/head) layers** while reusing the learned backbone — all driven by the existing
three-tier YAML configuration style (main → model → hyperparameters), with the model architecture and
task defined in config as they are today.

Two concrete workflows must be first-class and reproducible from YAML:

1. **Single-source → target.** Pretrain on NCI1, finetune on DHFR.
2. **Multi-source → target.** Pretrain jointly on NCI1 + NCI109 + Mutagenicity, finetune on DHFR.

---

## 2. Why ShareGNN transfer is not a plain `state_dict` load

This is the crux and the reason a ShareGNN-specific spec is needed on top of 07/07a.

### 2.1 The head is easy

The output width is **set by the last `linear` layer's explicit `out_features` in the model YAML**
(`model.py:435,464`), *not* auto-derived from `num_classes` (`self.out_dim = graph_data.num_classes`
at `model.py:214` only feeds an aggregation-layer default that does not size any weights). So
"replacing the final layers" is literally: reinitialize the last `linear` (and any post-aggregation
readout) for the target task. NCI1/NCI109/Mutagenicity/DHFR are all binary `graph_classification`
(num_classes = 2), so even the shape matches — but the head should still be reinitialized because the
class semantics differ. This part is exactly what 07/07a's `head.reinit: auto` already covers.

### 2.2 The invariant backbone is the hard part

Each `invariant_based_convolution` / `invariant_based_aggregation` layer owns two learnable tensors —
`Param_W` and `Param_b` — whose **lengths are functions of the dataset**:

- **Message-passing `Param_W`** length = the number of `(source_label, target_label)` pairs that
  *actually co-occur at each property value* in the dataset, after `rule_occurrence_threshold`
  filtering. Computed in `_build_distributions` (`inv_based_message_passing.py:349–438`): pairs are
  encoded as `a*max_label + b` and counted with `torch.unique` (`:396–409`); total length =
  `sum(weight_num)` (`_finalize_initialization`).
- **Message-passing `Param_b`** length = `sum(in_features × n_bias_labels[head])`
  (`:468`), i.e. driven by `num_unique_node_labels` of each bias label.
- **Aggregation `Param_W`** length = `sum(n_node_labels[i] × n_heads_per_label[i])`
  (`inv_based_pooling.py:64`).

`num_unique_node_labels` comes from `torch.unique` of that dataset's labels
(`NodeLabels.__init__`, `NodeLabels.py:11–12`). **Different datasets → different label alphabets and
different property co-occurrence → different `Param_W`/`Param_b` lengths.** NCI1 (~37 primary/atom
labels) and DHFR (~9) will not match.

### 2.3 The index buffers that give weights meaning are not even saved

The maps from a label/property combination to its weight slot (`_pv_<h>`, `_nw_<h>`, `_bias_idx_<s>`,
`_b_off`, aggregation `_agg_idx_<h>`, `_agg_col_offset`) are registered with **`persistent=False`**
(`inv_based_message_passing.py:443–444,459,476`; `inv_based_pooling.py:79–84`). They are **absent from
`state_dict`**; only the bare `Param_W`/`Param_b` values are saved
(`model_configuration.py:980,1010` — `torch.save(self.net.state_dict(), ...)`).

**Consequences:**

1. `load_model` rebuilds a fresh `GraphModel` on the *target* dataset and calls
   `net.load_state_dict(...)` **without `strict=False`** (`load_model.py:161`). A shape-mismatched
   invariant `Param_W` (the normal cross-dataset case) makes this **raise** — the current
   `pretrained_network` tuple path only works when source and target are the same dataset.
2. Even if lengths coincidentally matched, slot *k* means a different `(source_label, target_label,
   property)` tuple in each dataset, because the compaction from `torch.unique` is per-dataset. A
   positional copy would transfer garbage.

**Therefore cross-dataset invariant-weight transfer requires matching weights by their _semantic key_
(the label/property tuple they represent), not by position.** That mechanism does not exist yet and is
the core deliverable of this spec.

---

## 3. Design

### 3.1 Layer taxonomy for transfer

| Layer kind | `state_dict` params | Cross-dataset transfer behaviour |
|------------|--------------------|----------------------------------|
| `linear` (intermediate), `layer_norm`, `batch_norm`, activation, dropout, reshape | shape depends only on **feature dim** (config-driven) → dataset-independent when `input_features: constant` | **Copy by name+shape** (07/07a exact-match path). |
| final `linear` / readout head | shape = `out_features` (config) | **Reinitialize** (`head.reinit: auto`). |
| `invariant_based_convolution`, `invariant_based_aggregation` | `Param_W`/`Param_b`, **dataset-sized** | **Key-based remap** (§3.3) — copy the slots whose semantic key exists in both source and target; reinitialize the rest. |

Note the guardrail: with `input_features: {name: node_labels, transformation: one_hot}` the first
layer's `in_features` becomes dataset-dependent (one-hot width differs NCI1 vs DHFR,
`graph_dataset.py:765–774`), which breaks the "intermediate linear is transferable" assumption.
**Recommend `input_features: constant` for transferable ShareGNN configs** and validate/ warn otherwise
(§3.6).

### 3.2 Checkpoint contract extension: the weight-key manifest

The saved `state_dict` is insufficient for semantic transfer because the index buffers are
non-persistent. Add, next to each saved `.pt`, a sibling **weight-key manifest** JSON that records,
per invariant layer, the semantic identity of every weight/bias slot in the checkpoint:

```
{results}/{db}/Models/model_Best_Configuration_000000_run_0_val_step_0.pt
{results}/{db}/Models/model_Best_Configuration_000000_run_0_val_step_0.keys.json   # NEW
```

Manifest shape (illustrative):

```json
{
  "layers": {
    "2": {                                   // index into net_layers (the conv layer)
      "type": "invariant_based_convolution",
      "heads": [
        {
          "source_label": "induced_cycles_6_6",
          "target_label": "induced_cycles_6_6",
          "bias_label":   "primary",
          "property":     "distances",
          "weight_slots": [                   // one entry per Param_W slot for this head
            {"src": 3, "tgt": 3, "prop": 1, "offset": 0},
            {"src": 3, "tgt": 5, "prop": 1, "offset": 1}
          ],
          "bias_slots":   [ {"bias": 0, "feature": 0, "offset": 0} ]
        }
      ]
    }
  }
}
```

The keys use the **original label values** (`NodeLabels.original_node_labels`, col 0 of the label
file — `NodeLabels.py:9`), which carry dataset-independent structural meaning for structural label
types (cycle membership, betweenness bin, degree), not the per-dataset compacted `node_labels` (col
1). This is what makes the keys comparable across datasets. The manifest is produced during `__init__`
of each invariant layer (all the needed information — descriptions, unique values, offsets — is already
computed there) and written by the save path alongside `torch.save`.

> **Key-comparability caveat (must be documented, drives D1/D3):** original-value comparability holds
> cleanly for *structural* labels — `betweenness_centrality` (bin index), `simple_cycles`/
> `induced_cycles` (cycle-length membership), `degree`. It is **partial/unreliable** for `wl_*`
> (dataset-specific hash space) and for raw `primary`/atom-type labels (each TU dataset enumerates
> atom types independently). The manifest still records those keys; overlap is simply smaller. §3.5's
> union pretraining is the mitigation.

### 3.3 Key-based invariant weight transfer

New logic in the ShareGNN transfer path (owned by the `transfer.py` module from 07 §3.2, extended
here). For each target invariant layer:

1. Build the target layer normally (its `_build_distributions` runs on the **target** dataset,
   producing the target slot→key map, held in memory before/without persistence).
2. Load the source layer's manifest and `Param_W`/`Param_b`.
3. For each target weight slot, look up its key `(source_label_value, target_label_value, property_value)`
   in the source manifest. **Match → copy** the source weight into the target slot. **No match →
   keep the freshly initialized target weight.** Same for bias slots keyed by `(bias_label_value,
   feature)`.
4. Emit a `TransferReport` extension: per invariant layer, `matched / reinitialized / source_unused`
   slot counts (feeds provenance + logging, like 07 §3.5).

This degrades gracefully: full overlap → full backbone transfer; partial overlap (the realistic
cross-family case) → partial transfer with the rest reinitialized; zero overlap → equivalent to
training from scratch for that layer, logged loudly.

### 3.4 Where it plugs in

- Replace the verbatim `self.net = pretrained_network` (`model_configuration.py:533–535`) with:
  build target `GraphModel` from the target config, then call the transfer engine
  (name+shape copy for standard layers, key-based remap for invariant layers, head reinit).
  *(This is 07 §3.3 with the invariant-layer branch added.)*
- Reuse `load_model` (`load_model.py:96`) **only** to obtain the source `state_dict` + rebuild the
  source `para`; do not require the source net to be shape-compatible with the target. Load the source
  manifest from the sibling `.keys.json`.
- Keep everything config-driven via the `transfer:` block (07a). Add ShareGNN-specific sub-keys (§3.6).

### 3.5 Multi-source pretraining (NCI1 + NCI109 + Mutagenicity)

There is **no joint multi-dataset training today**: a `datasets:` list in `main.yml` trains each
dataset independently (`core.py:195`, per-dataset keys in `network_configurations`). Two viable ways to
get "one backbone from three source datasets":

**Option A — Union/merged pretraining dataset (recommended).** A single dataset entry whose `name` is
a **list** is concatenated into one `GraphDataset` (`core.py:562–574`, `preprocessing.py:233–255`).
Pretraining on the union `NCI1_NCI109_Mutagenicity` builds **one shared label alphabet** spanning all
three sources, which (a) is a single checkpoint to transfer from, and (b) **maximizes key overlap with
the target** since the union's alphabet is the largest. This is the cleanest fit for §3.3.

- **Blocker to fix first:** the union split path is broken — `load_configuration_splits`
  (`preprocessing.py:361–425`) calls `pretraining_finetuning(...)` (`:425`) and `create_splits(...)`
  (`:460`), **neither of which is defined or imported anywhere** → `NameError`. 07 Phase 0 already
  flags this. This spec **depends on implementing `create_splits`** (07 decision D3) so union datasets
  can generate splits. Until then, union pretraining requires a pre-supplied
  `{union}_splits.json` (the early-return path at `preprocessing.py:402–408` works when the file
  exists).

**Option B — Sequential pretraining.** Pretrain on NCI1, transfer to NCI109 (accumulating), transfer
to Mutagenicity, then to DHFR — each step a §3.3 transfer. More configs, no union-split dependency,
but each hop only transfers the overlapping keys, so structural knowledge can erode. **Secondary.**

Recommendation: **Option A**, contingent on `create_splits`. Ship Option B as the no-new-split
fallback.

### 3.6 Config schema (extends 07a's `transfer:` block)

Model architecture and task stay defined in YAML exactly as today. The `transfer:` block (hyperparameter
tier) gains ShareGNN-specific fields; everything else is inherited from 07a.

```yaml
transfer:
  source:
    results_path: ../pretrain_run/results/    # results dir of the pretraining run
    dataset: NCI1_NCI109_Mutagenicity          # db_name of the source checkpoint (union name here)
    select: best                               # best | {config_id, run_id, validation_id}
  strategy: finetune                           # finetune | linear_probe | differential_lr  (07 §3.4)
  head:
    reinit: auto                               # auto | always | never  (07a)
  invariant_transfer:                          # NEW — ShareGNN-specific
    match: keys                                # keys (semantic remap, §3.3) | none (reinit all invariant layers)
    on_missing: reinit                         # reinit (default) | zero  — for target slots with no source key
    min_overlap_warn: 0.10                     # warn if a layer transfers < this fraction of its slots
  freeze:                                      # optional (07a) — e.g. freeze the invariant backbone
    - "invariant_based_convolution.*"
```

Validation additions (`configuration_checks.py`, alongside 07a's rules):
- `invariant_transfer.match` ∈ {`keys`, `none`}; `on_missing` ∈ {`reinit`, `zero`}.
- If any model layer is `invariant_based_convolution`/`invariant_based_aggregation` **and**
  `input_features.name != constant`, **warn** that intermediate-layer transfer may be unsafe (§3.1).
- If `match: keys`, require the source checkpoint to have a sibling `.keys.json` manifest; error with
  a clear message pointing at §3.2 if absent (e.g. checkpoint predates manifest support).

---

## 4. Implementation phases

**Phase 0 — Prerequisites (from 07, ~0.5 day).** Land 07 Phase 0 (quarantine the dead
`pretraining_finetuning`/`create_splits` `NameError`s; make `'best'` string mode fail loudly) and 07
Phase 1's `transfer.py` skeleton + name/shape copy + head reinit for standard layers. This spec assumes
those exist.

**Phase 1 — Weight-key manifest (~1 day).**
- In each invariant layer `__init__`, assemble the slot→key structure (data already computed) and
  expose it (e.g. `layer.export_weight_keys() -> dict`).
- Write `<checkpoint>.keys.json` in the save path next to `torch.save`
  (`model_configuration.py:980,1010`).
- Unit test: manifest slot counts equal `Param_W`/`Param_b` lengths; keys use original label values.

**Phase 2 — Key-based transfer engine (~1.5 days).**
- `transfer_invariant_layer(target_layer, source_manifest, source_param_w, source_param_b, cfg)` →
  copies matched slots, reinit/zero the rest, returns per-layer report.
- Integrate into the reworked `initialize_model` invariant branch.
- Unit tests with tiny synthetic datasets: full overlap (all copied), partial overlap (subset copied,
  rest reinit), disjoint (all reinit), bias keying, `on_missing: zero`.

**Phase 3 — Config-driven sourcing + strategies (~1 day).**
- Parse `transfer.invariant_transfer`; wire `run_configuration` (`core.py:753–773`) to read the YAML
  block instead of `self.pretrained_network` (constructor kept as shim per 07 §7).
- `freeze` globs matching invariant layers; `linear_probe`/`differential_lr` param groups
  (`set_optimizer`, `model_configuration.py:588–617`) — reuse 07 Phase 3.
- Provenance: extend the result JSON (`core.py:777–786`) with the per-invariant-layer overlap report.

**Phase 4 — Multi-source union pretraining (~1 day).**
- Implement `create_splits` (07 D3) so the union dataset `NCI1_NCI109_Mutagenicity` can generate
  train/val/test splits; verify the union label alphabet is the superset of the three sources.
- Validate the union → finetune path end-to-end.

**Phase 5 — Example, tests, docs (~0.5 day).**
- `examples/transfer_learning_sharegnn/`: `pretrain_main.yml` (union NCI dataset), `finetune_main.yml`
  (DHFR with the `transfer:` block), shared `models_ShareGNN.yml`, two `parameters.yml`, and a `main.py`
  that runs pretrain → finetune (mirroring 07a §3's sketch).
- Integration smoke test on a tiny slice (few epochs) asserting: manifest written, non-zero key
  overlap NCI→DHFR for structural labels, head reinitialized, finetune completes, provenance recorded.
- Update `.claude/CLAUDE.md` "Important Patterns" with the ShareGNN transfer recipe; short `docs/` note.

---

## 5. Testing strategy

- **Unit:** manifest export correctness; key-based remap (overlap matrices → matched/reinit counts);
  config validation accepts good `invariant_transfer` blocks and rejects malformed ones; input-features
  guardrail warning fires.
- **Integration:** the new example runs pretrain (union) → finetune (DHFR) on tiny data in CI-friendly
  time; asserts checkpoint + `.keys.json` produced, structural-label overlap > 0, head reinit, run
  completes with provenance.
- **Regression:** loading a checkpoint that lacks a manifest under `match: keys` fails with the
  documented actionable error, not a bare shape `RuntimeError` from `load_state_dict`.

---

## 6. Risks & open decisions

| ID | Decision | Recommendation |
|----|----------|----------------|
| D1 | Which label types are declared cross-dataset transferable | **Structural only by default** (`betweenness_centrality`, `simple_cycles`, `induced_cycles`, `degree`). Treat `wl_*`/`primary` as best-effort (keys recorded, low expected overlap); rely on union pretraining. |
| D2 | Multi-source method | **Union pretraining (Option A)**, contingent on `create_splits`; sequential (Option B) as fallback. |
| D3 | Manifest storage | **Sibling `.keys.json`** next to the `.pt` (human-inspectable, decoupled from the tensor file). Alternative: make index buffers `persistent=True` — rejected (bloats every checkpoint, ties format to internal buffer layout). |
| D4 | `on_missing` default | **`reinit`** (fresh init for unmatched target slots); `zero` offered for ablations. |
| D5 | Backward compat with pre-manifest checkpoints | Under `match: keys`, **hard error** with guidance to re-save; `match: none` still works (reinit all invariant layers, transfer only standard layers + head). |

**Primary risk — low structural overlap across chemically dissimilar datasets.** Mitigated by (a) union
pretraining maximizing the source alphabet, (b) the graceful partial-transfer degradation, and (c) the
`min_overlap_warn` telemetry so a near-useless transfer is visible rather than silent.

---

## 7. Summary

The head swap the user asked for ("replace final layers") is trivial in this framework — the head width
is a YAML `out_features`, and 07/07a already handle head reinit. The real work is the **invariant
backbone**: its `Param_W`/`Param_b` are dataset-sized and their label→slot maps aren't even in the
checkpoint, so cross-dataset transfer must match weights by **semantic key**, not position. This spec
adds a weight-key manifest to the checkpoint, a key-based remap engine that copies overlapping
label/property slots and reinitializes the rest, a config-driven `transfer.invariant_transfer` block in
the existing three-tier YAML, and **union pretraining** on NCI1+NCI109+Mutagenicity to maximize overlap
with DHFR — delivered in phases on top of the 07/07a foundation.
