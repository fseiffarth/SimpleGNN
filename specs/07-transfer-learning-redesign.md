# Transfer Learning Redesign Spec

**Created:** 2026-06-15
**Priority:** HIGH
**Effort:** ~2-3 days (phased; Phase 1 is ~0.5 day)
**Impact:** Replaces a half-broken, conflated transfer-learning implementation with a single,
config-driven, reproducible, and testable pretrain → finetune workflow.

> **Status:** Plan only. No implementation in this document. Companion file:
> [`07a-transfer-learning-config-schema.md`](07a-transfer-learning-config-schema.md) (proposed YAML schema + examples).

---

## 1. Problem Statement

Transfer learning today is spread across two **unrelated concepts that are both called "transfer
learning"**, wired together inconsistently (part Python API, part YAML), and partly **non-functional**.

### 1.1 Concept A — Weight transfer (`pretrained_network`)

`FrameworkMain.__init__(main_config_path, pretrained_network=None)` accepts a pretrained model and
reuses its weights. Dispatch happens in `run_configuration()`:

- `core.py:730-744` — `tuple` form `(FrameworkMain, experiment_db_id)`: loads the *best* model from
  another live `FrameworkMain` via `load_model(...)`, then `train_configuration(pretrained_network=...)`.
- `core.py:745-748` — `str` form `'best'`: **unimplemented**, body is `pass` → silently trains from
  scratch with no warning.
- `core.py:749-750` — `Module`/`None` form: passed straight through.
- `model_configuration.py:507-540` `initialize_model()` — when a pretrained net is given it does
  `self.net = pretrained_network`, i.e. **reuses the entire model verbatim**.

### 1.2 Concept B — Dataset union (`pretraining_datasets` / `finetuning_datasets`)

A dataset `name` can be a **list**, which gets concatenated (`MUTAG_PROTEINS`) and the components
recorded in `single_datasets` (`core.py:539-551`). `data_generation_type` as a list triggers a
recursive **merge of several datasets into one union** `GraphDataset`
(`preprocessing.py:233-255`). `load_configuration_splits()` (`preprocessing.py:396-425`) is meant to
build combined pretrain/finetune split files.

### 1.3 What is actually broken

1. **Dead references — guaranteed `NameError`.** `preprocessing.py:425` calls
   `pretraining_finetuning(...)` and `preprocessing.py:460` calls `create_splits(...)`. **Neither is
   imported or defined anywhere** in `src/simplegnn/` (verified by grep). Any config that sets
   `pretraining_datasets`/`finetuning_datasets`, or any dataset needing split generation, crashes.
2. **`'best'` string mode is a silent no-op** (`core.py:746-748`).
3. **No head/readout replacement.** `initialize_model` reuses the whole net, so a different
   downstream task (different `#classes`, regression vs. classification) produces a dimension
   mismatch or trains the wrong output head. Nothing handles this.
4. **No fine-tuning strategy.** No backbone freezing, no head-only training (linear probing), no
   differential learning rates, no gradual unfreezing.
5. **No architecture-compatibility handling.** The pretrained net must match the new model config
   exactly; there is no partial `state_dict` load (transfer only the layers that match) and no
   validation/warning when they don't.
6. **Two configuration styles.** Weight transfer is Python-only (constructor arg, a live
   `FrameworkMain` passed as a tuple); dataset-union transfer is YAML-only. Neither lives in the
   three-tier YAML system, so experiments are hard to reproduce.
7. **Process-safety / coupling.** Passing `(FrameworkMain, id)` carries a whole live framework object
   into `run_configuration`, which joblib pickles into every parallel worker — heavy and fragile, and
   it interacts badly with the known split-loss bug (`TODO.md`, "Loaded splits are lost when
   `preprocessing(num_threads>1)`").
8. **No provenance.** Result/config JSONs (`core.py:753-763`) don't record the pretraining source, so
   a finetuned run can't be traced back to its checkpoint.
9. **No tests, no examples.** Nothing under `examples/` or `tests/` exercises transfer learning, so
   all of the above is invisible to CI.

### Affected files

| File | Lines | Role today |
|------|-------|-----------|
| `src/simplegnn/framework/core.py` | 87-93, 151, 186, 730-750, 986-998 | `pretrained_network` plumbing, dispatch, path appendix |
| `src/simplegnn/framework/model_configuration.py` | 272-349, 507-540 | `train_configuration`, `initialize_model` |
| `src/simplegnn/framework/utils/preprocessing.py` | 233-255, 361-471 | dataset union, split handling, dead calls |
| `src/simplegnn/framework/utils/load_model.py` | 93-157 | `load_model` (reused as-is) |
| `src/simplegnn/framework/utils/configuration_checks.py` | 84-100 | `single_datasets` validation |

---

## 2. Goals & Non-Goals

### Goals

- **One concept, one mechanism.** Transfer learning = "initialize (part of) a model from a previously
  trained checkpoint, then continue training", configured entirely in YAML.
- **Config-driven & reproducible.** A `transfer:` block in the hyperparameter config fully specifies
  the source checkpoint and the strategy. No live objects passed through constructors.
- **Correct head handling.** Detect output/head mismatch and replace + reinitialize the head while
  transferring the backbone.
- **Fine-tuning strategies.** Support: full finetune, frozen backbone (linear probing), and
  differential learning rates. Gradual unfreezing is optional/stretch.
- **Robust loading.** Partial `state_dict` loading with an explicit, logged report of matched /
  skipped / reinitialized parameters; clear errors on incompatibility.
- **Provenance.** Record the source checkpoint and strategy in the result JSON.
- **Tested.** Unit tests for the loader/strategy logic + one tiny end-to-end example.

### Non-Goals

- **Dataset union / multi-dataset joint training (Concept B)** is *out of scope for transfer
  learning* and is explicitly **decoupled** (see §6). It is a separate "merged dataset" feature; this
  plan removes its broken split code path from the TL story rather than fixing it inline.
- No new GNN architectures or self-supervised pretraining objectives.
- Not fixing the unrelated `num_threads>1` split-loss bug here (tracked in `TODO.md`), though the new
  design avoids passing live `FrameworkMain` objects, which sidesteps one of its triggers.

---

## 3. Proposed Design

### 3.1 Single entry point: a `transfer:` config block

Transfer learning becomes a block in the **hyperparameter config** (the existing third tier). Full
schema in [`07a`](07a-transfer-learning-config-schema.md). Shape:

```yaml
transfer:
  source:
    results_path: ../pretrain_run/Results   # where the pretrained checkpoint lives
    dataset: MUTAG                            # source dataset (db_name) of the checkpoint
    select: best                              # best | {config_id, run_id, validation_id}
  strategy: finetune        # finetune | linear_probe | differential_lr
  head:
    reinit: auto            # auto (reinit if output dim differs) | always | never
    match: prefix           # how backbone params are matched to the new model
  freeze:                   # optional explicit freeze list (layer indices or name globs)
    - "0"
    - "embedding.*"
  differential_lr:          # only used when strategy == differential_lr
    backbone: 1.0e-4
    head: 1.0e-3
```

`FrameworkMain.__init__` keeps the optional `pretrained_network` arg **only as a thin
backward-compat shim** (see §7); the canonical path is the YAML block.

### 3.2 New module: `framework/utils/transfer.py`

A small, pure, unit-testable module that owns all TL logic:

- `resolve_transfer_source(transfer_cfg, ...) -> TransferSource` — turns the `source` block into a
  concrete `(experiment_configuration, db_name, config_id, run_id, validation_id)` and reuses the
  existing `load_model()` (`load_model.py:93`) to get the pretrained `nn.Module`. No live
  `FrameworkMain` needed — it reconstructs the minimal experiment config from `results_path` +
  `dataset` (the run already archives `config.yml` there once `copy_experiment_config` is fixed; see
  §7 dependency note).
- `transfer_weights(target_model, pretrained_model, strategy, head_cfg, freeze_cfg) -> TransferReport`
  — partial `state_dict` copy + head reinit + freezing. Returns a `TransferReport`
  (matched / skipped / reinitialized / frozen parameter names) for logging and provenance.
- `build_param_groups(model, transfer_cfg) -> list[dict]` — produces optimizer param groups for the
  `differential_lr` strategy (consumed by `set_optimizer`).

This isolates TL from `core.py` and `model_configuration.py`, both of which only *call into* it.

### 3.3 Partial weight transfer + head handling

Replace the verbatim `self.net = pretrained_network` (`model_configuration.py:534`) with:

1. Always build the target `GraphModel` from the **current** model config (so architecture is
   defined by config, not inherited from the checkpoint).
2. Copy parameters from the pretrained `state_dict` where **name and shape match** (`strict=False`
   semantics, implemented explicitly so we can report and so we can decide head handling).
3. **Head reinit** (`head.reinit`): when the output layer's shape differs (or `always`), keep its
   freshly initialized weights and log it. `auto` is the default and the common case (new task →
   new head).
4. Skipped/extra params are logged in the `TransferReport`, never silently dropped.

### 3.4 Fine-tuning strategies

| Strategy | Behaviour |
|----------|-----------|
| `finetune` (default) | Transfer backbone, (re)init head per `head.reinit`, all params trainable. |
| `linear_probe` | Freeze the entire backbone (`requires_grad=False`), train only the head. |
| `differential_lr` | All trainable, but optimizer uses param groups: low LR for backbone, higher for head (`build_param_groups`). |
| `freeze:` list | Orthogonal modifier: freeze named/indexed layers under any strategy. |

`set_optimizer` (`model_configuration.py:578-607`) gains a branch: if a `transfer` block requests
param groups, build the optimizer from `build_param_groups(...)` instead of `self.net.parameters()`.

### 3.5 Provenance

Extend the result JSON written at `core.py:753-763` with a `transfer` section: source checkpoint
identity (`results_path`, `dataset`, resolved `config_id/run_id/validation_id`), `strategy`, and a
compact form of the `TransferReport` (counts of matched/skipped/reinit/frozen params).

---

## 4. Implementation Phases

### Phase 0 — Stop the bleeding (≈0.5 day)

- Remove or quarantine the **dead** `pretraining_finetuning` / `create_splits` call sites
  (`preprocessing.py:425`, `:460`). Either implement `create_splits` (it's needed independently of TL
  — split generation is a real gap) or raise a clear `NotImplementedError`/import error instead of a
  bare `NameError`. **Decision point D3.**
- Make `'best'` string mode (`core.py:746-748`) raise `NotImplementedError` instead of silently
  passing, until the new path replaces it.
- Add a regression test asserting the old broken paths now fail loudly (guards against silent no-ops).

### Phase 1 — Core weight-transfer engine (≈1 day)

- New `framework/utils/transfer.py` with `transfer_weights` + `TransferReport` (pure, no I/O).
- Rework `initialize_model` (`model_configuration.py:507-540`) to build target model + call
  `transfer_weights`.
- Unit tests with tiny synthetic models: name/shape matching, head reinit (`auto`/`always`/`never`),
  skipped-param reporting.

### Phase 2 — Config-driven sourcing (≈0.5 day)

- `resolve_transfer_source` + parse `transfer:` block; validation in `configuration_checks.py`.
- Wire `run_configuration` (`core.py:730-750`) to read `run_config.config['transfer']` instead of
  `self.pretrained_network`. Keep the constructor arg as a shim (§7).
- Reuse `load_model()` as the checkpoint loader.

### Phase 3 — Strategies + optimizer integration (≈0.5 day)

- `freeze` handling + `build_param_groups`; `set_optimizer` branch.
- Strategy tests: `linear_probe` (head-only grads), `differential_lr` (param-group LRs).

### Phase 4 — Provenance, example, docs (≈0.5 day)

- Extend result JSON (`core.py:753-763`).
- Add `examples/transfer_learning/` (tiny: pretrain on one small TU dataset, finetune on another) with
  `main.py` + the three YAML tiers.
- Update `CLAUDE.md` "Important Patterns" and a short `docs/` note.

---

## 5. Testing Strategy

- **Unit (`tests/`):** `transfer_weights` matching/reinit/report; `build_param_groups`; config
  validation accepts good blocks and rejects malformed ones; `resolve_transfer_source` path resolution.
- **Integration (smoke):** the new `examples/transfer_learning/main.py` runs pretrain →
  finetune end-to-end on a tiny dataset in CI-friendly time (few epochs), asserting a checkpoint is
  produced, transferred, and the finetuned run completes and writes provenance.
- **Regression:** Phase 0 "fails loudly" tests.

---

## 6. Decoupling the dataset-union feature (Concept B)

Concept B (merging datasets into one `GraphDataset`) is a legitimate but **separate** feature and is
*not* transfer learning. This plan:

- Keeps the union/merge data path (`preprocessing.py:233-255`, `single_datasets`) intact as a
  "merged dataset" capability.
- Removes the TL framing and the broken split glue (`load_configuration_splits` TL branch,
  `pretraining_finetuning`) from the transfer story.
- If joint pretraining on a union dataset is still wanted, it composes naturally with the new design:
  pretrain on the merged dataset, then point a `transfer.source` at that checkpoint. No special code.

A follow-up spec (`08-merged-dataset-feature.md`) should properly design split handling for unions
if that feature is to be kept. **Out of scope here.**

---

## 7. Backward Compatibility & Dependencies

- **Constructor shim:** keep `FrameworkMain(..., pretrained_network=...)`. If set, translate it into
  an in-memory `transfer` block (Module → direct transfer; tuple → resolve via the source path). Emit
  a `DeprecationWarning` pointing to the YAML block. Remove after one release.
- **Dependency on `copy_experiment_config`:** sourcing-by-path relies on the pretrain run having
  archived its `config.yml`. That archiver is currently a no-op bug (`core.py:1074`, `TODO.md` —
  "`copy_experiment_config` never copies the config"). **Fix that bug as a prerequisite for Phase 2**
  (small, already-scoped in `TODO.md`). Until fixed, `resolve_transfer_source` can fall back to
  requiring an explicit model/hyperparameter path in the `source` block.
- **No change** to `load_model.py` (reused as-is).

---

## 8. Risks & Open Decision Points

| ID | Decision | Recommendation |
|----|----------|----------------|
| D1 | Config location of `transfer:` block — hyperparameter tier vs main config | **Hyperparameter tier** (it's a training concern and benefits from grid search). |
| D2 | Default `head.reinit` | **`auto`** (reinit when output dim differs). |
| D3 | Phase 0 for `create_splits`: implement vs. error out | **Implement `create_splits`** — it's an independent real gap (split generation), low effort, unblocks normal runs too. |
| D4 | Source addressing — by results path vs. by passing a sibling `main.yml` | **By results path + dataset** (decoupled, picklable, reproducible). |
| D5 | Param matching strategy — exact name+shape vs. prefix/glob remap | Start with **exact name+shape**; add glob remap only if architectures legitimately differ in naming. |
| D6 | Gradual unfreezing schedule | **Defer** (stretch goal); the three strategies cover the common cases. |

- **Risk: state_dict key drift** across model configs (layer ordering / naming in
  `GraphModel`'s `nn.ModuleList`). Mitigated by the explicit, logged matching report rather than
  `load_state_dict(strict=True)`.
- **Risk: scope creep into Concept B.** Mitigated by §6's hard decoupling.

---

## 9. Summary

The current TL is two conflated ideas, one of which (`pretraining_datasets`/`finetuning_datasets`) is
**dead code that throws `NameError`**, and the other (`pretrained_network`) lacks head handling,
freezing, partial loading, provenance, and tests. The redesign collapses everything into a single,
YAML-driven **pretrain → finetune** workflow backed by a small testable `transfer.py` module,
decouples the unrelated dataset-union feature, and adds the missing strategies, robustness, and tests
— delivered in five low-risk phases starting with a half-day "stop the bleeding" pass.
