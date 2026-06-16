# Transfer Learning — Proposed Config Schema & Examples

**Created:** 2026-06-15
**Companion to:** [`07-transfer-learning-redesign.md`](07-transfer-learning-redesign.md)

> Plan only — this documents the *proposed* YAML schema for the redesigned, config-driven transfer
> learning. No implementation here.

---

## 1. The `transfer:` block

Lives in the **hyperparameter config** (third tier; same file as optimizer/lr/epochs). Absence of the
block ⇒ train from scratch (current default behaviour, unchanged).

```yaml
transfer:
  # --- Where to load the pretrained checkpoint from ---
  source:
    results_path: ../pretrain_run/Results   # dir produced by the pretraining experiment
    dataset: MUTAG                           # db_name of the source checkpoint
    select: best                             # "best"  OR  explicit selector below
    # select:                                #   (alternative explicit form)
    #   config_id: 0
    #   run_id: 0
    #   validation_id: 0
    # model_config: ../pretrain_run/models.yml          # optional fallback if results dir
    # hyperparameter_config: ../pretrain_run/params.yml #   has no archived config.yml

  # --- How to fine-tune ---
  strategy: finetune        # finetune | linear_probe | differential_lr

  head:
    reinit: auto            # auto (reinit head when output dim differs) | always | never
    match: prefix           # backbone-param matching: "exact" | "prefix" (default exact)

  freeze:                   # optional; layer indices (as strings) and/or name globs
    - "0"                   # freeze first layer in the ModuleList
    - "embedding.*"         # freeze by parameter-name glob

  differential_lr:          # only read when strategy == differential_lr
    backbone: 1.0e-4
    head: 1.0e-3
```

### Field semantics

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `source.results_path` | str/path | yes | — | Results dir of the pretraining run. |
| `source.dataset` | str | yes | — | `db_name` of the checkpoint inside that results dir. |
| `source.select` | `"best"` or map | no | `best` | `best` reuses `load_model(best=True)`; map gives explicit ids. |
| `source.model_config` / `hyperparameter_config` | path | no | — | Fallback when `config.yml` wasn't archived (see 07 §7). |
| `strategy` | enum | no | `finetune` | See 07 §3.4. |
| `head.reinit` | enum | no | `auto` | `auto` reinit iff output shape differs. |
| `head.match` | enum | no | `exact` | `prefix` enables glob remap of backbone keys. |
| `freeze` | list[str] | no | `[]` | Indices and/or name globs; orthogonal to `strategy`. |
| `differential_lr.{backbone,head}` | float | iff `differential_lr` | — | Per-group learning rates. |

---

## 2. Validation rules (to add in `configuration_checks.py`)

- `transfer` must be a mapping; `source` mandatory with `results_path` + `dataset`.
- `strategy` ∈ {`finetune`, `linear_probe`, `differential_lr`}.
- If `strategy == differential_lr` ⇒ `differential_lr.backbone` and `.head` required (floats).
- `head.reinit` ∈ {`auto`, `always`, `never`}; `head.match` ∈ {`exact`, `prefix`}.
- `select` is `"best"` or a map with integer `config_id`/`run_id`/`validation_id`.
- `freeze` entries are strings.
- Grid search: list-valued `strategy`/`differential_lr` values expand via the existing cartesian
  product, like any other hyperparameter.

---

## 3. End-to-end example (`examples/transfer_learning/`)

Proposed layout mirroring existing examples:

```
examples/transfer_learning/
├── main.py
├── pretrain_main.yml          # pretrain on source dataset
├── finetune_main.yml          # finetune on target dataset (params has transfer: block)
├── models.yml                 # shared backbone architecture
├── pretrain_params.yml        # no transfer block
└── finetune_params.yml        # contains the transfer: block above
```

`main.py` sketch (plan only — illustrative, not final):

```python
from pathlib import Path
from simplegnn.framework.core import FrameworkMain

# 1) Pretrain
pre = FrameworkMain(Path('examples/transfer_learning/pretrain_main.yml'))
pre.preprocessing(num_threads=1)
pre.run_configurations(num_threads=-1)
pre.evaluate_results()
pre.run_best_configuration(num_threads=-1)   # writes the checkpoint the finetune run points to

# 2) Finetune — transfer is fully described by finetune_params.yml's `transfer:` block.
ft = FrameworkMain(Path('examples/transfer_learning/finetune_main.yml'))
ft.preprocessing(num_threads=1)
ft.run_configurations(num_threads=-1)        # loads + transfers per the YAML block
ft.evaluate_results()
ft.run_best_configuration(num_threads=-1)
ft.evaluate_results(evaluate_best_model=True)
```

No live objects are passed between the two `FrameworkMain` instances — the link is the
`source.results_path` in `finetune_params.yml`.

---

## 4. Provenance recorded in result JSON

Extends the object written at `core.py:753-763`:

```json
{
  "experiment_time": "2026-06-15 12:00:00",
  "config_id": 0,
  "run_id": 0,
  "validation_id": 0,
  "transfer": {
    "source": {"results_path": "../pretrain_run/Results", "dataset": "MUTAG",
               "config_id": 3, "run_id": 0, "validation_id": 0},
    "strategy": "finetune",
    "report": {"matched": 12, "skipped": 0, "reinitialized": 2, "frozen": 0}
  }
}
```
