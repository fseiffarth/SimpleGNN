# Training Pipeline Optimization Spec

## 1. Training Evaluation Called Every Batch

**Location:** `src/framework/model_configuration.py:842-851`

**Problem:** `evaluate_results()` is called after every training batch, not just at epoch end. For 10 batches/epoch x 100 epochs = 1,000 evaluation calls. Each call includes console printing (lines 390-420).

**Proposed fix:** Accumulate batch metrics and call `evaluate_results()` once per epoch, or add a configurable `eval_frequency` parameter.

---

## 2. CSV I/O Every Epoch

**Location:** `src/framework/model_configuration.py:744-782`

**Problem:** `postprocess_writer()` opens, writes, and closes a CSV file every epoch:

```python
with open(final_path, "a") as file_obj:
    file_obj.write(res_str)
```

For 100 epochs x 10 folds x 5 runs = 5,000 file open/write/close operations.

**Proposed fix:**
- Buffer results in memory and write periodically (e.g., every 10 epochs).
- Or keep the file handle open for the duration of a run.
- Or write all results at the end of training.

---

## 3. Full Validation/Test Evaluation Every Epoch

**Location:** `src/framework/model_configuration.py:144-149`

**Problem:** Full validation and test set evaluation runs every epoch regardless of dataset size. For large datasets, evaluation dominates training time.

**Proposed fix:** Add configurable `validation_frequency` parameter (e.g., validate every 5 epochs). Only validate every epoch near convergence.

---

## 4. Hardcoded Evaluation Batch Size

**Location:** `src/framework/model_configuration.py:861`

**Problem:** `batches = [graph_ids[i:i + 512] for i in range(0, len(graph_ids), 512)]` - Fixed batch size of 512 regardless of available memory or graph sizes.

**Proposed fix:** Make evaluation batch size configurable via parameters.yml, or derive from training batch size.

---

## 5. Per-Epoch Batch Recreation

**Location:** `src/framework/model_configuration.py:112-124`

**Problem:** Training batches and class weights are recomputed every epoch:

```python
train_batches = self.get_train_batches(seeds, epoch)
if self.para.run_config.config.get('weighted_loss', False):
    self.class_weights = torch.zeros(...)
    for i in range(len(train_batches)):
        self.class_weights[i] = torch.unique(self.graph_data.y[train_batches[i]], return_counts=True)[1]
```

For default sampling (simple shuffle), the batch composition doesn't need to change structure, only order.

**Proposed fix:** For non-curriculum sampling strategies, pre-compute batch compositions once and only shuffle the order. Cache class weights if batch composition is unchanged.

---

## 6. Loss Function Recreated Per Batch

**Location:** `src/framework/model_configuration.py:812-819`

**Problem:** When `weighted_loss=True`, `set_loss_function()` is called every batch, recreating the `nn.CrossEntropyLoss` with new weights.

**Proposed fix:** Pre-compute all batch weights at epoch start; only update criterion weight attribute instead of recreating the module.

---

## 7. Grid Search Explosion

**Location:** `src/framework/run_configuration.py:138-146`

**Problem:** 7-level nested loop creates Cartesian product of all parameter lists:
- 3 batch_sizes x 5 LRs x 4 epochs x 3 dropouts x 4 optimizers x 2 weight_decays x 3 losses = 4,320 configurations

No pruning, no early termination of bad configs, no Bayesian optimization.

**Proposed fix:**
- Add `max_configurations` parameter to limit grid search.
- Implement random search (sample N configurations from the grid).
- Consider adding Optuna or similar hyperparameter optimization integration.

---

## 8. Joblib Serialization Overhead

**Location:** `src/framework/core.py:100-105`

**Problem:** Entire `graph_data` object is passed to each parallel job via joblib. For large datasets, this serialization/deserialization is expensive:

```python
joblib.Parallel(n_jobs=num_threads)(
    joblib.delayed(self.run_configuration)(
        graph_data=graph_data,  # Serialized per job!
        ...
    ) for i in range(len(run_loops)))
```

**Proposed fix:**
- Use `joblib.Memory` for caching or shared memory.
- Pass file paths instead of data objects; let each worker load data.
- Use `multiprocessing.shared_memory` for the dataset.

---

## 9. os.system() for File Copying

**Location:** `src/framework/core.py:505-517`

**Problem:** Uses `os.system()` with string formatting for file copies:

```python
if os.name == 'posix':
    os.system(f"cp {source_path} {destination_path}")
```

This is a shell injection vulnerability and slower than native Python.

**Proposed fix:** Use `shutil.copy2(source_path, destination_path)`.

---

## 10. Missing torch.no_grad() in Evaluation

**Location:** `src/framework/model_configuration.py:853-882`

**Problem:** `evaluate_graph_task()` does not consistently wrap evaluation in `torch.no_grad()`. Additionally, `self.net.train(False)` is called inside the loop.

**Proposed fix:**
```python
@torch.no_grad()
def evaluate_graph_task(self, ...):
    self.net.eval()
    ...
```

---

## 11. Sequential Dataset Evaluation

**Location:** `src/framework/core.py:107-141`

**Problem:** `evaluate_results()` processes datasets sequentially (no parallelization), unlike `run_configurations()` which uses joblib.

**Proposed fix:** Use joblib for parallel evaluation across datasets when evaluating multiple datasets.

---

## 12. Redundant Label/Property Loading Per Run

**Location:** `src/framework/utils/preprocessing.py:284-320`

**Problem:** Labels and properties are loaded from disk for each run/fold, even though they're identical across runs.

**Proposed fix:** Load once at the dataset level and share across runs via the `graph_data` object.

---

## Priority Order

1. **Section 1** - Evaluate per epoch not per batch (immediate, large impact)
2. **Section 10** - Add torch.no_grad() to evaluation (5 min, correctness + speed)
3. **Section 3** - Configurable validation frequency (30 min, large impact for big datasets)
4. **Section 9** - Replace os.system with shutil (5 min, security fix)
5. **Section 2** - Buffer CSV writes (30 min, reduces I/O)
6. **Section 5** - Cache batch compositions (1 hour)
7. **Section 7** - Add max_configurations / random search (2-4 hours)
8. **Section 8** - Optimize joblib serialization (2-4 hours)
