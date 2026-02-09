# Next Steps

**Last Updated:** 2026-02-09

Prioritized action items for SimpleGNN, ordered by impact and effort. Each item references the relevant spec file for full details.

---

## Implementation Status Summary

### ✅ Completed Items (Phase 1 Critical Fixes)
- ~~Fix ShareGNNLinear file~~ - **FIXED**: File removed from codebase
- ~~Add torch.no_grad() to evaluation path~~ - **FIXED**: Implemented at model_configuration.py:870
- ~~Replace os.system() with shutil~~ - **FIXED**: Using shutil.copy2() properly
- ~~Fix eval() in framework_layer.py~~ - **FIXED**: Using safe dictionary lookup
- ~~Buffer CSV writes~~ - **FIXED**: Buffering implemented with configurable flush interval
- ~~Fix Dropout instantiation~~ - **FIXED**: All conv layers properly use functional.dropout()
- ~~Remove dead code in model.py~~ - **NOT A BUG**: Only whitespace formatting

### ⚠️ Partially Completed
- **eval() usage**: Fixed in preprocessing.py, but still present in graph_dataset.py:876, 878

### 🔴 Outstanding Critical Items
- Type mismatch in unique property dict check (utils.py:131-134)
- enum checking pattern (layer_loader.py:18) - May not be a bug in Python 3.13+

---

## Phase 1: Remaining Critical Issues

### 1. Fix type mismatch in unique property dict check
- **File:** `src/models/ShareGNN/utils.py:131-134`
- **Issue:** Compares raw dict but appends PropertyDict object; uniqueness check always fails
- **Effort:** 10 min
- **Spec:** `03-model-infrastructure-bugs.md` Section 3
- **Status:** 🔴 NOT FIXED

### 2. Replace remaining eval() calls in graph_dataset.py
- **Files:** `src/datasets/graph_dataset.py:876, 878`
- **Issue:** Arbitrary code execution from transformation expressions
- **Effort:** 30 min
- **Spec:** `03-model-infrastructure-bugs.md` Section 4
- **Status:** ⚠️ PARTIAL - preprocessing.py fixed, graph_dataset.py remains

### 3. Verify enum checking for Python version compatibility
- **File:** `src/models/layers/utils/layer_loader.py:18`
- **Issue:** `if layer_type not in LayerTypes` works in Python 3.13+ but may fail in older versions
- **Effort:** 15 min (add version check or use [e.value for e in LayerTypes])
- **Spec:** N/A - discovered during analysis
- **Status:** ⚠️ NEEDS REVIEW - Works in Python 3.13+, may break in 3.10

---

## Phase 2: ShareGNN Quick Wins (high impact, low effort)

### 4. Cache config lookups in ShareGNN forward pass
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:304-308`
- **Issue:** Dict lookups on constant config values inside forward()
- **Effort:** 5 min
- **Spec:** `01-sharegnn-optimizations.md` Section 4
- **Status:** 🔴 NOT IMPLEMENTED

---

## Phase 3: ShareGNN Initialization Performance (high impact, moderate effort)

### 5. Parallelize head processing during layer initialization
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:78-177`
- **Issue:** Heads processed sequentially; each takes 5-30s. With 32 heads = 8+ minutes initialization time
- **Solution:** Parallelize outer loop over heads using joblib; add `parallel_loading` config parameter (default -1 = all cores)
- **Expected Gain:** 5-8x speedup on 8-core machines for multi-head models
- **Effort:** 2-4 hours
- **Spec:** `05-parallel-layer-initialization.md`
- **Status:** 🔴 NOT IMPLEMENTED - **NEW SPEC**

### 6. Pre-allocate weight distributions instead of quadratic torch.cat
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:117-141`
- **Issue:** Nested Python loops with repeated `torch.cat` cause O(n^2) allocation
- **Effort:** 1-2 hours
- **Spec:** `01-sharegnn-optimizations.md` Section 3
- **Status:** 🔴 NOT IMPLEMENTED
- **Note:** Can be combined with item #5 (parallel loading) for maximum initialization speedup

### 7. Pre-allocate bias distributions
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:151-161`
- **Issue:** Same quadratic pattern for bias; O(graphs x features) with repeated concat
- **Effort:** 1 hour (same pattern as step 6)
- **Spec:** `01-sharegnn-optimizations.md` Section 3
- **Status:** 🔴 NOT IMPLEMENTED

### 8. Pre-allocate pooling layer distributions
- **File:** `src/models/ShareGNN/layers/inv_based_pooling.py:54-68`
- **Issue:** Triple nested loop with same quadratic torch.cat pattern
- **Effort:** 1 hour (same pattern as step 6)
- **Spec:** `01-sharegnn-optimizations.md` Section 6
- **Status:** 🔴 NOT IMPLEMENTED

---

## Phase 4: ShareGNN Forward Pass Performance (high impact, moderate effort)

### 9. Eliminate per-forward dense matrix allocation
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:237, 254`
- **Issue:** `torch.zeros((heads, N, N))` allocated every forward call; should pre-allocate and zero in-place, or use sparse tensors
- **Effort:** 2-4 hours
- **Spec:** `01-sharegnn-optimizations.md` Section 2
- **Status:** 🔴 NOT IMPLEMENTED

### 10. Same fix for pooling layer dense allocation
- **File:** `src/models/ShareGNN/layers/inv_based_pooling.py:130`
- **Issue:** Same pattern as message passing
- **Effort:** 1 hour (same pattern as step 9)
- **Spec:** `01-sharegnn-optimizations.md` Section 6
- **Status:** 🔴 NOT IMPLEMENTED

---

## Phase 5: Training Loop Improvements (medium impact, low-moderate effort)

### 11. Move evaluation from per-batch to per-epoch
- **File:** `src/framework/model_configuration.py:842-851`
- **Issue:** `evaluate_results()` called after every batch; 10x unnecessary calls
- **Effort:** 30 min
- **Spec:** `02-training-pipeline-optimizations.md` Section 1
- **Status:** 🔴 NOT IMPLEMENTED

### 12. Add configurable validation frequency
- **File:** `src/framework/model_configuration.py:144-149`
- **Issue:** Full validation/test runs every epoch regardless of dataset size
- **Effort:** 30 min
- **Spec:** `02-training-pipeline-optimizations.md` Section 3
- **Status:** 🔴 NOT IMPLEMENTED

### 13. Make evaluation batch size configurable
- **File:** `src/framework/model_configuration.py:861`
- **Issue:** Hardcoded to 512
- **Effort:** 15 min
- **Spec:** `02-training-pipeline-optimizations.md` Section 4
- **Status:** 🔴 NOT IMPLEMENTED

---

## Phase 6: Configuration Robustness (medium impact, moderate effort)

### 14. Add semantic validation for config values
- **File:** `src/framework/utils/configuration_checks.py:153-207`
- **Issue:** Parameter values not validated; typos silently use defaults
- **Effort:** 1-2 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 6
- **Status:** 🔴 NOT IMPLEMENTED

### 15. Add missing config options (validation_frequency, eval_batch_size, etc.)
- **Files:** `src/framework/utils/parameters.py`, config checks
- **Issue:** Several useful parameters have no config support
- **Effort:** 2-4 hours (incremental)
- **Spec:** `04-dataset-and-config-improvements.md` Section 8
- **Status:** 🔴 NOT IMPLEMENTED

---

## Phase 7: Dataset & Preprocessing (medium impact, moderate effort)

### 16. Cache NetworkX graph conversions
- **File:** `src/datasets/utils/node_labeling.py:554-555`
- **Issue:** PyG-to-NetworkX conversion happens per graph per label type, never cached
- **Effort:** 1 hour
- **Spec:** `04-dataset-and-config-improvements.md` Section 4
- **Status:** 🔴 NOT IMPLEMENTED

### 17. Parallelize label computation across graphs
- **File:** `src/datasets/utils/node_labeling.py`
- **Issue:** Expensive label types (cycles, cliques) run sequentially
- **Effort:** 2-4 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 1
- **Status:** 🔴 NOT IMPLEMENTED

### 18. Reduce redundant label loading per run
- **File:** `src/framework/utils/preprocessing.py:284-320`
- **Issue:** Labels reloaded from disk for each fold/run
- **Effort:** 1 hour
- **Spec:** `02-training-pipeline-optimizations.md` Section 12
- **Status:** 🔴 NOT IMPLEMENTED

---

## Phase 8: Architecture Improvements (high impact, high effort)

### 19. Add ShareGNN batch processing support
- **Files:** `src/framework/model_configuration.py:798-807`, ShareGNN layer forward methods
- **Issue:** Per-graph Python loop; 10-50x slower than batched execution
- **Effort:** 1-2 days
- **Spec:** `01-sharegnn-optimizations.md` Section 1
- **Status:** 🔴 NOT IMPLEMENTED - **HIGHEST RUNTIME IMPACT ITEM**

### 20. Fix tensor shape convention consistency
- **Files:** `src/models/layers/framework_layer.py`, all layer implementations
- **Issue:** Documented (C,N,F) convention not followed; mixing 2D and 3D layers crashes
- **Effort:** 1-2 days
- **Spec:** `03-model-infrastructure-bugs.md` Section 6
- **Status:** 🔴 NOT IMPLEMENTED

### 21. Add residual projection for dimension mismatches
- **Files:** All conv layers
- **Issue:** Residual connections assume matching shapes without validation
- **Effort:** 2-4 hours
- **Spec:** `03-model-infrastructure-bugs.md` Section 7
- **Status:** 🔴 NOT IMPLEMENTED

### 22. Make BatchNorm handle 3D input
- **File:** `src/models/layers/nn_standard/batch_normalization.py:13-14`
- **Issue:** Only works with (N,F); fails on (C,N,F) from multi-head layers
- **Effort:** 30 min
- **Spec:** `03-model-infrastructure-bugs.md` Section 10
- **Status:** 🔴 NOT IMPLEMENTED

---

## Phase 9: Future Enhancements (when core is stable)

### 23. Add random search / Optuna for hyperparameter optimization
- **File:** `src/framework/run_configuration.py:138-146`
- **Issue:** Cartesian product grid search can explode combinatorially
- **Effort:** 4-8 hours
- **Spec:** `02-training-pipeline-optimizations.md` Section 7
- **Status:** 🔴 NOT IMPLEMENTED

### 24. Optimize joblib serialization for parallel runs
- **File:** `src/framework/core.py:100-105`
- **Issue:** Full graph_data serialized per parallel job
- **Effort:** 2-4 hours
- **Spec:** `02-training-pipeline-optimizations.md` Section 8
- **Status:** 🔴 NOT IMPLEMENTED

### 25. Add mixed-precision (AMP) training support
- **Effort:** 4-8 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 8
- **Status:** 🔴 NOT IMPLEMENTED

### 26. Add schema enforcement for YAML configs
- **Effort:** 4-8 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 7
- **Status:** 🔴 NOT IMPLEMENTED

### 27. Implement or remove empty positional encoding layer
- **File:** `src/models/ShareGNN/layers/inv_based_positional_encoding.py`
- **Issue:** Empty file; layer type registered in enum but has no code
- **Spec:** `01-sharegnn-optimizations.md` Section 9
- **Status:** 🔴 NOT IMPLEMENTED

---

## Recommended Next Actions (Priority Order)

1. **Parallelize layer initialization** (2-4 hours) - Item #5 - **5-8x faster initialization**
2. **Fix enum checking for backward compatibility** (15 min) - Item #3 - Ensure Python 3.10+ compatibility
3. **Fix type mismatch in property dict check** (10 min) - Item #1 - Critical correctness bug
4. **Replace eval() in graph_dataset.py** (30 min) - Item #2 - Security issue
5. **Cache config lookups in ShareGNN** (5 min) - Item #4 - Quick performance win
6. **Add ShareGNN batch processing** (1-2 days) - Item #19 - **10-50x runtime performance improvement**

### 10. Pre-allocate weight distributions instead of quadratic torch.cat
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:117-141`
- **Issue:** Nested Python loops with repeated `torch.cat` cause O(n^2) allocation
- **Effort:** 1-2 hours
- **Spec:** `01-sharegnn-optimizations.md` Section 3

### 11. Pre-allocate bias distributions
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:151-161`
- **Issue:** Same quadratic pattern for bias; O(graphs x features) with repeated concat
- **Effort:** 1 hour (same pattern as step 10)
- **Spec:** `01-sharegnn-optimizations.md` Section 3

### 12. Pre-allocate pooling layer distributions
- **File:** `src/models/ShareGNN/layers/inv_based_pooling.py:54-68`
- **Issue:** Triple nested loop with same quadratic torch.cat pattern
- **Effort:** 1 hour (same pattern as step 10)
- **Spec:** `01-sharegnn-optimizations.md` Section 6

---

## Phase 4: ShareGNN Forward Pass Performance (high impact, moderate effort)

### 13. Eliminate per-forward dense matrix allocation
- **File:** `src/models/ShareGNN/layers/inv_based_message_passing.py:237, 254`
- **Issue:** `torch.zeros((heads, N, N))` allocated every forward call; should pre-allocate and zero in-place, or use sparse tensors
- **Effort:** 2-4 hours
- **Spec:** `01-sharegnn-optimizations.md` Section 2

### 14. Same fix for pooling layer dense allocation
- **File:** `src/models/ShareGNN/layers/inv_based_pooling.py:130`
- **Issue:** Same pattern as message passing
- **Effort:** 1 hour (same pattern as step 13)
- **Spec:** `01-sharegnn-optimizations.md` Section 6

---

## Phase 5: Training Loop Improvements (medium impact, low-moderate effort)

### 15. Move evaluation from per-batch to per-epoch
- **File:** `src/framework/model_configuration.py:842-851`
- **Issue:** `evaluate_results()` called after every batch; 10x unnecessary calls
- **Effort:** 30 min
- **Spec:** `02-training-pipeline-optimizations.md` Section 1

### 16. Add configurable validation frequency
- **File:** `src/framework/model_configuration.py:144-149`
- **Issue:** Full validation/test runs every epoch regardless of dataset size
- **Effort:** 30 min
- **Spec:** `02-training-pipeline-optimizations.md` Section 3

### 17. Buffer CSV writes
- **File:** `src/framework/model_configuration.py:744-782`
- **Issue:** File opened/closed every epoch; 5,000+ I/O ops in typical runs
- **Effort:** 30 min
- **Spec:** `02-training-pipeline-optimizations.md` Section 2

### 18. Make evaluation batch size configurable
- **File:** `src/framework/model_configuration.py:861`
- **Issue:** Hardcoded to 512
- **Effort:** 15 min
- **Spec:** `02-training-pipeline-optimizations.md` Section 4

---

## Phase 6: Configuration Robustness (medium impact, moderate effort)

### 19. Add semantic validation for config values
- **File:** `src/framework/utils/configuration_checks.py:153-207`
- **Issue:** Parameter values not validated; typos silently use defaults
- **Effort:** 1-2 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 6

### 20. Add missing config options (validation_frequency, eval_batch_size, etc.)
- **Files:** `src/framework/utils/parameters.py`, config checks
- **Issue:** Several useful parameters have no config support
- **Effort:** 2-4 hours (incremental)
- **Spec:** `04-dataset-and-config-improvements.md` Section 8

---

## Phase 7: Dataset & Preprocessing (medium impact, moderate effort)

### 21. Cache NetworkX graph conversions
- **File:** `src/datasets/utils/node_labeling.py:554-555`
- **Issue:** PyG-to-NetworkX conversion happens per graph per label type, never cached
- **Effort:** 1 hour
- **Spec:** `04-dataset-and-config-improvements.md` Section 4

### 22. Parallelize label computation across graphs
- **File:** `src/datasets/utils/node_labeling.py`
- **Issue:** Expensive label types (cycles, cliques) run sequentially
- **Effort:** 2-4 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 1

### 23. Reduce redundant label loading per run
- **File:** `src/framework/utils/preprocessing.py:284-320`
- **Issue:** Labels reloaded from disk for each fold/run
- **Effort:** 1 hour
- **Spec:** `02-training-pipeline-optimizations.md` Section 12

---

## Phase 8: Architecture Improvements (high impact, high effort)

### 24. Add ShareGNN batch processing support
- **Files:** `src/framework/model_configuration.py:798-807`, ShareGNN layer forward methods
- **Issue:** Per-graph Python loop; 10-50x slower than batched execution
- **Effort:** 1-2 days
- **Spec:** `01-sharegnn-optimizations.md` Section 1

### 25. Fix tensor shape convention consistency
- **Files:** `src/models/layers/framework_layer.py`, all layer implementations
- **Issue:** Documented (C,N,F) convention not followed; mixing 2D and 3D layers crashes
- **Effort:** 1-2 days
- **Spec:** `03-model-infrastructure-bugs.md` Section 6

### 26. Add residual projection for dimension mismatches
- **Files:** All conv layers
- **Issue:** Residual connections assume matching shapes without validation
- **Effort:** 2-4 hours
- **Spec:** `03-model-infrastructure-bugs.md` Section 7

### 27. Make BatchNorm handle 3D input
- **File:** `src/models/layers/nn_standard/batch_normalization.py:13-14`
- **Issue:** Only works with (N,F); fails on (C,N,F) from multi-head layers
- **Effort:** 30 min
- **Spec:** `03-model-infrastructure-bugs.md` Section 10

---

## Phase 9: Future Enhancements (when core is stable)

### 28. Add random search / Optuna for hyperparameter optimization
- **File:** `src/framework/run_configuration.py:138-146`
- **Issue:** Cartesian product grid search can explode combinatorially
- **Effort:** 4-8 hours
- **Spec:** `02-training-pipeline-optimizations.md` Section 7

### 29. Optimize joblib serialization for parallel runs
- **File:** `src/framework/core.py:100-105`
- **Issue:** Full graph_data serialized per parallel job
- **Effort:** 2-4 hours
- **Spec:** `02-training-pipeline-optimizations.md` Section 8

### 30. Add mixed-precision (AMP) training support
- **Effort:** 4-8 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 8

### 31. Add schema enforcement for YAML configs
- **Effort:** 4-8 hours
- **Spec:** `04-dataset-and-config-improvements.md` Section 7

### 32. Implement or remove empty positional encoding layer
- **File:** `src/models/ShareGNN/layers/inv_based_positional_encoding.py`
- **Issue:** Empty file; layer type registered in enum but has no code
- **Spec:** `01-sharegnn-optimizations.md` Section 9
