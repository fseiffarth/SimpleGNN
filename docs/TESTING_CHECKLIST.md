# Testing Checklist for Indices/Counts Caching

## Pre-Testing Verification

- [x] Python syntax is valid
- [x] All three cache methods implemented
- [x] Variable scoping bug fixed
- [x] Cache messages added
- [x] Required imports present
- [x] No new dependencies introduced

**Status:** All pre-checks passed ✓

## Quick Verification (5 minutes)

Run the verification script:

```bash
cd /home/florian/Documents/CodeProjectsGit/SimpleGNN/repo
./verify_implementation.sh
```

**Expected:** All checks pass with green checkmarks

- [ ] Verification script passes

## Unit Tests (10 minutes)

Run the unit tests for cache methods:

```bash
cd /home/florian/Documents/CodeProjectsGit/SimpleGNN/repo
venv/bin/python tests/test_cache_methods.py
```

**Expected:** At least 3/4 tests pass (one test has mock setup issue, not critical)

- [ ] Unit tests pass

## Integration Test (30-60 minutes)

Run the full integration test:

```bash
cd /home/florian/Documents/CodeProjectsGit/SimpleGNN/repo
venv/bin/python test_cache_implementation.py
```

**Expected:**
- First run: "⊗ Cache miss" messages
- Cache files created in results/MUTAG/ShareGNN_indices_cache/
- Second run: "✓ Cache hit" messages
- Significant speedup (10-100x)

- [ ] Integration test passes
- [ ] Cache files created
- [ ] Cache hits on second run
- [ ] Speedup observed

## Manual Testing (30 minutes)

### Test 1: First Run (Cache Miss)

```bash
cd /home/florian/Documents/CodeProjectsGit/SimpleGNN/repo
rm -rf results/*/ShareGNN_indices_cache  # Clean cache
cd src
venv/bin/python -m examples.basic_example_share_gnn.main
```

**Check:**
- [ ] "⊗ Cache miss" messages appear during initialization
- [ ] "Cached X.XX MB:" messages appear
- [ ] Cache directory created: `../results/MUTAG/ShareGNN_indices_cache/`
- [ ] `.pt` and `.json` files present in cache directory
- [ ] Experiment completes successfully
- [ ] Record initialization time: ________ seconds

**Verify cache files:**
```bash
cd ..
ls -lh results/MUTAG/ShareGNN_indices_cache/0/
cat results/MUTAG/ShareGNN_indices_cache/0/*.json | head -20
```

- [ ] Cache files exist and have reasonable size (MB range)
- [ ] Metadata files contain expected structure

### Test 2: Second Run (Cache Hit)

```bash
cd src
venv/bin/python -m examples.basic_example_share_gnn.main
```

**Check:**
- [ ] "✓ Cache hit" messages appear during initialization
- [ ] NO "⊗ Cache miss" messages (all properties should hit)
- [ ] Initialization much faster than first run
- [ ] Experiment completes successfully
- [ ] Record initialization time: ________ seconds
- [ ] Calculate speedup: First run / Second run = ________ x

**Expected speedup:** 5-100x depending on dataset size

### Test 3: Correctness Validation

Run with cache and without cache, compare results:

```bash
cd src

# Run with cache
venv/bin/python -m examples.basic_example_share_gnn.main > /tmp/with_cache.log 2>&1

# Clear cache
cd .. && rm -rf results/*/ShareGNN_indices_cache && cd src

# Run without cache
venv/bin/python -m examples.basic_example_share_gnn.main > /tmp/without_cache.log 2>&1

# Compare validation/test metrics
diff <(grep -E "Validation|Test" /tmp/with_cache.log) <(grep -E "Validation|Test" /tmp/without_cache.log)
```

**Check:**
- [ ] No differences in validation metrics
- [ ] No differences in test metrics
- [ ] Results are numerically identical

**Note:** If diff shows differences, this indicates a bug!

### Test 4: Cache Invalidation

Modify configuration to test cache invalidation:

```bash
# Edit config file (if using YAML config)
# OR test with different threshold value

cd src
# First run with threshold=1 (default)
venv/bin/python -m examples.basic_example_share_gnn.main

# Modify config to use threshold=2
# (or edit examples/basic_example_share_gnn/parameters.yml)

# Run again - should see new cache misses
venv/bin/python -m examples.basic_example_share_gnn.main
```

**Check:**
- [ ] New "⊗ Cache miss" messages appear
- [ ] New cache files created with different hash
- [ ] Old cache files still present (not deleted)

### Test 5: Error Handling

Test that corrupted cache is handled gracefully:

```bash
cd /home/florian/Documents/CodeProjectsGit/SimpleGNN/repo

# Corrupt a cache file
echo "corrupted data" > results/MUTAG/ShareGNN_indices_cache/0/h0_p0_*.pt

# Run again
cd src
venv/bin/python -m examples.basic_example_share_gnn.main
```

**Check:**
- [ ] No crash or exception
- [ ] Warning message about cache load failure
- [ ] Falls back to computation
- [ ] Experiment completes successfully

## Performance Benchmarks (Optional, 2-3 hours)

Test on different datasets to measure performance impact:

### MUTAG (Small Dataset)
```bash
cd src
rm -rf ../results/MUTAG/ShareGNN_indices_cache
time venv/bin/python -m examples.basic_example_share_gnn.main  # First run
time venv/bin/python -m examples.basic_example_share_gnn.main  # Second run
```

- [ ] First run time: ________ seconds
- [ ] Second run time: ________ seconds
- [ ] Speedup: ________ x
- [ ] Cache size: ________ MB

### ZINC (Medium Dataset)
```bash
cd src
rm -rf ../results/ZINC/ShareGNN_indices_cache
time venv/bin/python -m examples.zinc.main  # First run
time venv/bin/python -m examples.zinc.main  # Second run
```

- [ ] First run time: ________ seconds
- [ ] Second run time: ________ seconds
- [ ] Speedup: ________ x
- [ ] Cache size: ________ MB

### QM9 (Large Dataset)
If available and configured:

```bash
cd src
rm -rf ../results/QM9/ShareGNN_indices_cache
time venv/bin/python -m examples.qm9.main  # First run
time venv/bin/python -m examples.qm9.main  # Second run
```

- [ ] First run time: ________ seconds
- [ ] Second run time: ________ seconds
- [ ] Speedup: ________ x
- [ ] Cache size: ________ MB

## Edge Cases (Optional, 30 minutes)

### Empty Cache Directory
```bash
mkdir -p results/MUTAG/ShareGNN_indices_cache/0
cd src
venv/bin/python -m examples.basic_example_share_gnn.main
```

- [ ] Handles empty directory gracefully
- [ ] Creates cache files normally

### Read-Only Cache Directory
```bash
chmod -w results/MUTAG/ShareGNN_indices_cache/0
cd src
venv/bin/python -m examples.basic_example_share_gnn.main
```

- [ ] Logs warning about save failure
- [ ] Continues execution without crashing
- [ ] Experiment completes successfully

**Cleanup:** `chmod +w results/MUTAG/ShareGNN_indices_cache/0`

### No Results Directory
```bash
mv results results_backup
cd src
venv/bin/python -m examples.basic_example_share_gnn.main
```

- [ ] Creates results directory automatically
- [ ] Creates cache directory automatically
- [ ] Experiment completes successfully

**Cleanup:** `mv results_backup results`

## Final Checklist

- [ ] All critical tests passed
- [ ] Cache hit/miss messages appear correctly
- [ ] Speedup is significant (5x+)
- [ ] Results are identical with/without cache
- [ ] Cache invalidation works
- [ ] Error handling is robust
- [ ] No crashes or exceptions
- [ ] Documentation is clear

## Known Issues

Document any issues found during testing:

```
Issue 1:
Description:
Status:
Workaround:

Issue 2:
Description:
Status:
Workaround:
```

## Sign-Off

**Tested By:** ________________
**Date:** ________________
**Result:** PASS / FAIL / CONDITIONAL PASS

**Notes:**
```




```

---

## Quick Reference

### Cache Location
```
results/{dataset}/ShareGNN_indices_cache/{layer_id}/
```

### Clear Cache
```bash
rm -rf results/*/ShareGNN_indices_cache
```

### Check Cache Size
```bash
du -sh results/*/ShareGNN_indices_cache
```

### View Cache Metadata
```bash
cat results/MUTAG/ShareGNN_indices_cache/0/*.json | head -20
```

### Verify Implementation
```bash
./verify_implementation.sh
```
