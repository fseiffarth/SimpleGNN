# 25 — Invariant layer load speedup

## Problem

On ZINC-full, model initialization stalls right after

```
Layer 1: estimated invariant index memory 10.3 GiB (151,201,923 shared pair rows, 2,371,647,955 pv entries)
```

That line is only the `_check_memory_budget` estimate; the time is spent in
`InvariantBasedMessagePassingLayer._build_distributions`. The ZINC-full
convolution layer has 16 head configs (15 on `distances` with 23 values,
1 on `edge_label_distances`), i.e. ~360 (head, property value) iterations, each
of which

1. `torch.load`s a fine-grained `(indices, counts)` cache file
   (`weights_only=False`, full unpickle + copy). `indices` is stored as
   **int64** although the number of uniques fits int32 (often int16) —
   ~15 GB read per layer build (`data/ZINC/caches` is 15 GB);
2. recomputes `cons.target_rows(key)` (`repeat_interleave` + `arange` +
   gathers, int64 over all rows of the value) — identical for all 15
   `distances` heads, so computed 15x redundantly;
3. remaps the indices in ~4 int64 full-length passes (`mapping[indices]`,
   `torch.where(indices >= 0, indices + base_offset, ...)`, `.to(int32)`)
   before the scatter into `pv`.

## Plan

### Step 0 — measure (done in this spec's first commit)

Per-phase wall-clock breakdown of the layer build, printed after
initialization when it takes >= 1 s (or always with `profile_layers: True`):
consolidation, cache load, cache-miss compute, threshold remap, target rows,
pv scatter, bias, `materialize_rows` (the optional precomputed-row cache) and
the rest of finalize. Splitting `initialize_model` into
`network_construction` / `network_to_device` belongs to the
`preprocessing_times` reporting work and lands with it.

Note: `parameters_ZINC_full.yml` forces `precompute_rows: true`, which
materializes **32.6 GiB** of assembled rows after the build (and moves them to
the GPU) — a second large init cost on top of the index build.

### Step 1 — bit-identical quick wins (no cache-format change)

- Memoize `target_rows` per (property description, value) for the duration of
  one build (int32; 151M rows fit), so shared descriptions compute it once.
- Fused lookup-table remap: one small int32 LUT of size `len(counts)` with
  `base_offset`, the threshold drop and the invalid bucket folded in;
  `pv[target] = lut[indices]` — one gather instead of ~4 int64 passes.
- `torch.load(..., mmap=True, weights_only=True)` for the indices cache
  (falls back to the old load for legacy files).
- Prefetch the next cache file on a background thread while the current one
  is processed.

Weight ordering, `weight_offset`, `_pv_*`, `_nw_*` and slot keys stay
identical, so checkpoints and transfer sidecars remain compatible.

### Step 2 — compact cache dtype (format bump, backward compatible)

Store `indices` in the smallest integer dtype that fits `len(counts)`
(int16/int32); legacy int64 files keep loading. 2–4x less I/O and disk.

### Step 3 (optional) — per-(description, source, target) cache in consolidated order

One int32 local index per consolidated row + per-value counts/uniques: one
mmap load and one gather per head instead of 23 loads + scatters. New cache
family; old per-value files orphaned (see `10-cache-cleanup.md`).

### Step 4 (optional) — reuse across runs

`run_configurations` and `run_best_configuration` (and every fold process)
rebuild the layer. Loading the opt-in `layer_distributions` cache with
`mmap=True` would make repeat builds nearly free.

## Testing

- Regression test: layer built with old vs. new code paths yields identical
  `_pv_*`, `_nw_*`, `weight_num`, `weight_offset`, slot keys (MUTAG + small
  ZINC subset); legacy int64 cache files still load.
- `pytest tests -q`; ZINC-full startup with step-0 timings before/after.
