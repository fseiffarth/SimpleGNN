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

### Step 0 measurements (ZINC-full, CPU, warm caches, `precompute_rows: false`)

| Phase | Time |
|---|---:|
| Network construction (all layers) | 235.0 s |
| Layer 1 (convolution) build | 51.9 s |
| ↳ bias | 20.2 s |
| ↳ target_rows | 17.5 s |
| ↳ scatter | 5.7 s |
| ↳ cache_load (316 files) | 3.6 s |
| ↳ consolidate | 2.7 s |
| ↳ remap | 1.5 s |

cProfile: 32 calls of `torch.unique(..., dim=0)` (`unique_dim`) took
**212.6 s** — one per head in the positional-encoding layer (14), the
aggregation layer (16) and the conv bias. On ZINC-full's 5.8M nodes the
row-wise `dim=0` path costs ~7–8 s per call versus ~0.1 s for the 1-D sort
unique, which yields the identical inverse for 1-D labels. Cache I/O is cheap
(3.6 s), so prefetching was dropped from step 1.

### Step 1 — bit-identical quick wins (no cache-format change)

- `label_inverse()` (`models/ShareGNN/utils.py`): 1-D sort unique instead of
  `unique(dim=0)` for the conv bias, aggregation and positional-encoding
  index vectors — the dominant cost.
- Memoize `target_rows` per (property description, value) for the duration of
  one build (int32; 151M rows fit), so shared descriptions compute it once.
- Fused lookup-table remap: one small int32 LUT of size `len(counts)` with
  `base_offset`, the threshold drop and the invalid bucket folded in;
  `pv[target] = lut[indices]` — one gather instead of ~4 int64 passes.
- `torch.load(..., mmap=True, weights_only=True)` for the indices cache
  (falls back to the old load for legacy files).

Weight ordering, `weight_offset`, `_pv_*`, `_nw_*` and slot keys stay
identical, so checkpoints and transfer sidecars remain compatible.

Result (same setup as the step 0 measurements):

| Phase | Step 0 | Step 1 |
|---|---:|---:|
| Network construction (all layers) | 235.0 s | **11.8 s** |
| Layer 1 build | 51.9 s | 8.8 s |
| ↳ bias | 20.2 s | 0.2 s |
| ↳ target_rows | 17.5 s | 1.4 s |
| ↳ scatter | 5.7 s | 3.8 s |
| ↳ cache_load | 3.6 s | 0.4 s |
| Peak RSS (whole run) | 20.1 GB | 19.1 GB |

What remains in layer 1 is the irreducible pv scatter (2.4B entries) and the
one-time consolidation; steps 2–3 would mainly save disk space now.

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

### Steps 2–4 — measured, not implemented

After step 1 the remaining layer-1 build is ~9 s; step 2 would save < 0.5 s
(disk only), step 3 ~3–4 s at the cost of a second cache family. Step 4
(opt-in `layer_distributions` cache) measured: +5.2 s and a 9.5 GB file on the
first run, 4.1 s saved per later build (~7 s net per ZINC-full experiment of
4 builds); mmap does not help since `net.to(device)` reads the full file.
All three left out.

### Step 5 — materialized rows (`precompute_rows: true`)

First CUDA run after step 1 (RTX 6000 + Threadripper):
`materialize_rows 62.81s` of a 77 s network construction, `move to cuda
4.45s`. The old `_materialize_rows` concatenated int64 row parts per graph
chunk and stable-argsorted all 2.19B rows by graph.

- Direct placement: the layout is fully determined by the per-(head-config,
  graph) row counts, so every row's destination is computed and written once.
  Bit-identical; 57.5 s -> 32.4 s on CPU (ZINC-full), peak RSS 56.3 -> 53.2 GB.
- Build on the GPU: on CUDA runs `_finalize_initialization` only records the
  row count; the layer's `_apply` override builds the rows right after
  `net.to(device)` on the device (lazy fallback in `_assemble_rows` if the net
  never reaches a GPU). Avoids the 32.6 GiB host buffer and most of the
  host-to-device copy. Same fits check as before; GPU memory in steady state
  is unchanged (the rows lived on the GPU before, too).
