# 10 — ShareGNN Cache Cleanup

Date: 2026-07-15. Follow-up to the caching added in spec 08 (fable invariant-layer
optimization), triggered by 42 GB of cache files in `data/ZINC/caches/` and
confusing terminal behavior (ZINC runs silent, ZINC-full runs spamming per-key
cache-hit lines).

## Diagnosis

`InvariantBasedMessagePassingLayer` has two cache tiers:

1. **Coarse per-layer cache** (`layerdist_<hash>.pt`): the fully merged
   weight/bias distribution tensors. Hits were **silent**; only the
   fine-grained tier printed. Files are 1.8–5.3 GB each on ZINC; the key hashes
   the head list structurally, so a semantically neutral config edit (e.g. two
   identical `num: 1` heads → one `num: 2` head) orphans the old file. Three
   byte-identical ZINC caches existed under different hashes. No eviction.
2. **Fine-grained (indices, counts) cache** (`<hash>.pt`, per head × property
   value): the `torch.unique` inverse/counts over all node pairs. This is the
   tier that saves real compute; its key excludes thresholds/layer_id by
   design, so it hits across config variants.

ZINC was silent because the coarse tier hit (skipping the build and all
prints); ZINC-full never had a usable coarse cache (would be ~50 GB), so every
run rebuilt and printed one ✓ line per (head, property value).

Both tiers wrote with plain `torch.save` (not atomic) while up to 30 joblib
workers share the same cache paths → occasional truncated files, "⚠ failed to
load" warnings, recompute ("sometimes not working").

## Changes (all in `src/simplegnn/models/ShareGNN/layers/inv_based_message_passing.py`)

- **Coarse cache is now opt-in**: only used when the hyperparameter config
  contains `cache: { layer_distributions: True }`. Default off — the rebuild
  from fine-cache hits is cheap tensor ops since the build was vectorized.
- **Coarse hits now log one line** (`✓ Layer N distribution cache hit: …`), so
  silence is never ambiguous.
- **All cache writes are atomic** (`tmp.<pid>` + `os.replace`), fixing the
  joblib-worker race on both `.pt` and `.json` files.
- **Sidecar `.json` files now store the exact cache-key dict** behind the
  filename hash, so unexpected misses can be diagnosed by diffing sidecars.
- **Fine-grained per-key ✓ prints replaced by a per-layer summary**
  (`Layer N indices cache: H hits, M misses`); miss lines are only printed for
  corrupt/format-mismatch files (a plain missing file is the normal cold path).
- Bug fixes found in the audit:
  - `torch.unique(..., sorted=True)` where `counts[-1] = 0` marks the invalid
    bucket — with `sorted=False` the "invalid bucket is last" assumption was an
    undocumented CPU implementation detail.
  - Coarse-cache load validates tensor types **before** `register_buffer`; a
    partial failure between the two registrations previously crashed the
    rebuild fallback (duplicate buffer registration).
  - `labeled_subdict` (a multi-million-row gather per property key) is now
    built only on a cache miss, not before the try.

### Related non-cache fixes landed in the same pass

- `generate_layer_options` (`framework/run_configuration.py`): grid options
  aliased one shared dict → every label-grid option silently became a copy of
  the last combination. Now copies per option.
- `config_paths_to_absolute` (`utils/path_conversions.py`): unconditionally
  clobbered `paths['labels']`/`paths['properties']` with `''` before the
  membership check (currently uncalled in src, but fixed).
- `load_model` / `load_model_old`: validate model paths before the expensive
  dataset preprocessing.
- Stale unit tests fixed accordingly (`tests/test_load_model_path_resolution_unit.py`
  imports the submodule to dodge the package-attr shadowing; `combine_node_labels`
  and `generate_layer_options` expectations corrected).

## Operational notes

- `data/<source>/caches/` is always safe to delete; everything regenerates on
  the next run. (`rm -rf data/ZINC/caches data/TUDatasets/caches` frees ~43 GB.)
- **Delete the caches after regenerating labels/properties**: fine-cache keys
  identify labels by *description string*, not content. If label or property
  files are rebuilt with different content under the same name and dataset
  size, stale caches would be silently reused.
- To get the old always-on coarse caching back for a small dataset, add
  `cache: { layer_distributions: True }` to that experiment's parameters yml.
