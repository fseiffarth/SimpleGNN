# Synthetic dataset splits (base_paper)

These are the **original paper splits** for the synthetic datasets, not regenerated.
They were recovered from the predecessor repository (`RuleGNN`, the old
`paper_experiments/` layout that `experiments/base_paper/` is a snapshot of) and
copied here verbatim. The framework now requires the split JSON to exist at
`FrameworkMain` construction (`configuration_checks.py`), and these files are
referenced directly via `paths.splits` in the migrated synthetic main configs.

| File | Dataset `source` | Folds | Graphs | Source of truth |
|------|------------------|-------|--------|-----------------|
| `CSL_splits.json` | `gnn_benchmark` | 5 | 150 | `RuleGNN/paper_experiments/Data/Splits/CSL_splits.json` |
| `EvenOddRings2_16_splits.json` | `generate_from_function` | 10 | 1200 | `RuleGNN/Data/Splits/` |
| `EvenOddRingsCount16_splits.json` | `generate_from_function` | 10 | 1200 | `RuleGNN/Data/Splits/` |
| `LongRings100_splits.json` | `generate_from_function` | 10 | 1200 | `RuleGNN/Data/Splits/` |
| `Snowflakes_splits.json` | `generate_from_function` | 10 | 1000 | `RuleGNN/Data/Splits/` |

## Notes

- **CSL** specifically uses the `paper_experiments/Data/Splits` copy. Every other
  CSL copy found in the predecessor repos had the validation set overlapping the
  test set (a known defect in some CSL split versions), which `load_splits`
  rejects as non-disjoint. The `paper_experiments` version is the only clean,
  disjoint one and matches this folder's provenance.
- Format matches the framework contract (`load_splits`,
  `src/simplegnn/framework/utils/preprocessing.py`): a list with one entry per
  fold, each `{ "test": [...], "model_selection": [ { "train": [...],
  "validation": [...] } ] }`.
- Validated: per fold the train/validation/test indices are pairwise disjoint and
  their union is exactly `0 .. N-1` for the dataset size `N` above.
- The synthetic datasets are generated deterministically (seed `764`, args copied
  from the old configs), so these index-based splits remain stable across
  regenerations of the underlying graphs.
