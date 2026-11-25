# Shared primary node tokens for ShareGNN transfer

## Objective

Give ShareGNN a primary node label with a consistent meaning across datasets. For molecular graphs, reuse the Random Walk Transformer's discrete per-atom encoding. For graphs without node labels, use a shared `UNLABELED` token. Keep meaningful domain labels where available. ShareGNN can consume these labels directly through its existing invariant-based heads. This proposal requires no walk sampling or transformer inference during ShareGNN training.

The transfer hypothesis has two parts. A primary-label head can reuse a weight for a chemical token observed in both source and target. A structure-only head can reuse a weight for the same structural role even when the two domains have disjoint primary tokens. Evaluate these claims separately.

## 1. Define the token vocabulary

First identify the reference tokenizer implementation and revision. Record its atom fields, normalization rules, hydrogen conventions, and treatment of missing or unsupported values. Take representative molecules from its examples and record the exact token emitted for each atom. These are the reference fixtures for the adapters. The exact atom schema and dataset coverage are open until this comparison is done.

| Node input | Primary token in shared mode | Sharing rule |
|---|---|---|
| Atom with sufficient chemical information | Reference atom token, qualified by tokenizer schema and version | Share across molecular datasets that reproduce the token |
| Node with no semantic label | One shared `UNLABELED` token | Share across datasets with genuinely unlabeled nodes |
| Node with a meaningful domain label | Domain vocabulary and decoded category | Share where the category definitions agree. Use a dataset-specific vocabulary otherwise |
| Atom with incomplete information | Reference fallback if defined. Otherwise retain an identifiable native atom category in its dataset vocabulary, or a separate unknown-atom token if none exists | Count these cases separately from exact atom matches |

Detect absent labels from the source schema. Integer zero can mean a valid atom category, a valid domain class, or a placeholder inserted by the loader. Preserve continuous node attributes independently of the primary token.

## 2. Add dataset adapters

Introduce shared tokens as an opt-in primary-label mode. Resolve each node's token during preprocessing while the source features and their meanings are available. For merged datasets, resolve tokens in each component before combining their vocabularies. Existing configs continue to select native labels by default.

| Source | First implementation check |
|---|---|
| OGB molecular and Peptides | Start here. Both loaders expose the nine OGB atom fields. Decode field values and compare per-atom tokens with the reference on the same molecules. |
| QM9 | Use the atomic numbers and molecular records. Check explicit hydrogens and atom ordering when matching reference tokens to graph nodes. |
| ZINC variants | Decode the benchmark atom-type table and bond information. Check whether each required distinction is recoverable. Record ambiguous cases. |
| Molecular TU and custom datasets | Verify the atom and bond codebooks per source. Retain native categories where a mapping is ambiguous. |
| Unlabeled and nonchemical datasets | Verify a genuinely unlabeled benchmark and a benchmark with meaningful categorical labels. Synthetic constant input features alone do not establish the meaning of a node label. |

The adapter output should include how many nodes received exact reference tokens, fallback tokens, native categories, and `UNLABELED`. Dataset usability and chemical-token coverage are separate results.

## 3. Connect the tokens to ShareGNN

Reuse the [canonical label hashing](../src/simplegnn/datasets/utils/label_hashing.py) and [checkpoint transfer](../src/simplegnn/framework/utils/transfer.py) machinery. Hash the token meaning, including its vocabulary and version. Keep dense row IDs local to each dataset. Match checkpoint rows by canonical hash, allowing a token to occupy different local rows in source and target. Keep missing labels and the frequency-truncation bucket distinct from `UNLABELED`. Include the primary-label mode and schema revision in cache identities for primary labels and every derived labeling that consumes them.

Start with two head types in the same model:

- A primary-token head in the invariant-based encoder or pooling layer learns atom and domain-specific weights.
- A structure-only head, such as unlabeled closed-walk counts, shares structural weights across molecular, categorical, and unlabeled graphs.

For example, a carbon atom and an unlabeled node can have the same closed-walk signature. Their primary rows differ, while the structure-only row can transfer. Test joint heads such as `(primary, closed_walks)` afterward. Their weights transfer when both components match. For derived labels and convolution heads, verify the semantics of every component, including edge labels and properties, before enabling cross-dataset row matching.

## 4. Verify correctness and measure transfer

1. **Token parity.** Compare each molecular adapter with the reference fixtures. Include equivalent atoms with different dataset-local IDs, distinct atom tokens that share an element, and incomplete records. Verify node permutation only permutes the token assignments.
2. **Weight transfer.** Save a checkpoint with distinguishable row weights. Load it into a target with reordered local IDs and one unseen token. Check the exact copied weights and fresh target row. Include a valid category zero, an unlabeled node, and two unrelated domain categories with the same integer ID. Exercise separate and merged datasets, cache reloads, and native-mode configs.
3. **Molecule-to-molecule experiment.** Start with a verified OGB molecular source and a Peptides target. Compare native primary labels plus structure, shared primary tokens plus structure, and structure alone. For each target architecture compare source initialization with training from scratch. Use a target-specific prediction head, matched budgets and seeds, and validation-based model selection.
4. **Molecule-to-unlabeled experiment.** Choose an unlabeled target with a predictive task and an established split. Report the effect of transferring the structure-only heads. Compare with the same target model trained from scratch. Include an unlabeled-to-unlabeled pair to exercise shared `UNLABELED` rows directly.
5. **Report coverage alongside scores.** Report exact tokenizer coverage, the fraction of target nodes whose token appeared in source training, matched parameter rows per head, and target performance by seed. Break out fallback and unlabeled coverage so a frequent placeholder cannot inflate the reported chemical overlap.

The first milestone is one verified molecular adapter pair, one unlabeled adapter, and a checkpoint test demonstrating both chemical and structural row reuse. Expand the adapter matrix after that milestone. Adopt shared tokens as a default for a benchmark only after the transfer comparison supports it.
