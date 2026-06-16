# Roadmap: integrating modern GNNs into SimpleGNN

## Context
SimpleGNN today ships classical message-passing convs (GCN, GIN, GAT, GATv2, SAGE) plus
the proprietary ShareGNN invariant layers. The goal of this task is **not to implement a
specific model now**, but to produce a vetted roadmap: (a) document the *best entry point*
for adding new architectures, (b) catalog candidate "modern" GNNs (SIN/CIN/PIN, graph
transformers, and others), and (c) rank them by how easily they slot into the current
framework. Scope per user: **roadmap only — no code committed in this plan.**

Key environment note: the project `venv` interpreter is currently broken (dangling
`python3.13` symlink) but its packages are intact (torch 2.10 CPU, **PyG 2.7**). Any later
implementation/verification can use `/opt/blender-5.1.2-linux-x64/5.1/python/bin/python3.13`
with `PYTHONPATH=src:venv/lib/python3.13/site-packages`, or repoint the venv symlink.

## Integration surface (the best entry point)
The model is a **sequential stack**: `GraphModel.forward` runs `x = layer(x, batch_data)`
over an `nn.ModuleList` (`src/simplegnn/models/model.py:302-317`). Every layer subclasses
`FrameworkLayer` (`src/simplegnn/models/layers/framework_layer.py`); `in_features` is
auto-propagated from the previous layer's `out_features`
(`model.py:430-433`). Per-batch data available to a layer: `batch_data.x`,
`.edge_index`, `.edge_attributes`, `.batch` (+ ShareGNN `.node_labels`, `.properties`).

**The proven recipe to add a layer (4 touch points)** — used by GCN/GIN/GAT/SAGE:
1. `src/simplegnn/models/layers/utils/layer_types.py` — add a `LayerTypes` enum value.
2. `src/simplegnn/models/layers/utils/layer_loader.py` — add a `check_layer` branch with
   required keys (mirror the `GCN_CONVOLUTION` branch at lines 187-211).
3. `src/simplegnn/models/model.py:get_model_layer` (dispatch at lines 453-489) — import the
   wrapper and add an `elif ... return MyConv(layer_args).type(self.precision).requires_grad_(self.convolution_grad)`.
4. New wrapper `src/simplegnn/models/layers/mpnn_classical/<name>_conv.py` extending
   `GNNConvLayer` (`gnn_conv.py`); `forward(node_representation, batch_data)` calls a PyG
   conv with `batch_data.edge_index` (+ `.edge_attributes` / `.batch`). Pattern reference:
   `gin_conv.py` (incl. edge-feature/GINE path), `gat_conv.py` (multi-head), `global_pooling.py`.

YAML then references the new `layer_type`. Activation strings are exact-match only
(`framework_layer.py:281-296`).

**This 4-point recipe is the entry point** for everything that is "an MPNN-style layer
consuming `(x, edge_index[, edge_attr])`". Two structural gaps limit what fits cleanly:
- **No positional/structural encodings (PE/SE)** and no place to inject them — node `x`
  is built from `input_features` in `graph_dataset.preprocess_share_gnn_data`
  (`graph_dataset.py:736+`). Graph transformers need PE; this is the main enabler to build.
- **No complex/higher-order data structures** (no simplicial/cell adjacencies) — required
  by topological networks (CIN/SIN); the single node/edge tensor model can't represent them.

Confirmed available in the installed **PyG 2.7**: `GPSConv, TransformerConv, PNAConv,
GINEConv, ResGatedGraphConv, GeneralConv, GraphConv, SGConv, TAGConv, ARMAConv, ChebConv,
EdgeConv, GENConv, FiLMConv, SuperGATConv, GATv2Conv, SAGEConv`; transforms
`AddLaplacianEigenvectorPE, AddRandomWalkPE, VirtualNode`. Topological convs are **not** in
PyG core (need external libs).

## Architecture catalog, ranked by integration difficulty

### Tier 1 — Trivial (4-point recipe only; ~1 file/~30 lines each; no data-pipeline change)
| Architecture | PyG class | Notes |
|---|---|---|
| GatedGCN (Bresson&Laurent'17) | `ResGatedGraphConv` | edge-gated; optional `edge_dim`. |
| Graph Transformer (sparse) (Shi'20) | `TransformerConv` | local attention over edges; `heads`, optional `edge_dim`. Mirror `gat_conv.py` head handling. |
| GINE (full) | `GINEConv` | already half-wired via `gin_conv.py` `edge_dim`; expose as its own type. |
| GENConv / DeeperGCN (Li'20) | `GENConv` | deep residual MPNN. |
| Weighted GraphConv, SGC, TAGCN, ARMA, Cheb, FiLM, SuperGAT, GeneralConv, EdgeConv | resp. PyG classes | all consume `(x, edge_index)`; identical wrapper. |

### Tier 1.5 — Trivial layer + one small precompute
| Architecture | PyG class | Extra |
|---|---|---|
| PNA (Corso'20) | `PNAConv` | needs in-degree histogram (`deg`) computed once at preprocessing and passed to the layer. |

### Tier 2 — Moderate (needs the PE/SE enabler and/or dense-batch attention)
| Architecture | PyG class / approach | What must be built |
|---|---|---|
| **GraphGPS** (Rampášek'22) | `GPSConv` (MPNN+global attn per layer) | PE/SE preprocessing (LapPE via `AddLaplacianEigenvectorPE`, RWSE via `AddRandomWalkPE`) + a way to fold PE into `x` (extend `input_features` in `graph_dataset.py:736+`). Uses `batch` (already present). |
| Global Graph Transformer / SAN-style | custom layer using `to_dense_batch(x, batch)` + `nn.MultiheadAttention` | dense-batch + mask helper; PE as above. Self-contained `FrameworkLayer`. |
| Virtual-node augmentation | `VirtualNode` transform | cheap global-context add-on; complements any MPNN. |

### Tier 3 — Hard (new data model / pipeline; likely external deps)
| Architecture | Why hard |
|---|---|
| **CIN / CWN / CIN++** (cell networks, Bodnar'21) and **MPSN / "SIN"** (simplicial, Bodnar'21) | require lifting graphs to simplicial/cell complexes, computing boundary/upper/lower adjacencies and per-rank features, and multi-adjacency message passing. Cannot be expressed in the current single node/edge tensors or the sequential `x=layer(x,batch_data)` flow. Needs a new complex data container + lifting preprocessing + a parallel layer family; consider external libs (`cwn` by Bodnar et al., or TopoX / `TopoModelX`). |
| **Graphormer (full)** (Ying'21) | centrality + spatial (shortest-path) + edge attention biases; needs all-pairs SP precompute and a dense attention-bias path. |
| **"PIN"** | ambiguous — no single canonical "PIN" in the literature (likely a path-/persistence-based topological isomorphism network). Tentatively Tier 3; see Open question. |

## Recommended phased roadmap (for later execution)
- **Phase A — prove the entry point (Tier 1).** Add 2–3 PyG convs (suggest `TransformerConv`,
  `PNAConv`, `ResGatedGraphConv`) via the 4-point recipe + example model YAMLs; smoke-train on
  a small TU dataset (MUTAG/DHFR).
- **Phase B — transformer enabler (Tier 2).** Build reusable PE/SE preprocessing (LapPE, RWSE)
  and an `input_features` path to concatenate PE into `x`; wrap `GPSConv` and a dense global
  `TransformerConv`. This unlocks the whole transformer family.
- **Phase C — topological (Tier 3, stretch).** Design a complex data container + graph→complex
  lifting; integrate CIN/SIN, ideally via an external lib, as a *parallel* model family rather
  than forcing it through the sequential `GraphModel`.

## Cross-cutting enablers to build (shared across phases)
- PE/SE infrastructure (LapPE, RWSE) + injection into node features (reuse PyG transforms).
- In-degree histogram precompute (PNA).
- Dense-batch attention helper (`to_dense_batch` / `to_dense_adj`) for global transformers.
- (Already present) edge-feature plumbing via `edge_attributes` and `edge_dim`.

## Verification approach (when a layer is actually implemented)
1. Config-load check: construct `FrameworkMain(main.yml)` and expand `get_run_configs` for a
   model YAML using the new `layer_type` (catches missing `check_layer` keys / dispatch).
2. Smoke train: one short run on MUTAG or DHFR; confirm forward/backward and a results CSV.
3. `pytest tests -q` stays green. Run via the recovered interpreter (venv interpreter broken).

## Open question (non-blocking; resolve before Phase C)
"PIN" is ambiguous. Confirm the intended paper (e.g., a path-based or persistence-based
isomorphism network) so it can be placed precisely; currently catalogued under Tier 3.

## Critical files (touch points for any future implementation)
- `src/simplegnn/models/layers/utils/layer_types.py` — enum.
- `src/simplegnn/models/layers/utils/layer_loader.py` — `check_layer` validation (lines 187-211 pattern).
- `src/simplegnn/models/model.py` — `get_model_layer` dispatch (lines 453-489) + imports.
- `src/simplegnn/models/layers/mpnn_classical/*_conv.py` + `gnn_conv.py` — wrapper pattern.
- `src/simplegnn/datasets/graph_dataset.py` (`preprocess_share_gnn_data`, ~736) — PE/feature injection point for transformers.
