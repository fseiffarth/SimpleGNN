# Adding Custom Node Labeling Functions

## Overview

The SimpleGNN framework's ShareGNN layers use node labeling functions to create structural features for invariant-based message passing. Node labels capture graph-theoretic properties (degree, cycles, cliques, etc.) that help the model learn meaningful representations.

**Existing label types:**
- `primary`: Original node features from dataset
- `trivial`: All nodes labeled 0 (baseline)
- `index`: Unique label per node position
- `degree`: Node degree
- `wl`: Weisfeiler-Leman iteration
- `labeled_degree`: Degree with original labels
- `wl_labeled`: WL with original labels
- `wl_labeled_edges`: WL with edge labels
- `cycle`: Cycle counting
- `subgraph`: Subgraph pattern matching
- `clique`: Clique membership

**When to create custom labeling functions:**
- Your domain requires specific structural features (e.g., betweenness centrality, PageRank)
- Existing labels don't capture the graph properties you need
- You want to experiment with novel graph-theoretic metrics

**When to combine existing labels instead:**
- Multiple existing label types can be used together in a single layer
- The `combine_node_labels()` function concatenates labels
- Example: `['degree', 'wl']` provides both local and global structural information

## System Architecture

### Two-File Modification Pattern

Adding a custom labeling function requires changes to two files:

1. **`src/datasets/utils/node_labeling.py`**: Define the labeling class and save function
2. **`src/models/ShareGNN/preprocessing/preprocessing.py`**: Register the label type in the dispatcher

### NodeLabelingBase Abstract Class

All labeling functions extend `NodeLabelingBase`, which provides:

```python
class NodeLabelingBase(ABC):
    def __init__(self, base_name, graph_data, label_path=None,
                 max_labels=None, optional_parameters=None, save_times=None):
        # Sets up label naming, paths, and parameter tracking

    @abstractmethod
    def generate(self) -> List[List[int]]:
        # Your labeling algorithm - returns list of label lists

    def create_and_save_labels(self) -> str:
        # Generates, discretizes (if needed), and saves labels to .pt file
```

**Key responsibilities:**
- **`__init__`**: Configure label naming and optional parameters
- **`generate()`**: Compute raw labels (can be continuous or discrete)
- **`create_and_save_labels()`**: Orchestrates generation, discretization, and saving

### Caching Mechanism

Labels are saved as PyTorch `.pt` files in `data/labels/<dataset>/`:

```python
# File: MUTAG_labels_betweenness_10.pt
{
    'dataset_name': 'MUTAG',
    'label_name': 'betweenness_10',
    'node_labels': torch.Tensor([[orig_label, freq_sorted_label], ...])
}
```

**File naming convention:** `{dataset}_labels_{base_name}_{params}.pt`

**Caching behavior:**
- If `.pt` file exists, labels are loaded (no recomputation)
- Changing parameters generates a new file with different name
- Delete cached files to force regeneration

### Frequency-Based Discretization

The `relabel_node_labels()` function automatically discretizes labels:

1. Count frequency of each unique label value across dataset
2. Sort by frequency (most frequent = label 0, next = label 1, etc.)
3. If `max_labels` is set, keep only top-k most frequent, set rest to -1
4. Returns `(original_label, frequency_sorted_label)` pairs

**Purpose:** Ensures low-index labels represent common structural patterns (more important for learning).

## Step-by-Step Implementation Guide

### Step 1: Define the Labeling Class

**Location:** `src/datasets/utils/node_labeling.py`

Add your class after the existing labeling classes (around line 700-800):

```python
class CustomNodeLabeling(NodeLabelingBase):
    """
    Brief description of what this labeling measures.

    Parameters
    ----------
    graph_data : GraphDataset
        The dataset to label.
    label_path : Path, optional
        Directory where labels will be saved.
    max_labels : int, optional
        Maximum number of distinct labels (for frequency-based filtering).
    custom_param : type, optional
        Description of your custom parameter.
    save_times : Path, optional
        Path to file for logging generation times.

    Notes
    -----
    - Add any important algorithmic details
    - Explain discretization strategy if applicable
    """
    def __init__(self, graph_data, label_path=None, max_labels=None,
                 custom_param=None, save_times=None):
        # Register optional parameters for file naming
        optional_params = []
        if custom_param is not None:
            optional_params.append(('custom_param', custom_param))

        super().__init__(
            base_name='custom',  # Used in filename
            graph_data=graph_data,
            label_path=label_path,
            max_labels=max_labels,
            optional_parameters=optional_params,
            save_times=save_times
        )

        # Store instance variables
        self.custom_param = custom_param

    def generate(self) -> List[List[int]]:
        """
        Compute labels for all nodes in all graphs.

        Returns
        -------
        List[List[int]]
            List of label lists, one per graph. Each inner list contains
            one label per node (in node order).
        """
        # Ensure NetworkX graphs are available (if needed)
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)

        # Compute labels for each graph
        node_labels = []
        for graph in self.graph_data.nx_graphs:
            # Your labeling algorithm here
            labels = []
            for node in graph.nodes():
                label = compute_your_metric(graph, node, self.custom_param)
                labels.append(label)
            node_labels.append(labels)

        return node_labels
```

**Key design decisions:**

1. **`base_name`**: Short identifier for filename (e.g., 'degree', 'wl', 'betweenness')
2. **`optional_parameters`**: List of `(name, value)` tuples appended to filename
   - Example: `optional_parameters=[('bins', 10)]` → `custom_10.pt`
3. **NetworkX graphs**: Access via `self.graph_data.nx_graphs`
   - Call `create_nx_graphs(directed=False)` if not already created
4. **Return format**: List of lists matching graph/node structure
   - `node_labels[i][j]` = label for node j in graph i

### Step 2: Create the Save Function

**Location:** Same file (`node_labeling.py`), right after your class:

```python
def save_custom_labels(graph_data, label_path=None, max_labels=None,
                      custom_param=None, save_times=None):
    """
    Generate and save custom node labels.

    Parameters
    ----------
    graph_data : GraphDataset
        Graph dataset to label.
    label_path : Path, optional
        Directory where labels will be saved.
    max_labels : int, optional
        Maximum number of distinct labels.
    custom_param : type, optional
        Description of your custom parameter.
    save_times : Path, optional
        Path to file for logging generation times.

    Returns
    -------
    str
        Path to the saved .pt file.

    See Also
    --------
    CustomNodeLabeling : Implementation details
    """
    labeling = CustomNodeLabeling(graph_data, label_path, max_labels,
                                  custom_param, save_times)
    return labeling.create_and_save_labels()
```

**Purpose:** Provides a simple function interface for the preprocessing dispatcher.

### Step 3: Register in Preprocessing Dispatcher

**Location:** `src/models/ShareGNN/preprocessing/preprocessing.py`

**Add import** at the top of the file (around line 6):

```python
from datasets.utils.node_labeling import (
    get_label_string, save_labels_to_file, save_primary_labels,
    save_trivial_labels, save_index_labels, save_degree_labels,
    save_wl_labels, save_labeled_degree_labels, save_wl_labeled_labels,
    save_wl_labeled_edges_labels, save_cycle_labels, save_subgraph_labels,
    save_clique_labels,
    save_custom_labels,  # Add your function here
    load_labels, combine_node_labels
)
```

**Add elif branch** in the `layer_to_labels()` function (around line 174):

```python
def layer_to_labels(layer, graph_data, label_path, generation_times_labels_path):
    # ... existing if/elif branches for other label types ...

    elif layer['label_type'] == 'custom':
        # Set default values for optional parameters
        if 'max_labels' not in layer:
            layer['max_labels'] = None
        if 'custom_param' not in layer:
            layer['custom_param'] = None

        # Call your save function
        file_path = save_custom_labels(
            graph_data=graph_data,
            label_path=label_path,
            max_labels=layer.get('max_labels', None),
            custom_param=layer.get('custom_param', None),
            save_times=generation_times_labels_path
        )

    # ... existing else block ...
```

**Parameter handling:**
- Use `layer.get('param_name', default_value)` to safely extract YAML parameters
- Set defaults for optional parameters that might not be in config
- Pass parameters to your save function in the same order as the function signature

## Handling Continuous Values (Discretization)

Graph neural networks require discrete labels as input. If your metric produces continuous values (e.g., centrality scores), you need to discretize them.

### Approach 1: Use Framework's Frequency-Based Discretization

**When to use:**
- Values are naturally discrete or ordinal
- You want the most frequent values prioritized (lower label indices)
- You want consistent behavior with other label types

**How it works:**

1. Return raw labels from `generate()` (can be int or float)
2. The framework's `relabel_node_labels()` automatically discretizes
3. Set `max_labels` in YAML to cap number of distinct labels

**Example:**

```python
def generate(self) -> List[List[int]]:
    # Return discrete values directly
    node_labels = []
    for graph in self.graph_data.nx_graphs:
        labels = [graph.degree(node) for node in graph.nodes()]  # Discrete
        node_labels.append(labels)
    return node_labels
```

**YAML usage:**

```yaml
head:
  labels:
    head:
      - label_type: custom
        max_labels: 20  # Keep only top-20 most frequent values
```

### Approach 2: Manual Binning Before Saving

**When to use:**
- Values are continuous (centrality, scores, probabilities)
- You want control over bin boundaries
- You need domain-specific binning (e.g., percentiles, log-scale)

**How it works:**

1. Collect all values across dataset
2. Compute bins using your chosen strategy
3. Discretize values in `generate()` before returning

**Example: Percentile-Based Binning**

```python
def generate(self) -> List[List[int]]:
    import numpy as np

    if self.graph_data.nx_graphs is None:
        self.graph_data.create_nx_graphs(directed=False)

    # Step 1: Collect all values across dataset
    all_values = []
    graph_values = []

    for graph in self.graph_data.nx_graphs:
        values = [compute_continuous_metric(graph, node) for node in graph.nodes()]
        graph_values.append(values)
        all_values.extend(values)

    # Step 2: Compute percentile bins
    num_bins = self.max_labels or 10
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(all_values, percentiles)

    # Handle edge case: all values identical
    if len(set(bins)) == 1:
        return [[0] * len(vals) for vals in graph_values]

    # Step 3: Discretize each graph using global bins
    node_labels = []
    for values in graph_values:
        # digitize returns 1-indexed bins, subtract 1 for 0-indexing
        labels = np.digitize(values, bins[1:-1], right=False)
        node_labels.append(labels.tolist())

    return node_labels
```

**Alternative binning strategies:**

**Equal-width bins:**
```python
min_val, max_val = min(all_values), max(all_values)
bins = np.linspace(min_val, max_val, num_bins + 1)
```

**Log-scale bins (for heavy-tailed distributions):**
```python
log_vals = np.log1p(all_values)  # log(1 + x) avoids log(0)
log_bins = np.linspace(log_vals.min(), log_vals.max(), num_bins + 1)
bins = np.expm1(log_bins)  # exp(x) - 1
```

**Custom thresholds:**
```python
bins = [0, 0.1, 0.5, 0.9, 1.0]  # Domain-specific boundaries
```

## Testing Your Implementation

### Basic Testing

1. **Create a minimal test configuration:**

```yaml
# examples/test_custom/main.yml
datasets:
  - MUTAG

task: graph_classification

model_config_paths:
  - examples/test_custom/model.yml

parameter_config_paths:
  - examples/test_custom/parameters.yml

# ... other standard config ...
```

```yaml
# examples/test_custom/model.yml
model_configs:
  - head:
      labels:
        head:
          - label_type: custom
            max_labels: 20
            custom_param: test_value
      architecture:
        head:
          - layer_type: inv_based_message_passing
            # ... rest of layer config ...
```

2. **Run preprocessing:**

```python
from pathlib import Path
from framework.core import FrameworkMain

experiment = FrameworkMain(Path('examples/test_custom/main.yml'))
experiment.preprocessing(num_threads=1)
```

3. **Verify `.pt` file creation:**

```bash
ls data/labels/MUTAG/
# Should see: MUTAG_labels_custom_<params>.pt
```

4. **Inspect labels:**

```python
import torch

labels = torch.load('data/labels/MUTAG/MUTAG_labels_custom_<params>.pt')
print(labels.keys())  # ['dataset_name', 'label_name', 'node_labels']
print(labels['node_labels'].shape)  # (N, 2) where N = total nodes
print(labels['node_labels'][:10])  # Check first 10 labels
```

### Integration Testing

1. **Add layer to full ShareGNN model config**
2. **Run complete training pipeline:**

```python
experiment.run_configurations(num_threads=-1)
experiment.evaluate_results()
```

3. **Verify:**
   - No errors during label loading
   - Model initializes successfully
   - Training runs without crashes
   - Results are reasonable (not NaN loss)

### Validation Checklist

- [ ] `.pt` file created in correct directory
- [ ] File contains `dataset_name`, `label_name`, `node_labels` keys
- [ ] `node_labels` tensor shape is `(N, 2)` where N = total nodes across dataset
- [ ] Labels are non-negative integers (except -1 for filtered labels)
- [ ] Label distribution makes sense (check with `torch.bincount()`)
- [ ] Caching works: second preprocessing run skips recomputation
- [ ] Changing parameters generates new file
- [ ] Model training completes successfully

### Debugging Tips

**Problem: Labels are all the same value**
- Check if your metric is computed correctly
- Verify NetworkX graph creation (directed vs undirected)
- Inspect a few graphs manually to ensure variation

**Problem: Too many distinct labels**
- Use `max_labels` to cap frequency-based filtering
- Consider manual binning for continuous values
- Check for numerical precision issues (e.g., float equality)

**Problem: Labels are -1**
- These are filtered out by `max_labels` (too infrequent)
- Increase `max_labels` or adjust discretization strategy

**Problem: File not being cached**
- Check filename includes all parameters
- Verify `optional_parameters` list is correct
- Look for typos in parameter names

## Example: Betweenness Centrality Implementation

This complete example implements betweenness centrality as a node labeling function. Betweenness centrality measures how often a node lies on shortest paths between other nodes—high values indicate "bridge" nodes connecting different graph regions.

### Implementation in node_labeling.py

```python
class BetweennessCentralityNodeLabeling(NodeLabelingBase):
    """
    Node labeling based on betweenness centrality.

    Betweenness centrality measures how often a node lies on shortest paths
    between other nodes. High values indicate "bridge" nodes that connect
    different parts of the graph.

    The continuous centrality values are discretized into bins using
    percentile-based binning for meaningful distribution across labels.

    Parameters
    ----------
    graph_data : GraphDataset
        The dataset to label.
    label_path : Path, optional
        Directory where labels will be saved.
    max_labels : int, optional
        Maximum number of distinct labels (for frequency-based filtering after binning).
    num_bins : int, optional
        Number of bins for discretization. If not specified, uses max_labels
        or defaults to 10.
    save_times : Path, optional
        Path to file for logging generation times.

    Notes
    -----
    - Centrality values range from 0.0 (peripheral) to 1.0 (central bridge)
    - Percentile-based binning ensures balanced label distribution
    - Nodes with same centrality receive same label
    - Computational complexity: O(n³) for dense graphs, O(nm) for sparse graphs
    """
    def __init__(self, graph_data, label_path=None, max_labels=None,
                 num_bins=None, save_times=None):
        optional_params = []
        if num_bins is not None:
            optional_params.append(('bins', num_bins))

        super().__init__(
            base_name='betweenness',
            graph_data=graph_data,
            label_path=label_path,
            max_labels=max_labels,
            optional_parameters=optional_params,
            save_times=save_times
        )

        self.num_bins = num_bins or max_labels or 10

    def generate(self) -> List[List[int]]:
        """
        Compute betweenness centrality labels for all nodes.

        Returns
        -------
        List[List[int]]
            List of discretized centrality labels, one list per graph.
        """
        if self.graph_data.nx_graphs is None:
            self.graph_data.create_nx_graphs(directed=False)

        # Step 1: Collect all betweenness centrality values across dataset
        all_centralities = []
        graph_centralities = []

        for graph in self.graph_data.nx_graphs:
            # Compute betweenness centrality for all nodes
            centrality = nx.betweenness_centrality(graph, normalized=True)
            # Store as list maintaining node order
            cent_list = [centrality[node] for node in graph.nodes()]
            graph_centralities.append(cent_list)
            all_centralities.extend(cent_list)

        # Step 2: Compute percentile-based bins across entire dataset
        import numpy as np
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        bins = np.percentile(all_centralities, percentiles)

        # Handle edge case where all values are identical
        if len(set(bins)) == 1:
            return [[0] * len(gc) for gc in graph_centralities]

        # Step 3: Discretize each graph using the global bins
        node_labels = []
        for cent_list in graph_centralities:
            # digitize returns 1-indexed bins, subtract 1 for 0-indexing
            # Example: bins=[0, 0.2, 0.5, 1.0] with num_bins=3 gives labels [0, 1, 2]
            labels = np.digitize(cent_list, bins[1:-1], right=False)
            node_labels.append(labels.tolist())

        return node_labels


def save_betweenness_centrality_labels(graph_data, label_path=None,
                                       max_labels=None, num_bins=None,
                                       save_times=None):
    """
    Generate and save betweenness centrality-based node labels.

    Parameters
    ----------
    graph_data : GraphDataset
        Graph dataset to label.
    label_path : Path, optional
        Directory where labels will be saved.
    max_labels : int, optional
        Maximum number of distinct labels (used as num_bins if num_bins not set).
    num_bins : int, optional
        Number of bins for discretization (overrides max_labels if set).
    save_times : Path, optional
        Path to file for logging generation times.

    Returns
    -------
    str
        Path to the saved .pt file.

    See Also
    --------
    BetweennessCentralityNodeLabeling : Implementation details
    """
    labeling = BetweennessCentralityNodeLabeling(
        graph_data, label_path, max_labels, num_bins, save_times
    )
    return labeling.create_and_save_labels()
```

### Registration in preprocessing.py

**Add import:**

```python
from datasets.utils.node_labeling import (
    get_label_string, save_labels_to_file, save_primary_labels,
    save_trivial_labels, save_index_labels, save_degree_labels,
    save_wl_labels, save_labeled_degree_labels, save_wl_labeled_labels,
    save_wl_labeled_edges_labels, save_cycle_labels, save_subgraph_labels,
    save_clique_labels,
    save_betweenness_centrality_labels,  # Add this
    load_labels, combine_node_labels
)
```

**Add elif branch in `layer_to_labels()` function:**

```python
elif layer['label_type'] == 'betweenness_centrality':
    if 'max_labels' not in layer:
        layer['max_labels'] = None
    if 'num_bins' not in layer:
        layer['num_bins'] = None

    file_path = save_betweenness_centrality_labels(
        graph_data=graph_data,
        label_path=label_path,
        max_labels=layer.get('max_labels', None),
        num_bins=layer.get('num_bins', None),
        save_times=generation_times_labels_path
    )
```

### YAML Configuration Example

```yaml
# In model config
head:
  labels:
    head:
      - label_type: betweenness_centrality
        num_bins: 20          # Discretize into 20 percentile bins
        max_labels: 100       # Optional: further filter to 100 most frequent
```

### Expected Output

After running preprocessing:

```bash
data/labels/MUTAG/
└── MUTAG_labels_betweenness_20.pt
```

Loading and inspecting:

```python
import torch
labels = torch.load('data/labels/MUTAG/MUTAG_labels_betweenness_20.pt')

print(labels['dataset_name'])  # 'MUTAG'
print(labels['label_name'])    # 'betweenness_20'
print(labels['node_labels'].shape)  # (3371, 2) for MUTAG
print(labels['node_labels'][:5])
# tensor([[ 4.,  7.],
#         [ 2.,  5.],
#         [ 8.,  3.],
#         [ 0., 12.],
#         [ 1.,  9.]])
# Column 0: original bin labels
# Column 1: frequency-sorted labels
```

### Performance Considerations

Betweenness centrality has high computational complexity:
- **Sparse graphs**: O(nm) where n=nodes, m=edges
- **Dense graphs**: O(n³)

For large datasets (>1000 nodes per graph or >10k graphs):
- Consider sampling a subset for binning
- Use approximate algorithms (e.g., NetworkX `betweenness_centrality(k=100)`)
- Cache labels aggressively (file naming includes all parameters)
- Consider alternative centrality measures (eigenvector, closeness)

## Summary

Adding a custom labeling function requires:

1. **Create class** extending `NodeLabelingBase` in `node_labeling.py`
   - Implement `generate()` to compute labels
   - Handle discretization (manual binning or rely on framework)

2. **Create save function** in same file
   - Simple wrapper calling `create_and_save_labels()`

3. **Register in dispatcher** in `preprocessing.py`
   - Add import
   - Add elif branch in `layer_to_labels()`

4. **Test thoroughly**
   - Verify label generation and caching
   - Check discretization quality
   - Validate integration with ShareGNN training

The betweenness centrality example demonstrates this pattern with continuous values requiring percentile-based binning. Follow this template for your domain-specific graph metrics.
