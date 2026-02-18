"""
Graph Neural Network Model: Core PyTorch model for GNN architectures.

This module provides the main GraphModel class that dynamically constructs
GNN architectures from YAML configuration files. It supports both classical
message-passing GNNs (GCN, GIN, GAT, GATv2, GraphSAGE) and the proprietary
ShareGNN architecture with invariant-based layers.

The model architecture is specified as a sequential list of layers in YAML
configuration files. GraphModel automatically handles:
- Layer instantiation and parameter propagation
- Input/output feature dimension matching between layers
- Multi-head/multi-channel tensor shape handling
- Precision (float/double) and device (CPU/CUDA) management
- Random input variation for data augmentation

Key Classes
-----------
GraphModel : Main GNN model class
    PyTorch nn.Module that builds and executes GNN architectures
    dynamically from configuration specifications.

Usage Examples
--------------
Create and use a GraphModel:

>>> from simplegnn.models.model import GraphModel
>>> from simplegnn.framework.utils.parameters import Parameters
>>> model = GraphModel(graph_data, parameters, seed=42, device='cpu')
>>> output = model(batch_data)

See Also
--------
models.layers.framework_layer.FrameworkLayer : Base class for all layers
models.layers.utils.layer_types.LayerTypes : Enumeration of layer types
"""
import torch
import torch.nn as nn

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.models.ShareGNN.layers.inv_based_message_passing import InvariantBasedMessagePassingLayer
from simplegnn.models.ShareGNN.layers.inv_based_pooling import InvariantBasedAggregationLayer
from simplegnn.models.layers.mpnn_classical.gat_conv import GATConv
from simplegnn.models.layers.mpnn_classical.gatv2_conv import GATv2Conv
from simplegnn.models.layers.mpnn_classical.gcn_conv import GCNConv
from simplegnn.models.layers.mpnn_classical.gin_conv import GINConv
from simplegnn.models.layers.mpnn_classical.global_pooling import GlobalPooling
from simplegnn.models.layers.mpnn_classical.sage_conv import SAGEConv
from simplegnn.models.layers.nn_standard.activation import ActivationLayer
from simplegnn.models.layers.nn_standard.batch_normalization import BatchNormLayer
from simplegnn.models.layers.nn_standard.dropout import DropoutLayer
from simplegnn.models.layers.nn_standard.layer_normalization import LayerNormalization
from simplegnn.models.layers.nn_standard.linear import LinearLayer
from simplegnn.models.layers.nn_standard.reshape import Reshape
from simplegnn.models.layers.utils.layer_types import LayerTypes
from simplegnn.utils.timer import TimeClass


class GraphModel(torch.nn.Module):
    """
    Dynamic GNN model builder and executor.

    GraphModel is the main PyTorch model class in the SimpleGNN framework.
    It constructs GNN architectures dynamically from layer specifications
    provided in YAML configuration files. The model supports:

    - Classical GNN layers: GCN, GIN, GAT, GATv2, GraphSAGE
    - ShareGNN invariant-based layers: Message passing and pooling
    - Standard layers: Linear, activation, batch norm, dropout, reshape
    - Multi-head and multi-channel processing
    - Float and double precision
    - CPU and CUDA execution
    - Reproducible random seeding

    The architecture is specified as a sequential list of layers in
    parameters.layers, where each layer is instantiated with automatic
    feature dimension propagation from previous layers.

    Parameters
    ----------
    graph_data : GraphDataset
        Dataset containing graphs, node features, labels, and optional
        properties (for ShareGNN). Provides num_node_features and
        num_classes for model configuration.
    para : Parameters
        Parameter container with model architecture (para.layers),
        hyperparameters (dropout, learning rate), and run configuration.
    seed : int
        Random seed for reproducibility. Sets torch.manual_seed().
    device : str or torch.device
        Device for model execution ('cpu' or 'cuda').

    Attributes
    ----------
    net_layers : nn.ModuleList
        Sequential list of instantiated layer modules. Populated from
        para.layers during __init__.
    graph_data : GraphDataset
        Reference to input dataset.
    para : Parameters
        Reference to parameters container.
    config_parameters : dict
        Shortcut to para.run_config.config for quick access.
    device : str or torch.device
        Execution device (CPU or CUDA).
    seed : int
        Random seed used for initialization.
    precision : torch.dtype
        Tensor precision (torch.float or torch.double) based on config.
    out_dim : int
        Output dimension equal to graph_data.num_classes.
    random_variation_bool : bool or None
        Whether to apply random noise to inputs for data augmentation.
    convolution_grad : bool
        Whether convolution layers require gradients (default: True).
    aggregation_grad : bool
        Whether aggregation/pooling layers require gradients (default: True).
    epoch : int
        Current training epoch (initialized to 0).
    timer : TimeClass
        Utility for tracking execution time.

    Examples
    --------
    Build a GNN model from configuration:

    >>> model = GraphModel(graph_data, parameters, seed=42, device='cpu')
    >>> print(f"Model has {len(model.net_layers)} layers")
    >>> model.to(device)

    Forward pass on batch:

    >>> output = model(batch_data)
    >>> print(f"Output shape: {output.shape}")

    Notes
    -----
    **Layer Instantiation:**

    Layers are instantiated sequentially from para.layers. Each layer
    receives in_features and in_channels from the previous layer (or from
    graph_data.num_node_features for the first layer).

    **Tensor Shape Conventions:**

    - Standard layers: (N, F) where N=nodes, F=features
    - Multi-head/channel layers: (C, N, F) where C=channels/heads

    **Random Input Variation:**

    If config['input_features']['random_variation'] is set, Gaussian noise
    is added to inputs during forward pass:
    x = x + N(mean, std)

    **YAML Configuration Example:**

    ```yaml
    layers:
      - layer_type: gcn_convolution
        out_features: 64
      - layer_type: activation
        activation_function: torch.nn.ReLU()
      - layer_type: linear
        out_features: 10
    ```

    See Also
    --------
    models.layers.framework_layer.FrameworkLayer : Base class for layers
    framework.utils.parameters.Parameters : Configuration container
    """

    def __init__(self, graph_data: GraphDataset, para: Parameters, seed,
                 device):
        """
        Initialize GraphModel and build layer architecture.

        Constructs the GNN model by sequentially instantiating layers
        from the configuration. Sets random seed, precision, device, and
        initializes all layer modules.

        Parameters
        ----------
        graph_data : GraphDataset
            Dataset providing num_node_features and num_classes.
        para : Parameters
            Configuration containing layers list and hyperparameters.
        seed : int
            Random seed for torch.manual_seed().
        device : str or torch.device
            Target device ('cpu' or 'cuda').

        Notes
        -----
        Layers are added to self.net_layers in the order they appear in
        para.layers. Each layer's in_features and in_channels are
        automatically set from the previous layer's out_features and
        out_channels.
        """
        super(GraphModel, self).__init__()
        # set torch seed
        torch.manual_seed(seed)
        self.device = device
        self.seed = seed
        self.graph_data = graph_data
        self.para = para
        self.config_parameters = para.run_config.config
        self.print_weights = self.para.net_print_weights
        dropout = self.para.dropout
        self.convolution_grad =self.config_parameters.get('convolution_grad', True)
        self.aggregation_grad =self.config_parameters.get('aggregation_grad', True)
        self.out_dim = self.graph_data.num_classes
        self.precision = torch.float
        if self.para.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double

        self.random_variation_bool = self.para.run_config.config.get('input_features', None).get('random_variation', None)
        self.aggregation_out_dim = 0


        # Initializing the graph model by adding layers as per the configuration
        self.net_layers = nn.ModuleList()
        in_features = self.graph_data.num_node_features
        num_heads = 0
        for layer in self.para.layers:
            model_layer = self.get_model_layer(layer=layer)
            self.net_layers.append(model_layer)

        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, batch_data, *args, **kwargs):
        """
        Forward pass through the GNN model.

        Executes the complete forward pass by sequentially applying all
        layers in net_layers. Optionally adds random Gaussian noise to
        input features for data augmentation if configured.

        Parameters
        ----------
        batch_data : torch_geometric.data.Batch or GraphDataset item
            Batched graph data containing:
            - x: Node features, shape (N, F) or (C, N, F)
            - edge_index: Graph connectivity
            - batch: Batch assignment vector (for graph-level tasks)
            - Additional attributes used by specific layers
        *args
            Variable positional arguments passed to all layers.
        **kwargs
            Variable keyword arguments passed to all layers.

        Returns
        -------
        torch.Tensor
            Model output after all layers.
            - Graph classification: (batch_size, num_classes)
            - Node classification: (N, num_classes)
            - Graph regression: (batch_size, output_dim)

        Notes
        -----
        **Random Input Variation:**

        If config['input_features']['random_variation'] is enabled,
        adds Gaussian noise to input features:

        x_augmented = x + N(mean, std)

        where mean and std are specified in configuration.

        **Layer Processing:**

        Each layer receives:
        - x: Current feature tensor
        - batch_data: Full batch data for accessing edge_index, batch, etc.
        - *args, **kwargs: Additional arguments

        Layers update x sequentially: x = layer(x, batch_data)

        **Tensor Shape Flow:**

        Shape transformations depend on layer types:
        - Convolution layers: May change features, preserve node count
        - Pooling layers: Reduce node count (graph-level)
        - Linear layers: Change feature dimension
        - Activation/Dropout: Preserve shape

        Examples
        --------
        Standard forward pass:

        >>> output = model.forward(batch_data)
        >>> loss = criterion(output, batch_data.y)

        With random variation enabled in config:

        >>> # Input features get Gaussian noise added automatically
        >>> output = model(batch_data)
        """
        x = batch_data.x
        if self.random_variation_bool:
            mean = self.para.run_config.config['input_features']['random_variation'].get('mean', 0.0)
            std = self.para.run_config.config['input_features']['random_variation'].get('std', 0.1)
            if self.para.run_config.config.get('precision', 'double') == 'float':
                random_variation = torch.normal(mean=mean, std=std, size=x.size(),
                                                dtype=torch.float)
            else:
                random_variation = torch.normal(mean=mean, std=std, size=x.size(),
                                                dtype=torch.double)
            x = x + random_variation

        for i, layer in enumerate(self.net_layers):
            x = layer(x, batch_data, *args, **kwargs)

        return x




    def get_model_layer(self, layer):
        """
        Instantiate a single layer from configuration specification.

        Creates and configures a layer module based on layer_type and
        parameters. Automatically propagates in_features and in_channels
        from the previous layer, sets output dimensions, handles multi-head
        configurations, and applies precision/device settings.

        Parameters
        ----------
        layer : LayerSpecification
            Layer specification object containing:
            - layer_type: str matching LayerTypes enum value
            - layer_dict: dict of layer-specific parameters
            - layer_id: int unique layer identifier

        Returns
        -------
        nn.Module
            Instantiated layer module (subclass of FrameworkLayer or
            standard PyTorch layer). Layer is configured with:
            - Correct dtype (float/double)
            - Device placement
            - Gradient requirements
            - Automatic feature dimension propagation

        Raises
        ------
        ValueError
            If layer_type is not recognized or if head_mode configuration
            is invalid (e.g., num_heads != in_channels when
            head_mode='same_as_in_channels').

        Notes
        -----
        **Feature Dimension Propagation:**

        For the first layer:
        - in_features = graph_data.num_node_features
        - in_channels = 1

        For subsequent layers:
        - in_features = previous_layer.out_features
        - in_channels = previous_layer.out_channels

        **Multi-Head Handling:**

        Two head modes are supported:

        1. extend_in_channels (default):
           out_channels = in_channels * num_heads

        2. same_as_in_channels:
           out_channels = in_channels
           (requires num_heads == in_channels)

        **Layer Types and Instantiation:**

        - ShareGNN layers:
          - INVARIANT_BASED_CONVOLUTION → InvariantBasedMessagePassingLayer
          - INVARIANT_BASED_AGGREGATION → InvariantBasedAggregationLayer

        - Classical GNN layers:
          - GCN_CONVOLUTION → GCNConv
          - GAT_CONVOLUTION → GATConv
          - GATv2_CONVOLUTION → GATv2Conv
          - GIN_CONVOLUTION → GINConv
          - SAGE_CONVOLUTION → SAGEConv

        - Pooling/Aggregation:
          - GLOBAL_POOLING → GlobalPooling

        - Standard layers:
          - LINEAR → LinearLayer
          - ACTIVATION → ActivationLayer
          - BATCH_NORM → BatchNormLayer
          - LAYER_NORM → LayerNormalization
          - DROPOUT → DropoutLayer
          - RESHAPE → Reshape

        **Gradient Control:**

        - Convolution layers: Use self.convolution_grad
        - Aggregation/pooling layers: Use self.aggregation_grad
        - Standard layers: Gradients enabled by default

        Examples
        --------
        Layer instantiation happens automatically in __init__:

        >>> # In __init__:
        >>> for layer_spec in self.para.layers:
        ...     model_layer = self.get_model_layer(layer_spec)
        ...     self.net_layers.append(model_layer)

        See Also
        --------
        models.layers.utils.layer_types.LayerTypes : Layer type enumeration
        models.layers.framework_layer.FrameworkLayer : Base layer class
        """
        layer_args = layer.layer_dict
        # Update common layer arguments
        layer_args['layer_id'] = layer.layer_id
        layer_args['device'] = self.device
        layer_args['seed'] = self.seed
        prev_layer = (None if len(self.net_layers) == 0 else self.net_layers[-1])
        if prev_layer is None:
            layer_args['in_features'] = self.graph_data.num_node_features
            layer_args['in_channels'] = 1
        else:
            layer_args['in_features'] = prev_layer.out_features
            layer_args['in_channels'] = prev_layer.out_channels

        layer_args['out_features'] = layer_args.get('out_features', layer_args['in_features'])
        # TODO how to get the out channels from in_channels and num_heads. At the moment extent each in_channel by num_heads, but this may not be the case for all layers
        # another option is apply one head to one in_channel (only if num_heads == in_channels)
        if 'heads' in layer_args:
            layer_args['num_heads'] = len(layer_args['heads'])

        num_heads = layer_args.get('num_heads', 1)
        head_mode = layer_args.get('head_mode', 'extend_in_channels')
        if head_mode == 'extend_in_channels':
            layer_args['out_channels'] = layer_args['in_channels'] * num_heads
        elif head_mode == 'same_as_in_channels':
            if num_heads != layer_args['in_channels']:
                raise ValueError(f'num_heads must be equal to in_channels when head_mode is same_as_in_channels, but got num_heads={num_heads} and in_channels={in_channels}')
            layer_args['out_channels'] = layer_args['in_channels']
        layer_args['dtype'] = self.precision
        layer_args['out_features'] = layer_args.get('out_features', layer_args['in_features'])

        if layer.layer_type == LayerTypes.INVARIANT_BASED_CONVOLUTION.value:
            return InvariantBasedMessagePassingLayer(layer=layer, parameters=self.para, graph_data=self.graph_data).type(self.precision).requires_grad_(self.convolution_grad)

        elif layer.layer_type == LayerTypes.INVARIANT_BASED_AGGREGATION.value:
            self.aggregation_out_dim = layer_args.get('out_dim', self.out_dim)
            return InvariantBasedAggregationLayer(layer=layer,
                                                                  parameters=self.para,
                                                                  graph_data=self.graph_data,).requires_grad_(self.aggregation_grad)
        # GNN specific layers
        elif layer.layer_type == LayerTypes.GCN_CONVOLUTION.value:
            return GCNConv(layer_args).type(self.precision).requires_grad_(self.convolution_grad)
        elif layer.layer_type == LayerTypes.GAT_CONVOLUTION.value:
            return GATConv(layer_args).type(self.precision).requires_grad_(self.convolution_grad)
        elif layer.layer_type == LayerTypes.GATv2_CONVOLUTION.value:
            return GATv2Conv(layer_args).type(self.precision).requires_grad_(self.convolution_grad)
        elif layer.layer_type == LayerTypes.GIN_CONVOLUTION.value:
            return GINConv(layer_args).type(self.precision).requires_grad_(self.convolution_grad)
        elif layer.layer_type == LayerTypes.SAGE_CONVOLUTION.value:
            return SAGEConv(layer_args).type(self.precision).requires_grad_(self.convolution_grad)
        elif layer.layer_type == LayerTypes.GLOBAL_POOLING.value:
            layer_args['mode'] = layer_args.get('mode', 'mean')
            return GlobalPooling(layer_args).type(self.precision).requires_grad_(self.aggregation_grad)
        elif layer.layer_type == LayerTypes.DROPOUT.value:
            return DropoutLayer(layer_args)
            # Dropout does not change feature dimension
        elif layer.layer_type == LayerTypes.ACTIVATION.value:
            layer_args['activation_function'] = layer_args.get('activation_function', torch.nn.ReLU())
            return ActivationLayer(layer_args)
            # Activation does not change feature dimension
        elif layer.layer_type == LayerTypes.BATCH_NORM.value:
            return BatchNormLayer(layer_args).type(self.precision)
            # BatchNorm does not change feature dimension
        elif layer.layer_type == LayerTypes.LAYER_NORM.value:
            return LayerNormalization(layer_args=layer_args).type(self.precision)
        elif layer.layer_type == LayerTypes.LINEAR.value:
            return LinearLayer(layer_args=layer_args).type(self.precision)
        elif layer.layer_type == LayerTypes.RESHAPE.value:
            return Reshape(layer_args=layer_args).type(self.precision)
        else:
            raise ValueError(f'Layer type {layer.layer_type} not recognized in GraphModel')

    def return_info(self):
        """
        Return model class type information.

        Returns
        -------
        type
            The class type of this model instance (GraphModel).

        Notes
        -----
        Utility method for introspection and debugging.
        """
        return type(self)