from abc import abstractmethod, ABC

import torch
import torch_geometric
from torch._C.cpp import nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

from simplegnn.datasets.graph_dataset import GraphDataset


class FrameworkLayer(torch.nn.Module, ABC):
    """
    Abstract base class for all custom GNN layers in the SimpleGNN framework.

    This class provides a common interface and shared functionality for all layer
    implementations. It handles standard configurations like activation functions,
    batch normalization, dropout, residual connections, and precision settings.

    All custom layers must inherit from this class and implement the forward() method.

    Parameters
    ----------
    layer_args : dict
        Configuration dictionary containing layer parameters. Required keys:
        - 'layer_id' : int
            Unique identifier for this layer within the network.
        - 'name' : str
            Human-readable layer name (e.g., 'GCNConv', 'LinearLayer').
        - 'seed' : int
            Random seed for reproducibility.
        - 'dtype' : torch.dtype
            Precision for computations (torch.float32 or torch.float64).
        - 'in_features' : int
            Number of input features per node.
        - 'out_features' : int
            Number of output features per node.
        - 'in_channels' : int
            Number of input channels/heads.
        - 'out_channels' : int
            Number of output channels/heads.

        Optional keys:
        - 'num_heads' : int
            Number of attention heads or parallel computations (default: 1).
        - 'heads' : list of dict
            Alternative to num_heads for multi-head configurations.
        - 'residual' : bool
            Whether to use residual connections (default: False).
        - 'batch_norm' : bool
            Whether to apply batch normalization (default: False).
        - 'batch_norm_eps' : float
            Epsilon for batch norm stability (default: 1e-5).
        - 'batch_norm_momentum' : float
            Momentum for batch norm running statistics (default: 0.1).
        - 'batch_norm_affine' : bool
            Learnable affine parameters in batch norm (default: True).
        - 'batch_norm_track_running_stats' : bool
            Track running mean/var in batch norm (default: True).
        - 'batch_norm_allow_single_element' : bool
            Allow batch norm with single-element batches (default: False).
        - 'dropout' : float
            Dropout probability, 0.0 to 1.0 (default: 0.0).
        - 'device' : str
            Device to run layer on, 'cpu' or 'cuda' (default: 'cpu').
        - 'bias' : bool
            Whether to use bias terms in linear transformations (default: True).
        - 'activation' : str or torch.nn.Module
            Activation function (default: torch.nn.Identity()). Supported strings:
            'torch.nn.Identity()', 'torch.nn.ReLU()', 'torch.nn.LeakyReLU()',
            'torch.nn.ELU()', 'torch.nn.GELU()', 'torch.nn.Tanh()',
            'torch.nn.Sigmoid()', 'torch.nn.Softmax(dim=1)', 'torch.nn.SELU()'.

    Attributes
    ----------
    layer_id : int
        Unique layer identifier.
    name : str
        Layer name.
    seed : int
        Random seed.
    precision : torch.dtype
        Computation precision.
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_heads : int
        Number of heads (computed from num_heads or heads configuration).
    residual : bool
        Whether residual connections are enabled.
    batch_norm : bool
        Whether batch normalization is enabled.
    batch_norm_args : dict or None
        Batch normalization configuration (only if batch_norm=True).
    batch_norm_layer : torch_geometric.nn.BatchNorm or None
        Batch normalization layer instance (only if batch_norm=True).
    dropout : float
        Dropout probability.
    device : str
        Device for layer computation.
    bias : bool
        Whether bias terms are used.
    activation : torch.nn.Module
        Activation function instance.

    Notes
    -----
    **Tensor Shape Conventions:**
    - Input: (C, N, F) or (N, F)
        - C: Number of channels/heads
        - N: Number of nodes (concatenated across batch)
        - F: Number of features per node
    - Output: (C', N, F') or (N, F')
        - C': Number of output channels/heads
        - F': Number of output features
        - N: Node count remains unchanged

    **Batch Processing:**
    For batched graphs, nodes from different graphs are concatenated along the
    N dimension. The batch structure is tracked separately in batch_data.

    **Activation Functions:**
    If activation is provided as a string, it must exactly match one of the
    supported activation names (including parentheses). Alternatively, pass
    a torch.nn.Module instance directly.

    Raises
    ------
    ValueError
        If any required layer_arg is missing or if an unsupported activation
        function is specified.

    See Also
    --------
    models.layers.mpnn_classical : Classical GNN layer implementations
    models.ShareGNN.layers : ShareGNN-specific layer implementations

    Examples
    --------
    >>> layer_args = {
    ...     'layer_id': 0,
    ...     'name': 'MyCustomLayer',
    ...     'seed': 42,
    ...     'dtype': torch.float32,
    ...     'in_features': 16,
    ...     'out_features': 32,
    ...     'in_channels': 1,
    ...     'out_channels': 1,
    ...     'activation': 'torch.nn.ReLU()',
    ...     'dropout': 0.5,
    ...     'residual': True
    ... }
    >>> # Subclass must implement forward()
    >>> class MyLayer(FrameworkLayer):
    ...     def forward(self, x, batch_data):
    ...         # Custom implementation
    ...         return x
    """
    def __init__(self, layer_args=None):
        """
        Initialize the framework layer with configuration parameters.

        Validates and sets all required and optional layer parameters, including
        dimensions, activation functions, normalization, and regularization settings.

        Parameters
        ----------
        layer_args : dict or None
            Layer configuration dictionary. See class docstring for required and
            optional keys.

        Raises
        ------
        ValueError
            If layer_args is None or any required parameter is missing.
            If an unsupported activation function is specified.

        Notes
        -----
        The initialization performs the following steps:
        1. Validates presence of all mandatory parameters
        2. Extracts layer ID, name, and seed
        3. Sets precision (dtype)
        4. Configures input/output dimensions (features and channels)
        5. Computes number of heads from num_heads or heads configuration
        6. Sets up residual connection flag
        7. Initializes batch normalization if requested
        8. Sets dropout rate
        9. Determines compute device
        10. Configures bias usage
        11. Parses and instantiates activation function

        **Heads Configuration:**
        If 'heads' key is present in layer_args, num_heads is computed by summing
        the 'num' field from each head dictionary. Otherwise, num_heads defaults to 1.

        **Batch Normalization:**
        When batch_norm=True, creates a torch_geometric.nn.BatchNorm layer with
        configurable parameters for eps, momentum, affine transforms, running stats,
        and single-element batch handling.
        """
        super(FrameworkLayer, self).__init__()
        # Mandatory layer arguments
        if layer_args is None:
            raise ValueError("layer_args must be provided")
        self.layer_args = layer_args

        # Get layer ID and name
        self.layer_id = self.layer_args.get('layer_id', None)
        if self.layer_id is None:
            raise ValueError("layer_id must be provided")
        self.name = self.layer_args.get('name', None)
        if self.name is None:
            raise ValueError("layer name must be provided")

        # Get layer seed for reproducibility
        self.seed = self.layer_args.get('seed', None)
        if self.seed is None:
            raise ValueError("seed must be provided")

        # Set layer precision
        self.precision = layer_args.get('dtype', None)
        if self.precision is None:
            raise ValueError("precision setting is not supported in this framework layer implementation")


        # Get input and output dimensions of the layer
        self.in_features = layer_args.get('in_features', None)
        if self.in_features is None:
            raise ValueError("in_features must be provided")
        self.out_features = layer_args.get('out_features', None)
        if self.out_features is None:
            raise ValueError("out_features must be provided")
        self.in_channels = layer_args.get('in_channels', None)
        if self.in_channels is None:
            raise ValueError("in_channels must be provided")
        self.out_channels = layer_args.get('out_channels', None)
        if self.out_channels is None:
            raise ValueError("out_channels must be provided")

        self.num_heads = layer_args.get('num_heads', 1)
        if 'heads' in layer_args:
            self.num_heads = 0
            for head in layer_args['heads']:
                self.num_heads += head['num']


        # Whether to use residual connections in this layer
        self.residual = layer_args.get('residual', False)
        # Whether to use batch normalization in this layer
        self.batch_norm = layer_args.get('batch_norm', False)
        if self.batch_norm:
            self.batch_norm_args = {
                'in_channels': self.in_channels,
                'eps': layer_args.get('batch_norm_eps', 1e-5),
                'momentum': layer_args.get('batch_norm_momentum', 0.1),
                'affine': layer_args.get('batch_norm_affine', True),
                'track_running_stats': layer_args.get('batch_norm_track_running_stats', True),
                'allow_single_element': layer_args.get('batch_norm_allow_single_element', False),
            }
            self.batch_norm_layer = torch_geometric.nn.BatchNorm(**self.batch_norm_args)
        # Dropout rate for this layer
        self.dropout = layer_args.get('dropout', 0.0)
        # Device to run the layer on
        self.device = layer_args.get('device', 'cpu')
        # bias for this layer
        self.bias = layer_args.get('bias', True) # whether to use bias in this layer, default is True

        # Activation function for this layer
        if 'activation' not in layer_args:
            self.activation = torch.nn.Identity()
        else:
            activation_value = layer_args['activation']
            if isinstance(activation_value, torch.nn.Module):
                self.activation = activation_value
            elif isinstance(activation_value, str):
                ACTIVATION_MAP = {
                    'torch.nn.Identity()': torch.nn.Identity(),
                    'torch.nn.ReLU()': torch.nn.ReLU(),
                    'torch.nn.LeakyReLU()': torch.nn.LeakyReLU(),
                    'torch.nn.ELU()': torch.nn.ELU(),
                    'torch.nn.GELU()': torch.nn.GELU(),
                    'torch.nn.Tanh()': torch.nn.Tanh(),
                    'torch.nn.Sigmoid()': torch.nn.Sigmoid(),
                    'torch.nn.Softmax(dim=1)': torch.nn.Softmax(dim=1),
                    'torch.nn.SELU()': torch.nn.SELU(),
                }
                if activation_value in ACTIVATION_MAP:
                    self.activation = ACTIVATION_MAP[activation_value]
                else:
                    raise ValueError(f"Unsupported activation function: '{activation_value}'. "
                                     f"Supported values: {list(ACTIVATION_MAP.keys())}")
            else:
                raise ValueError(f"Activation must be a string or torch.nn.Module, got {type(activation_value)}")

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: GraphDataset, *args, **kwargs):
        """
        Forward pass of the layer (must be implemented by subclasses).

        Processes node representations through the layer's transformation, producing
        updated node features. This is an abstract method that must be overridden
        by all concrete layer implementations.

        Parameters
        ----------
        node_representation : torch.Tensor
            Input node features. Shape is either:
            - (N, F) for single-channel layers, where:
                - N: number of nodes (concatenated across batch)
                - F: number of input features
            - (C, N, F) for multi-channel/multi-head layers, where:
                - C: number of input channels/heads
                - N: number of nodes
                - F: number of input features
        batch_data : GraphDataset
            Graph dataset object containing:
            - Edge indices and attributes
            - Batch assignment for nodes
            - Graph-level information
            - Node labels and properties (for ShareGNN layers)
        *args
            Additional positional arguments (layer-specific).
        **kwargs
            Additional keyword arguments (layer-specific).

        Returns
        -------
        torch.Tensor
            Updated node representations. Shape is either:
            - (N, F') for single-channel output
            - (C', N, F') for multi-channel/multi-head output
            where F' is the number of output features and C' is the number of
            output channels/heads.

        Notes
        -----
        Implementations should:
        1. Apply the layer's transformation (e.g., message passing, linear, pooling)
        2. Apply activation function if specified
        3. Apply dropout if specified and in training mode
        4. Apply batch normalization if configured
        5. Apply residual connection if configured and dimensions match

        The batch_data object provides access to graph structure via:
        - batch_data.edge_index: Edge connectivity (COO format)
        - batch_data.batch: Node-to-graph assignment
        - batch_data.node_labels: Node labels (for invariant-based layers)
        - batch_data.properties: Edge properties (for invariant-based layers)

        See Also
        --------
        models.layers.mpnn_classical.gcn.GCNConv : Example implementation
        models.ShareGNN.layers.inv_based_message_passing.InvariantBasedMessagePassingLayer : Complex implementation
        """
        pass






















