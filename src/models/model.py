import torch
import torch.nn as nn

from datasets.graph_dataset import GraphDataset
from framework.utils.parameters import Parameters
from models.ShareGNN.layers.inv_based_message_passing import InvariantBasedMessagePassingLayer
from models.ShareGNN.layers.inv_based_pooling import InvariantBasedAggregationLayer
from models.layers.mpnn_classical.gat_conv import GATConv
from models.layers.mpnn_classical.gatv2_conv import GATv2Conv
from models.layers.mpnn_classical.gcn_conv import GCNConv
from models.layers.mpnn_classical.gin_conv import GINConv
from models.layers.mpnn_classical.global_pooling import GlobalPooling
from models.layers.mpnn_classical.sage_conv import SAGEConv
from models.layers.nn_standard.activation import ActivationLayer
from models.layers.nn_standard.batch_normalization import BatchNormLayer
from models.layers.nn_standard.dropout import DropoutLayer
from models.layers.nn_standard.layer_normalization import LayerNormalization
from models.layers.nn_standard.linear import LinearLayer
from models.layers.nn_standard.reshape import Reshape
from models.layers.utils.layer_types import LayerTypes
from utils.timer import TimeClass


class GraphModel(torch.nn.Module):
    def __init__(self, graph_data: GraphDataset, para: Parameters, seed, device):
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
            self.aggregation_out_dim = layer.layer_dict.get('out_dim', self.out_dim)
            return InvariantBasedAggregationLayer(layer=layer,
                                                                  parameters=self.para,
                                                                  graph_data=self.graph_data,).requires_grad_(self.aggregation_grad)
        elif layer.layer_type in [LayerTypes.GCN_CONVOLUTION.value,
                                  LayerTypes.GAT_CONVOLUTION.value,
                                  LayerTypes.GATv2_CONVOLUTION.value,
                                  LayerTypes.GIN_CONVOLUTION.value,
                                  LayerTypes.SAGE_CONVOLUTION.value]:
            layer_args = layer.layer_dict
            # GNN specific layers
            if layer.layer_type == LayerTypes.GCN_CONVOLUTION.value:
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
            layer_args = {'mode': layer.layer_dict.get('mode', 'mean')}
            return GlobalPooling(layer_args).type(self.precision).requires_grad_(self.aggregation_grad)
        elif layer.layer_type == LayerTypes.DROPOUT.value:
            layer_args = {'p': layer.layer_dict.get('p', 0.5)}
            return DropoutLayer(layer_args)
            # Dropout does not change feature dimension
        elif layer.layer_type == LayerTypes.ACTIVATION.value:
            layer_args = {'activation_function': layer.layer_dict.get('activation_function', torch.nn.ReLU())}
            return ActivationLayer(layer_args)
            # Activation does not change feature dimension
        elif layer.layer_type == LayerTypes.BATCH_NORM.value:
            layer_args = {'batch_norm': True, 'in_channels': in_features}
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
        return type(self)