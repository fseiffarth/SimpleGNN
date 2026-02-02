import torch
import torch.nn as nn

from datasets.graph_dataset import GraphDataset
from framework.utils.parameters import Parameters
from models.ShareGNN.layers.inv_based_message_passing import InvariantBasedMessagePassingLayer
from models.ShareGNN.layers.inv_based_pooling import InvariantBasedAggregationLayer
from models.ShareGNN.layers.share_gnn_layer_norm import ShareGNNLayerNorm
from models.ShareGNN.layers.share_gnn_reshape import ShareGNNReshapeLayer
from models.layers.mpnn_classical.gat_conv import GATConv
from models.layers.mpnn_classical.gatv2_conv import GATv2Conv
from models.layers.mpnn_classical.gcn_conv import GCNConv
from models.layers.mpnn_classical.gin_conv import GINConv
from models.layers.mpnn_classical.global_pooling import GlobalPooling
from models.layers.mpnn_classical.sage_conv import SAGEConv
from models.layers.nn_standard.activation import ActivationLayer
from models.layers.nn_standard.batch_normalization import BatchNormLayer
from models.layers.nn_standard.dropout import DropoutLayer
from models.layers.nn_standard.linear import LinearLayer
from models.layers.utils.layer_types import LayerTypes
from utils.timer import TimeClass


class GraphModel(torch.nn.Module):
    def __init__(self, graph_data: GraphDataset, para: Parameters, seed, device):
        super(GraphModel, self).__init__()
        # set torch seed
        torch.manual_seed(seed)
        self.graph_data = graph_data
        self.para = para
        self.config_parameters = para.run_config.config
        self.print_weights = self.para.net_print_weights
        dropout = self.para.dropout
        self.convolution_grad =self.config_parameters.get('convolution_grad', True)
        self.aggregation_grad =self.config_parameters.get('aggregation_grad', True)
        self.out_dim = self.graph_data.num_classes
        precision = para.run_config.config.get('precision', 'float')
        self.module_precision = torch.float
        if precision == 'double':
            self.module_precision = torch.double

        self.aggregation_out_dim = 0

        nn.Sequential(

        )

        # Define the layers
        self.net_layers = nn.ModuleList()
        input_features = self.graph_data.num_node_features
        num_heads = 0
        current_feature_dimension = input_features
        for i, layer in enumerate(para.layers):
            prev_layer = (None if len(self.net_layers) == 0 else self.net_layers[-1])
            if layer.layer_type == 'invariant_based_convolution':
                if i != 0 and self.config_parameters.get('use_feature_transformation', None) is not None:
                    input_features = self.config_parameters['use_feature_transformation'].get('out_dimension', 16)
                self.net_layers.append(
                    InvariantBasedMessagePassingLayer(layer_id=i,
                                                                    seed=seed,
                                                                    layer=layer,
                                                                    parameters=para,
                                                                    graph_data=self.graph_data,
                                                                    device=device,
                                                                     input_features=current_feature_dimension,
                                                                     output_features=current_feature_dimension).type(self.module_precision).requires_grad_(self.convolution_grad))

            elif layer.layer_type == 'invariant_based_aggregation':
                self.aggregation_out_dim = layer.layer_dict.get('out_dim', self.out_dim)
                self.net_layers.append(
                    InvariantBasedAggregationLayer(layer_id=i,
                                                                 seed=seed,
                                                                 layer=layer,
                                                                 parameters=para,
                                                                 out_dim=self.aggregation_out_dim,
                                                                 graph_data=self.graph_data,
                                                                 device=device,
                                                                  input_features=current_feature_dimension,
                                                                  output_features=current_feature_dimension).requires_grad_(self.aggregation_grad))
            elif layer.layer_type == 'linear':
                output_features = layer.layer_dict.get('out_features', 16)
                layer_args = {
                    'in_features': current_feature_dimension,
                    'out_features': output_features,
                    'bias': layer.layer_dict.get('bias', True),
                    'dtype': self.module_precision
                }
                self.net_layers.append(LinearLayer(layer_args))
                current_feature_dimension = output_features
            elif layer.layer_type == 'reshape':
                if isinstance(prev_layer, InvariantBasedAggregationLayer):
                    output_features = prev_layer.num_heads * prev_layer.output_features * prev_layer.output_dimension
                self.net_layers.append(ShareGNNReshapeLayer(layer_id=i,
                                                                           seed=seed,
                                                                           layer=layer,
                                                                           parameters=para,
                                                                           graph_data=self.graph_data,
                                                                           num_heads=num_heads,
                                                                           input_features=input_features,
                                                                           output_features=output_features).type(self.module_precision))
                current_feature_dimension = output_features
            elif layer.layer_type == 'layer_norm':
                self.net_layers.append(ShareGNNLayerNorm(layer_id=i,
                                                                        num_heads=num_heads,
                                                                        input_features=input_features,
                                                                        output_features=output_features).type(self.module_precision))
                current_feature_dimension = output_features
            elif layer.layer_type in [LayerTypes.GCN_CONVOLUTION.value,
                                      LayerTypes.GAT_CONVOLUTION.value,
                                      LayerTypes.GATv2_CONVOLUTION.value,
                                      LayerTypes.GIN_CONVOLUTION.value,
                                      LayerTypes.SAGE_CONVOLUTION.value]:
                layer_args = layer.layer_dict
                layer_args['in_channels'] = current_feature_dimension
                current_feature_dimension = layer_args['out_channels']
                # GNN specific layers
                if layer.layer_type == LayerTypes.GCN_CONVOLUTION.value:
                    self.net_layers.append(GCNConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.GAT_CONVOLUTION.value:
                    self.net_layers.append(GATConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.GATv2_CONVOLUTION.value:
                    self.net_layers.append(GATv2Conv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.GIN_CONVOLUTION.value:
                    self.net_layers.append(GINConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.SAGE_CONVOLUTION.value:
                    self.net_layers.append(SAGEConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))

            elif layer.layer_type == LayerTypes.GLOBAL_POOLING.value:
                layer_args = {'mode': layer.layer_dict.get('mode', 'mean')}
                self.net_layers.append(GlobalPooling(layer_args).type(self.module_precision).requires_grad_(self.aggregation_grad))
            elif layer.layer_type == LayerTypes.DROPOUT.value:
                layer_args = {'p': layer.layer_dict.get('p', 0.5)}
                self.net_layers.append(DropoutLayer(layer_args))
                # Dropout does not change feature dimension
            elif layer.layer_type == LayerTypes.ACTIVATION.value:
                layer_args = {'activation_function': layer.layer_dict.get('activation_function', torch.nn.ReLU())}
                self.net_layers.append(ActivationLayer(layer_args))
                # Activation does not change feature dimension
            elif layer.layer_type == LayerTypes.BATCH_NORM.value:
                layer_args = {'batch_norm': True, 'in_channels': current_feature_dimension}
                self.net_layers.append(BatchNormLayer(layer_args).type(self.module_precision))
                # BatchNorm does not change feature dimension
            else:
                raise ValueError(f'Layer type {layer.layer_type} not recognized in GraphModel')
        self.dropout = nn.Dropout(dropout)

        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, batch_data, *args, **kwargs):
        x = batch_data.x
        representation_list = []
        for i, layer in enumerate(self.net_layers):
            x = layer(x, batch_data, *args, **kwargs)
            #if isinstance(layer, GNNConvLayer):
            #    representation_list.append(x)
            #    if i == len(self.net_layers) - 1 or not isinstance(self.net_layers[i + 1], GNNConvLayer):
            #        x = torch.squeeze(torch.mean(torch.stack(representation_list), 0, True), 0)

        return x

    def return_info(self):
        return type(self)
