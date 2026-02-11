import numpy as np

from simplegnn.models.layers.utils.layer_types import LayerTypes


def layer_from_yml(layer_id, layer_yml, layers_per_architecture, network_architecture):
    """
    Creates a layer from a yml definition and adds it to the layers_per_architecture list
    :param layer_id:
    :param layer_yml:
    :param layers_per_architecture:
    :param network_architecture:
    :return:
    """
    if 'layer_type' not in layer_yml:
        raise ValueError("layer_type is required in layer_yml")
    layer_type = layer_yml['layer_type']
    if layer_type not in LayerTypes:
        raise ValueError(f"layer_type {layer_type} is not supported")
    elif layer_type in [LayerTypes.INVARIANT_BASED_CONVOLUTION.value, LayerTypes.INVARIANT_BASED_AGGREGATION.value]:
        layer_from_yml_invariant_based(layer_id, layer_yml, layers_per_architecture, network_architecture)
    else:
        if len(layers_per_architecture) <= layer_id:
            while len(layers_per_architecture) <= layer_id:
                layers_per_architecture.append([])
        layers_per_architecture[layer_id].append(network_architecture[layer_id])


def layer_from_yml_invariant_based(layer_id, layer_yml, layers_per_architecture, network_architecture):
    correct, error = check_layer(layer_id, layer_yml)
    if correct:
        if len(layers_per_architecture) <= layer_id:
            while len(layers_per_architecture) <= layer_id:
                layers_per_architecture.append([])
        layers_per_architecture[layer_id].append(network_architecture[layer_id])
    else:
        short_type, error = check_layer_short_type(layer_id, layer_yml)
        if short_type:
            if isinstance(layer_yml['heads'], int):
                channel_combinations = [layer_yml['heads']]
            elif isinstance(layer_yml['heads'], list):
                channel_combinations = layer_yml['heads']
            else:
                raise ValueError(f'Channels in layer {layer_id} is not an int or a list')
            label_combinations = len(layer_yml['labels'])
            key_combinations_per_label = []
            key_list_per_label = [None] * label_combinations
            for j, label in enumerate(layer_yml['labels']):
                # get all keys except label_type
                key_list = list(label.keys())
                key_list.remove('label_type')
                key_list_per_label[j] = key_list
                # get all lenghts of the values of the keys
            for j, key_list in enumerate(key_list_per_label):
                if len(key_list) == 0:
                    key_combinations_per_label.append([1])
                else:
                    key_combinations_per_label.append([])
                    for key in key_list:
                        key_combinations_per_label[-1].append(len(layer_yml['labels'][j][key]))
            if layer_yml['layer_type'] == 'invariant_based_convolution':
                property_combinations = len(layer_yml['properties'])
            else:
                property_combinations = 0

            # get all possible combinations of the values in triples
            option_dicts = []
            for num_channels in channel_combinations:
                for label_id in range(label_combinations):
                    num_different_keys = len(key_list_per_label[label_id])
                    # generate all choices from a list of counts
                    all_choices = []
                    for j in range(num_different_keys):
                        if len(all_choices) == 0:
                            for k in range(key_combinations_per_label[label_id][j]):
                                all_choices.append([k])
                        else:
                            new_choices = []
                            for choice in all_choices:
                                for k in range(key_combinations_per_label[label_id][j]):
                                    new_choices.append(choice + [k])
                            all_choices = new_choices
                    for key_combinations in range(np.prod(key_combinations_per_label[label_id])):
                        if layer_yml['layer_type'] == 'invariant_based_convolution':
                            for property_id in range(property_combinations):
                                option_dict = {}
                                option_dict['layer_type'] = layer_yml['layer_type']
                                option_dict['activation'] = layer_yml.get('activation', None)
                                option_dict['activation_kwargs'] = layer_yml.get('activation_kwargs', {})
                                option_dict['heads'] = []
                                option_dict['concatenate_heads'] = layer_yml.get('concatenate_heads', True)
                                head_dict = {}
                                head_dict['bias'] = layer_yml['bias']
                                head_dict['num'] = num_channels
                                head_dict['activation'] = layer_yml.get('activation', None)
                                head_dict['activation_kwargs'] = layer_yml.get('activation_kwargs', {})
                                label_dict = {}
                                label_dict['head'] = {'label_type': layer_yml['labels'][label_id]['label_type']}
                                label_dict['tail'] = {'label_type': layer_yml['labels'][label_id]['label_type']}
                                label_dict['bias'] = {'label_type': layer_yml['labels'][label_id]['label_type']}
                                if len(all_choices) > 0:
                                    choice = all_choices[key_combinations]
                                    for c_idx, value in enumerate(choice):
                                        key = key_list_per_label[label_id][c_idx]
                                        label_dict['head'][key] = layer_yml['labels'][label_id][key][value]
                                        label_dict['tail'][key] = layer_yml['labels'][label_id][key][value]
                                        label_dict['bias'][key] = layer_yml['labels'][label_id][key][value]
                                head_dict['labels'] = label_dict.copy()
                                head_dict['properties'] = {}
                                head_dict['properties']['name'] = layer_yml['properties'][property_id]['name']
                                head_dict['properties']['values'] = layer_yml['properties'][property_id][
                                    'values']
                                if layer_yml['properties'][property_id].get('cutoff', None) is not None:
                                    head_dict['properties']['cutoff'] = layer_yml['properties'][property_id][
                                        'cutoff']
                                option_dict['heads'].append(head_dict)
                                option_dicts.append(option_dict)
                        else:
                            option_dict = {}
                            option_dict['layer_type'] = layer_yml['layer_type']
                            option_dict['activation'] = layer_yml.get('activation', None)
                            option_dict['activation_kwargs'] = layer_yml.get('activation_kwargs', {})
                            if layer_yml.get('out_dim', None) is not None:
                                option_dict['out_dim'] = layer_yml['out_dim']
                            option_dict['heads'] = []
                            option_dict['concatenate_heads'] = layer_yml.get('concatenate_heads', True)
                            head_dict = {}
                            head_dict['bias'] = layer_yml['bias']
                            head_dict['num'] = num_channels
                            head_dict['activation'] = layer_yml.get('activation', None)
                            head_dict['activation_kwargs'] = layer_yml.get('activation_kwargs', {})
                            label_dict = {'label_type': layer_yml['labels'][label_id]['label_type']}
                            if len(all_choices) > 0:
                                choice = all_choices[key_combinations]
                                for c_idx, value in enumerate(choice):
                                    key = key_list_per_label[label_id][c_idx]
                                    label_dict[key] = layer_yml['labels'][label_id][key][value]
                            head_dict['labels'] = label_dict.copy()
                            option_dict['heads'].append(head_dict)
                            option_dicts.append(option_dict)

            if len(layers_per_architecture) <= layer_id:
                while len(layers_per_architecture) <= layer_id:
                    layers_per_architecture.append([])
            layers_per_architecture[layer_id] = option_dicts.copy()
        else:
            raise ValueError(f'Layer {layer_yml} with id: {layer_id} is not correctly defined: {error}')


def check_layer(i:int, layer: dict)->(bool, str):
    if 'layer_type' not in layer:
        return False, f'Layer type not defined in layer {i}, it must be convolution or aggregation'
    if layer['layer_type'] not in LayerTypes:
        return False, f'Layer type {layer["layer_type"]} not supported in layer {i}'

    if layer['layer_type'] == LayerTypes.LINEAR.value:
        required = ['out_features']
        for req in required:
            if req not in layer:
                return False, f'{req} not defined in layer {i}'
    elif layer['layer_type'] == LayerTypes.RESHAPE.value:
        pass
    elif layer['layer_type'] == LayerTypes.LAYER_NORM.value:
        pass
    elif layer['layer_type'] == LayerTypes.GLOBAL_POOLING.value:
        required = ['mode']
        for req in required:
            if req not in layer:
                return False, f'{req} not defined in layer {i}'
            else:
                if layer['mode'] not in ['mean', 'max', 'sum']:
                    return False, f'Mode must be mean, max or sum in layer {i}'
    elif layer['layer_type'] == LayerTypes.ACTIVATION.value:
        pass
    elif layer['layer_type'] == LayerTypes.DROPOUT.value:
        pass
    elif layer['layer_type'] == LayerTypes.BATCH_NORM.value:
        pass
    elif layer['layer_type'] == LayerTypes.GCN_CONVOLUTION.value:
        required=['bias', 'out_channels']
        for req in required:
            if req not in layer:
                return False, f'{req} not defined in layer {i}'
    elif layer['layer_type'] == LayerTypes.GAT_CONVOLUTION.value:
        required = ['bias', 'out_channels']
        for req in required:
            if req not in layer:
                return False, f'{req} not defined in layer {i}'
    elif layer['layer_type'] == LayerTypes.GATv2_CONVOLUTION.value:
        required = ['bias', 'out_channels']
        for req in required:
            if req not in layer:
                return False, f'{req} not defined in layer {i}'
    elif layer['layer_type'] == LayerTypes.GIN_CONVOLUTION.value:
        required = ['bias', 'out_channels']
        for req in required:
            if req not in layer:
                return False, f'{req} not defined in layer {i}'
    elif layer['layer_type'] == LayerTypes.SAGE_CONVOLUTION.value:
        required = ['bias', 'out_channels']
        for req in required:
            if req not in layer:
                return False, f'{req} not defined in layer {i}'
    elif layer['layer_type'] in [LayerTypes.INVARIANT_BASED_CONVOLUTION.value, LayerTypes.INVARIANT_BASED_AGGREGATION.value]:
        if 'heads' not in layer:
            return False, f'Heads not defined in layer {i}'
        else:
            if not isinstance(layer['heads'], list):
                return False, f'Heads must be a list in layer {i}'
            else:
                for j, head in enumerate(layer['heads']):
                    if not isinstance(head, dict):
                        return False, f'Channel must be a dictionary in layer {i}'
                    if 'bias' not in head:
                        return False, f'Bias not defined in layer {i}, it must be True or False'
                    if 'num' not in head:
                        return False, f'Number of heads must be defined in head {j}'
                    if 'labels' not in head:
                        return False, f'Labels not defined in head {j}'
                    else:
                        if layer['layer_type'] == 'invariant_based_convolution':
                            if 'head' not in head['labels']:
                                return False, f'Head not defined in head {i}'
                            else:
                                if 'label_type' not in head['labels']['head']:
                                    return False, f'Label type not defined in head {i} for head'
                            if 'tail' not in head['labels']:
                                return False, f'Tail not defined in head {i}'
                            else:
                                if 'label_type' not in head['labels']['tail']:
                                    return False, f'Label type not defined in head {i} for tail'
                            if 'bias' not in head['labels']:
                                return False, f'Bias not defined in head {i}'
                            else:
                                if 'label_type' not in head['labels']['bias']:
                                    return False, f'Label type not defined in head {i} for bias'
                            if 'properties' not in head:
                                return False, f'Properties not defined in head {i}'
                            else:
                                if 'name' not in head['properties']:
                                    return False, f'Property name not defined in head {i}'
                                if 'values' not in head['properties']:
                                    return False, f'Property values not defined in head {i}'
                                else:
                                    if not isinstance(head['properties']['values'], list):
                                        return False, f'Property values must be a list in head {i}'
                        elif layer['layer_type'] == 'invariant_based_aggregation':
                            if 'label_type' not in head['labels']:
                                return False, f'Label type not defined in head {i}'
    else:
        return False, f'Layer type {layer["layer_type"]} not supported in layer {i}'

    return True, ''

def check_layer_short_type(i, layer):
    if 'layer_type' not in layer:
        return False, f'Layer type not defined in layer {i}, it must be convolution or aggregation'
    if 'bias' not in layer:
        return False, f'Bias not defined in layer {i}, it must be True or False'
    if 'heads' not in layer:
        return False, f'Heads not defined in layer {i}, it must be a list of ints of parallel Heads used'
    else:
        if not isinstance(layer['heads'], list):
            return False, f'Heads must be a list in layer {i}'
    if 'labels' not in layer:
        return False, f'Labels not defined in layer {i}'
    else:
        if not isinstance(layer['labels'], list):
            return False, f'Labels must be a list in layer {i}'
        else:
            for l in layer['labels']:
                if not isinstance(l, dict):
                    return False, f'Label must be a dictionary in layer {i}'
                if 'label_type' not in l:
                    return False, f'Label type not defined in label in layer {i}'

    if 'properties' not in layer:
        if layer['layer_type'] == 'convolution':
            return False, f'Properties not defined in convolution layer {i}'
    else:
        if not isinstance(layer['properties'], list):
            return False, f'Properties must be a list in layer {i}'
        else:
            for prop in layer['properties']:
                if not isinstance(prop, dict):
                    return False, f'Property must be a dictionary in layer {i}'
                if 'name' not in prop:
                    return False, f'Property name not defined in layer {i}'
                if 'values' not in prop:
                    return False, f'Property values not defined in layer {i}'
                else:
                    if not isinstance(prop['values'], list):
                        return False, f'Property values must be a list'
    return True, ''

def check_network_architectures(network_architectures, print_errors=False):
    '''
    Checks if the network architectures is in the correct format
    '''
    invalid_architectures = []
    for i, architecture in enumerate(network_architectures):
        invalid_layers = []
        for i, layer in enumerate(architecture):
            correct, error = check_layer(i, layer)
            if not correct:
                invalid_layers.append(error)
        if len(invalid_layers) > 0:
            invalid_architectures.append(i)
            if print_errors:
                for error in invalid_layers:
                        print(error)
    if len(invalid_architectures) > 0:
        if print_errors:
            for i in invalid_architectures:
                print(f'Architecture {i} is invalid')
        return False
    return True


