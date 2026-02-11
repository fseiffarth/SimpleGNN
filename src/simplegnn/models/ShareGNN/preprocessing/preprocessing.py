# generate preprocessing by scanning the config file
import json
from pathlib import Path

from simplegnn.datasets.graph_dataset import GraphDataset
from simplegnn.datasets.utils.node_labeling import get_label_string, save_labels_to_file, save_primary_labels, \
    save_trivial_labels, save_index_labels, save_degree_labels, save_wl_labels, save_labeled_degree_labels, \
    save_wl_labeled_labels, save_wl_labeled_edges_labels, save_cycle_labels, save_subgraph_labels, save_clique_labels, \
    save_betweenness_centrality_labels, load_labels, combine_node_labels
from simplegnn.framework.run_configuration import get_run_configs
from simplegnn.models.ShareGNN.preprocessing.properties import write_distance_properties, write_distance_edge_properties


def layer_to_labels(experiment_configuration, layer_strings: json, graph_data: GraphDataset, generation_times_labels_path=None) -> Path:
    file_path = None
    layer = json.loads(layer_strings)
    label_path = experiment_configuration['paths']['labels'].joinpath(f'{graph_data.name}')
    # check if the path exists, otherwise create it
    if not label_path.exists():
        label_path.mkdir()
    # if label_type is a list, then the layer is a combination of different label types
    if type(layer['label_type']) == list and len(layer['label_type']) > 1:
        # recursively call the function for each label type
        labels = []
        label_names = []
        for label_type in layer['label_type']:
            new_layer_string = layer.copy()
            # remove the label_type key and replace it with the new label_type
            new_layer_string['label_type'] = label_type
            l_path = layer_to_labels(experiment_configuration, json.dumps(new_layer_string), graph_data,
                                     generation_times_labels_path)
            # get all after last /
            label_name = '_'.join(l_path.stem.split('_')[1:-1])
            label_names.append(label_name)
            labels.append(load_labels(l_path))
        # combine the labels
        combined_labels = combine_node_labels(labels)
        l = f'{combined_labels.label_name}'
        max_labels = layer.get('max_labels', None)
        if max_labels is not None:
            l += f'_{max_labels}'
        file_path = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
        save_labels_to_file(file_path, combined_labels.dataset_name, l, combined_labels.node_labels,
                            max_labels=layer.get('max_labels', None))
    else:
        if isinstance(layer['label_type'], list):
            layer['label_type'] = layer['label_type'][0]
        # switch case for the different layers
        if layer['label_type'] == 'primary':
            file_path = save_primary_labels(graph_data=graph_data,
                                            label_path=label_path,
                                            max_labels=layer.get('max_labels', None),
                                            save_times=generation_times_labels_path)
        elif layer['label_type'] == 'trivial':
            file_path = save_trivial_labels(graph_data=graph_data,
                                            label_path=label_path,
                                            save_times=generation_times_labels_path)
        elif layer['label_type'] == 'index':
            file_path = save_index_labels(graph_data=graph_data,
                                          max_labels=layer.get('max_labels', None),
                                          label_path=label_path,
                                          save_times=generation_times_labels_path)
        elif layer['label_type'] == 'index_text':
            file_path = save_index_labels(graph_data=graph_data,
                                          max_labels=layer.get('max_labels', None),
                                          label_path=label_path,
                                          save_times=generation_times_labels_path,
                                          index_text=True)
        elif layer['label_type'] == 'degree':
            file_path = save_degree_labels(graph_data=graph_data,
                                           label_path=label_path,
                                           max_labels=layer.get('max_labels', None),
                                           save_times=generation_times_labels_path)
        elif layer['label_type'] == 'wl':
            layer['max_labels'] = layer.get('max_labels', None)
            layer['depth'] = layer.get('depth', 3)
            if layer['depth'] == 0:
                file_path = save_degree_labels(graph_data=graph_data,
                                               label_path=label_path,
                                               max_labels=layer.get('max_labels', None),
                                               save_times=generation_times_labels_path)
            else:
                file_path = save_wl_labels(graph_data=graph_data,
                                           depth=layer.get('depth', 3),
                                           max_labels=layer['max_labels'],
                                           label_path=label_path,
                                           save_times=generation_times_labels_path)
        elif layer['label_type'] == 'wl_labeled':
            layer['max_labels'] = layer.get('max_labels', None)
            layer['depth'] = layer.get('depth', 3)
            base_labels = None
            if 'base_labels' in layer:
                base_labels = dict()
                base_label_path = experiment_configuration['paths']['labels'].joinpath(
                    f'{graph_data.name}').joinpath(
                    f"{graph_data.name}_labels_{get_label_string(layer['base_labels'])}.pt")
                base_labels['layer_dict'] = layer['base_labels']
                base_labels['layer_string'] = get_label_string(layer['base_labels'])
                base_labels['labels'] = load_labels(base_label_path)

            if layer['depth'] == 0:
                file_path = save_labeled_degree_labels(graph_data=graph_data,
                                                       label_path=label_path,
                                                       max_labels=layer.get('max_labels', None),
                                                       save_times=generation_times_labels_path)
            else:
                file_path = save_wl_labeled_labels(graph_data=graph_data,
                                                   depth=layer.get('depth', 3),
                                                   max_labels=layer['max_labels'],
                                                   label_path=label_path,
                                                   base_labels=base_labels,
                                                   save_times=generation_times_labels_path)
        elif layer['label_type'] == 'wl_labeled_edges':
            layer['max_labels'] = layer.get('max_labels', None)
            layer['depth'] = layer.get('depth', 3)
            base_labels = None
            if 'base_labels' in layer:
                base_labels = dict()
                base_label_path = experiment_configuration['paths']['labels'].joinpath(
                    f'{graph_data.name}').joinpath(
                    f"{graph_data.name}_labels_{get_label_string(layer['base_labels'])}.pt")
                base_labels['layer_dict'] = layer['base_labels']
                base_labels['layer_string'] = get_label_string(layer['base_labels'])
                base_labels['labels'] = load_labels(base_label_path)

            file_path = save_wl_labeled_edges_labels(graph_data=graph_data,
                                                     depth=layer.get('depth', 3),
                                                     max_labels=layer['max_labels'],
                                                     label_path=label_path,
                                                     base_labels=base_labels,
                                                     save_times=generation_times_labels_path)
        elif layer['label_type'] == 'simple_cycles' or layer['label_type'] == 'induced_cycles':
            cycle_type = 'simple' if layer['label_type'] == 'simple_cycles' else 'induced'
            if 'max_labels' not in layer:
                layer['max_labels'] = None
            if 'max_cycle_length' not in layer:
                layer['max_cycle_length'] = None
            if 'min_cycle_length' not in layer:
                layer['min_cycle_length'] = None
            file_path = save_cycle_labels(graph_data=graph_data,
                                          min_cycle_length=layer['min_cycle_length'],
                                          max_cycle_length=layer['max_cycle_length'],
                                          max_labels=layer["max_labels"],
                                          cycle_type=cycle_type,
                                          label_path=label_path,
                                          save_times=generation_times_labels_path)
        elif layer['label_type'] == 'subgraph':
            if 'id' in layer:
                if layer['id'] > len(experiment_configuration['subgraphs']):
                    raise ValueError(
                        f'Please specify the subgraphs in the config files under the key "subgraphs" as folllows: subgraphs: - "[nx.complete_graph(4)]"')
                else:
                    import ast
                    subgraph_list = ast.literal_eval(experiment_configuration['subgraphs'][layer['id']])
                    file_path = save_subgraph_labels(graph_data=graph_data,
                                                     subgraphs=subgraph_list,
                                                     subgraph_id=layer['id'],
                                                     max_labels=layer.get('max_labels', None),
                                                     label_path=label_path,
                                                     save_times=generation_times_labels_path)
            else:
                raise ValueError(
                    f'Please specify the id of the subgraph in the layer with description {layer_strings}.')
        elif layer['label_type'] == 'cliques':
            if 'max_labels' not in layer:
                layer['max_labels'] = None
            if 'max_clique_size' not in layer:
                layer['max_clique_size'] = None
            file_path = save_clique_labels(graph_data=graph_data,
                                           max_clique=layer['max_clique_size'],
                                           max_labels=layer.get('max_labels', None),
                                           label_path=label_path,
                                           save_times=generation_times_labels_path)
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
        else:
            # print in red in the console
            print(f'The automatic generation of labels for the layer type {layer["label_type"]} is not supported yet.')
    return file_path


def property_to_properties(experiment_configuration, property_strings: json, graph_data: GraphDataset, generation_times_properties_path=None) -> None:
    properties_path = experiment_configuration['paths']['properties'].joinpath(f'{graph_data.name}')
    # check if the path exists, otherwise create it
    if not properties_path.exists():
        properties_path.mkdir()
    # switch case for the different properties
    properties = json.loads(property_strings)
    if properties['name'] == 'distances':
        if 'cutoff' not in properties:
            properties['cutoff'] = None
        print(f'Generating distance properties for {graph_data.name} with cutoff {properties["cutoff"]}')
        write_distance_properties(graph_data, out_path=properties_path, cutoff=properties['cutoff'],
                                  save_times=generation_times_properties_path)
    # TODO: change the edge_label_distances to the new torch format
    elif properties['name'] == 'edge_label_distances':
        if 'cutoff' not in properties:
            properties['cutoff'] = None
        print(
            f'Generating edge label distance properties for {graph_data.name} with cutoff {properties["cutoff"]}')
        write_distance_edge_properties(graph_data, out_path=properties_path, cutoff=properties['cutoff'],
                                       save_times=generation_times_properties_path)


def preprocessing_from_config(experiment_configuration, graph_data: GraphDataset,
                              generation_times_labels_path: Path = None,
                              generation_times_properties_path: Path = None) -> None:
    # get the layers from the config file
    run_configs = get_run_configs(experiment_configuration)
    # preprocessed layers
    preprocessed_label_dicts = set()
    proprocessed_label_dicts_first = set()
    preprocessed_properties = set()
    # iterate over the layers
    for run_config in run_configs:
        for layer in run_config.layers:
            for property_dict in layer.get_unique_property_dicts():
                p_dict = property_dict.property_dict.copy()
                p_dict.pop('values')
                json_property = json.dumps(p_dict, sort_keys=True)
                preprocessed_properties.add(json_property)
            for label_dict in layer.get_unique_layer_dicts():
                json_layer = json.dumps(label_dict, sort_keys=True)
                preprocessed_label_dicts.add(json_layer)
                # if key base_labels in label_dict, then add the base_labels to the preprocessed_label_dicts
                if 'base_labels' in label_dict:
                    json_layer = json.dumps(label_dict['base_labels'], sort_keys=True)
                    proprocessed_label_dicts_first.add(json_layer)
    # generate all necessary labels and properties, first need to create the nx graphs to run the algorithms on
    # graph_data.create_nx_graphs(directed=False)
    for layer in proprocessed_label_dicts_first:
        layer_to_labels(layer)
    for layer in preprocessed_label_dicts:
        layer_to_labels(experiment_configuration=experiment_configuration, layer_strings=layer, graph_data=graph_data, generation_times_labels_path=generation_times_labels_path)
    for preprocessed_property in preprocessed_properties:
        property_to_properties(experiment_configuration=experiment_configuration, property_strings=preprocessed_property, graph_data=graph_data, generation_times_properties_path=generation_times_properties_path)


