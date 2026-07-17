import gzip
import os
import pickle
from pathlib import Path
from typing import List

import torch

from simplegnn.utils.utils import convert_to_tuple


class ConsolidatedProperties:
    """
    Graph-major consolidated view of a Properties object.

    The raw Properties store one (K_p, 2) global node-pair tensor per property
    value p. The invariant layers need per-graph slices across *all* property
    values of a description at once, so this view interleaves the per-value
    pair lists into a single graph-major layout (all pairs of graph g
    contiguous), with graph-local int32 node ids and an int16 property-value id
    per row. Built once per Properties object and shared by every invariant
    layer that uses the description — the per-head data reduces to one int32
    parameter-index vector aligned to these rows.
    """

    def __init__(self, prop: 'Properties', x_slices: torch.Tensor):
        keys = list(prop.properties.keys())
        self.keys = keys
        self.key_index = {key: idx for idx, key in enumerate(keys)}
        num_graphs = x_slices.shape[0] - 1
        lens_per_key = torch.stack([prop.properties_slices[k].diff() for k in keys])  # (K, G)
        total_per_graph = lens_per_key.sum(dim=0)
        self.slices = torch.cat([torch.zeros(1, dtype=torch.int64), total_per_graph.cumsum(dim=0)])
        self.total_rows = int(self.slices[-1])
        # start offset of each key's rows inside every graph's block: rows of
        # graph g are ordered by key (in `keys` order), so key k starts after
        # the lengths of all previous keys
        self.prior_per_key = torch.zeros((len(keys), num_graphs), dtype=torch.int64)
        if len(keys) > 1:
            self.prior_per_key[1:] = lens_per_key[:-1].cumsum(dim=0)

        lp = torch.empty((self.total_rows, 2), dtype=torch.int32)
        kid_dtype = torch.int16 if len(keys) <= torch.iinfo(torch.int16).max else torch.int32
        key_id = torch.empty(self.total_rows, dtype=kid_dtype)
        x_starts = x_slices[:-1]
        graph_ids = torch.arange(num_graphs, dtype=torch.int64)
        for k_idx, key in enumerate(keys):
            pairs = prop.properties[key]
            target, graph_of_row = self._target_rows_impl(prop, k_idx)
            lp[target] = (pairs - x_starts[graph_of_row].unsqueeze(1)).to(torch.int32)
            key_id[target] = k_idx
        self.lp = lp
        self.key_id = key_id
        self._prop = prop

    def _target_rows_impl(self, prop, k_idx: int):
        """Consolidated row index (and graph id) of every row of property value k_idx."""
        key = self.keys[k_idx]
        sl = prop.properties_slices[key]
        lens_k = sl.diff()
        num_graphs = lens_k.shape[0]
        graph_of_row = torch.repeat_interleave(torch.arange(num_graphs, dtype=torch.int64), lens_k)
        pos_in_graph = torch.arange(int(sl[-1]), dtype=torch.int64) - sl[:-1][graph_of_row]
        target = self.slices[:-1][graph_of_row] + self.prior_per_key[k_idx][graph_of_row] + pos_in_graph
        return target, graph_of_row

    def target_rows(self, property_key) -> torch.Tensor:
        """Consolidated row index of every row of the given property value."""
        return self._target_rows_impl(self._prop, self.key_index[property_key])[0]



class Properties:
    def __init__(self, path: Path, db_name: str, property_name: str, valid_values: dict[tuple[int, int], list[int]]):
        self.name = property_name
        self.db = db_name
        self.valid_values = {}
        self.all_values = None
        # load the properties from a file, first decompress the file with gzip and then load the pickle file
        self.properties = None
        self.properties_slices = None
        self.num_properties = {}
        self.valid_property_map = {}

        # path to the data
        data_path = path.joinpath(db_name).joinpath(f'{db_name}_properties_{property_name}.pt')
        # path to the info file
        info_path = path.joinpath(db_name).joinpath(f'{db_name}_properties_{property_name}.yml')

        # check if the file exists, otherwise raise an error
        if os.path.isfile(data_path) and os.path.isfile(info_path):
            with gzip.open(data_path, 'rb') as f:
                self.all_values, self.properties, self.properties_slices = pickle.load(f)
        else:
            raise FileNotFoundError(f'File {data_path} or {info_path} not found')

        for (layer_id, channel_id), values in valid_values.items():
            self.add_properties(layer_id=layer_id, channel_id=channel_id, valid_values=values)

        self._consolidated = None

    def consolidated(self, x_slices) -> ConsolidatedProperties:
        """Lazily built graph-major view shared by all invariant layers."""
        if self._consolidated is None:
            self._consolidated = ConsolidatedProperties(self, x_slices)
        return self._consolidated


    def add_properties(self, valid_values: List[int], layer_id: int, channel_id: int):
        self.valid_values[(layer_id, channel_id)] = []
        self.valid_property_map[(layer_id, channel_id)] = {}
        # if property name is edge_label_distance, and the valid values is a list of values interpret them as the distances and take all the values from self.all_values with first entry equal to the distance
        if 'edge_label_distances' in self.name:
            # check if valid_values is a list of ints
            if type(valid_values[0]) == int:
                tmp_valid_values = []
                for v in self.all_values:
                    if v[0] in valid_values:
                        tmp_valid_values.append(v)
                self.valid_values[(layer_id, channel_id)] = tmp_valid_values
            else:
                self.valid_values[(layer_id, channel_id)] = valid_values
        elif 'circle_distances' in self.name:
            if type(valid_values[0]) == str:
                for v in valid_values:
                    if v == 'no_circles':
                        for x in self.all_values:
                            if x[1] == 0 and x[2] == 0:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'circles':
                        for x in self.all_values:
                            if x[1] == 1 and x[2] == 1:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'in_circles':
                        for x in self.all_values:
                            if x[1] == 0 and x[2] == 1:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'out_circles':
                        for x in self.all_values:
                            if x[1] == 1 and x[2] == 0:
                                self.valid_values[(layer_id, channel_id)].append(x)
            else:
                self.valid_values[(layer_id, channel_id)] = valid_values
        else:
            self.valid_values[(layer_id, channel_id)] = valid_values

        # check if all the valid values are in the valid properties, if not raise an error
        invalid_values = []
        for value in self.valid_values[(layer_id, channel_id)]:
            if value not in self.all_values:
                invalid_values.append(value)
        if len(invalid_values) > 0:
            # remove invalid values from the valid values
            self.valid_values[(layer_id, channel_id)] = [v for v in self.valid_values[(layer_id, channel_id)] if v not in invalid_values]
            print(f'There are properties that are not arising in the dataset: {invalid_values}')

        # number of valid properties
        self.num_properties[(layer_id, channel_id)] = len(self.valid_values[(layer_id, channel_id)])
        for i, value in enumerate(self.valid_values[(layer_id, channel_id)]):
            try:
                property_value = int(value)
                self.valid_property_map[(layer_id, channel_id)][property_value] = i
            except:
                # check if the length of the value is 1, if not iterate over the values
                try:
                    len(value[0])
                    for v in value:
                        self.valid_property_map[(layer_id, channel_id)][convert_to_tuple(v)] = i
                except:
                    self.valid_property_map[(layer_id, channel_id)][convert_to_tuple(value)] = i
