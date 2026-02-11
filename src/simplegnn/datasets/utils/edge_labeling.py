import gzip
import os
import pickle
from pathlib import Path
from typing import List

from simplegnn.utils.utils import convert_to_tuple





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
