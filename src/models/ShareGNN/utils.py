from typing import Optional

from datasets.utils.node_labeling import get_label_string




class LabelDict:
    """
    LabelDict is a class that holds the information of the labels (coming from an invariant) of a head in a ShareGNN
    param: label_dict: the dictionary that contains the information of the labels
    """
    def __init__(self, label_dict: dict):
        self.label_dict = label_dict
        self.source_labels = label_dict.get('head', None)
        self.target_labels = label_dict.get('tail', None)
        self.bias_labels = label_dict.get('bias', None)

    def get_source_string(self)->str:
        """
        outputs the string representation of the source labels, i.e., the source regarding the message passing direction
        :return: the string representation of the source labels
        """
        return get_label_string(self.source_labels)
    def get_target_string(self)->str:
        """
        outputs the string representation of the target labels, i.e., the target regarding the message passing direction
        :return: the string representation of the target labels
        """
        return get_label_string(self.target_labels)
    def get_bias_string(self)->str:
        """
        outputs the string representation of the bias labels
        :return: the string representation of the bias labels
        """
        return get_label_string(self.bias_labels)

class PropertyDict:
    """
    PropertyDict is a class that holds the information of the pairwise properties of a head in a ShareGNN
    parameters:
    property_dict: the dictionary that contains the information of the properties
    """
    def __init__(self, property_dict: dict):
        self.property_dict = property_dict

    def get_values(self)-> Optional[list]:
        """
        outputs all possible values of the property (e.g., the distances between two nodes which are considered for message passing)
        :return: a list of all possible values
        """
        return self.property_dict.get('values', None)

    def get_property_string(self)->str:
        """
        outputs the string representation of the property
        :return: the string representation of the property
        """
        string_name = self.property_dict['name']
        if 'cutoff' in self.property_dict:
            string_name += f"_cutoff_{self.property_dict['cutoff']}"
        return string_name



class LayerHead:
    """
    LayerHead defines one head (regarding multi-heads) of a layer in a ShareGNN, i.e., the type of the labels e.t.c.
    :param info_dict: the dictionary that contains the information of the head
    :param head_id: the id of the head in the layer (from 0 to n-1) where n is the total number of heads
    """
    def __init__(self, info_dict: dict, head_id):
        self.head_id = head_id
        self.label_dict = LabelDict(info_dict.get('labels', None))
        self.property_dict = PropertyDict(info_dict.get('properties', None))
        # if head, tail, bias is not specified, set head tail bias to the same value
        self.source_labels = self.label_dict.source_labels
        self.head_node_labels = -1
        self.target_labels = self.label_dict.target_labels
        self.target_node_labels = -1
        self.bias_labels = self.label_dict.bias_labels
        self.bias_node_labels = -1
        self.bias = info_dict.get('bias', False)
        if self.source_labels is None:
            self.source_labels = self.label_dict.label_dict
        if self.target_labels is None:
            self.target_labels = self.source_labels
        if self.bias_labels is None:
            self.bias_labels = self.source_labels


class Layer:
    """
    This class holds the information of a layer of a ShareGNN
    :param: layer_dict: the dictionary that contains the layer information
    :param: layer_id: the id of the layer in the ShareGNN (from 0 to n-1) where n is the total number of layers
    """
    def __init__(self, layer_dict, layer_id):
        """
        Constructor of the Layer
        :param layer_dict: the dictionary that contains the layer information
        """
        self.layer_type = layer_dict["layer_type"]
        self.layer_dict = layer_dict
        self.layer_heads = []
        self.layer_id = layer_id
        if self.layer_type in ['invariant_based_convolution', 'invariant_based_aggregation']:
            for c_id, head_entry in enumerate(layer_dict.get('heads', [])):
                self.layer_heads.append(LayerHead(head_entry, c_id))

    def get_unique_layer_dicts(self) -> list[dict]:
        """
        :return: the unique label dictionaries of the layer. This is used for preprocessing and loading of the label information.
        """
        unique_dicts = []
        for head in self.layer_heads:
            if not isinstance(head.source_labels, dict):
                raise ValueError("Source labels must be a dict")
            if head.source_labels not in unique_dicts:
                unique_dicts.append(head.source_labels)
            if head.target_labels not in unique_dicts:
                unique_dicts.append(head.target_labels)
            if head.bias_labels not in unique_dicts:
                unique_dicts.append(head.bias_labels)
        return unique_dicts

    def get_unique_property_dicts(self) -> list[dict]:
        """
        :returns: the unique property dictionaries of the layer. This is used for preprocessing and loading of the property information.
        """
        unique_dicts = []
        for head in self.layer_heads:
            if head.property_dict.property_dict not in unique_dicts and head.property_dict.property_dict is not None:
                unique_dicts.append(head.property_dict.property_dict)
        return unique_dicts

    def get_source_string(self, head_id=0):
        """
        :return: the string representation of the source labels of the head
        """
        return get_label_string(self.layer_heads[head_id].source_labels)
    def get_target_string(self, head_id=0):
        """
        :return: the string representation of the target labels of the head
        """
        return get_label_string(self.layer_heads[head_id].target_labels)
    def get_bias_string(self, head_id=0):
        """
        :return: the string representation of the bias labels of the head
        """
        return get_label_string(self.layer_heads[head_id].bias_labels)

    def get_layer_label_strings(self)->list[str]:
        label_string_list = set()
        for head in range(len(self.layer_heads)):
            label_string_list.add(self.get_source_string(head))
            label_string_list.add(self.get_target_string(head))
            label_string_list.add(self.get_bias_string(head))
        return list(label_string_list)

    def num_heads(self):
        """
        :return: the number of heads of the layer
        """
        return len(self.layer_heads)
