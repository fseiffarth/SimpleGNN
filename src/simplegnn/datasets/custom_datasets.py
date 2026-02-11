from typing import List

import networkx as nx
import numpy as np

from simplegnn.datasets.custom_benchmarks.counting_rings import CountingRings
from simplegnn.datasets.custom_benchmarks.even_odd_rings import EvenOddRings
from simplegnn.datasets.custom_benchmarks.string_graphs import EvenPairs, ParityCheck, FirstChar, LastChar
from simplegnn.datasets.custom_benchmarks.long_rings import LongRings
from simplegnn.datasets.custom_benchmarks.node_classification_test import NodeClassificationTest
from simplegnn.datasets.custom_benchmarks.ring_diagonals import RingDiagonals
from simplegnn.datasets.custom_benchmarks.ring_transfer import RingTransfer
from simplegnn.datasets.custom_benchmarks.snowflakes import Snowflakes


def long_rings(data_size=1200, ring_size=100, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    return LongRings(data_size=data_size, ring_size=ring_size, seed=seed, *args, **kwargs)

def even_odd_rings(data_size=1200, ring_size=100, difficulty=1, count=False, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    """
    Create a benchmark dataset consisting of labeled rings with ring_size nodes and labels.
    The label of the graph is determined by the following:
    - Select the node with label and the node with distance ring_size//2 say x and the ones with distances ring_size//4, ring_size//8, say y_1, y_2 and z_1, z_2
    Now consider the numbers:
    a = 1 + x
    b = y_1 + y_2
    c = z_1 + z_2
    and distinct the cases odd and even. This defines the 8 possible labels of the graphs.
    """
    return EvenOddRings(data_size=data_size, ring_size=ring_size, difficulty=difficulty, count=count, seed=seed, *args, **kwargs)

def ring_diagonals( data_size=1200, ring_size=100,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    """
    Create a dataset of ring graphs with diagonals.
    """
    return RingDiagonals(data_size=data_size, ring_size=ring_size, *args, **kwargs)

def snowflakes(smallest_snowflake=3, largest_snowflake=12, flakes_per_size=100, seed=764, generation_type='binary',*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    """
    Create a dataset of snowflake graphs.
    """
    return Snowflakes(smallest_snowflake=smallest_snowflake, largest_snowflake=largest_snowflake, flakes_per_size=flakes_per_size, plot=False, seed=seed, generation_type=generation_type)


def counting_rings(data_size=1000, ring_size=3, min_rings=0, max_rings=9, seed=42, *args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    return CountingRings(data_size=data_size, ring_size=ring_size, min_rings=min_rings, max_rings=max_rings, seed=seed, *args, **kwargs)

def ring_transfer(data_size=1200, node_dimension=10, ring_size=100, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[np.ndarray[float]]]:
    return RingTransfer(data_size=data_size, node_dimension=node_dimension, ring_size=ring_size, seed=seed)



def node_classification_test(data_size=1, max_size=1000, num_node_features=1,seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    return NodeClassificationTest(data_size=data_size, max_size=max_size, num_node_features=num_node_features,seed=seed, *args, **kwargs)

# Represent strings as graphs

# Parity check: graphs represent bit strings, label is 1 if number of 1s is even, else 0
def parity_check(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    return ParityCheck(data_size=data_size, max_size=max_size, seed=seed, *args, **kwargs)

# Even pairs: graphs represent bit strings, label is 1 if first and last bit are the same, else 0
def even_pairs(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    return EvenPairs(data_size=data_size, max_size=max_size, seed=seed, *args, **kwargs)

# First Char: graphs represent bit strings, label is 1 if first bit is 1, else 0
def first_char(data_size=1500, max_size=40, seed=764, *args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    return FirstChar(data_size=data_size, max_size=max_size, seed=seed, *args, **kwargs)

# Last Char: graphs represent bit strings, label is 1 if last bit is 1, else 0
def last_char(data_size=1500, max_size=40, seed=764, *args, **kwargs) -> tuple[List[nx.Graph], List[int]]:
    return LastChar(data_size=data_size, max_size=max_size, seed=seed, *args, **kwargs)
