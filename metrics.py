import numpy as np
import torch
import random
import sys
import os
import os
import pytz
import logging
from scipy import spatial
import numpy as np
import random
import torch
import networkx as nx
import copy
from rdkit import Chem
from datetime import datetime
from torch_geometric.utils import subgraph, to_dense_adj, to_networkx
from torch_geometric.data import Data, Batch, Dataset, DataLoader


def sparsity(explainer_graph_set, original_graph_set):
    num_edges = 0
    num_nodes = 0
    tot_edges = 0
    tot_nodes = 0
    for graph in explainer_graph_set:
        num_edges += graph.edge_index.shape[1]
        num_nodes += graph.x.shape[0]
    for graph in original_graph_set:
        tot_edges += graph.edge_index.shape[1]
        tot_nodes += graph.x.shape[0]
    return 1 - (num_nodes + num_edges) / (tot_nodes + tot_edges)


def contrastivity(first_set, second_set):
    vector_first = []
    vector_second = []
    for graph in first_set:
        vector_first.append(nx.average_clustering(to_networkx(graph)))
    for graph in second_set:
        vector_second.append(nx.average_clustering(to_networkx(graph)))
    cos_sim = 1 - spatial.distance.cosine(vector_first, vector_second)
    return cos_sim





