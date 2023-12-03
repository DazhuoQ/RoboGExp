from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.utils import is_undirected, barabasi_albert_graph, one_hot
import torch
import numpy as np
from torch_geometric.data import Data

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)


dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=500, num_edges=5),
    motif_generator=HouseMotif(),
    num_motifs=10,
    num_graphs=1,
)

for graph in dataset:
    print(graph)
    x = one_hot(graph.y, num_classes=4)
    data = Data(x=x, edge_index=graph.edge_index, y=graph.node_mask)
    print(data)
    torch.save(data, '/home/cs.aau.dk/yj25pu/RoboGExp/datasets/BAHouse/data.pt')
