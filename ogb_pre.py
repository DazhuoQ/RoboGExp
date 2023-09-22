from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph
import torch
from torch_geometric.loader import DataLoader

pyg_dataset = PygPCQM4Mv2Dataset(root = 'datasets/', smiles2graph = smiles2graph)
split_dict = pyg_dataset.get_idx_split()
train_idx = split_dict['train']
valid_idx = split_dict['valid']
train_set = pyg_dataset[split_dict['train']]
valid_set = pyg_dataset[split_dict['valid']]

train_loader = DataLoader(pyg_dataset[train_idx])

for graph in train_loader:
    print(graph)


# dataset = torch.load('datasets/Mutagenicity/processed/data.pt')
# print(dataset)
# print('============================')
# dataset = torch.load('datasets/pcqm4m-v2/processed/geometric_data_processed.pt')
# print(dataset)