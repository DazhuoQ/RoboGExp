import torch_geometric.transforms as T
from torch import default_generator
from torch.utils.data import random_split
from torch_geometric.datasets import Reddit, PPI, Planetoid
import torch
import os


def get_dataset(dataset_root, dataset_name):
    if dataset_name.lower() in ['reddit']:
        return Reddit(os.path.join(dataset_root, dataset_name))
    elif dataset_name.lower() in ['bahouse']:
        return torch.load(os.path.join(dataset_root, dataset_name, 'data.pt'))
    elif dataset_name.lower() in ['ppi']:
        return PPI(os.path.join(dataset_root, dataset_name))
    elif dataset_name.lower() in ['citeseer']:
        return Planetoid(dataset_root, dataset_name)
    else:
        raise ValueError(f"{dataset_name} is not defined.")

def get_dim(dataset_name):
    if dataset_name.lower() in ['reddit']:
        return 602, 41
    elif dataset_name.lower() in ['bahouse']:
        return 0, 4
    elif dataset_name.lower() in ['ppi']:
        return 50, 121
    elif dataset_name.lower() in ['citeseer']:
        return 3703, 6
    else:
        raise ValueError(f"{dataset_name} is not defined.")    