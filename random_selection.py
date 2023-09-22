import time
import logging
import random
import networkx
import pandas as pd
import torch
import os
import networkx as nx
import numpy as np
import hydra
import scipy.sparse as sp
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import add_remaining_self_loops, to_scipy_sparse_matrix, \
    k_hop_subgraph, to_networkx, subgraph
from gnnNets import get_gnnNets
from visualization import PlotUtils
from dataset import get_dataset, get_dataloader
from gspan.gspan import gSpan
from utils import set_seed
from warnings import simplefilter
from tqdm import tqdm


class Sample(object):
    def __init__(self, bounds, size):
        self._size = size
        self._bounds = bounds
        self._counter = [0 for _ in range(len(bounds))]
        self._sample = []

    def feed(self, item):
        l_b = self._bounds[item[1]][0]
        u_b = self._bounds[item[1]][1]
        if len(self._sample) == self._size:
            return
        if self._counter[item[1]] < l_b or (self._counter[item[1]] > l_b and self._bounds < u_b):
            self._sample.append(item)
            self._counter[item[1]] += 1

    def get_result(self):
        return self._sample


def evaluation(graph, solution, model, num_classes, device):
    ex_labels = model(graph).to('cpu')
    result = F.softmax(ex_labels, -1).argmax(-1)
    group_count = [0 for _ in range(num_classes)]
    fidelity_value = [0 for _ in range(num_classes)]
    graph_group_set = [[] for _ in range(num_classes)]
    for element in solution:
        group_count[element[1]] += 1
        ground_truth = graph.y[graph.batch[element[0]].item()]
        subset, _, _, _ = k_hop_subgraph(element[0], 1, graph.edge_index)
        edge_index = subgraph(subset, graph.edge_index, relabel_nodes=True)[0]
        G = Data(graph.x[subset], edge_index)
        graph_group_set[element[1]].append(G)
        G.to(device)
        fidelity_value[element[1]] += (int(ground_truth == result[graph.batch[element[0]].item()].item())
                                       - int(ground_truth == F.softmax(model(G), -1).argmax(-1)))

    faithfulness_score = 0
    fidelity_score = 100

    for i in range(num_classes):
        if group_count[i] == 0:
            continue

        fidelity_score = 1.0 * fidelity_value[i] / group_count[i]
        logging.info(f'fidelity- score label {i}: {fidelity_score}')

    graph_set = []
    for i in range(num_classes):
        if len(graph_group_set[i]) == 0:
            graph_set.append((nx.Graph(), None))
            continue
        big_graph = Batch.from_data_list(graph_group_set[i])
        graph_set.append((to_networkx(big_graph, to_undirected=True), big_graph.x))

    return fidelity_score


def data_record(dataset_name, budget, fidelity_score, cost_time):
    project_name = os.path.dirname(__file__)
    exp_name = os.path.join(project_name, 'experiments/data_record')
    data_name = os.path.join(exp_name, f'random_{dataset_name}')
    data = {'fidelity': fidelity_score, 'cost_time': f'{cost_time:.4f}'}
    data_frame = pd.DataFrame(data, index=[0])
    if not os.path.exists(data_name):
        data_frame.to_csv(data_name, header=True, mode='a', index=True, sep=',')
    else:
        data_frame.to_csv(data_name, header=False, mode='a', index=True, sep=',')


def pre_process(graph, model):
    ex_labels = model(graph).to('cpu')
    result = F.softmax(ex_labels, -1).argmax(-1)
    elements = []
    graph.to('cpu')
    for i in range(graph.num_nodes):
        elements.append((i, result[graph.batch[i].item()].item()))
    return elements


def random_selection(graph, model, bounds, cardinality_k, num_classes, dataset_name):
    elements = pre_process(graph, model)
    random.shuffle(elements)
    random_algorithm = Sample(bounds, cardinality_k)
    counter = 0
    for element in tqdm(elements, desc='Process', leave=True, ncols=100, unit='B', unit_scale=True):
        if element[1] == 1:
            counter += 1
        random_algorithm.feed(element)

    result_elements = random_algorithm.get_result()
    return result_elements


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.params = config.models.params[config.datasets.dataset_name]
    dataset = get_dataset(dataset_root='datasets', dataset_name=config.datasets.dataset_name)
    dataset_params = {
        'batch_size': config.models.params.batch_size,
        'data_split_ratio': config.datasets.data_split_ratio,
        'seed': config.datasets.seed
    }
    set_seed(3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = get_dataloader(dataset, **dataset_params)
    test_indices = loader['test'].dataset.indices
    graph = Batch.from_data_list(dataset[test_indices]).to(device)
    num_node = graph.num_nodes
    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)
    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.params.gnn_latent_dim)}l_best.pth'))['net']
    model.load_state_dict(state_dict)
    model.eval()
    model.to('cpu')

    bounds_list = config.datasets.bounds
    bounds = []
    bounds_list = OmegaConf.to_container(bounds_list)
    for i in range(config.datasets.num_classes):
        bounds.append((bounds_list[2*i], bounds_list[2*i+1]))
    num_classes = config.datasets.num_classes

    graph.to('cpu')

    logging.info("Start greedy node selection process")
    t = time.perf_counter()
    result_elements = random_selection(graph=graph,
                                       model=model,
                                       bounds=bounds,
                                       cardinality_k=int(config.datasets.budget),
                                       num_classes=num_classes,
                                       dataset_name=config.datasets.dataset_name)
    
    logging.info("Finish greedy node selection process")
    logging.info("Start evaluation process")

    fidelity_score = evaluation(graph, result_elements, model, num_classes, device='cpu')
    logging.info("End evaluation process")
    cost_time = time.perf_counter() - t
    logging.info(f'Cost:{time.perf_counter() - t:.4f}s')
    data_record(config.datasets.dataset_name, config.datasets.budget, fidelity_score, cost_time)


if __name__ == '__main__':
    import sys

    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
