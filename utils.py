import numpy as np
import torch
import random
import sys
import os
import os
import pytz
import logging
import numpy as np
import random
import torch
import networkx as nx
import copy
from rdkit import Chem
from datetime import datetime
from torch_geometric.utils import subgraph, to_dense_adj
from torch_geometric.data import Data, Batch, Dataset, DataLoader

# For associated game
from itertools import combinations
from scipy.sparse.csgraph import connected_components as cc

# For superadditive extension
from more_itertools import set_partitions

# For visualization
from typing import Union, List
from textwrap import wrap
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_dirs(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)

def timetz(*args):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz).timetuple()


def get_logger(log_path, log_file, console_log=False, log_level=logging.INFO):
    check_dirs(log_path)

    tz = pytz.timezone("US/Pacific")
    logger = logging.getLogger(__name__)
    logger.propagate = False  # avoid duplicate logging
    logger.setLevel(log_level)

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_graph_build_func(build_method):
    if build_method.lower() == "zero_filling":
        return graph_build_zero_filling
    elif build_method.lower() == "split":
        return graph_build_split
    elif build_method.lower() == "remove":
        return graph_build_remove
    else:
        raise NotImplementedError


"""
Graph building/Perturbation
`graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
"""


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through masking the unselected nodes with zero features"""
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through spliting the selected nodes from the original graph"""
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index


def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through removing the unselected nodes from the original graph"""
    ret_X = X[node_mask == 1]
    ret_edge_index, _ = subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
    return ret_X, ret_edge_index


"""
Associated game of the HN value
Implementated using sparse tensor
"""


def get_ordered_coalitions(n):
    coalitions = sum(
        [[set(c) for c in combinations(range(n), k)] for k in range(1, n + 1)], []
    )
    return coalitions


def get_associated_game_matrix_M(coalitions, n, tau):
    indices = []
    values = []
    for i, s in enumerate(coalitions):
        for j, t in enumerate(coalitions):
            if i == j:
                indices += [[i, j]]
                values += [1 - (n - len(s)) * tau]
            elif len(s) + 1 == len(t) and s.issubset(t):
                indices += [[i, j]]
                values += [tau]
            elif len(t) == 1 and not t.issubset(s):
                indices += [[i, j]]
                values += [-tau]

    indices = torch.Tensor(indices).t()
    size = (2**n - 1, 2**n - 1)
    M = torch.sparse_coo_tensor(indices, values, size)
    return M


def get_associated_game_matrix_P(coalitions, n, adj):
    indices = []
    for i, s in enumerate(coalitions):
        idx_s = torch.LongTensor(list(s))
        num_cc, labels = cc(adj[idx_s, :][:, idx_s])
        cc_s = []
        for k in range(num_cc):
            cc_idx_s = (labels == k).nonzero()[0]
            cc_s += [set((idx_s[cc_idx_s]).tolist())]
        for j, t in enumerate(coalitions):
            if t in cc_s:
                indices += [[i, j]]

    indices = torch.Tensor(indices).t()
    values = [1.0] * indices.shape[-1]
    size = (2**n - 1, 2**n - 1)

    P = torch.sparse_coo_tensor(indices, values, size)
    return P


def get_limit_game_matrix(H, exp_power=7, tol=1e-3, is_sparse=True):
    """
    Speed up the power computation by
    1. Use sparse matrices
    2. Put all tensors on cuda
    3. Compute powers exponentially rather than linearly
        i.e. H -> H^2 -> H^4 -> H^8 -> H^16 -> ...
    """
    i = 0
    diff_norm = tol + 1
    while i < exp_power and diff_norm > tol:
        if is_sparse:
            H_tilde = torch.sparse.mm(H, H)
        else:
            H_tilde = torch.mm(H, H)
        diff_norm = (H_tilde - H).norm()
        H = H_tilde
        i += 1
    return H_tilde


"""
khop or random sampling to generate subgraphs
"""


def sample_subgraph(
    data, max_sample_size, sample_method, target_node=None, k=0, adj=None
):
    if sample_method == "khop":
        # pick nodes within k-hops of target node. Hop by hop until reach max_sample_size
        if adj is None:
            adj = (
                to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                .detach()
                .cpu()
            )

        adj_self_loop = adj + torch.eye(data.num_nodes)
        k_hop_adj = adj_self_loop
        sampled_nodes = set()
        m = max_sample_size
        l = 0
        while k > 0 and l < m:
            k_hop_nodes = k_hop_adj[target_node].nonzero().view(-1).tolist()
            next_hop_nodes = list(set(k_hop_nodes) - sampled_nodes)
            sampled_nodes.update(next_hop_nodes[: m - l])
            l = len(sampled_nodes)
            k -= 1
            k_hop_adj = torch.mm(k_hop_adj, adj_self_loop)
        sampled_nodes = torch.tensor(list(sampled_nodes))

    elif sample_method == "random":  # randomly pick #max_sample_size nodes
        sampled_nodes = torch.randperm(data.num_nodes)[:max_sample_size]
    else:
        ValueError("Unknown sample method")

    sampled_x = data.x[sampled_nodes]
    sampled_edge_index, _ = subgraph(
        sampled_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index)
    sampled_adj = adj[sampled_nodes, :][:, sampled_nodes]

    return sampled_nodes, sampled_data, sampled_adj


"""
Payoff computation
"""


def get_char_func(model, target_class, payoff_type="norm_prob", payoff_avg=None):
    def char_func(data):
        with torch.no_grad():
            logits = model(data=data)
            if payoff_type == "raw":
                payoff = logits[:, target_class]
            elif payoff_type == "prob":
                payoff = logits.softmax(dim=-1)[:, target_class]
            elif payoff_type == "norm_prob":
                prob = logits.softmax(dim=-1)[:, target_class]
                payoff = prob - payoff_avg[target_class]
            elif payoff_type == "log_prob":
                payoff = logits.log_softmax(dim=-1)[:, target_class]
            else:
                raise ValueError("unknown payoff type")
        return payoff

    return char_func


class MaskedDataset(Dataset):
    def __init__(self, data, mask, subgraph_building_func):
        super().__init__()

        self.num_nodes = data.num_nodes
        self.x = data.x
        self.edge_index = data.edge_index
        self.device = data.x.device
        self.y = data.y

        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)

        self.mask = mask.type(torch.float32).to(self.device)
        self.subgraph_building_func = subgraph_building_func

    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, idx):
        masked_x, masked_edge_index = self.subgraph_building_func(
            self.x, self.edge_index, self.mask[idx]
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)
        return masked_data


def get_coalition_payoffs(data, coalitions, char_func, subgraph_building_func):
    n = data.num_nodes
    masks = []
    for coalition in coalitions:
        mask = torch.zeros(n)
        mask[list(coalition)] = 1.0
        masks += [mask]

    coalition_mask = torch.stack(masks, axis=0)
    masked_dataset = MaskedDataset(data, coalition_mask, subgraph_building_func)
    masked_dataloader = DataLoader(
        masked_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    masked_payoff_list = []
    for masked_data in masked_dataloader:
        masked_payoff_list.append(char_func(masked_data))

    masked_payoffs = torch.cat(masked_payoff_list, dim=0)
    return masked_payoffs


"""
Superadditive extension
"""


class TrieNode:
    def __init__(self, player, payoff=0, children=[]):
        self.player = player
        self.payoff = payoff
        self.children = children


class CoalitionTrie:
    def __init__(self, coalitions, n, v):
        self.n = n
        self.root = self.get_node(None, 0)
        for i, c in enumerate(coalitions):
            self.insert(c, v[i])

    def get_node(self, player, payoff):
        return TrieNode(player, payoff, [None] * self.n)

    def insert(self, coalition, payoff):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                curr.children[player] = self.get_node(player, 0)
            curr = curr.children[player]
        curr.payoff = payoff

    def search(self, coalition):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                return None
            curr = curr.children[player]
        return curr.payoff

    def visualize(self):
        self._visualize(self.root, 0)

    def _visualize(self, node, level):
        if node:
            print(f"{'-'*level}{node.player}:{node.payoff}")
            for child in node.children:
                self._visualize(child, level + 1)


def superadditive_extension(n, v):
    """
    n (int): number of players
    v (list of floats): dim = 2 ** n - 1, each entry is a payoff
    """
    coalition_sets = get_ordered_coalitions(n)
    coalition_lists = [sorted(list(c)) for c in coalition_sets]
    coalition_trie = CoalitionTrie(coalition_lists, n, v)
    v_ext = v[:]
    for i, coalition in enumerate(coalition_lists):
        partition_payoff = []
        for part in set_partitions(coalition, 2):
            subpart_payoff = []
            for subpart in part:
                subpart_payoff += [coalition_trie.search(subpart)]
            partition_payoff += [sum(subpart_payoff)]
        v_ext[i] = max(partition_payoff + [v[i]])
        coalition_trie.insert(coalition, v_ext[i])
    return v_ext


"""
Evaluation functions
"""


def scores2coalition(scores, sparsity):
    scores_tensor = torch.tensor(scores)
    top_idx = scores_tensor.argsort(descending=True).tolist()
    cutoff = int(len(scores) * (1 - sparsity))
    cutoff = min(cutoff, (scores_tensor > 0).sum().item())
    coalition = top_idx[:cutoff]
    return coalition


def evaluate_coalition(explainer, data, coalition):
    device = explainer.device
    data = data.to(device)
    pred_prob = explainer.model(data).softmax(dim=-1)
    target_class = pred_prob.argmax(-1).item()
    original_prob = pred_prob[:, target_class].item()

    num_nodes = data.num_nodes
    if len(coalition) == num_nodes:
        # Edge case: pick the graph itself as the explanation, for synthetic data
        masked_prob = original_prob
        maskout_prob = 0
    elif len(coalition) == 0:
        # Edge case: pick the empty set as the explanation, for synthetic data
        masked_prob = 0
        maskout_prob = original_prob
    else:
        mask = torch.zeros(num_nodes).type(torch.float32).to(device)
        mask[coalition] = 1.0
        masked_x, masked_edge_index = explainer.subgraph_building_func(
            data.x, data.edge_index, mask
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index).to(device)
        masked_prob = (
            explainer.model(masked_data).softmax(dim=-1)[:, target_class].item()
        )

        maskout_x, maskout_edge_index = explainer.subgraph_building_func(
            data.x, data.edge_index, 1 - mask
        )
        maskout_data = Data(x=maskout_x, edge_index=maskout_edge_index).to(device)
        maskout_prob = (
            explainer.model(maskout_data).softmax(dim=-1)[:, target_class].item()
        )

    fidelity = original_prob - maskout_prob
    inv_fidelity = original_prob - masked_prob
    sparsity = 1 - len(coalition) / num_nodes
    return fidelity, inv_fidelity, sparsity


def fidelity_normalize_and_harmonic_mean(fidelity, inv_fidelity, sparsity):
    """
    The idea is similar to the F1 score, two measures are summarized to one through harmonic mean.

    Step1: normalize both scores with sparsity
        norm_fidelity = fidelity * sparsity
        norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    Step2: rescale both normalized scores from [-1, 1] to [0, 1]
        rescaled_fidelity = (1 + norm_fidelity) / 2
        rescaled_inv_fidelity = (1 - norm_inv_fidelity) / 2
    Step3: take the harmonic mean of two rescaled scores
        2 / (1/rescaled_fidelity + 1/rescaled_inv_fidelity)

    Simplifying these three steps gives the formula
    """
    norm_fidelity = fidelity * sparsity
    norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    harmonic_fidelity = (
        (1 + norm_fidelity)
        * (1 - norm_inv_fidelity)
        / (2 + norm_fidelity - norm_inv_fidelity)
    )
    return norm_fidelity, norm_inv_fidelity, harmonic_fidelity


def evaluate_scores_list(explainer, data_list, scores_list, sparsity, logger=None):
    """
    Evaluate the node importance scoring methods, where each node has an associated score,
    i.e. GStarX and GraphSVX.

    Args:
    data_list (list of PyG data)
    scores_list (list of lists): each entry is a list with scores of nodes in a graph

    """

    assert len(data_list) == len(scores_list)

    f_list = []
    inv_f_list = []
    n_f_list = []
    n_inv_f_list = []
    sp_list = []
    h_f_list = []
    for i, data in enumerate(data_list):
        node_scores = scores_list[i]
        coalition = scores2coalition(node_scores, sparsity)
        f, inv_f, sp = evaluate_coalition(explainer, data, coalition)
        n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)

        f_list += [f]
        inv_f_list += [inv_f]
        n_f_list += [n_f]
        n_inv_f_list += [n_inv_f]
        sp_list += [sp]
        h_f_list += [h_f]

    f_mean = np.mean(f_list).item()
    inv_f_mean = np.mean(inv_f_list).item()
    n_f_mean = np.mean(n_f_list).item()
    n_inv_f_mean = np.mean(n_inv_f_list).item()
    sp_mean = np.mean(sp_list).item()
    h_f_mean = np.mean(h_f_list).item()

    if logger is not None:
        logger.info(
            f"Fidelity Mean: {f_mean:.4f}\n"
            f"Inv-Fidelity Mean: {inv_f_mean:.4f}\n"
            f"Norm-Fidelity Mean: {n_f_mean:.4f}\n"
            f"Norm-Inv-Fidelity Mean: {n_inv_f_mean:.4f}\n"
            f"Sparsity Mean: {sp_mean:.4f}\n"
            f"Harmonic-Fidelity Mean: {h_f_mean:.4f}\n"
        )

    return sp_mean, f_mean, inv_f_mean, n_f_mean, n_inv_f_mean, h_f_mean


"""
Visualization
"""


def coalition2subgraph(coalition, data, relabel_nodes=True):
    sub_data = copy.deepcopy(data)
    node_mask = torch.zeros(data.num_nodes)
    node_mask[coalition] = 1

    sub_data.x = data.x[node_mask == 1]
    sub_data.edge_index, _ = subgraph(
        node_mask.bool(), data.edge_index, relabel_nodes=relabel_nodes
    )
    return sub_data


def to_networkx(
    data,
    node_index=None,
    node_attrs=None,
    edge_attrs=None,
    to_undirected=False,
    remove_self_loops=False,
):
    r"""
    Extend the PyG to_networkx with extra node_index argument, so subgraphs can be plotted with correct ids

    Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)


        node_index (iterable): Pass in it when there are some nodes missing.
                 max(node_index) == max(data.edge_index)
                 len(node_index) == data.num_nodes
    """
    import networkx as nx

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    if node_index is not None:
        """
        There are some nodes missing. The max(data.edge_index) > data.x.shape[0]
        """
        G.add_nodes_from(node_index)
    else:
        G.add_nodes_from(range(data.num_nodes))

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G