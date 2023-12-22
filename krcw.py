import hydra
import torch
import torch_geometric.utils as tgu
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from dataset import get_dataset, get_dim
from warnings import simplefilter
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from train_node_gnn import GCN
from graph_cert.utils import get_fragile, standardize, propagation_matrix
from graph_cert.certify import k_squared_parallel, worst_margins_given_k_squared
from torch.autograd.functional import jacobian
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from grakel import Graph as GrakelGraph
from grakel import RandomWalkLabeled
from collections import defaultdict
import random
import time
from torch_geometric.nn import GCNConv



class EvaGNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(EvaGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 2)  # Output is an embedding of size 2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def get_reddit(graph):

    # Step 1: Select Random Nodes
    num_nodes_to_select = 5000
    num_nodes = graph.num_nodes
    selected_nodes = torch.randperm(num_nodes)[:num_nodes_to_select]

    # Step 2: Extract Subgraph
    sub_data, _ = tgu.subgraph(selected_nodes, graph.edge_index, num_nodes=num_nodes, relabel_nodes=True)
    new_sub = Data(x=graph.x[selected_nodes], edge_index=sub_data, y=graph.y[selected_nodes])

    return new_sub



def aug_normalized_adj(adj_matrix):
    """
    Args:
        adj_matrix: input adj_matrix
    Returns:
        a normalized_adj which follows influence spread idea
    """
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_matrix_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_matrix_inv_sqrt.dot(adj_matrix).dot(d_matrix_inv_sqrt).tocoo()



def visual(G, dataset):

    nx_G = tgu.to_networkx(G, to_undirected=True)
    for i in range(nx_G.number_of_nodes()):
        label = np.where(G.x[i].cpu().numpy() == 1)
        nx_G.nodes[i]['label'] = label[0][0]

    pos = nx.kamada_kawai_layout(nx_G)

    if dataset=='Mutagenicity':
        node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 112: 'Li', 13: 'Ca'}
        node_idxs = {idx: int(label['label']) for idx, label in nx_G.nodes.data(True)}
        # node_idxs = {k: int(v) for k, v in enumerate(np.where(G.x.numpy() == 1)[1])}
        node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00', 'lime', 'maroon', 'blue', 'green', 'yellow', 'red', 'black']
        node_colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
    elif dataset=='MUTAG':
        node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
        node_idxs = {idx: int(label['label']) for idx, label in nx_G.nodes.data(True)}
        # node_idxs = {k: int(v) for k, v in enumerate(np.where(G.x.numpy() == 1)[1])}
        node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
        node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
        node_colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

    nx.draw(nx_G, pos=pos, node_color=node_colors, node_size=500, width=2.0, arrows=False, labels=node_labels)
    plt.title("PyG Graph Visualization")
    plt.show()



def Kruskal_ST(G):
    nx_G = tgu.to_networkx(G, to_undirected=True)
    SpanningTree = nx.minimum_spanning_tree(nx_G)
    pyg_SpanningTree = tgu.from_networkx(SpanningTree)
    pyg_SpanningTree.edge_index = tgu.to_undirected(pyg_SpanningTree.edge_index)
    return pyg_SpanningTree



def remove_edges(edge_index, edges_to_remove):
    # Convert edge_index to a set of tuples
    edge_set = {tuple(sorted((i.item(), j.item()))) for i, j in edge_index.t()}

    # Convert edges_to_remove to a set of tuples
    edges_to_remove_set = {tuple(sorted((i.item(), j.item()))) for i, j in edges_to_remove.t()}

    # Remove the specified edges
    edge_set = edge_set - edges_to_remove_set

    # Recreate the edge_index tensor
    edge_list = [[n1, n2] for n1, n2 in edge_set]
    updated_edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # Update data.edge_index
    return updated_edge_index



def InitTestMask(splitG):
    unique_nodes = torch.unique(splitG.edge_label_index)
    test_mask = torch.zeros(splitG.x.size(0), dtype=torch.bool)
    splitG.test_mask = test_mask
    splitG.test_mask[unique_nodes] = True
    return splitG


def AddEdge(test_data, top_sensitive_edges):

    test_data.edge_label_index = torch.cat([test_data.edge_label_index, top_sensitive_edges], dim=1)
    test_data.edge_index = remove_edges(test_data.edge_index, top_sensitive_edges)
    unique_nodes = torch.unique(top_sensitive_edges)
    test_data.test_mask[unique_nodes] = True

    return test_data




def Validator(adj, logits, nodes, labels):

    fragile = get_fragile(adj=adj, threat_model='rem')

    deg = adj.sum(1).A1.astype(np.int32)
    local_budget = np.maximum(deg - 5, 0)

    # precomputed the K x K perturbed graphs
    k_squared_pageranks = k_squared_parallel(
        adj=adj, alpha=0.8, fragile=fragile, local_budget=local_budget, logits=logits, nodes=nodes)

    # compute the exact worst-case margins for all test nodes
    worst_margins = worst_margins_given_k_squared(
        k_squared_pageranks=k_squared_pageranks, labels=labels[nodes], logits=logits)
    
    return (worst_margins>0).mean()




def SensiEdge(test_data, model, k):

    test_data.x.requires_grad = True
    output = model(test_data.x, test_data.edge_index)  # Forward pass
    loss = output.sum()  # Compute loss
    loss.backward()

    # Extract and aggregate edge gradients
    edge_gradients = {}
    for i, edge in enumerate(test_data.edge_index.t()):
        node_a, node_b = edge
        edge_key = tuple(sorted([node_a.item(), node_b.item()]))  # Sort the nodes to treat both directions the same
        gradient = (test_data.x.grad[node_a] + test_data.x.grad[node_b]).mean()  # Or use .sum() or .max()
        if edge_key in edge_gradients:
            edge_gradients[edge_key] += gradient
        else:
            edge_gradients[edge_key] = gradient

    # Sort edges by their aggregated gradients
    sorted_edges = sorted(edge_gradients.items(), key=lambda x: x[1], reverse=True)

    # Select top 5 sensitive edges
    top_5_edge_tuples = [edge[0] for edge in sorted_edges[:k]]
    flat_edges = [[n1, n2] for n1, n2 in top_5_edge_tuples]
    top_5_edges = torch.tensor(flat_edges, dtype=torch.long).t()

    return top_5_edges


def get_subgraph(splitG, graph):

    subset = splitG.test_mask.nonzero(as_tuple=True)[0]

    sub_edge_index, _ = tgu.subgraph(subset, graph.edge_index, num_nodes=graph.num_nodes, relabel_nodes=True)
    if splitG.edge_label_index != None:
        new_sub = Data(x=graph.x[subset], edge_index=splitG.edge_label_index)
    else:
        new_sub = Data(x=graph.x[subset], edge_index=sub_edge_index)

    return new_sub


def SimilarityEvaluation(g1, g2):

    def pyg_to_networkx(pyg_graph):
        edges = pyg_graph.edge_index.t().tolist()
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(edges)
        return nx_graph

    # Convert PyG graphs to NetworkX graphs
    nx_graph1 = pyg_to_networkx(g1)
    nx_graph2 = pyg_to_networkx(g2)

    # Calculate GED
    ged = nx.graph_edit_distance(nx_graph1, nx_graph2)

    # Normalize GED
    max_ged = nx_graph1.number_of_nodes() + nx_graph1.number_of_edges()
    normalized_ged = ged / max_ged

    print("Normalized GED:", normalized_ged)

    return normalized_ged

def get_fidelity_graph(sub_nodes, graph):

    _, test_index, _, _ = tgu.k_hop_subgraph(sub_nodes, 1, graph.edge_index, relabel_nodes=False, directed=False)
    subg = Data(x=graph.x, edge_index=test_index)

    return subg



def get_fidelity_plus(mask, sub_nodes, graph):

    sub_edge_index, _ = tgu.subgraph(sub_nodes, graph.edge_index, num_nodes=graph.num_nodes, relabel_nodes=False)
    subg = Data(x=graph.x, edge_index=sub_edge_index)
    with torch.no_grad():
        subg.x[mask] = torch.zeros_like(graph.x[0])

    return subg



# def FidelityEvaluation(out, dataset, test_data, model, graph):

#     test_nodes = test_data.test_mask.nonzero(as_tuple=True)[0]

#     remaining_graph_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
#     remaining_graph_mask[test_nodes] = False
#     remaining_nodes = remaining_graph_mask.nonzero(as_tuple=True)[0]

#     remaining_graph = get_fidelity_plus(test_data.test_mask, remaining_nodes, graph)
#     test_graph = get_fidelity_graph(test_nodes, graph)
#     # remaining_graph, test_graph = get_fidelity_graph(remaining_nodes, test_nodes, graph)
#     model.eval()
#     with torch.no_grad():
#         # minus_out = model(test_graph.x, test_graph.edge_index)
#         # # minus_out = model(test_graph.x, g.edge_label_index)
#         plus_out = model(remaining_graph.x, remaining_graph.edge_index)

#     model.eval()
#     with torch.no_grad():
#         minus_out = model(test_graph.x, test_graph.edge_index)
#         # # minus_out = model(test_graph.x, g.edge_label_index)
#         # plus_out = model(remaining_graph.x, remaining_graph.edge_index)

#     minus_predict_lst = []
#     plus_predict_lst = []
#     # print(f'test_nodes:{test_nodes}')
#     for node in test_nodes:
#         # print(f'node:{node}')
#         node_pred = out[node]
#         minus_pred = minus_out[node]
#         plus_pred = plus_out[node]
#         # print(f'node_pred:{node_pred}')
#         # print(f'minus_pred:{minus_pred}')
#         # print(f'plus_pred:{plus_pred}')

#         original_predictions = node_pred.argmax(dim=-1)
#         minus_predictions = minus_pred.argmax(dim=-1)
#         plus_predictions = plus_pred.argmax(dim=-1)
#         ground_truth = graph.y[node]
#         # print(f'original_predictions:{original_predictions}')
#         # print(f'minus_predictions:{minus_predictions}')
#         # print(f'plus_predictions:{plus_predictions}')
#         # print(f'ground_truth:{ground_truth}')

#         origin_fide = (original_predictions == ground_truth)
#         minus_fide = (minus_predictions == ground_truth)
#         plus_fide = (plus_predictions == ground_truth)
#         # print(f'origin_fide:{origin_fide}')
#         # print(f'minus_fide:{minus_fide}')
#         # print(f'plus_fide:{plus_fide}')

#         if dataset == 'PPI':
#             original_predictions = tgu.one_hot(torch.tensor([original_predictions]), num_classes=121)
#             minus_predictions = tgu.one_hot(torch.tensor([minus_predictions]), num_classes=121)
#             plus_predictions = tgu.one_hot(torch.tensor([plus_predictions]), num_classes=121)

#             origin_fide = torch.tensor(True) if origin_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)
#             minus_fide = torch.tensor(True) if minus_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)
#             plus_fide = torch.tensor(True) if plus_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)

#         minus_predict_lst.append(abs(origin_fide.item() - minus_fide.item()))
#         plus_predict_lst.append(abs(origin_fide.item() - plus_fide.item()))
#         # minus_predict_lst.append(origin_fide.item() - minus_fide.item())
#         # plus_predict_lst.append(origin_fide.item() - plus_fide.item())

#         # break

#     minus_result = np.mean(minus_predict_lst)
#     plus_result = np.mean(plus_predict_lst)

#     return plus_result, minus_result

def SimilarityEvaluation(g1, g2):
    normalized_ged = (g1.num_edges - g2.num_edges)/(g1.num_nodes + g1.num_edges)
    return normalized_ged

def FidelityEvaluationPlus(out, dataset, test_data, model, graph):

    test_nodes = test_data.test_mask.nonzero(as_tuple=True)[0]

    remaining_graph_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    remaining_graph_mask[test_nodes] = False
    remaining_nodes = remaining_graph_mask.nonzero(as_tuple=True)[0]

    remaining_graph = get_fidelity_plus(test_data.test_mask, remaining_nodes, graph)
    model.eval()
    with torch.no_grad():
        plus_out = model(remaining_graph.x, remaining_graph.edge_index)

    plus_predict_lst = []
    for node in test_nodes:
        # print(f'node:{node}')
        node_pred = out[node]
        plus_pred = plus_out[node]

        original_predictions = node_pred.argmax(dim=-1)
        plus_predictions = plus_pred.argmax(dim=-1)
        ground_truth = graph.y[node]

        origin_fide = (original_predictions == ground_truth)
        plus_fide = (plus_predictions == ground_truth)

        if dataset == 'PPI':
            original_predictions = tgu.one_hot(torch.tensor([original_predictions]), num_classes=121)
            plus_predictions = tgu.one_hot(torch.tensor([plus_predictions]), num_classes=121)

            origin_fide = torch.tensor(True) if origin_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)
            plus_fide = torch.tensor(True) if plus_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)

        plus_predict_lst.append(abs(origin_fide.item() - plus_fide.item()))

        # break

    plus_result = np.mean(plus_predict_lst)

    return plus_result

def FidelityEvaluationMinus(out, dataset, test_data, model, graph):

    test_nodes = test_data.test_mask.nonzero(as_tuple=True)[0]
    test_graph = get_fidelity_graph(test_nodes, graph)

    minus_out = model(test_graph.x, test_graph.edge_index)
    out = model(graph.x, graph.edge_index)

    minus_predict_lst = []
    for node in test_nodes:

        node_pred = out[node]
        minus_pred = minus_out[node]

        original_predictions = node_pred.argmax(dim=-1)
        minus_predictions = minus_pred.argmax(dim=-1)
        ground_truth = graph.y[node]

        origin_fide = (original_predictions == ground_truth)
        minus_fide = (minus_predictions == ground_truth)

        if dataset == 'PPI':
            original_predictions = tgu.one_hot(torch.tensor([original_predictions]), num_classes=121)
            minus_predictions = tgu.one_hot(torch.tensor([minus_predictions]), num_classes=121)

            origin_fide = torch.tensor(True) if origin_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)
            minus_fide = torch.tensor(True) if minus_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)

        minus_predict_lst.append(abs(origin_fide.item() - minus_fide.item()))

        # break

    minus_result = np.mean(minus_predict_lst)

    return minus_result

# def FidelityEvaluationMinus(out, dataset, test_data, model, graph):

#     test_nodes = test_data.test_mask.nonzero(as_tuple=True)[0]
#     test_graph = get_fidelity_graph(test_nodes, graph)

#     minus_out = model(test_graph.x, test_graph.edge_index)
#     out = model(graph.x, graph.edge_index)

#     minus_predict_lst = []
#     for node in test_nodes:
    
#         node_pred = out[node]

#         minus_pred = minus_out[node]
#         minus_predict_lst.append(abs((node_pred.argmax(dim=-1) == graph.y[node]).item() - (minus_pred.argmax(dim=-1) == graph.y[node]).item()))

#     minus_result = np.mean(minus_predict_lst)

#     return minus_result



@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):

    seed = 1124
    torch.manual_seed(seed)
    random.seed(seed)
    vt = 20
    b = 10
    k = 20
    ## dataset

    config.models.params = config.models.params[config.datasets.dataset_name]
    input_dim, output_dim = get_dim(config.datasets.dataset_name)
    dataset = get_dataset(dataset_root='datasets', dataset_name=config.datasets.dataset_name)
    if config.datasets.dataset_name == 'BAHouse':
        bigG = dataset
    elif config.datasets.dataset_name == 'Reddit':
        bigG = get_reddit(dataset[0])
    else:
        if dataset._data.x is not None:
            dataset._data.x = dataset._data.x.float()
        dataset._data.y = dataset._data.y.squeeze().long()
        print(f'dataset[0]:{dataset[0]}')
        bigG = dataset[0]

    split = T.RandomLinkSplit(num_val=0.01, num_test=vt, is_undirected=True, add_negative_train_samples=False)
    _, _, test_data = split(bigG)
    test_data = InitTestMask(test_data)
    original_test_data = deepcopy(test_data)


    ## model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GCN(input_dim, output_dim)
    model.load_state_dict(torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.params.gnn_latent_dim)}l_best.pth')))
    model.eval()
    model.to(device)

    model1 = GCN(input_dim, output_dim)
    model1.load_state_dict(torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.params.gnn_latent_dim)}l_best.pth')))
    model1.eval()
    model1.to(device)

    ## krcw

    out = model(bigG.x, bigG.edge_index)
    out1 = model1(bigG.x, bigG.edge_index)
    logits = out.detach().numpy()
    alpha = 0.85

    cnt = 0
    robust_ratio = 0
    t = time.perf_counter()
    # while robust_ratio<0.97 and cnt<10:
    while cnt<k:

        adj_matrix = tgu.add_remaining_self_loops(bigG.edge_index, num_nodes=bigG.num_nodes)[0]
        coo_adj_matrix = tgu.to_scipy_sparse_matrix(adj_matrix)
        csr_adj_matrix = coo_adj_matrix.tocsr()

        ppr_clean = propagation_matrix(adj=csr_adj_matrix, alpha=alpha)
        weighted_logits = ppr_clean @ logits
        predicted = weighted_logits.argmax(1)

        Validator(csr_adj_matrix, 
                logits, 
                test_data.test_mask.nonzero(as_tuple=True)[0].tolist(), 
                predicted)

        top_sensitive_edges = SensiEdge(test_data, model, b)

        test_data = AddEdge(test_data, top_sensitive_edges)
        
        print(f'finish epoch: {cnt}')
        cnt += 1
    cost_time = time.perf_counter() - t
    print(f'time cost:{cost_time:.4f}s')
    print('find the k-rcw')

    g1 = get_subgraph(test_data, bigG)
    g2 = get_subgraph(original_test_data, bigG)

    NormGED = SimilarityEvaluation(g1, g2)
    f_plus = FidelityEvaluationPlus(out, config.datasets.dataset_name, test_data, model, bigG)
    f_minus = FidelityEvaluationMinus(out1, config.datasets.dataset_name, test_data, model1, bigG)

    print(f'GED: {NormGED}')
    print(f'fidelity+: {f_plus}')
    print(f'fidelity-: {f_minus}')

    print(f'stat: k: {k}, b: {b} vt: {vt}')


if __name__ == '__main__':
    import sys

    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
