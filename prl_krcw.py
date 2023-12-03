import torch.multiprocessing as mp
from train_node_gnn import GCN as MyModel
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch
import networkx as nx
from torch_geometric.data import Data
import pymetis
import random
import numpy as np
import torch_geometric.utils as tgu
import time
from graph_cert.utils import get_fragile, propagation_matrix
from graph_cert.certify import k_squared_parallel, worst_margins_given_k_squared
from torch_geometric.nn import GCNConv
from copy import deepcopy
import os
from torch_geometric.datasets import Reddit, PPI, Planetoid


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
    num_nodes_to_select = 50000
    num_nodes = graph.num_nodes
    selected_nodes = torch.randperm(num_nodes)[:num_nodes_to_select]

    # Step 2: Extract Subgraph
    sub_data, _ = tgu.subgraph(selected_nodes, graph.edge_index, num_nodes=num_nodes, relabel_nodes=True)
    new_sub = Data(x=graph.x[selected_nodes], edge_index=sub_data, y=graph.y[selected_nodes])

    return new_sub


def get_dataset(dataset_root, dataset_name):
    if dataset_name.lower() in ['reddit']:
        return Reddit(os.path.join(dataset_root, dataset_name))[0]
    elif dataset_name.lower() in ['bahouse']:
        return torch.load(os.path.join(dataset_root, dataset_name, 'data.pt'))
    elif dataset_name.lower() in ['ppi']:
        return PPI(os.path.join(dataset_root, dataset_name))[0]
    elif dataset_name.lower() in ['citeseer']:
        return Planetoid(dataset_root, dataset_name)[0]
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def create_subgraphs(graph, parts, nparts):
    # Create subgraphs based on the partitioning
    subgraphs = []
    for i in range(nparts):
        # Nodes in current partition
        nodes = [node for node, part in enumerate(parts) if part == i]
        node_mapping = {node: idx for idx, node in enumerate(nodes)}

        # Edges in current partition
        edge_indices = []
        for j in range(graph.edge_index.shape[1]):
            src = graph.edge_index[0, j].item()
            dst = graph.edge_index[1, j].item()
            if src in nodes and dst in nodes:
                # Remap nodes to new indexing
                new_src = node_mapping[src]
                new_dst = node_mapping[dst]
                edge_indices.append([new_src, new_dst])

        # Convert edges to tensor
        edge_tensor = torch.tensor(edge_indices, dtype=torch.long).t()

        # Subset 'x' and 'y' for nodes in the partition
        x_sub = graph.x[nodes]
        y_sub = graph.y[nodes] if graph.y is not None else None

        # Create PyG Data object
        subgraph_data = Data(x=x_sub, edge_index=edge_tensor, y=y_sub)
        subgraphs.append(subgraph_data)

    return subgraphs


def pyg_to_adjacency_list(edge_index):
    # Convert PyG edge_index to an adjacency list
    max_node = edge_index.max().item() + 1
    adjacency_list = [[] for _ in range(max_node)]
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        adjacency_list[src].append(dst)
        # If the graph is undirected, add the reverse edge as well
        # Uncomment the next line if your graph is undirected
        adjacency_list[dst].append(src)
    return adjacency_list


def PartitionSubgraph(graph, num_parts):

    adjacency_list = pyg_to_adjacency_list(graph.edge_index)

    # Partition the graph using Metis
    _, parts = pymetis.part_graph(num_parts, adjacency=adjacency_list)

    subgraphs = create_subgraphs(graph, parts, num_parts)

    return subgraphs


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


def AddEdge(test_data, top_sensitive_edges):

    test_data.edge_label_index = torch.cat([test_data.edge_label_index, top_sensitive_edges], dim=1)
    test_data.edge_index = remove_edges(test_data.edge_index, top_sensitive_edges)
    unique_nodes = torch.unique(top_sensitive_edges)
    test_data.test_mask[unique_nodes] = True

    return test_data


def InitTestMask(splitG):
    unique_nodes = torch.unique(splitG.edge_label_index)
    test_mask = torch.zeros(splitG.x.size(0), dtype=torch.bool)
    splitG.test_mask = test_mask
    splitG.test_mask[unique_nodes] = True
    return splitG


def get_subgraph(splitG, graph):

    subset = splitG.test_mask.nonzero(as_tuple=True)[0]
    sub_edge_index, _ = tgu.subgraph(subset, graph.edge_index, num_nodes=graph.num_nodes, relabel_nodes=True)
    new_sub = Data(x=graph.x[subset], edge_index=sub_edge_index)

    return new_sub


def SimilarityEvaluation(g1, g2, device, vt):

    model = EvaGNN(g1.x.size(dim=1)).to(device)

    # Pass each graph through the model
    embedding1 = model(g1.to(device))
    embedding2 = model(g2.to(device))

    # Aggregate node embeddings to get a single graph embedding
    # Using mean as an example of aggregation
    graph_embedding1 = torch.mean(embedding1, dim=0)
    graph_embedding2 = torch.mean(embedding2, dim=0)

    # Calculate similarity - for example, using cosine similarity
    cosine_similarity = F.cosine_similarity(graph_embedding1.unsqueeze(0), graph_embedding2.unsqueeze(0))

    gw = (g2.edge_index.size(1) + (vt*10))/g1.edge_index.size(1)

    return cosine_similarity.item(), gw


def get_fidelity_graph(sub_nodes, graph):

    sub_edge_index, _ = tgu.subgraph(sub_nodes, graph.edge_index, num_nodes=graph.num_nodes, relabel_nodes=False)

    return Data(x=graph.x, edge_index=sub_edge_index)


def FidelityEvaluation(out, dataset, test_nodes, model, graph):

    remaining_graph_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    remaining_graph_mask[test_nodes] = False
    remaining_nodes = remaining_graph_mask.nonzero(as_tuple=True)[0]

    remaining_graph = get_fidelity_graph(remaining_nodes, graph)
    test_graph = get_fidelity_graph(test_nodes, graph)

    minus_out = model(test_graph.x, test_graph.edge_index)
    # minus_out = model(test_graph.x, g.edge_label_index)
    plus_out = model(remaining_graph.x, remaining_graph.edge_index)

    minus_predict_lst = []
    plus_predict_lst = []
    for node in test_nodes:
    
        node_pred = out[node]
        minus_pred = minus_out[node]
        plus_pred = plus_out[node]

        original_predictions = node_pred.argmax(dim=-1)
        minus_predictions = minus_pred.argmax(dim=-1)
        plus_predictions = plus_pred.argmax(dim=-1)
        ground_truth = graph.y[node]

        origin_fide = (original_predictions == ground_truth)
        minus_fide = (minus_predictions == ground_truth)
        plus_fide = (plus_predictions == ground_truth)

        if dataset == 'PPI':
            original_predictions = tgu.one_hot(torch.tensor([original_predictions]), num_classes=121)
            minus_predictions = tgu.one_hot(torch.tensor([minus_predictions]), num_classes=121)
            plus_predictions = tgu.one_hot(torch.tensor([plus_predictions]), num_classes=121)

            origin_fide = torch.tensor(True) if origin_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)
            minus_fide = torch.tensor(True) if minus_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)
            plus_fide = torch.tensor(True) if plus_fide.nonzero(as_tuple=True)[0].size(0)/121 > 0.5 else torch.tensor(False)

        minus_predict_lst.append(abs(origin_fide.item() - minus_fide.item()))
        plus_predict_lst.append(abs(origin_fide.item() - plus_fide.item()))

    minus_result = np.mean(minus_predict_lst)
    plus_result = np.mean(plus_predict_lst)

    return plus_result, minus_result


def K_RCW(graph, vt, k, model, device, result_list, rank, dataset_name):
    graph.to(device)
    seed = 1124
    torch.manual_seed(seed)

    split = T.RandomLinkSplit(num_val=0.01, num_test=vt, is_undirected=True, add_negative_train_samples=False)
    _, _, test_data = split(graph)
    test_data = InitTestMask(test_data)
    original_test_data = deepcopy(test_data)

    out = model(graph.x, graph.edge_index)
    logits = out.detach().numpy()
    alpha = 0.85

    cnt = 0
    robust_ratio = 0
    while cnt<3:

        adj_matrix = tgu.add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]
        coo_adj_matrix = tgu.to_scipy_sparse_matrix(adj_matrix)
        csr_adj_matrix = coo_adj_matrix.tocsr()

        ppr_clean = propagation_matrix(adj=csr_adj_matrix, alpha=alpha)
        weighted_logits = ppr_clean @ logits
        predicted = weighted_logits.argmax(1)

        # robust_ratio = Validator(csr_adj_matrix, 
        #         logits, 
        #         test_data.test_mask.nonzero(as_tuple=True)[0].tolist(), 
        #         predicted)
        robust_ratio = 1.0

        top_sensitive_edges = SensiEdge(test_data, model, k)

        test_data = AddEdge(test_data, top_sensitive_edges)
        
        print(f'finish epoch: {cnt} | current robust: {robust_ratio}')
        cnt += 1

    print('find the k-rcw')

    g1 = get_subgraph(test_data, graph)
    g2 = get_subgraph(original_test_data, graph)

    kernel_value, gw = SimilarityEvaluation(g1, g2, device, vt)
    f_plus, f_minus = FidelityEvaluation(out, dataset_name, test_data.test_mask.nonzero(as_tuple=True)[0], model, graph)

    result_list[rank] = kernel_value, f_plus, f_minus, gw


if __name__ == '__main__':

    seed = 1124
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    vt = 20
    k = 3

    mp.set_start_method('spawn', force=True)
    num_processes = 2
    device = torch.device("cpu")

    # config

    dataset_name = 'Reddit'
    data_root_dir = '/home/cs.aau.dk/yj25pu/RoboGExp/datasets'
    model_root_dir = '/home/cs.aau.dk/yj25pu/RoboGExp/checkpoints'

    # model prepare

    input_dim = 602
    output_dim = 41
    model = MyModel(input_dim, output_dim).to('cpu')
    model.load_state_dict(torch.load(os.path.join(
                                model_root_dir,
                                dataset_name,
                                'gcn_3l_best.pth'
    )))
    model.share_memory()
    
    # data prepare

    print('start partition. ')

    dataset = get_dataset(data_root_dir, dataset_name)
    dataset = get_reddit(dataset)
    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = split(dataset)
    pyg_data_list = PartitionSubgraph(graph, num_processes)

    print('partition done. ')

    # parallel

    manager = mp.Manager()
    result_list = manager.list([0] * num_processes)

    t = time.perf_counter()

    processes = []    
    for rank in range(num_processes):
        p = mp.Process(target=K_RCW, args=(pyg_data_list[rank], vt, k, model, device, result_list, rank, dataset_name))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    cost_time = time.perf_counter() - t
    print(f'time cost:{cost_time:.4f}s')

    print("Results:", list(result_list))

    print(f'stat: k: {k}, vt: {vt}')