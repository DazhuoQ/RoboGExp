import copy
import time as tm
import logging
import torch
import os
import pandas as pd
import networkx as nx
import numpy as np
import hydra
import scipy.sparse as sp
import torch.nn.functional as F
import numpy.linalg as la
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import add_remaining_self_loops, to_scipy_sparse_matrix, \
    k_hop_subgraph, to_networkx, subgraph
from gnnNets import get_gnnNets
from networkx.algorithms import isomorphism
from visualization import PlotUtils
from gspan.gspan_s import gSpan
from dataset import get_dataset, get_dataloader
from warnings import simplefilter
from tqdm import tqdm
from utils import get_logger


class StreamingAlgorithm(object):
    def __init__(self, cardinality_k, bounds, sub_function, graph, model, k, dataset_name, graph_id_begin):
        self.cardinality_k = cardinality_k
        self.bounds = bounds
        self.sub_function = sub_function
        self.solution = []
        self.graph = graph
        self.model = model
        self.k = k
        self.dataset_name = dataset_name
        self.bucket = [[] for _ in range(len(bounds))]
        self.hop_subgraph = []
        self.pattern_set = []
        self.graph_id_begin = graph_id_begin
        self.run_time = 0.0
        self.pat_set = []
        self.pat_can = []
        self.ex_sub_set = []


    def verify(self, subgraph, counter_subgraph, label):
        return F.softmax(self.model(subgraph).to('cpu'), -1).argmax(-1) == label and \
               F.softmax(self.model(counter_subgraph).to('cpu'), -1).argmax(-1) != label


    def insert(self, element):
        subset, _, _, _ = k_hop_subgraph(element[0], 3, self.graph.edge_index)
        t = tm.perf_counter()
        edge_index = subgraph(subset, self.graph.edge_index, relabel_nodes=True)[0]
        G = Data(self.graph.x[subset], edge_index)

        origin_graph_id = self.graph.batch[element[0]].item()
        origin_graph = self.graph.__getitem__(origin_graph_id)
        x_cf = copy.deepcopy(origin_graph.x)
        for node in subset:
            node_id = node - self.graph_id_begin[origin_graph_id]
            x_cf[node_id] = 0
        G_cf = Data(x_cf, origin_graph.edge_index)

        if not self.verify(G, G_cf, element[1]):
            self.run_time += tm.perf_counter() - t
            return
        
        self.solution.append(element)
        flag = self.feasible(self.solution)
        if not flag:
            node_pair = self.find_simple(self.solution, len(self.solution))
            if node_pair[0] != -1:
                self.solution.remove(node_pair)
                self.update_value(node_pair)
                flag = True
            else:
                self.solution.pop()
        self.run_time += tm.perf_counter() - t

        if flag:
            self.PatternMatch(G, element)

        if len(self.bucket[element[1]]) < self.bounds[element[1]][0]:
            self.bucket[element[1]].append(element[0])
        

    def feasible(self, elements):
        element_from_colors = [0] * len(self.bounds)
        for element in elements:
            element_from_colors[element[1]] += 1
        extra_elements_needed = 0
        for i in range(len(element_from_colors)):
            if self.bounds[i][1] < element_from_colors[i]:
                return False
            extra_elements_needed += max(0, self.bounds[i][0] - element_from_colors[i])

        if len(elements) + extra_elements_needed > self.cardinality_k:
            return False

        return True
    

    def find(self, solution, length):
        node_pair = (-1, -1)
        all_value = self.sub_function.cal(solution)
        min_value = 1000000000.0
        e_value = 0
        for i in range(length):
            temp_solution = copy.deepcopy(solution)
            element = temp_solution[i]
            temp_solution.remove(element)
            if i == length - 1:
                e_value = all_value - self.sub_function.cal(temp_solution)
                continue
            if not self.feasible(temp_solution):
                continue
            now_value = all_value - self.sub_function.cal(temp_solution)
            if now_value < min_value:
                min_value = now_value
                node_pair = element
        if e_value > 2 * min_value:
            return node_pair
        return tuple((-1, -1))


    def find_simple(self, solution, length):
        node_pair = (-1, -1)
        min_value = 100000000.0
        e_value = 0
        label = solution[-1][1]
        for i in range(length):
            if i == length - 1:
                e_value = self.sub_function.cal([solution[i]])
                continue
            if label != solution[i][1]:
                continue
            now_value = self.sub_function.cal([solution[i]])
            if now_value < min_value:
                min_value = now_value
                node_pair = solution[i]
        if e_value > 2 * min_value:
            return node_pair
        return tuple((-1, -1))


    def update_value(self, node_pair):
        node_id = node_pair[0]
        for pattern in self.pattern_set:
            if node_id in pattern[1]:
                pattern[1].remove(node_id)


    def check_isomorphism(self, graph_one, graph_two):
        graph_match = isomorphism.GraphMatcher(graph_one, graph_two)
        return graph_match.subgraph_is_isomorphic()


    def IsFullCover(self, x_sub):
        Flag = True
        for re_s in nx.connected_components(x_sub):
            if len(list(re_s)) > 3:
                Flag = False
        return Flag


    def GspanPreprocess(self, component):
        save_list = []
        save_list.append('t # ' + str(0))
        for idv, label in component.nodes.data(True):
            save_list.append('v ' + str(idv) + ' ' + str(label['label']))
        for e in component.edges:
            save_list.append('e ' + str(e[0]) + ' ' + str(e[1]) + ' 2')
        save_list.append('t # -1')
        return save_list


    def NewPattern(self, pattern):
        Flag = True
        for old_p in self.pat_set:
            GM = isomorphism.GraphMatcher(old_p, pattern)
            if GM.is_isomorphic():
                Flag = False
        return Flag
    

    def NewCandi(self, pattern):
        Flag = True
        for old_p in self.pat_can:
            GM = isomorphism.GraphMatcher(old_p, pattern)
            if GM.is_isomorphic():
                Flag = False
        return Flag
    

    def GspanFinding(self, save_list):
        gs = gSpan(
            database_file_name=save_list,
            min_support=2,
            min_num_vertices=3,
            max_num_vertices=6,
            is_undirected=False,
            max_ngraphs=1
        )
        gs.run()
        for g_str in gs.p_set:
            lst = g_str.split(' ')
            pattern = nx.Graph()
            for i in range(len(lst)):
                if lst[i] == 'v':
                    pattern.add_node(lst[i+1], label=lst[i+2])
                elif lst[i] == 'e':
                    pattern.add_edge(lst[i+1], lst[i+2])
            pattern = nx.convert_node_labels_to_integers(pattern)
            if self.NewCandi(pattern):
                self.pat_can.append(pattern)


    def OneCover(self, x_sub):
        edge_x_sub = copy.deepcopy(x_sub)
        self.ex_sub_set.append(edge_x_sub)
        def sort_func(a):
            return a.number_of_nodes()
        self.pat_can.sort(key=sort_func, reverse=True)
        for pattern in self.pat_can:
            GM = isomorphism.GraphMatcher(edge_x_sub, pattern)
            if GM.subgraph_is_isomorphic():
                if self.NewPattern(pattern):
                    self.pat_set.append(pattern)
                unique_lst = []
                for mapping in GM.subgraph_isomorphisms_iter():
                    ids = [int(id) for id in mapping]
                    unique_lst.extend(ids)
                id_lst = list(set(unique_lst))
                x_sub.remove_nodes_from(id_lst)
        for re_s in nx.connected_components(x_sub):
            node_list = list(re_s)
            if len(node_list)<3:
                continue
            re_p = nx.Graph(nx.induced_subgraph(x_sub, node_list))
            re_p = nx.convert_node_labels_to_integers(re_p)
            self.pat_set.append(re_p)


    def EdgeLoss(self, x_sub):
        edge_start = x_sub.number_of_edges()
        for pattern in self.pat_set:
            GM = isomorphism.GraphMatcher(x_sub, pattern)
            if GM.subgraph_is_isomorphic():
                unique_lst = []
                for mapping in GM.subgraph_isomorphisms_iter():
                    ids = [int(id) for id in mapping]
                    unique_lst.extend(ids)
                id_lst = list(set(unique_lst))
                x_sub.remove_nodes_from(id_lst)
        edge_end = x_sub.number_of_edges()
        edge_loss = edge_end/edge_start
        return edge_loss


    def PatternMatch(self, G, element):
        nx_graph = to_networkx(G, to_undirected=True)
        for i in range(nx_graph.number_of_nodes()):
            label = np.where(G.x[i].cpu().numpy() == 1)
            nx_graph.nodes[i]['label'] = label[0][0]
        self.hop_subgraph.append((nx_graph, element[0]))
        save_list = self.GspanPreprocess(nx_graph)
        self.GspanFinding(save_list)
        x_sub = copy.deepcopy(nx_graph)
        self.OneCover(x_sub)


    def post_process(self):
        pass

    def get_result(self):
        return self.pattern_set
    
    def get_solution(self):
        return self.solution

    def get_runtime(self):
        return self.run_time
    
    def get_pat_set(self):
        return self.pat_set
    
    def get_x_sub_num(self):
        return self.x_sub_num
    
    def get_ex_sub_set(self):
        return self.ex_sub_set


class SubFunction(object):
    def __init__(self, graph, influence_state, diversity_state, gamma, elements):
        self.graph = graph
        self.influence_state = influence_state
        self.diversity_state = diversity_state
        self.gamma = gamma
        self.elements = elements

    def cal(self, elements):
        counter = set()
        diverse_counter = set()
        for element in elements:
            counter = counter.union(self.influence_state[element[0]])
            diverse_counter = diverse_counter.union(self.diversity_state[element[0]])
        return float(len(counter)) / self.graph.num_nodes + \
               self.gamma * float(len(diverse_counter)) / self.graph.num_nodes


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


def pre_process(graph, model):
    ex_labels = model(graph).to('cpu')
    result = F.softmax(ex_labels, -1).argmax(-1)
    elements = []
    graph.to('cpu')
    for i in range(graph.num_nodes):
        elements.append((i, result[graph.batch[i].item()].item()))
    return elements


def visual_pattern(pattern_set, dataset_name):
    plot_util = PlotUtils(dataset_name)
    counter = 0
    for graph, _ in pattern_set:
        max_id = 0
        for node_id in graph.nodes():
            max_id = max(int(graph.nodes[node_id]['label']), max_id)
        x = torch.zeros(graph.number_of_nodes(), max_id + 1)
        for node_id in graph.nodes():
            x[int(node_id)][int(graph.nodes[node_id]['label'])] = 1
        # plot_util.plot(graph, x=x, fig_name=f'{dataset_name}_pattern_{counter}')
        counter += 1


def EdgeLoss(x_sub_set, pattern_set):
    def sort_func(a):
        return a.number_of_nodes()
    edge_loss_lst = []
    for x_sub in tqdm(x_sub_set):
        if x_sub.number_of_nodes() < 3 or x_sub.number_of_edges() < 1:
            continue
        edge_start = x_sub.number_of_edges()
        pattern_set.sort(key=sort_func, reverse=True)
        for pattern in pattern_set:
            GM = isomorphism.GraphMatcher(x_sub, pattern)
            if GM.subgraph_is_isomorphic():
                unique_lst = []
                for mapping in GM.subgraph_isomorphisms_iter():
                    ids = [int(id) for id in mapping]
                    unique_lst.extend(ids)
                id_lst = list(set(unique_lst))
                x_sub.remove_nodes_from(id_lst)
        edge_end = x_sub.number_of_edges()
        edge_loss_lst.append(edge_end/edge_start)
    edge_loss = np.mean(edge_loss_lst)
    return edge_loss


def evaluation(graph, pattern_set, solution, model, pat_set, ex_sub_set, num_classes, dataset_name, graph_id_begin):
    ex_labels = model(graph).to('cpu')
    result = F.softmax(ex_labels, -1)
    st = set()
    for _, cover in pattern_set:
        st = st.union(cover)

    group_count = 0
    fidelity_minus = 0
    graph_group_set = [[] for _ in range(num_classes)]

    for element in solution:
        group_count += 1
        ground_truth = graph.y[graph.batch[element[0]].item()]
        subset, _, _, _ = k_hop_subgraph(element[0], 3, graph.edge_index)
        edge_index = subgraph(subset, graph.edge_index, relabel_nodes=True)[0]
        G = Data(graph.x[subset], edge_index)
        graph_group_set[element[1]].append(G)
  
        plot_util = PlotUtils(dataset_name)
        # plot_util.plot(to_networkx(G), x=G.x, fig_name=f'graph_{element[0]}_label_{element[1]}')

        fidelity_minus += (result[graph.batch[element[0]].item()][ground_truth].item() 
                           - F.softmax(model(G), -1)[0][ground_truth])

    fidelity_minus_score = 1.0 * fidelity_minus / group_count

    print(f'fidelity- score : {fidelity_minus_score}')
    logging.info(f'fidelity- score : {fidelity_minus_score}')
    
    print(f'# of patterns: {len(pat_set)}')
    pattern_node_num = 0
    for pattern in pat_set:
        pattern_node_num = pattern_node_num + pattern.number_of_nodes()
    x_sub_num = np.sum([x_sub.number_of_nodes() for x_sub in ex_sub_set])
    print(f'# of nodes of all patterns: {pattern_node_num}')
    print(f'# of nodes of all ex_sub: {x_sub_num}')
    
    print(f'len(ex_sub_set): {len(ex_sub_set)}')
    edge_loss = EdgeLoss(ex_sub_set, pat_set)
    print(f'Edge loss ratio: {edge_loss}')

    return fidelity_minus_score, _, _


def data_record(dataset_name, fidelity_minus_score, fidelity_plus_score, sparsity_score, cost_time):
    project_name = os.path.dirname(__file__)
    exp_name = os.path.join(project_name, 'experiments/data_record')
    data_name = os.path.join(exp_name, f'streaming_{dataset_name}')
    data = {'fidelity_minus': fidelity_minus_score, 
            'fidelity_plus': fidelity_plus_score, 
            'sparsity_score': sparsity_score,
            'cost_time': f'{cost_time:.4f}'}
    data_frame = pd.DataFrame(data, index=[0])
    if not os.path.exists(data_name):
        data_frame.to_csv(data_name, header=True, mode='a', index=True, sep=',')
    else:
        data_frame.to_csv(data_name, header=False, mode='a', index=True, sep=',')


def streaming_pattern_generation(graph, model, bounds, influence_state, diversity_state, gamma,
                                 cardinality_k, num_classes, k, dataset_name, graph_id_begin):
    elements = pre_process(graph, model)
    sub_funcion = SubFunction(graph, influence_state, diversity_state, gamma, elements)
    algorithm = StreamingAlgorithm(cardinality_k, bounds, sub_funcion, 
                                   graph, model, k, dataset_name,
                                   graph_id_begin)

    for element in tqdm(elements, desc='Process', leave=True, ncols=100, unit='B', unit_scale=True):
        algorithm.insert(element)

    algorithm.post_process()
    print(f'Cost:{algorithm.get_runtime():.4f}s')
    logging.info(f'Cost:{algorithm.get_runtime():.4f}s')
    pattern_set = algorithm.get_result()
    solution = algorithm.get_solution()

    return pattern_set, solution, algorithm.get_runtime(), algorithm.get_pat_set(), algorithm.get_ex_sub_set()


def many_graphs_to_one_graph(graphs, threshold, radium, device):
    influenc_state = []
    diversity_state = []
    tot_node = 0
    counter = 0
    graph_id_begin = []
    for graph in graphs:    
        counter += 1
        adj_matrix = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]
        coo_adj_matrix = to_scipy_sparse_matrix(adj_matrix)
        aug_normalized_adj_matrix = aug_normalized_adj(coo_adj_matrix)
        influence_matrix = torch.FloatTensor(aug_normalized_adj_matrix.todense()).to(device)
        influence_matrix2 = torch.mm(influence_matrix, influence_matrix)
        influence_matrix3 = torch.mm(influence_matrix, influence_matrix2).cpu()
        for i in range(influence_matrix3.shape[0]):
            act_set = set()
            div_set = set()
            for j in range(influence_matrix3.shape[1]):
                if influence_matrix3[i][j] > threshold:
                    act_set.add(tot_node+j)
                minus = influence_matrix3[i].numpy() - influence_matrix3[j].numpy()
                if np.linalg.norm(minus) < radium:
                    div_set.add(tot_node+j)
            influenc_state.append(act_set)
            diversity_state.append(div_set)
        graph_id_begin.append(tot_node)
        tot_node += graph.num_nodes
    return Batch.from_data_list(graphs).to('cpu'), influenc_state, diversity_state, graph_id_begin


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.params = config.models.params[config.datasets.dataset_name]
    dataset = get_dataset(dataset_root='datasets', dataset_name=config.datasets.dataset_name)
    dataset_params = {
        'batch_size': config.models.params.batch_size,
        'data_split_ratio': config.datasets.data_split_ratio,
        'seed': config.datasets.seed
    }
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    log_file = (
        f"streaming_{config.datasets.dataset_name}_{config.models.gnn_name}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.debug(OmegaConf.to_yaml(config))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loader = get_dataloader(dataset, **dataset_params)
    test_indices = loader['test'].dataset.indices
    graph, influence_state, diversity_state, graph_id_begin = many_graphs_to_one_graph(dataset[test_indices], 
                                                      threshold=float(config.datasets.threshold),
                                                      radium=float(config.datasets.radium),
                                                      device=device)

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

    graph.to('cpu')

    bounds_list = config.datasets.bounds
    bounds = []
    bounds_list = OmegaConf.to_container(bounds_list)
    for i in range(config.datasets.num_classes):
        bounds.append((bounds_list[2 * i], bounds_list[2 * i + 1]))

    num_classes = config.datasets.num_classes
    print("Start streaming process")
    logging.debug("Start streaming process")
    pattern_set, solution, cost_time, pat_set, ex_sub_set = streaming_pattern_generation(graph=graph,
                                                                    model=model,
                                                                    bounds=bounds,
                                                                    influence_state=influence_state,
                                                                    diversity_state=diversity_state,
                                                                    gamma=float(config.datasets.gamma),
                                                                    cardinality_k=int(config.datasets.budget),
                                                                    num_classes=num_classes,
                                                                    k=int(config.datasets.k),
                                                                    dataset_name=config.datasets.dataset_name,
                                                                    graph_id_begin=graph_id_begin)
    print("Finish streaming process")
    logging.debug("Finish streaming process")
    print("Start evaluation")
    logging.debug("Start evaluation")
    print(f'cost-time: {cost_time:.4f}s')
    fidelity_minus_score, fidelity_plus_score, sparsity_score = evaluation(
                                                                    graph, pattern_set, solution, 
                                                                    model, pat_set, ex_sub_set, num_classes,
                                                                    dataset_name=config.datasets.dataset_name, 
                                                                    graph_id_begin=graph_id_begin)
    # data_record(config.datasets.dataset_name, 
    #             fidelity_minus_score, fidelity_plus_score, sparsity_score, cost_time=cost_time)
    print("End evaluation")
    logging.info("End evaluation")


if __name__ == '__main__':
    import sys

    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
