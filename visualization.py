import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
from typing import List, Dict, Union


class PlotUtils(object):
    """
    PlotUtils For datasets
    """

    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, node_list=None, fig_name=None, title_sentence=None, **kwargs):
        """
        Args:
            graph: origin graph
            node_list: choose node
            fig_name: fig title name
            title_sentence: describe sentence for this fig
            **kwargs: such as node_attribute matrix x and prediction label y
        Returns: No return
        """
        if self.dataset_name in ['Mutagenicity', 'Mutag', 'NCI1']:
            x = kwargs.get('x')
            self.plot_molecule(graph, node_list, x, title_sentence=title_sentence, fig_name=fig_name)
        elif self.dataset_name in ['BA-2MOTIFS']:
            x = kwargs.get('x')
            self.plot_ba2motifs(graph, node_list, x, title_sentence=title_sentence, fig_name=fig_name)
        elif self.dataset_name in ['ENZYMES', 'PROTEINS']:
            x = kwargs.get('x')
            self.plot_protein(graph, node_list, x, title_sentence=title_sentence, fig_name=fig_name)
        elif self.dataset_name in ['REDDIT-MULTI-5K', 'REDDIT-BINARY', 'IMDB-MULTI']:
            self.plot_reddit(graph, node_list, title_sentence=title_sentence, fig_name=fig_name)
        elif self.dataset_name in ['TRIANGLES', 'Synthie']:
            x = kwargs.get('x')
            self.plot_synthetic(graph, node_list, x, title_sentence=title_sentence, fig_name=fig_name)
        else:
            raise NotImplemented

    def plot_subgraph(self,
                      graph,
                      node_list,
                      colors: Union[None, str, List[str]] = '#EEBF6D',
                      labels=None,
                      edge_list=None,
                      edge_colors='gray',
                      subgraph_edge_colors='black',
                      fig_name=None,
                      title_sentence=None
                      ):
        """
        ----------------------------------
        if node_list == None:
            plot origin subgraph only
        else :
            plot explainable subgraph too
        ----------------------------------
        """
        if edge_list is None and node_list is not None:
            edge_list = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in node_list and n_to in node_list]

        pos = nx.kamada_kawai_layout(graph)
        if node_list is not None:
            pos_nodelist = {k: v for k, v in pos.items() if k in node_list}
        else:
            pos_nodelist = {}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=graph.nodes(),
                               node_color=colors,
                               node_size=350,
                               alpha=1)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_colors, arrows=False)
        if edge_list is not None:
            nx.draw_networkx_edges(graph, pos=pos_nodelist,
                                   edgelist=edge_list, width=3,
                                   edge_color=subgraph_edge_colors,
                                   arrows=False)

        if node_list is not None:
            pass

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels=labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))
        if fig_name is not None:
            # fig_record_name = f'../results/figures/{self.dataset_name}' \
            #                   f'/{fig_name}_{time.strftime("%Y%m%d-%H%M", time.localtime())}'
            fig_record_name = f'imgs/{self.dataset_name}' \
                              f'/{fig_name}_{time.strftime("%Y%m%d-%H%M", time.localtime())}'
            plt.savefig(fig_record_name)
        if self.is_show:
            plt.show()
        if fig_name is not None:
            plt.close()

    def plot_molecule(self, graph, node_list, x, edge_list=None, title_sentence=None, fig_name=None):
        node_idx = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        if self.dataset_name == "Mutag":
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_labels = {k: node_dict[v] for k, v in node_idx.items()}
            node_color = ['#EEBF6D', '#4EAB90', '#834026', '#29A329', '#EDDDC3', '#8EB69C', '#D94F33']
            colors = [node_color[v % len(node_color)] for k, v in node_idx.items()]
        elif self.dataset_name == 'Mutagenicity':
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P',
                         9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            node_labels = {k: node_dict[v] for k, v in node_idx.items()}
            node_color = ['#EEBF6D', '#834026', '#8EB69C', '#E6F1F3', '#4EAB90', '#29A329', '#D94F33', '#90BEE0',
                          '#A9A9A9', '#EDDDC3', '#4B74B2', '#BA55D3', '#7B68EE', '#DAA520']
            colors = [node_color[v % len(node_color)] for k, v in node_idx.items()]
        elif self.dataset_name == 'NCI1':
            node_dict = {i: i for i in range(37)}
            node_labels = {k: node_dict[v] for k, v in node_idx.items()}
            node_color = ['#EEBF6D', '#834026', '#8EB69C', '#FC2E92', '#FD3030', '#DB25F8', '#FB6720',
                          '#27A6FC', '#4D1EF7', '#FC04CD', '#32F6A7', '#2C62F4', '#FEF829', '#F8340E',
                          '#33F567', '#2BF8E7', '#FC8F28', '#62FB05', '#3FF40D', '#28F5C3', '#E1FC2F',
                          '#F634B2', '#8038F6', '#1AF532', '#A034FC', '#FD25F7', '#33CCFD', '#9FFC33',
                          '#F7B130', '#2013FE', '#FC2A6E', '#B50AFE', '#36E5F4', '#FED730', '#33F328',
                          '#132CF5', '#0BFD73']
            colors = [node_color[v % len(node_color)] for k, v in node_idx.items()]

        else:
            raise NotImplementedError
        self.plot_subgraph(graph, node_list,
                           colors=colors,
                        #    labels=node_labels,
                           edge_list=edge_list,
                           edge_colors='gray',
                           subgraph_edge_colors='black',
                           title_sentence=title_sentence,
                           fig_name=fig_name
                           )

    def plot_ba2motifs(self, graph, node_list, x, edge_list=None, title_sentence=None, fig_name=None):
        node_idx = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        if self.dataset_name.lower() in ['ba-2motifs']:
            node_colors = ['#EEBF6D', '#4EAB90', '#834026']
            colors = [node_colors[v % len(node_colors)] for k, v in node_idx.items()]
        else:
            raise NotImplementedError
        self.plot_subgraph(graph, node_list,
                           colors=colors,
                           # labels=node_labels,
                           edge_list=edge_list,
                           edge_colors='gray',
                           subgraph_edge_colors='black',
                           title_sentence=title_sentence,
                           fig_name=fig_name
                           )

    def plot_protein(self, graph, node_list, x, edge_list=None, title_sentence=None, fig_name=None):
        node_idx = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        if self.dataset_name.lower() in ['enzymes', 'proteins']:
            node_dict = {0: 'Y', 1: 'G', 2: 'B'}
            node_labels = {k: node_dict[v] for k, v in node_idx.items()}
            node_colors = ['#EEBF6D', '#4EAB90', '#834026']
            colors = [node_colors[v % len(node_colors)] for k, v in node_idx.items()]
        else:
            raise NotImplementedError
        self.plot_subgraph(graph, node_list,
                           colors=colors,
                           # labels=node_labels,
                           edge_list=edge_list,
                           edge_colors='gray',
                           subgraph_edge_colors='black',
                           title_sentence=title_sentence,
                           fig_name=fig_name
                           )

    def plot_reddit(self, graph, node_list=None, edge_list=None, title_sentence=None, fig_name=None):
        return self.plot_subgraph(graph, node_list,
                                  edge_list=edge_list,
                                  title_sentence=title_sentence,
                                  fig_name=fig_name)

    def plot_synthetic(self, graph, node_list, x, edge_list=None, title_sentence=None, fig_name=None):
        node_idx = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
        if self.dataset_name.lower() in ['triangles', 'synthie']:
            node_dict = {i: i for i in range(11)}
            node_labels = {k: node_dict[v] for k, v in node_idx.items()}
            node_colors = ['#EEBF6D', '#834026', '#8EB69C', '#FC2E92', '#FD3030', '#DB25F8', '#FB6720',
                           '#27A6FC', '#4D1EF7', '#FC04CD', '#32F6A7']
            colors = [node_colors[v % len(node_colors)] for k, v in node_idx.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, node_list,
                           colors=colors,
                           labels=None,
                           edge_list=edge_list,
                           edge_colors='gray',
                           subgraph_edge_colors='black',
                           title_sentence=title_sentence,
                           fig_name=fig_name
                           )