import torch_geometric.transforms as T

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import torch
from dataset import get_dataset, get_dim
from warnings import simplefilter
import os
import hydra

from torch_geometric.utils import one_hot, subgraph
from torch_geometric.data import Data

def get_reddit(graph):

    # Step 1: Select Random Nodes
    num_nodes_to_select = 5000
    num_nodes = graph.num_nodes
    selected_nodes = torch.randperm(num_nodes)[:num_nodes_to_select]

    # Step 2: Extract Subgraph
    sub_data, _ = subgraph(selected_nodes, graph.edge_index, num_nodes=num_nodes, relabel_nodes=True)
    new_sub = Data(x=graph.x[selected_nodes], edge_index=sub_data, y=graph.y[selected_nodes])

    return new_sub


def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        acc = eval_node_classifier(model, graph, graph.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    return model


def eval_node_classifier(model, graph, mask):

    model.eval()
    pred = model(graph.x, graph.edge_index).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        output = self.conv2(x, edge_index)

        return output



@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):

    seed = 1124
    torch.manual_seed(seed)

    config.models.params = config.models.params[config.datasets.dataset_name]
    if config.datasets.dataset_name == 'BAHouse':
        data = get_dataset(dataset_root='datasets', dataset_name=config.datasets.dataset_name)
    else:
        data = get_dataset(dataset_root='datasets', dataset_name=config.datasets.dataset_name)[0]
    input_dim, output_dim = get_dim(config.datasets.dataset_name)
    if data.x is not None:
        data.x = data.x.float()
    data.y = data.y.squeeze()
    print(data)
    if config.datasets.dataset_name == 'Reddit':
        data = get_reddit(data)

    split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
    graph = split(data)
    graph.y = graph.y.to(torch.long)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    gcn = GCN(input_dim, output_dim).to(device)
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    print('start training')
    gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)

    test_acc = eval_node_classifier(gcn, graph, graph.test_mask)
    print(f'Test Acc: {test_acc:.3f}')

    save_dir = os.path.join('/home/cs.aau.dk/yj25pu/RoboGExp/checkpoints/', config.datasets.dataset_name, 'gcn_3l_best.pth')

    torch.save(gcn.state_dict(), save_dir)


if __name__ == '__main__':
    import sys

    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
    
