# This model utilizes PyTorch Geometric's SAGEConv to aggregate information from neighboring transactions.


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGENet(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr="mean"
    ):
        super(GraphSAGENet, self).__init__()
        # We allow changing the aggregator (mean, max, lstm) as described in the paper
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x: Node feature matrix
        # edge_index: Graph connectivity matrix
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
