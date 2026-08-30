# This is a standard Multi-Layer Perceptron (MLP) that will try to predict fraud using only the node 
# features, completely ignoring the network graph.

import torch
import torch.nn.functional as F


class MLPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(MLPNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        # We accept edge_index to keep the API consistent, but we IGNORE it here.
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
