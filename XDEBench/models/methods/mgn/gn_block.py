"""GN Block."""
import torch.nn as nn
from .gn_layer import GNLayer


class GNBlock(nn.Module):
    """GNBlock consists of multiple GNLayer."""
    def __init__(self, hidden_dim, n_layer):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.net = self.build_net(hidden_dim, n_layer)
    
    def build_net(self, hidden_dim, n_layer):
        net = nn.ModuleList()
        for i in range(n_layer):
            net.add_module(f'gn_layer_{i}', GNLayer(hidden_dim=hidden_dim))
        return net

    def forward(self, vertex_features, edge_features, edge_index):
        for layer in self.net:
            vertex_updated, edge_updated = layer(
                vertex_features=vertex_features,
                edge_features=edge_features,
                edge_index=edge_index
            )
            vertex_features = vertex_features + vertex_updated
            edge_features = edge_features + edge_updated
        return vertex_features, edge_features
