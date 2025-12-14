"""GN Model."""
import torch.nn as nn

from .utils import build_mlp
from .gn_block import GNBlock


class GNModel(nn.Module):
    def __init__(self, vertex_dim, edge_dim, hidden_dim, out_dim, n_layer):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.encoder = Encoder(vertex_dim, edge_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, out_dim)
        self.processor = GNBlock(hidden_dim, n_layer)

    def forward(self, graph):
        vertex_attr = graph.vertex_attr
        edge_attr = graph.edge_attr
        vertex_features, edge_features = self.encoder(
            vertex_attr=vertex_attr,
            edge_attr=edge_attr
        )
        vertex_features, _ = self.processor(
            vertex_features=vertex_features,
            edge_features=edge_features,
            edge_index=graph.edge_index
        )
        vertex_attr = self.decoder(vertex_features)
        return vertex_attr
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class Encoder(nn.Module):
    """Encoder of vertex, edge."""
    def __init__(self, vertex_dim, edge_dim, hidden_dim):
        super().__init__()
        self.vertex_mlp = build_mlp(in_dim=vertex_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.edge_mlp = build_mlp(in_dim=edge_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)

    def forward(self, vertex_attr, edge_attr):
        vertex_features = self.vertex_mlp(vertex_attr)
        edge_features = self.edge_mlp(edge_attr)
        return vertex_features, edge_features


class Decoder(nn.Module):
    """Decoder of vertex."""
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.out_mlp = build_mlp(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim, \
                                 layer_norm=False)

    def forward(self, input_features):
        return self.out_mlp(input_features)
