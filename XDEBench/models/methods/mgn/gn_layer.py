"""GN Layer."""
import torch
import torch.nn as nn
from torch_scatter import scatter

from .utils import build_mlp


class GNLayer(nn.Module):
    def __init__(self, in_dim=None, hidden_dim=32, out_dim=None, build_mlp=build_mlp, reduce='sum'):
        super().__init__()
        self.edge_module = EdgeModule(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                      build_mlp=build_mlp, reduce=reduce)
        self.vertex_module = VertexModule(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                          build_mlp=build_mlp, reduce=reduce)
    
    def forward(self, vertex_features, edge_features, edge_index):
        edge_features = self.edge_module(vertex_features, edge_features, edge_index)
        vertex_features = self.vertex_module(vertex_features, edge_features, edge_index)
        return vertex_features, edge_features


class EdgeModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, build_mlp=build_mlp, reduce='sum'):
        super().__init__()
        if in_dim is None and out_dim is None:
            self.mlp = build_mlp(hidden_dim * 3, hidden_dim, hidden_dim)
        else:
            self.mlp = build_mlp(in_dim, hidden_dim, out_dim)
        self.reduce = reduce
    
    def forward(self, vertex_features, edge_features, edge_index):
        senders_idx, receivers_idx = edge_index
        senders_features = vertex_features[senders_idx]
        receivers_features = vertex_features[receivers_idx]
        collected_features = torch.cat([senders_features, receivers_features, edge_features],
                                       dim=-1)
        edge_features = self.mlp(collected_features)  # (num_edge, h)
        return edge_features


class VertexModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, build_mlp=build_mlp, reduce='sum'):
        super().__init__()
        if in_dim is None and out_dim is None:
            self.mlp = build_mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        else:
            self.mlp = build_mlp(in_dim, hidden_dim, out_dim)
        self.reduce = reduce
    
    def forward(self, vertex_features, edge_features, edge_index):
        num_vertex = vertex_features.shape[0]
        _, receivers_idx = edge_index
        aggr_edge_features = scatter(edge_features, receivers_idx, dim=0,
                                          dim_size=num_vertex, reduce=self.reduce)  # (num_vertex, h)
        collected_features = torch.cat([vertex_features, aggr_edge_features], dim=-1)
        return self.mlp(collected_features)
