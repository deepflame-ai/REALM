"""Simulator."""
import torch
import torch.nn as nn

from .gn_model import GNModel


class Simulator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = GNModel(
            vertex_dim=args.vertex_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.width,
            out_dim=args.out_dim,
            n_layer=args.n_layers
        )
        self.noise_std = args.noise_std
    
    def forward(self, graph, roll_out=False):
        fields = graph.input_fields

        if not roll_out:
            fields_noise = self.get_fields_noise(graph)
            fields_noised = fields + fields_noise
            graph.vertex_attr = fields_noised
            predicted = self.model(graph)
            predicted_fields = fields_noised + predicted
            return predicted_fields
        else:
            graph.vertex_attr = fields
            predicted = self.model(graph)
            predicted_fields = fields + predicted
            return predicted_fields

    def get_fields_noise(self, graph):
        fields = graph.input_fields
        noise = torch.normal(std=self.noise_std, mean=0.0, size=fields.shape).to(fields.device)
        return noise
