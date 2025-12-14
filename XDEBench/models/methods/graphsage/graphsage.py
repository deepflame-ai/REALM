import torch
import torch.nn as nn
import torch_geometric.nn as nng
from models.base._Basic import MLP


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'GraphSAGE'

        self.nb_hidden_layers = args.n_layers
        self.size_hidden_layers = args.width
        self.bn_bool = True
        self.activation = nn.ReLU()
        self.in_dim = args.vertex_dim
        self.out_dim = args.out_dim

        self.encoder = MLP(self.in_dim, args.width * 2, args.width, n_layers=0, res=False,
                           act=args.act)
        self.decoder = MLP(args.width, args.width * 2, self.out_dim, n_layers=0, res=False, act=args.act)

        self.in_layer = nng.SAGEConv(
            in_channels=args.width,
            out_channels=self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(
                in_channels=self.size_hidden_layers,
                out_channels=self.size_hidden_layers
            ))

        self.out_layer = nng.SAGEConv(
            in_channels=self.size_hidden_layers,
            out_channels=self.size_hidden_layers
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))

    def forward(self, data, roll_out=False):
        graph = data
        z, edge_index = graph.input_fields, graph.edge_index
        
        z = self.encoder(z)
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)
        z = self.out_layer(z, edge_index)
        z = self.decoder(z)
        return z
