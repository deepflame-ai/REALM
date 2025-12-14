import torch
import torch.nn as nn
from deepxde.nn import DeepONet  # pip install deepxde


class Model(nn.Module):  
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'DeepONet_deepxde3d'
        
        self.in_channels = args.in_dim-3
        self.out_channels = args.out_dim
        self.hidden_channels = args.width
        self.n_layers = args.n_layers
        self.dot_dims = args.modes   # temp use modes
        
        self.layers_branch = [self.in_channels] + [self.hidden_channels] * self.n_layers + [self.dot_dims*self.out_channels]
        self.layers_trunk = [4] + [self.hidden_channels] * self.n_layers + [self.dot_dims] # [x,y,z,t]
        
        self.net = DeepONet(
            layer_sizes_branch=self.layers_branch, 
            layer_sizes_trunk=self.layers_trunk, 
            activation="gelu", 
            kernel_initializer="Glorot normal", 
            num_outputs=self.out_channels, 
            multi_output_strategy='split_branch'
        )

    def forward(self, x, coords, time=None):
        # x: [batchSize, in_channels, nx, ny, nz]
        # time: [batchSize, 1, nx, ny, nz] (optional)
        # coords: [1, 3, nx, ny, nz]
        x_func = x.permute(0, 2, 3, 4, 1).reshape(-1, self.in_channels)        # x_func: [batchSize, nx, ny, nz, in_channels] -> [batchSize*nx*ny*nz, in_channels]
        
        if coords.shape[0] != x.shape[0]:
            coords = coords.repeat(x.shape[0], 1, 1, 1, 1)                      # x_loc: [1, 3, nx, ny] -> [batchSize, 3, nx, ny]
        x_loc = torch.cat((coords, time), dim=1)                                # [batchSize, 4, nx, ny, nz]        
        x_loc = x_loc.permute(0, 2, 3, 4, 1).reshape(-1, 4)                    # x_loc: [batchSize, nx, ny, nz, 4] -> [batchSize*nx*ny*nz, 4]
        
        out = self.net([x_func, x_loc])                                        # out: [batchSize*nx*ny*nz, out_channels]
        out = out.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[4], -1)  # out: [batchSize, nx, ny, nz, out_channels]
        out = out.permute(0, 4, 1, 2, 3)                                       # out: [batchSize, out_channels, nx, ny, nz]

        return out