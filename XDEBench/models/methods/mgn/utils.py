"""Data-driven utils."""
import torch.nn as nn


def activation_func(name):
    """Return activation function by name."""
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unknown activation function: {name}')

def build_mlp(in_dim, hidden_dim, out_dim, activation='relu', layer_norm=True):
    module = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        activation_func(activation),
        nn.Linear(hidden_dim, hidden_dim),
        activation_func(activation),
        nn.Linear(hidden_dim, hidden_dim),
        activation_func(activation),
        nn.Linear(hidden_dim, out_dim)
    )
    if layer_norm:
        module.add_module('layer_norm', nn.LayerNorm(out_dim))
    return module
