import torch
import copy
import torch.nn as nn
import torch.nn.functional as F



class BranchNet(nn.Module):
    """Branch net with residual connection on conv_blocks output"""
    def __init__(self, in_channels, hidden_channels, p, q, num_conv_layers=4, kernel_size=3):
        super().__init__()
        self.use_residual = in_channels == hidden_channels  # only if shape match
        layers = []
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_conv_layers - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
        self.conv_blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, p * q)
        )
        # optional 1x1 projection if in_channels != hidden_channels
        self.res_proj = None
        if not self.use_residual:
            self.res_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

    def forward(self, u):  # u: (b, c, s, s)
        residual = u if self.use_residual else self.res_proj(u)
        x = self.conv_blocks(u)
        x = x + residual  # residual connection
        x = self.pool(x)
        return self.fc(x.squeeze(-1).squeeze(-1))  # (b, p*q)
    

class TrunkNet(nn.Module):
    """Trunk net processes spatial coordinates"""
    def __init__(self, p, q, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        in_dim = 2
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, p * q))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):  # coords: (s*s, 2)
        return self.net(coords)  # (s*s, p*q)

class Model(nn.Module):  
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'DeepONetv2'
        self.in_channels = args.in_dim
        self.out_channels = args.out_dim
        self.p = args.width
        self.q = args.out_dim
        self.num_chemical = args.num_chemical
        self.hidden_channels = args.width
        self.trunk_hidden = args.width
        self.trunk_layers = 5
        self.branch_layers = args.n_layers

        self.branch = BranchNet(self.in_channels, self.hidden_channels, self.p, self.out_channels,
                                num_conv_layers=self.branch_layers)
        self.trunk = TrunkNet(self.p, self.out_channels, hidden_dim=self.trunk_hidden, num_layers=self.trunk_layers)

    def forward(self, x, coords):
        # u: (b, s, s, c) --> permute to (b, c, s, s)
        x_0 = copy.copy(x[:, :self.out_channels])
        batch_size = x.shape[0]
        x = torch.cat((x, coords.repeat(batch_size, 1, 1, 1)), dim=1)
        b, c, s, _ = x.shape
        # branch output: (b, p*q)
        B = self.branch(x)
        # trunk output: (s*s, p*q)
        coords = coords.reshape(-1, 2)
        T = self.trunk(coords)
        # reshape for multiplication
        B = B.view(b, self.q, self.p)       # (b, q, p)
        T = T.view(s*s, self.q, self.p)     # (s*s, q, p)
        # dot over p: (b, q, s*s)
        out = torch.einsum('bqp,iqp->bqi', B, T)
        # reshape to (b, q, s, s)
        out = out.view(b, self.q, s, s)
        return out + x_0