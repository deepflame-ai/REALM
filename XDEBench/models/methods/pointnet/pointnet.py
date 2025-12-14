import torch
import torch.nn as nn
import torch_geometric.nn as nng
from models.base._Embedding import unified_pos_embedding
from models.base._Basic import MLP


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'PointNet'

        self.n_hidden = args.width
        self.fun_dim = args.out_dim
        self.space_dim = args.in_dim - args.out_dim
        self.act = args.act
        self.out_dim = args.out_dim

        self.in_block = MLP(self.n_hidden, self.n_hidden * 2, self.n_hidden * 2, n_layers=0, res=False,
                            act=self.act)
        self.max_block = MLP(self.n_hidden * 2, self.n_hidden * 8, self.n_hidden * 32, n_layers=0, res=False,
                             act=self.act)

        self.out_block = MLP(self.n_hidden * (2 + 32), self.n_hidden * 16, self.n_hidden * 4, n_layers=0, res=False,
                             act=self.act)

        self.encoder = MLP(self.fun_dim + self.space_dim, self.n_hidden * 2, self.n_hidden, n_layers=0, res=False,
                           act=self.act)
        self.decoder = MLP(self.n_hidden, self.n_hidden * 2, self.out_dim, n_layers=0, res=False, act=self.act)

        self.fcfinal = nn.Linear(self.n_hidden * 4, self.n_hidden)

    def forward(self, x, coords):
        b, c, g = x.shape  # (batch, feat_dim, n_points)

        # 拼接坐标和函数值
        z = torch.cat((x, coords.repeat(b, 1, 1)), dim=1).permute(0, 2, 1)  # (b, g, c+2)

        # 构造 batch index
        batch = torch.arange(b, device=x.device).repeat_interleave(g)  # (b*g,)

        # 展平送进 encoder
        z = z.reshape(b*g, -1)
        z = self.encoder(z)
        z = self.in_block(z)

        # 全局特征
        global_coef = self.max_block(z)
        global_coef = nng.global_max_pool(global_coef, batch=batch)  # (b, hidden_big)

        # broadcast 回每个点
        global_coef = global_coef[batch]  # (b*g, hidden_big)

        # 拼接
        z = torch.cat([z, global_coef], dim=-1)  # (b*g, hidden+hidden_big)
        z = self.out_block(z)
        z = self.fcfinal(z)
        z = self.decoder(z)

        # reshape 回 (b, out_dim, g)
        return z.view(b, g, -1).permute(0, 2, 1)