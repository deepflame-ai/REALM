import torch.nn as nn
from torch_geometric.nn import GATConv

# GAT模型定义
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'GAT'
        self.in_dim = args.vertex_dim
        self.out_dim = args.out_dim
        self.edge_features = args.edge_dim
        self.hidden_channels = args.width
        self.heads = args.n_heads
        self.dropout = 0.5
        
        # 第一层GAT：处理节点特征，考虑边特征
        self.conv1 = GATConv(in_channels=self.in_dim, 
                             out_channels=self.hidden_channels, 
                             heads=self.heads, 
                             edge_dim=self.edge_features, 
                             dropout=self.dropout)
        # 第二层GAT：输出预测
        self.conv2 = GATConv(in_channels=self.hidden_channels * self.heads, 
                             out_channels=self.out_dim, 
                             heads=1, 
                             edge_dim=self.edge_features, 
                             dropout=self.dropout)
        
    def forward(self, data, roll_out=False):
        graph = data
        x, edge_index, edge_attr = graph.input_fields, graph.edge_index, graph.edge_attr
        
        # 第一层GAT
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层GAT
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x