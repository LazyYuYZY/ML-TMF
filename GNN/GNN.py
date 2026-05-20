import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GNNWithDNN(nn.Module):
    """
    图神经网络 + 后续 DNN 的组合模型。
    图卷积部分提取节点嵌入，最后的 DNN 对每个节点的嵌入进行回归预测。
    """
    def __init__(self,
                 in_dim=3,                     # 输入特征维度（CM Sketch 的行数）
                 gnn_hidden_dim=64,            # GNN 隐藏层维度
                 gnn_num_layers=2,             # GNN 层数
                 gnn_type='gat',               # 'gcn' 或 'gat'
                 heads=4,                      # GAT 多头注意力头数（仅当 gnn_type='gat' 时有效）
                 dropout=0.0,                  # GNN 中的 dropout
                 dnn_hidden_dim=128,           # 最终 DNN 的隐藏层维度
                 dnn_num_layers=2,             # 最终 DNN 的隐藏层数（不包括输出层）
                 output_dim=1):                # 输出维度（回归值为 1）
        super().__init__()

        self.gnn_num_layers = gnn_num_layers
        self.gnn_type = gnn_type.lower()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()           # 可选：层归一化
        self.dropout = dropout

        # ---------- 构建 GNN 层 ----------
        current_dim = in_dim
        for i in range(gnn_num_layers):
            if self.gnn_type == 'gcn':
                conv = GCNConv(current_dim, gnn_hidden_dim)
                current_dim = gnn_hidden_dim
            elif self.gnn_type == 'gat':
                # 最后一层通常使用单头输出，前面用多头拼接
                if i == gnn_num_layers - 1:
                    conv = GATConv(current_dim, gnn_hidden_dim, heads=1, dropout=dropout)
                    current_dim = gnn_hidden_dim
                else:
                    conv = GATConv(current_dim, gnn_hidden_dim, heads=heads, dropout=dropout)
                    current_dim = gnn_hidden_dim * heads
            else:
                raise ValueError("gnn_type must be 'gcn' or 'gat'")

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(current_dim))   # 稳定训练

        # ---------- 最终 DNN（回归头） ----------
        dnn_layers = []
        dnn_layers.append(nn.Linear(current_dim, dnn_hidden_dim))
        dnn_layers.append(nn.ELU())
        for _ in range(dnn_num_layers - 1):
            dnn_layers.append(nn.Linear(dnn_hidden_dim, dnn_hidden_dim))
            dnn_layers.append(nn.ELU())
        dnn_layers.append(nn.Linear(dnn_hidden_dim, output_dim))
        # 输出层不加激活函数（回归任务，损失为 MSE）
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, data):
        """
        输入 data 为元组 (node_features, edge_index) 或 torch_geometric.data.Data 对象。
        返回形状为 (num_nodes, output_dim) 的预测张量。
        """
        if isinstance(data, tuple):
            x, edge_index = data
        else:
            x, edge_index = data.x, data.edge_index

        # 逐层进行图卷积
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后通过 DNN 预测每个节点的目标值
        out = self.dnn(x)
        # 保持与参考代码接口一致，返回 (node_features, edge_index) 元组（但此处 edge_index 未修改）
        return out, edge_index