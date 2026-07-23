"""FedGCN 使用的两层图卷积网络。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    """实现先线性变换、再按邻接矩阵传播的基础图卷积层。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """初始化图卷积权重、可选偏置和参数范围。"""

        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """按照原FedGCN实现的均匀分布方式初始化参数。"""

        limit = 1.0 / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight, -limit, limit)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -limit, limit)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """根据节点特征和归一化邻接矩阵计算一层图卷积输出。"""

        support = torch.mm(features, self.weight)
        if adjacency.is_sparse:
            output = torch.sparse.mm(adjacency, support)
        else:
            output = torch.mm(adjacency, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        """返回便于日志查看的输入与输出维度。"""

        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features
        )


class GCN(nn.Module):
    """用于Cora/Citeseer节点分类的两层GCN。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        """创建两层图卷积、ReLU和dropout结构。"""

        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """输出每个节点在全部类别上的对数概率。"""

        hidden = F.relu(self.gc1(features, adjacency))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.gc2(hidden, adjacency)
        return F.log_softmax(logits, dim=1)

