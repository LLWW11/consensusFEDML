"""提供兼容 FedML 旧 FEMNIST CNN 参数结构的纯PyTorch模型。"""

from __future__ import absolute_import

import torch
import torch.nn as nn


class FEMNISTChannelsLastCNN(nn.Module):
    """保留旧CNN层和初始化，仅修正输入通道维处理。"""

    def __init__(self):
        """按旧 FedML FEMNIST CNN 的62类结构初始化全部参数。"""
        super(FEMNISTChannelsLastCNN, self).__init__()
        # 层创建顺序与 FedML 0.7.600 CNN_DropOut(False) 保持一致。
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 62)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        """接受[N,28,28]或[N,1,28,28]并输出62类logits。"""
        if inputs.ndim == 3:
            inputs = torch.unsqueeze(inputs, 1)
        if inputs.ndim != 4 or int(inputs.shape[1]) != 1:
            raise ValueError(
                "FEMNIST CNN输入必须为[N,28,28]或[N,1,28,28]。"
            )
        values = self.conv2d_1(inputs)
        values = self.relu(values)
        values = self.conv2d_2(values)
        values = self.relu(values)
        values = self.max_pooling(values)
        values = self.dropout_1(values)
        values = self.flatten(values)
        values = self.linear_1(values)
        values = self.relu(values)
        values = self.dropout_2(values)
        return self.linear_2(values)
