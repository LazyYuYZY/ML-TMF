import torch.nn as nn
import torch

class DNN0(nn.Module):
    def __init__(self, input_size=3, deep_l=2, num=100):
        super(DNN0, self).__init__()
        layers = nn.ModuleList()
        # 输入层
        layers.append(nn.Linear(in_features=input_size, out_features=num))
        layers.append(nn.ReLU())
        for i in range(deep_l - 1):
            layers.append(nn.Linear(num, num))
            layers.append(nn.ReLU())
        # 输出层
        layers.append(nn.Linear(num, 1))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, input: torch.FloatTensor):
        return self.net(input)

class DNN1(nn.Module):
    def __init__(self, input_size=3, deep_l=2, num=100,drop_out=0.0):
        super(DNN1, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)
        layers = nn.ModuleList()
        # 输入层
        layers.append(nn.Linear(in_features=input_size, out_features=num))
        layers.append(nn.ReLU())
        for i in range(deep_l - 1):
            layers.append(self.dropout)
            layers.append(nn.Linear(num, num))
            layers.append(nn.ReLU())
        # 输出层
        layers.append(nn.Linear(num, 1))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, input: torch.FloatTensor):
        return self.net(input)
