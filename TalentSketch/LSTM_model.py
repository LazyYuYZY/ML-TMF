# -*- coding:UTF-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size=1, hidden_size=16, output_size=1, num_layers=1,classif_type=False):
        super(LstmRNN, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
        self.classify_type=classif_type
        if self.classify_type:
            self.sigmoid = nn.Sigmoid()

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        if self.classify_type:
            x = self.sigmoid(x)
        return x[-1]