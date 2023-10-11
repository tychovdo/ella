import torch
from torch import nn

import numpy as np

from asdfghjkl.operations.linear2d import LinearUV

import math


class FLinear2d(nn.Module):
    def __init__(self, in_channels, out_channels, in_shape=(28, 28), out_shape=(14, 14), bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_size = in_shape[0] * in_shape[1]
        self.out_size = out_shape[0] * out_shape[1]
        self.lin_U = LinearUV(in_channels*self.in_size, out_channels, bias=bias)
        self.lin_V = LinearUV(out_channels*in_channels, self.out_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        bound = math.sqrt(3 / self.in_size)
        self.lin_U.weight.data.uniform_(-bound, bound)
        if self.lin_U.bias is not None:
            fan_in = self.in_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.lin_U.bias, -bound, bound)

        bound = math.sqrt(3 / self.in_channels)
        self.lin_V.weight.data.uniform_(-bound, bound)
        if self.lin_V.bias is not None:
            fan_in = self.in_channels
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.lin_V.bias, -bound, bound)

        # self.lin_U.weight.data = torch.sign(self.lin_U.weight.data) * torch.sqrt(torch.abs(self.lin_U.weight.data)) * math.sqrt(2)
        # self.lin_V.weight.data = torch.sign(self.lin_V.weight.data) * torch.sqrt(torch.abs(self.lin_V.weight.data)) * math.sqrt(2)

    def forward(self, x):
        b = x.shape[0]

        assert x.shape[1] == self.in_channels, f"Input does not have right amount of channels ({x.shape[1]} != {self.in_channels})"
        assert x.shape[2] * x.shape[3] == self.in_size, f"Input does not have a total of {self.in_size} values in spatial dimension ({x.shape})"

        x = x.reshape(-1, self.in_channels, self.in_size)
        x = x.permute(1, 0, 2) # cbx
        x = torch.block_diag(*x) #zv = (cb)(cx)
        x = x.view(self.in_channels, b, -1) # cb(cx)
        x = x.permute(1, 0, 2)  # bc(cx)
        x = self.lin_U(x) # zd = bc(d)
        x = x.permute(2, 0, 1) # dbc
        x = torch.block_diag(*x) # wu = (db)(dc)
        x = x.view(self.out_channels, b, -1) # db(dc)
        x = x.permute(1, 0, 2) # bd(dc)
        x = self.lin_V(x) # bd(y)

        x = x.view(-1, self.out_channels, *self.out_shape)

        return x

