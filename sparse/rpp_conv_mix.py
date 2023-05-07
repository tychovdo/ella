import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse.flinear2d import FLinear2d
from sparse.slinear2d import SLinear2d
from sparse.sconv2d import SConv2d

from asdfghjkl.operations import Conv2dAug


def subsample2(x):
    x1 = x[:, :, ::2, ::2]
    x2 = x[:, :, 1::2, ::2]
    x3 = x[:, :, ::2, 1::2]
    x4 = x[:, :, 1::2, 1::2]
    
    x1m, x2m, x3m, x4m = x1.max(), x2.max(), x3.max(), x4.max()
    maxval = max(x1m, x2m, x3m, x4m)
    if x1m == maxval:
        return x1
    elif x2m == maxval:
        return x2
    elif x3m == maxval:
        return x3
    else:
        return x4


def lift_p4(x):
    x = torch.stack([torch.rot90(x, k=k, dims=[-1, -2]) for k in range(4)], 1)
    x = x.reshape(x.shape[0], -1, *x.shape[3:])
    
    return x


class GConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, G=1, stride=1, padding=1, padding_mode='circular'):
        super(GConv2d, self).__init__()

        self.conv = Conv2dAug(in_ch, out_ch, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.G = G
        
    def forward(self, x):
        B, _, H, W = x.shape
        C = x.shape[1] // self.G
        
        x_aug = x.view(B, self.G, C, H, W)
        y_aug = self.conv(x_aug)
        y = y_aug.view(B, -1, *y_aug.shape[3:])
        
        return y
    

class MixBlock1(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, G=4, stride=1):
        super(MixBlock1, self).__init__()
        self.gconv = GConv2d(in_ch, out_ch, 3, G=G, stride=1, padding=1, padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.gconv.conv.weight, nonlinearity='relu')
        self.gconv.conv.bias.data.zero_()

        self.conv = nn.Conv2d(in_ch*G, out_ch*G, 3, stride=1, padding=1, padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        self.conv.bias.data.zero_()

        self.linear = FLinear2d(in_ch*G, out_ch*G, (h, w), (h//stride, w//stride))
        self.linear.lin_U.bias.data.zero_()
        self.linear.lin_V.bias.data.zero_()

        self.stride = stride
        self.activation = F.relu
        
    def forward(self, x):
        gconvout = self.gconv(x)
        convout = self.conv(x)
        linout = self.linear(x)

        if self.stride == 2:
            gconvout = subsample2(gconvout)
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")

        print(gconvout.shape, convout.shape, linout.shape)
        convout = (gconvout + convout + linout) / 3.0
        
        return self.activation(convout)
    
class Mixer1(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, G=4):
        super(Mixer1, self).__init__()

        self.G = G

        modules = [MixBlock1(in_ch, alpha, h=h, w=w, G=G, stride=1),
                   MixBlock1(alpha, 2*alpha, h=h, w=w, G=G, stride=2),
                   MixBlock1(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), G=G, stride=1),
                   MixBlock1(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), G=G, stride=2),
                   MixBlock1(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), G=G, stride=1),
                   MixBlock1(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), G=G, stride=2),
                   MixBlock1(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), G=G, stride=2),
                   MixBlock1(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), G=G, stride=2)]
        self.conv_net = nn.Sequential(*modules)

        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        # lift to p4
        x = lift_p4(x)

        # equivariant network
        out = self.conv_net(x)

        # maxpool p4
        B, _, H, W = out.shape
        C = out.shape[1] // self.G
        out = out.view(B, self.G, C, H, W).amax(1)

        # final layers
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)

        return out


