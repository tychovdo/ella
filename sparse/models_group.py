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
    

class GConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, G=4, stride=1):
        super(GConvBlock, self).__init__()
        self.conv = GConv2d(in_ch, out_ch, 3, G=G, stride=1, padding=1, padding_mode='circular')
        self.conv.conv.bias.data.zero_()
        self.stride = stride
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)
        
        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")
        
        return self.activation(convout)
    
class D_GConv(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, G=4):
        super(D_GConv, self).__init__()

        self.G = G

        modules = [GConvBlock(in_ch, alpha, h=h, w=w, G=G, stride=1),
                   GConvBlock(alpha, 2*alpha, h=h, w=w, G=G, stride=2),
                   GConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), G=G, stride=1),
                   GConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), G=G, stride=2),
                   GConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), G=G, stride=1),
                   GConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), G=G, stride=2),
                   GConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), G=G, stride=2),
                   GConvBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), G=G, stride=2)]
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


class GConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, G=4, stride=1, padding=1, padding_mode='circular', groups=1):
        super(GConv3d, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups)
        self.G = G
        
    def forward(self, x):
        B, _, H, W = x.shape
        C = x.shape[1] // self.G
        
        x_aug = x.view(B, self.G, C, H, W)
        x_aug = x_aug.transpose(1, 2)
        
        y_aug = self.conv(x_aug)
        y_aug = y_aug.transpose(1, 2)
        y = y_aug.reshape(B, -1, *y_aug.shape[3:])
        
        return y

class GConvBlockFull(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, G=4, stride=1, padding=1):
        super(GConvBlockFull, self).__init__()
        self.conv = GConv3d(in_ch, out_ch, (4, 3, 3), G=G, stride=(1, 1, 1), padding='same', padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv.conv.weight, nonlinearity='relu')
        self.conv.conv.bias.data.zero_()
        self.stride = stride
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)
        
        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")
        
        return self.activation(convout)

class D_GConvFull(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, G=4):
        super(D_GConvFull, self).__init__()

        self.G = G

        modules = [GConvBlockFull(in_ch, alpha, h=h, w=w, G=G, stride=1),
                   GConvBlockFull(alpha, 2*alpha, h=h, w=w, G=G, stride=2),
                   GConvBlockFull(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), G=G, stride=1),
                   GConvBlockFull(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), G=G, stride=2),
                   GConvBlockFull(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), G=G, stride=1),
                   GConvBlockFull(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), G=G, stride=2),
                   GConvBlockFull(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), G=G, stride=2),
                   GConvBlockFull(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), G=G, stride=2)]
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

class GConvBlockPointwise(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, G=4, stride=1, padding=1):
        super(GConvBlockPointwise, self).__init__()
        self.conv = GConv3d(in_ch, out_ch, (1, 3, 3), G=G, stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv.conv.weight, nonlinearity='relu')
        self.conv.conv.bias.data.zero_()
        self.stride = stride
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)
        
        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")
        
        convout = self.activation(convout)

        return convout


class D_GConvPointwise(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, G=4):
        super(D_GConvPointwise, self).__init__()

        self.G = G

        modules = [GConvBlockPointwise(in_ch, alpha, h=h, w=w, G=G, stride=1),
                   GConvBlockPointwise(alpha, 2*alpha, h=h, w=w, G=G, stride=2),
                   GConvBlockPointwise(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), G=G, stride=1),
                   GConvBlockPointwise(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), G=G, stride=2),
                   GConvBlockPointwise(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), G=G, stride=1),
                   GConvBlockPointwise(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), G=G, stride=2),
                   GConvBlockPointwise(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), G=G, stride=2),
                   GConvBlockPointwise(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), G=G, stride=2)]
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


class GConvBlockGSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, G=4, stride=1, padding=1):
        super(GConvBlockGSeparable, self).__init__()
        self.in_ch = in_ch
        self.conv1 = GConv3d(in_ch, in_ch, (1, 3, 3), G=G, stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='circular', groups=in_ch)
        torch.nn.init.kaiming_uniform_(self.conv1.conv.weight, nonlinearity='relu') 
        self.conv1.conv.bias.data.zero_()
        self.conv2 = GConv3d(in_ch, out_ch, (G, 1, 1), G=G, stride=(1, 1, 1), padding='same', padding_mode='circular')
        self.conv2.conv.weight.data.fill_(1 / np.sqrt(in_ch) / G)
        self.conv2.conv.bias.data.zero_()
        self.stride = stride
        self.activation = F.relu
        self.in_ch = in_ch
        
    def forward(self, x):
        convout = self.conv1(x)
        convout = self.conv2(convout)
        
        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")
        
        convout = self.activation(convout)
        return convout


class D_GConvGSeparable(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, G=4):
        super(D_GConvGSeparable, self).__init__()

        self.G = G

        modules = [GConvBlockGSeparable(in_ch, alpha, h=h, w=w, G=G, stride=1),
                   GConvBlockGSeparable(alpha, 2*alpha, h=h, w=w, G=G, stride=2),
                   GConvBlockGSeparable(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), G=G, stride=1),
                   GConvBlockGSeparable(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), G=G, stride=2),
                   GConvBlockGSeparable(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), G=G, stride=1),
                   GConvBlockGSeparable(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), G=G, stride=2),
                   GConvBlockGSeparable(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), G=G, stride=2),
                   GConvBlockGSeparable(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), G=G, stride=2)]
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

    
class GConvBlockDGSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, G=4, stride=1, padding=1):
        super(GConvBlockDGSeparable, self).__init__()
        self.conv1 = GConv3d(in_ch, in_ch, (1, 3, 3), G=G, stride=(1, 1, 1), groups=in_ch, padding=(0, 1, 1), padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv1.conv.weight, nonlinearity='relu') 
        self.conv1.conv.bias.data.zero_()
        self.conv2 = GConv3d(in_ch, in_ch, (G, 1, 1), G=G, stride=(1, 1, 1), groups=in_ch, padding='same', padding_mode='circular')
        self.conv2.conv.weight.data.fill_(1 / G)
        self.conv2.conv.bias.data.zero_()
        self.conv3 = GConv3d(in_ch, out_ch, (1, 1, 1), G=G, stride=(1, 1, 1), padding=(0, 0, 0), padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv3.conv.weight, nonlinearity='linear') 
        self.conv3.conv.bias.data.zero_()
        self.stride = stride
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv1(x)
        convout = self.conv2(convout)
        convout = self.conv3(convout)
        
        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")
        
        convout = self.activation(convout)
        return convout


class D_GConvDGSeparable(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, G=4):
        super(D_GConvDGSeparable, self).__init__()

        self.G = G

        modules = [GConvBlockDGSeparable(in_ch, alpha, h=h, w=w, G=G, stride=1),
                   GConvBlockDGSeparable(alpha, 2*alpha, h=h, w=w, G=G, stride=2),
                   GConvBlockDGSeparable(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), G=G, stride=1),
                   GConvBlockDGSeparable(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), G=G, stride=2),
                   GConvBlockDGSeparable(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), G=G, stride=1),
                   GConvBlockDGSeparable(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), G=G, stride=2),
                   GConvBlockDGSeparable(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), G=G, stride=2),
                   GConvBlockDGSeparable(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), G=G, stride=2)]
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

    

