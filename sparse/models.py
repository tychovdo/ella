import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse.flinear2d import FLinear2d
from sparse.slinear2d import SLinear2d
from sparse.sconv2d import SConv2d


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

class BNLinearBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(BNLinearBlock, self).__init__()
        self.linear = nn.Linear(int(in_ch*h*w), int(out_ch*h*w/(stride**2)))
        self.linear.bias.data.zero_()
        self.stride=stride
        self.out_ch = out_ch
        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        linout = self.linear(x.reshape(x.shape[0], -1))
        linout = linout.view(x.shape[0], self.out_ch, 
                             int(self.h/self.stride), int(self.w/self.stride))
        out = linout
        return self.activation(out)

class BNConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(BNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        self.conv.bias.data.zero_()
        self.stride = stride
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)
        
        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")
        
        return self.activation(convout)
    
class BNLinearConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(BNLinearConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        self.conv.bias.data.zero_()
        self.stride = stride
        self.linear = nn.Linear(int(in_ch*h*w), int(out_ch*h*w/(stride**2)))
        self.linear.bias.data.zero_()
        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)

        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")

        linout = self.linear(x.reshape(x.shape[0], -1))
        linout = linout.view(convout.shape)
        
        out = 0.5 * (linout + convout)
        return self.activation(out)

class FLinearBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(FLinearBlock, self).__init__()
        self.linear = FLinear2d(in_ch, out_ch, (h, w), (h//stride, w//stride))
        self.linear.lin_U.bias.data.zero_()
        self.linear.lin_V.bias.data.zero_()
        self.stride=stride
        self.out_ch = out_ch
        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        linout = self.linear(x)
        out = linout
        out = self.activation(out)
        return out


class FLinearConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(FLinearConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='circular')
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        self.conv.bias.data.zero_()
        self.stride = stride
        self.linear = FLinear2d(in_ch, out_ch, (h, w), (h//stride, w//stride))
        self.linear.lin_U.bias.data.zero_()
        self.linear.lin_V.bias.data.zero_()
        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)

        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")

        linout = self.linear(x)
        
        out = 0.5 * (linout + convout)
        out = self.activation(out)

        return out
    
    
class SConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1, learn_omega=False):
        super(SConvBlock, self).__init__()
        self.conv = SConv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='circular', bias=True, learn_omega=learn_omega)

        tmp_conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='circular')
        torch.nn.init.kaiming_uniform_(tmp_conv.weight, nonlinearity='relu')
        self.conv._set_u_from_weight(tmp_conv.weight)

        self.conv.bias.data.zero_()
        self.stride = stride
        self.conv.reduce(0.8)
        self.activation = F.relu
        
    def forward(self, x):
        convout  = self.conv(x)

        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")

        return self.activation(convout)

class SLinearBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1, learn_omega=False):
        super(SLinearBlock, self).__init__()
        self.linear = SLinear2d(in_ch, out_ch, (h, w), (h//stride, w//stride), bias=False, learn_omega=learn_omega)

        self.linear.reduce(0.5)
        self.linear.lin_U.bias.data.zero_()
        # self.linear.lin_V.bias.data.zero_()

        self.stride=stride
        self.out_ch = out_ch
        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        linout = self.linear(x)
        out = linout
        out = self.activation(out)

        return out

class SLinearConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1, learn_omega=False):
        super(SLinearConvBlock, self).__init__()
        self.conv = SConv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='circular', bias=True, learn_omega=learn_omega)

        tmp_conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='circular')
        torch.nn.init.kaiming_uniform_(tmp_conv.weight, nonlinearity='relu')
        self.conv._set_u_from_weight(tmp_conv.weight)
        self.conv.reduce(0.5)

        self.conv.bias.data.zero_()

        self.stride = stride
        self.linear = SLinear2d(in_ch, out_ch, (h, w), (h//stride, w//stride), learn_omega=learn_omega)

        self.linear.reduce(0.5)
        self.linear.lin_U.bias.data.zero_()
        # self.linear.lin_V.bias.data.zero_()

        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)

        if self.stride == 2:
            convout = subsample2(convout)
        elif self.stride > 2:
            raise NotImplementedError(f"Equivariance for stride {self.stride} not implemented yet")

        linout = self.linear(x)
        
        out = 0.5 * (linout + convout)
        out = self.activation(out)

        return out


#### NETWORKS: 

class D_FC(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(D_FC, self).__init__()
        modules = [BNLinearBlock(in_ch, alpha, h=h, w=w, stride=1),
                  BNLinearBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                  BNLinearBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                  BNLinearBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                  BNLinearBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                  BNLinearBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                  BNLinearBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2),
                  BNLinearBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out
    
class D_Conv(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(D_Conv, self).__init__()
        modules = [BNConvBlock(in_ch, alpha, h=h, w=w, stride=1),
                   BNConvBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   BNConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   BNConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   BNConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   BNConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   BNConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2),
                   BNConvBlock(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out
    
class D_Both(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(D_Both, self).__init__()
        modules = [BNLinearConvBlock(in_ch, alpha, h=h, w=w, stride=1),
                   BNLinearConvBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   BNLinearConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   BNLinearConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   BNLinearConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   BNLinearConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   BNLinearConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2),
                   BNLinearConvBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out

class F_FC(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(F_FC, self).__init__()
        modules = [FLinearBlock(in_ch, alpha, h=h, w=w, stride=1),
                   FLinearBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   FLinearBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   FLinearBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   FLinearBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   FLinearBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   FLinearBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2),
                   FLinearBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), stride=2)]
        self.conv_net = nn.Sequential(*modules)

        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out
    

class F_Both(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(F_Both, self).__init__()
        modules = [FLinearConvBlock(in_ch, alpha, h=h, w=w, stride=1),
                   FLinearConvBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   FLinearConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   FLinearConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   FLinearConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   FLinearConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   FLinearConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2),
                   FLinearConvBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), stride=2),]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out
    

class S_Conv(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, learn_omega=False):
        super(S_Conv, self).__init__()
        modules = [SConvBlock(in_ch, alpha, h=h, w=w, stride=1, learn_omega=learn_omega),
                   SConvBlock(alpha, 2*alpha, h=h, w=w, stride=2, learn_omega=learn_omega),
                   SConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1, learn_omega=learn_omega),
                   SConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2, learn_omega=learn_omega),
                   SConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1, learn_omega=learn_omega),
                   SConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2, learn_omega=learn_omega),
                   SConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2, learn_omega=learn_omega),
                   SConvBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), stride=2, learn_omega=learn_omega)]

        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out


class S_FC(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, learn_omega=False):
        super(S_FC, self).__init__()
        modules = [SLinearBlock(in_ch, alpha, h=h, w=w, stride=1, learn_omega=learn_omega),
                   SLinearBlock(alpha, 2*alpha, h=h, w=w, stride=2, learn_omega=learn_omega),
                   SLinearBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1, learn_omega=learn_omega),
                   SLinearBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2, learn_omega=learn_omega),
                   SLinearBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1, learn_omega=learn_omega),
                   SLinearBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2, learn_omega=learn_omega),
                   SLinearBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2, learn_omega=learn_omega),
                   SLinearBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), stride=2, learn_omega=learn_omega)]
        self.conv_net = nn.Sequential(*modules)

        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out
    

class S_Both(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32, learn_omega=False):
        super(S_Both, self).__init__()
        modules = [SLinearConvBlock(in_ch, alpha, h=h, w=w, stride=1, learn_omega=learn_omega),
                   SLinearConvBlock(alpha, 2*alpha, h=h, w=w, stride=2, learn_omega=learn_omega),
                   SLinearConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1, learn_omega=learn_omega),
                   SLinearConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2, learn_omega=learn_omega),
                   SLinearConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1, learn_omega=learn_omega),
                   SLinearConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2, learn_omega=learn_omega),
                   SLinearConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=2, learn_omega=learn_omega),
                   SLinearConvBlock(8*alpha, 16*alpha,h=int(h/16), w=int(w/16), stride=2, learn_omega=learn_omega),]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.final1 = nn.Linear(16*alpha, 64*alpha)
        self.final1.bias.data.zero_()
        self.final2 = nn.Linear(64*alpha, num_classes)
        self.final2.bias.data.zero_()

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(*out.shape[:2])
        out = self.final1(out)
        out = self.activation(out)
        out = self.final2(out)
        return out



