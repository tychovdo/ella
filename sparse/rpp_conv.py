import torch.nn as nn
import torch.nn.functional as F
from asdfghjkl.operations import Bias, Scale


class LinearConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w):
        super(LinearConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.linear = nn.Linear(in_ch*h*w, out_ch*h*w)
        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)
        linout = self.linear(x.view(x.shape[0], -1))
        linout = linout.view(convout.shape)
        
        return self.activation(linout + convout)
    
class RPPConv(nn.Module):
    def __init__(self, num_layers=3, ch=32, in_ch=3, num_classes=10, h=3, w=2):
        super(RPPConv, self).__init__()
        layers = [LinearConvBlock(in_ch, ch, h, w)]
        for _ in range(num_layers-1):
            layers.append( LinearConvBlock(ch, ch, h, w) )
        
        self.conv_net = nn.Sequential(*layers)
        self.classifier = nn.Linear(ch*h*w, num_classes)
        
    
    def forward(self, x):
        conv_out = self.conv_net(x)
        return self.classifier(conv_out.view(x.shape[0], -1))

    
    
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,3,padding=1)
        self.activation = F.relu
        
    def forward(self, x):
        convout = self.conv(x)
        return self.activation(convout)
        
class ConvNet(nn.Module):
    def __init__(self, num_layers=3, ch=32, outdim=1, h=3, w=2):
        super(ConvNet, self).__init__()
        layers = [ConvBlock(1, ch)]
        for _ in range(num_layers-1):
            layers.append( ConvBlock(ch, ch) )
        
        self.conv_net = nn.Sequential(*layers)
        self.classifier = nn.Linear(ch*h*w, outdim)
        
    def forward(self, x):
        conv_out = self.conv_net(x)
        return self.classifier(conv_out.view(x.shape[0], -1))
    
    
class LinearBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_ch*h*w, out_ch*h*w)
        self.h = h
        self.w = w
        self.activation = F.relu
        
    def forward(self, x):
        linout = self.linear(x.view(x.shape[0], -1))
        return self.activation(linout)
        
class LinearNet(nn.Module):
    def __init__(self, num_layers=3, ch=32, outdim=1, h=3, w=2):
        super(LinearNet, self).__init__()
        layers = [LinearBlock(1, ch, h, w)]
        for _ in range(num_layers-1):
            layers.append( LinearBlock(ch, ch, h, w) )
        
        self.conv_net = nn.Sequential(*layers)
        self.classifier = nn.Linear(ch*h*w, outdim)
        
    def forward(self, x):
        conv_out = self.conv_net(x)
        return self.classifier(conv_out.view(x.shape[0], -1))

class NNLinearBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(NNLinearBlock, self).__init__()
        self.linear = nn.Linear(int(in_ch*h*w), int(out_ch*h*w/(stride**2)))
        self.stride=stride
        self.out_ch = out_ch
        self.h = h
        self.w = w
        self.activation = F.relu
        self.linscale = Scale()
        # self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        # linout = self.linear(x.view(x.shape[0], -1))
        linout = self.linear(x.reshape(x.shape[0], -1))
        linout = linout.view(x.shape[0], self.out_ch, 
                             int(self.h/self.stride), int(self.w/self.stride))
        linout = self.linscale(linout)
        # out = self.bn(linout)
        out = linout
        return self.activation(out)
    
class NNLinearConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(NNLinearConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.linear = nn.Linear(int(in_ch*h*w), int(out_ch*h*w/(stride**2)))
        self.h = h
        self.w = w
        self.activation = F.relu

        self.linscale = Scale()
        self.convscale = Scale()
        # self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        convout = self.conv(x)
        convout = self.convscale(convout)

        #linout = self.linear(x.view(x.shape[0], -1))
        linout = self.linear(x.reshape(x.shape[0], -1))
        linout = linout.view(convout.shape)
        linout = self.linscale(linout)
        
        # out = self.bn(linout + convout)
        out = linout + convout
        return self.activation(out)
    
class NNConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(NNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.activation = F.relu
        self.convscale = Scale()
        # self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        # convout = self.bn(self.conv(x))
        convout  = self.conv(x)
        convout  = self.convscale(convout)
        return self.activation(convout)

class BNLinearBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(BNLinearBlock, self).__init__()
        self.linear = nn.Linear(int(in_ch*h*w), int(out_ch*h*w/(stride**2)))
        self.stride=stride
        self.out_ch = out_ch
        self.h = h
        self.w = w
        self.activation = F.relu
        # self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        # linout = self.linear(x.view(x.shape[0], -1))
        linout = self.linear(x.reshape(x.shape[0], -1))
        linout = linout.view(x.shape[0], self.out_ch, 
                             int(self.h/self.stride), int(self.w/self.stride))
        # out = self.bn(linout)
        out = linout
        return self.activation(out)
    
class BNLinearConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(BNLinearConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.linear = nn.Linear(int(in_ch*h*w), int(out_ch*h*w/(stride**2)))
        self.h = h
        self.w = w
        self.activation = F.relu
        # self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        convout = self.conv(x)
        #linout = self.linear(x.view(x.shape[0], -1))
        linout = self.linear(x.reshape(x.shape[0], -1))
        linout = linout.view(convout.shape)
        
        # out = self.bn(linout + convout)
        out = linout + convout
        return self.activation(out)
    
class BNConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, stride=1):
        super(BNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.activation = F.relu
        # self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        # convout = self.bn(self.conv(x))
        convout  = self.conv(x)
        return self.activation(convout)

class RPP_D_Conv(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(RPP_D_Conv, self).__init__()
        modules = [BNLinearConvBlock(in_ch, alpha, h=h, w=w, stride=1),
                   BNLinearConvBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   BNLinearConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   BNLinearConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   BNLinearConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   BNLinearConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   BNLinearConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=1),
                   BNLinearConvBlock(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.linear1 = nn.Linear(16*alpha * int(h/16) * int(w/16), 64*alpha)
        self.linear2 = nn.Linear(64*alpha, num_classes)
        
    def forward(self, x):
        out = self.conv_net(x)
        out = self.activation(self.linear1(out.view(x.shape[0], -1)))
        return self.linear2(out)
    

class D_Conv(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(D_Conv, self).__init__()
        modules = [BNConvBlock(in_ch, alpha, h=h, w=w, stride=1),
                   BNConvBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   BNConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   BNConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   BNConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   BNConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   BNConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=1),
                   BNConvBlock(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.linear1 = nn.Linear(16*alpha * int(h/16) * int(w/16), 64*alpha)
        self.linear2 = nn.Linear(64*alpha, num_classes)
        
    def forward(self, x):
        out = self.conv_net(x)
        out = self.activation(self.linear1(out.view(x.shape[0], -1)))
        return self.linear2(out)
    
    
class D_FC(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(D_FC, self).__init__()
        modules = [BNLinearBlock(in_ch, alpha, h=h, w=w, stride=1),
                  BNLinearBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                  BNLinearBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                  BNLinearBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                  BNLinearBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                  BNLinearBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                  BNLinearBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=1),
                  BNLinearBlock(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.linear1 = nn.Linear(16*alpha * int(h/16) * int(w/16), 64*alpha)
        self.linear2 = nn.Linear(64*alpha, num_classes)
        
    def forward(self, x):
        out = self.conv_net(x)
        out = self.activation(self.linear1(out.view(x.shape[0], -1)))
        return self.linear2(out)

class RPP_D_Conv_B(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(RPP_D_Conv_B, self).__init__()
        modules = [NNLinearConvBlock(in_ch, alpha, h=h, w=w, stride=1),
                   NNLinearConvBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   NNLinearConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   NNLinearConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   NNLinearConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   NNLinearConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   NNLinearConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=1),
                   NNLinearConvBlock(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.linear1 = nn.Linear(16*alpha * int(h/16) * int(w/16), 64*alpha)
        self.linear2 = nn.Linear(64*alpha, num_classes)
        
    def forward(self, x):
        out = self.conv_net(x)
        out = self.activation(self.linear1(out.view(x.shape[0], -1)))
        return self.linear2(out)
    

class D_Conv_B(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(D_Conv_B, self).__init__()
        modules = [NNConvBlock(in_ch, alpha, h=h, w=w, stride=1),
                   NNConvBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   NNConvBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   NNConvBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   NNConvBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   NNConvBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   NNConvBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=1),
                   NNConvBlock(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.linear1 = nn.Linear(16*alpha * int(h/16) * int(w/16), 64*alpha)
        self.linear2 = nn.Linear(64*alpha, num_classes)
        
    def forward(self, x):
        out = self.conv_net(x)
        out = self.activation(self.linear1(out.view(x.shape[0], -1)))
        return self.linear2(out)
    
    
class D_FC_B(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, alpha=1, h=32, w=32):
        super(D_FC_B, self).__init__()
        modules = [NNLinearBlock(in_ch, alpha, h=h, w=w, stride=1),
                   NNLinearBlock(alpha, 2*alpha, h=h, w=w, stride=2),
                   NNLinearBlock(2*alpha, 2*alpha, h=int(h/2), w=int(w/2), stride=1),
                   NNLinearBlock(2*alpha, 4*alpha, h=int(h/2), w=int(w/2), stride=2),
                   NNLinearBlock(4*alpha, 4*alpha, h=int(h/4), w=int(w/4), stride=1),
                   NNLinearBlock(4*alpha, 8*alpha, h=int(h/4), w=int(w/4), stride=2),
                   NNLinearBlock(8*alpha, 8*alpha, h=int(h/8), w=int(w/8), stride=1),
                   NNLinearBlock(8*alpha, 16*alpha,h=int(h/8), w=int(w/8), stride=2)]
        self.conv_net = nn.Sequential(*modules)
        
        self.activation = F.relu
        self.linear1 = nn.Linear(16*alpha * int(h/16) * int(w/16), 64*alpha)
        self.linear2 = nn.Linear(64*alpha, num_classes)
        
    def forward(self, x):
        out = self.conv_net(x)
        out = self.activation(self.linear1(out.view(x.shape[0], -1)))
        return self.linear2(out)
    
    

def RPPConv_L2(mdl, conv_wd, basic_wd):
    conv_l2 = 0.
    basic_l2 = 0.
    for block in mdl.conv_net:
        if hasattr(block, 'conv'):
            conv_l2 += sum([p.pow(2).sum() for p in block.conv.parameters()])
        if hasattr(block, 'linear'):
            basic_l2 += sum([p.pow(2).sum() for p in block.linear.parameters()])
        
    return conv_wd*conv_l2  + basic_wd*basic_l2

def RPPConv_L1(mdl, conv_wd, basic_wd):
    conv_l1 = 0.
    basic_l1 = 0.
    for block in mdl.conv_net:
        if hasattr(block, 'conv'):
            conv_l1 += sum([p.abs().sum() for p in block.conv.parameters()])
        if hasattr(block, 'linear'):
            basic_l1 += sum([p.abs().sum() for p in block.linear.parameters()])
        
    return conv_wd*conv_l1  + basic_wd*basic_l1


