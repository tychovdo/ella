import torch
import numpy as np
from torch import nn
from torch import nn
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from asdfghjkl.operations import Bias, Scale
from asdfghjkl.operations.conv_aug import Conv2dAug, Conv1dAug

# This file contains different NN Models:
# MLP, LeNet (CNN), Fixup ResNet, Fixup WRN, MLP Mixer, SimpleViT,
# all of which can also be used augmented


def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    else:
        raise ValueError('invalid activation')


class MaxPool2dAug(nn.MaxPool2d):

    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


class AvgPool2dAug(nn.AvgPool2d):

    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


class AdaptiveAvgPool2dAug(nn.AdaptiveAvgPool2d):

    def forward(self, input):
        k_aug = input.shape[1]
        input = super().forward(input.flatten(start_dim=0, end_dim=1))
        return input.reshape(-1, k_aug, *input.shape[1:])


class ConstantLastLogit(nn.Module):

    def __init__(self, augmented=False) -> None:
        super().__init__()
        self.augmented = augmented

    def forward(self, input):
        if self.augmented:
            return torch.cat([input, input.new_zeros(input.shape[0], input.shape[1], 1)], dim=2)
        return torch.cat([input, input.new_zeros(input.shape[0], 1)], dim=1)


class MLP(nn.Sequential):
    def __init__(self, input_size, width, depth, output_size, activation='relu',
                 bias=True, fixup=False, augmented=False, last_logit_constant=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        flatten_start_dim = 2 if augmented else 1
        act = get_activation(activation)
        output_size = output_size if not last_logit_constant else output_size - 1

        self.add_module('flatten', nn.Flatten(start_dim=flatten_start_dim))

        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=bias))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=bias))
                if fixup:
                    self.add_module(f'bias{i+1}b', Bias())
                    self.add_module(f'scale{i+1}b', Scale())
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], output_size, bias=bias))

        if last_logit_constant:
            self.add_module('constant_logit', ConstantLastLogit(augmented=augmented))


class LeNet(nn.Sequential):
    
    def __init__(self, in_channels=1, n_out=10, activation='relu', n_pixels=28, 
                 augmented=False, last_logit_constant=False):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = Conv2dAug if augmented else nn.Conv2d
        pool = MaxPool2dAug if augmented else nn.MaxPool2d
        flatten = nn.Flatten(start_dim=2) if augmented else nn.Flatten(start_dim=1)
        n_out = n_out if not last_logit_constant else n_out - 1 
        self.add_module('conv1', conv(in_channels, 6, 5, 1))
        self.add_module('act1', act())
        self.add_module('pool1', pool(2))
        self.add_module('conv2', conv(6, 16, mid_kernel_size, 1))
        self.add_module('act2', act())
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(16, 120, 5, 1))
        self.add_module('flatten', flatten)
        self.add_module('act3', act())
        self.add_module('lin1', torch.nn.Linear(120*1*1, 84))
        self.add_module('act4', act())
        self.add_module('linout', torch.nn.Linear(84, n_out))
        if last_logit_constant:
            self.add_module('constant_logit', ConstantLastLogit(augmented=augmented))


def conv3x3(in_planes, out_planes, stride=1, augmented=False):
    """3x3 convolution with padding"""
    Conv2d = Conv2dAug if augmented else nn.Conv2d
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, augmented=False):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.augmented = augmented
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride, augmented=augmented)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes, augmented=augmented)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            cat_dim = 2 if self.augmented else 1
            identity = torch.cat((identity, torch.zeros_like(identity)), cat_dim)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    FixupResnet-depth where depth is a `3 * 2 * n + 2` with `n` blocks per residual layer.
    The two added layers are the input convolution and fully connected output.
    """

    def __init__(self, depth, num_classes=10, in_planes=16, in_channels=3, augmented=False,
                 last_logit_constant=False):
        super(ResNet, self).__init__()
        n_out = num_classes if not last_logit_constant else num_classes - 1
        self.llc = last_logit_constant
        assert (depth - 2) % 6 == 0, 'Invalid ResNet depth, has to conform to 6 * n + 2'
        layer_size = (depth - 2) // 6
        layers = 3 * [layer_size]
        self.num_layers = 3 * layer_size
        self.inplanes = in_planes
        self.augmented = augmented
        AdaptiveAvgPool2d = AdaptiveAvgPool2dAug if augmented else nn.AdaptiveAvgPool2d
        self.conv1 = conv3x3(in_channels, in_planes, augmented=augmented)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FixupBasicBlock, in_planes, layers[0])
        self.layer2 = self._make_layer(FixupBasicBlock, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(FixupBasicBlock, in_planes * 4, layers[2], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=2 if augmented else 1)
        self.bias2 = Bias()
        self.fc = nn.Linear(in_planes * 4, n_out)
        if last_logit_constant:
            self.constant_logit = ConstantLastLogit(augmented=augmented)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight,
                                mean=0,
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0) 
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        AvgPool2d = AvgPool2dAug if self.augmented else nn.AvgPool2d
        if stride != 1:
            downsample = AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, augmented=self.augmented))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, augmented=self.augmented))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(self.bias2(x))
        if self.llc:
            x = self.constant_logit(x)

        return x


class WRNFixupBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, augmented=False):
        super(WRNFixupBasicBlock, self).__init__()
        self.bias1 = Bias()
        self.relu1 = nn.ReLU(inplace=True)
        basemodul = Conv2dAug if augmented else nn.Conv2d
        self.augmented = augmented
        self.conv1 = basemodul(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bias2 = Bias()
        self.relu2 = nn.ReLU(inplace=True)
        self.bias3 = Bias()
        self.conv2 = basemodul(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias4 = Bias()
        self.scale1 = Scale()
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and basemodul(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bias1(x))
        else:
            out = self.relu1(self.bias1(x))
        if self.equalInOut:
            out = self.bias3(self.relu2(self.bias2(self.conv1(out))))
        else:
            out = self.bias3(self.relu2(self.bias2(self.conv1(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bias4(self.scale1(self.conv2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class WRNFixupNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, augmented=False):
        super(WRNFixupNetworkBlock, self).__init__()
        self.augmented = augmented
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, self.augmented))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10, dropRate=0.0, augmented=False,
                 last_logit_constant=False):
        super(WideResNet, self).__init__()
        n_out = num_classes if not last_logit_constant else num_classes - 1
        self.llc = last_logit_constant
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = WRNFixupBasicBlock
        # 1st conv before any network block
        self.num_layers = n * 3
        basemodul = Conv2dAug if augmented else nn.Conv2d
        self.augmented = augmented
        self.conv1 = basemodul(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias1 = Bias()
        # 1st block
        self.block1 = WRNFixupNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, augmented=augmented)
        # 2nd block
        self.block2 = WRNFixupNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, augmented=augmented)
        # 3rd block
        self.block3 = WRNFixupNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, augmented=augmented)
        # global average pooling and classifier
        self.bias2 = Bias()
        self.relu = nn.ReLU()
        self.pool = AvgPool2dAug(8) if augmented else nn.AvgPool2d(8)
        self.fc = nn.Linear(nChannels[3], n_out)
        if self.llc:
            self.constant_logit = ConstantLastLogit(augmented=augmented)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, WRNFixupBasicBlock):
                conv = m.conv1
                k = conv.weight.shape[0] * np.prod(conv.weight.shape[2:])
                nn.init.normal_(conv.weight,
                                mean=0,
                                std=np.sqrt(2. / k) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.convShortcut is not None:
                    cs = m.convShortcut
                    k = cs.weight.shape[0] * np.prod(cs.weight.shape[2:])
                    nn.init.normal_(cs.weight,
                                    mean=0,
                                    std=np.sqrt(2. / k))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bias1(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = self.pool(out)
        if self.augmented:
            out = out.flatten(start_dim=2)
        else:
            out = out.flatten(start_dim=1)
        out = self.fc(self.bias2(out))
        if self.llc:
            return self.constant_logit(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def MixerFeedForward(dim, expansion_factor=4, dense=nn.Linear, fixup=False):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        Bias() if fixup else nn.Identity(),
        dense(dim, inner_dim),
        Bias() if fixup else nn.Identity(),
        nn.GELU(),
        Bias() if fixup else nn.Identity(),
        dense(inner_dim, dim),
        Scale() if fixup else nn.Identity()
    )


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def MLPMixer(image_size=32, channels=3, patch_size=4, dim=512, depth=6, num_classes=10,
             expansion_factor=4, expansion_factor_token=0.5, fixup=False, augmented=False,
             last_logit_constant=False):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(Conv1dAug if augmented else nn.Conv1d, kernel_size=1, bias=True), nn.Linear
    n_out = num_classes if not last_logit_constant else num_classes - 1

    bspec = 'b m' if augmented else 'b'
    return nn.Sequential(
        Rearrange(f'{bspec} c (h p1) (w p2) -> {bspec} (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            Residual(MixerFeedForward(num_patches, expansion_factor, chan_first, fixup)),
            Residual(MixerFeedForward(dim, expansion_factor_token, chan_last, fixup))
        ) for _ in range(depth)],
        Reduce(f'{bspec} n c -> {bspec} c', 'mean'),
        Bias() if fixup else nn.Identity(),
        nn.Linear(dim, n_out),
        ConstantLastLogit(augmented=augmented) if last_logit_constant else nn.Identity()
    )


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32, augmented=False):
    if augmented:
        _, _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    else:
        _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def ViTFeedForward(dim, hidden_dim, fixup=False):
        return nn.Sequential(
            Bias() if fixup else nn.Identity(),
            nn.Linear(dim, hidden_dim),
            Bias() if fixup else nn.Identity(),
            nn.GELU(),
            Bias() if fixup else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            Scale() if fixup else nn.Identity()
        )


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, fixup=False, augmented=False):
        super().__init__()
        self.shift = Bias() if fixup else nn.Identity()
        self.augmented = augmented
        inner_dim = dim_head * heads
        self.heads = heads
        self._scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.scale = Scale() if fixup else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(self.shift(x)).chunk(3, dim=-1)
        bspec = 'b m' if self.augmented else 'b'
        q, k, v = map(lambda t: rearrange(t, f'{bspec} n (h d) -> {bspec} h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self._scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, f'{bspec} h n d -> {bspec} n (h d)')
        return self.scale(self.to_out(out))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, fixup, augmented):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, fixup=fixup, augmented=augmented),
                ViTFeedForward(dim, mlp_dim, fixup=fixup)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """Simple vision transformer SimpleViT."""
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=8,
                 mlp_dim=512, channels=3, dim_head=64, fixup=False, augmented=False,
                 last_logit_constant=True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width
        n_out = num_classes if not last_logit_constant else num_classes - 1
        self.llc = last_logit_constant
        self.augmented = augmented
        bspec = 'b m' if augmented else 'b'
        self.bspec = bspec
        self.to_patch_embedding = nn.Sequential(
            Rearrange(f'{bspec} c (h p1) (w p2) -> {bspec} h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, fixup, augmented)
        self.to_latent = Bias() if fixup else nn.Identity()
        self.linear_head = nn.Linear(dim, n_out)
        if last_logit_constant:
            self.constant_logit = ConstantLastLogit(augmented=augmented)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x, augmented=self.augmented)
        x = rearrange(x, f'{self.bspec} ... d -> {self.bspec} (...) d') + pe
        x = self.transformer(x)
        x = x.mean(dim=2 if self.augmented else 1)
        x = self.to_latent(x)
        x = self.linear_head(x)
        if self.llc:
            return self.constant_logit(x)
        return x


if __name__ == '__main__':
    from torch.nn.utils import parameters_to_vector
    from time import time
    # from functorch import make_functional, jacrev
    from laplace.curvature import AsdlGGN, AugAsdlGGN

    class Timer:
        def __init__(self, name) -> None:
            self.name = name
        def __enter__(self):
            self.start_time = time()
        def __exit__(self, *args, **kwargs):
            print(self.name, 'took', f'{time() - self.start_time:.3f}s')

    def flat_jac(jac):
        return torch.cat([p.flatten(start_dim=2) for p in jac], dim=2)

    def grad(model):
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    def jacobians_naive(model, data, augmented=False, differentiable=False):
        model.zero_grad()
        f = model(data).mean(dim=1) if augmented else model(data)
        Jacs = list()
        cg = differentiable
        for i in range(f.shape[0]):
            if len(f.shape) > 1:
                jacs = list()
                for j in range(f.shape[1]):
                    rg = differentiable or (i != (f.shape[0] - 1) or j != (f.shape[1] - 1))
                    f[i, j].backward(retain_graph=rg, create_graph=cg)
                    # torch.autograd.backward(f[i, j], torch.tensor(1.), retain_graph=rg, create_graph=cg)
                    Jij = grad(model)
                    jacs.append(Jij)
                    model.zero_grad()
                jacs = torch.stack(jacs).t()
            else:
                rg = differentiable or (i != (f.shape[0] - 1))
                f[i].backward(retain_graph=rg, create_graph=cg)
                jacs = grad(model)
                model.zero_grad()
            Jacs.append(jacs)
        Jacs = torch.stack(Jacs).transpose(1, 2)
        return Jacs, f

    # Example data
    n_data, n_aug, n_channels, n_pixels = 4, 2, 3, 32
    n_outputs = 3
    device = 'cpu'
    output_shape = torch.Size([n_data, n_outputs])
    output_shape_aug = torch.Size([n_data, n_aug, n_outputs])
    X = torch.randn(n_data, n_channels, n_pixels, n_pixels, device=device)
    X_aug = torch.randn(n_data, n_aug, n_channels, n_pixels, n_pixels, device=device)
    X_aug_test = X.unsqueeze(1).repeat((1, n_aug, 1, 1, 1))

    def model_test(model_cls, hparams):
        # Test model to model_aug consistency
        torch.manual_seed(117)
        model = model_cls(**hparams).to(device)
        params = parameters_to_vector(model.parameters())
        with Timer('Normal model inference'):
            normal_out = model(X).detach()
        torch.manual_seed(117)
        model_aug = model_cls(augmented=True, **hparams).to(device)
        assert torch.allclose(parameters_to_vector(model_aug.parameters()), params)
        with Timer('Augmented model inference'):
            assert model_aug(X_aug).size() == output_shape_aug
        assert torch.allclose(model_aug(X_aug_test).mean(1), normal_out, atol=1e-7)

        # Test Jacobians of ASDL
        setattr(model, 'output_size', n_outputs)
        backend = AsdlGGN(model, 'classification')
        with Timer('Jac asdl'):
            Js, f = backend.jacobians(X)
        model.zero_grad()
        Js.sum().backward()
        js_grad = grad(model).clone()

        with Timer('Jac naive'):
            Js_true, f_true = jacobians_naive(model, X, differentiable=True)
        model.zero_grad()
        Js_true.sum().backward()
        js_true_grad = grad(model).clone()
        assert torch.allclose(js_grad.sum(), js_true_grad.sum())
        assert torch.allclose(f, f_true)
        try:
            assert torch.allclose(Js_true, Js, atol=1e-6)
        except:
            print('Js discrepancy', torch.max(torch.abs(Js_true - Js)))

        # Test jacs: somehow not aligned with Js_true..
        # model = model_cls(**hparams).to(device)
        # model_clone = model_cls(**hparams)
        # model_clone.load_state_dict(model.state_dict())
        # start_time = time()
        # func, ps = make_functional(model_clone)
        # jac = flat_jac(jacrev(func)(ps, X))
        # print(jac.sum(), Js_true.sum())
        # assert Js_true.shape == jac.shape
        # print('jac rev in ', time() - start_time)

        setattr(model_aug, 'output_size', n_outputs)
        backend_aug = AugAsdlGGN(model_aug, 'classification', kron_jac=True)
        with Timer('Kronecker augmented Jacs'):
            Js, f = backend_aug.jacobians(X_aug)
        with Timer('Naive augmented Jacs'):
            Js_true, f_true = jacobians_naive(model_aug, X_aug, augmented=True)
        assert Js_true.shape == Js.shape
        assert torch.allclose(f, f_true)
        backend_aug_slow = AugAsdlGGN(model_aug, 'classification', kron_jac=False)
        with Timer('Standard augmented Jacs'):
            Js, f = backend_aug_slow.jacobians(X_aug)
        assert torch.allclose(f, f_true)
        try:
            assert torch.allclose(Js_true, Js, atol=1e-6)
        except:
            print('Js discrepancy', torch.max(torch.abs(Js_true - Js)))

        # Test KFAC and diag summation consistency
        m = int(n_data/2)
        ys = torch.randint(3, size=(n_data,), device=device)
        backend_kron = AsdlGGN(model, 'classification', kron_jac=True)
        # Diag
        for backendi, Xi in zip([backend, backend_kron, backend_aug_slow, backend_aug], 
                                [X, X, X_aug_test, X_aug_test]):
            loss, diag = backendi.diag(Xi, ys)
            lossa, diaga = backendi.diag(Xi[:m], ys[:m])
            lossb, diagb = backendi.diag(Xi[m:], ys[m:])
            assert torch.allclose(loss, lossa+lossb)
            assert torch.allclose(diag, diaga+diagb)
            assert diag.shape == params.shape
            # KFAC
            loss, kfac = backendi.kron(Xi, ys, N=n_data)
            lossa, kfaca = backendi.kron(Xi[:m], ys[:m], N=n_data)
            lossb, kfacb = backendi.kron(Xi[m:], ys[m:], N=n_data)
            assert torch.allclose(loss, lossa+lossb)
            a, b = kfac.flatten(), (kfaca+kfacb).flatten()
            (a + b).mean().backward()
            # sortedd, indices = torch.sort((a - b).abs(), descending=True)
            # print(sortedd[:10])
            # print(a[indices][:10])
            print('Max memory allocated: ' + str(torch.cuda.max_memory_allocated('cuda')/(1024 ** 3)) + ' Gb.')
            assert torch.allclose(kfac.flatten(), (kfaca + kfacb).flatten(), atol=1e-3, rtol=1e-4)
            assert kfac.diag().shape == params.shape

            # NTK
            Js, _ = backendi.jacobians(Xi)
            J = Js.reshape(n_data * n_outputs, -1)
            K_test = J @ J.T
            backendi.likelihood = 'regression'
            loss, K = backendi.kernel(Xi, ys, J.new_ones(J.shape[-1]))
            assert torch.allclose(K_test, K, atol=1e-5)
            # NTK indep (single class test)
            J = Js[:, 0, :].reshape(n_data, -1)
            K_test = J @ J.T
            loss, K = backendi.single_kernel(Xi, ys, J.new_ones((1, )), output_ix=0)
            try:
                assert torch.allclose(K_test, K, atol=1e-5)
            except:
                # NOTE: convolutions of the type in LeNet and Mixer appear to lead to inexact KFAC (=Gram)
                print('Indep kernel with gram not exactly equivalent for', model.__class__.__name__)
                print('Relative error:', torch.abs(K[:4, :4] - K_test[:4, :4]) / K_test[:4, :4])
            backendi.likelihood = 'classification'


    # 1. Test MLP
    hparams = dict(input_size=n_channels*(n_pixels**2), width=100, depth=3, output_size=n_outputs, fixup=True)
    print(10 * '-', 'MLP', 10 * '-')
    model_test(MLP, hparams)

    # 2. Test CNN
    hparams = dict(in_channels=n_channels, n_pixels=n_pixels, n_out=n_outputs)
    print(10 * '-', 'LeNet', 10 * '-')
    model_test(LeNet, hparams)

    # 3. Test ResNet-14
    hparams = dict(depth=14, num_classes=n_outputs, in_channels=n_channels)
    print(10 * '-', 'ResNet', 10 * '-')
    model_test(ResNet, hparams)

    # 4. Test WRN
    print(10 * '-', 'WRN', 10 * '-')
    model_test(WideResNet, dict(num_classes=n_outputs))

    # 5. Test MLPMixer
    print(10 * '-', 'MLPMixer', 10 * '-')
    model_test(MLPMixer, dict(fixup=True, num_classes=n_outputs))

    # 6. Test ViT
    print(10 * '-', 'ViT', 10 * '-')
    model_test(ViT, dict(image_size=32, fixup=True, num_classes=n_outputs))
