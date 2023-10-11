import numpy as np

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair

from sparse.regression import Regression
from sparse.kernels import RBF

class SLinearU(nn.Linear):
    # (in_channels*in_size)x(out_channels) linear layer
    # sparsified spatially within the 'in_size' subdimension

    def __init__(self, in_channels, in_size, out_channels, bias=True, device=None, dtype=None,
                 free_anchor_loc=False, n_anchor=None, order=0,
                 scale=1.0, omega=1.0, noise_std=0.01, kernel_type='rbf',
                 learn_omega=False, learn_scale=False, learn_noise=False,
                 solver='solve'):
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_channels * in_size
        self.out_features = out_channels

        super().__init__(self.in_features, self.out_features)

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.in_shape = (int(math.sqrt(in_size)), int(math.sqrt(in_size)))
        self.free_anchor_loc = free_anchor_loc

        # set kernel
        if kernel_type == 'rbf':
            kernel = RBF(scale=scale, omega=omega, learn_scale=learn_scale, learn_omega=learn_omega, device=device)
        else:
            raise NotImplementedError(f"Unknown kernel: {kernel_type}")
        
        # sample grid
        s = ((self.in_shape[0] - 1) / 2, (self.in_shape[1] - 1) / 2)
        
        ys = torch.linspace(-s[0], s[0], steps=self.in_shape[0], device=device)
        xs = torch.linspace(-s[1], s[1], steps=self.in_shape[1], device=device)

        grid = torch.stack(torch.meshgrid(ys, xs, indexing='xy'), 2) 
        grid = grid.view(-1, 2) # (HW, 2)
        self.register_buffer('grid', grid)
        
        # init anchor point locations at grid
        assert n_anchor in (0, None, self.in_shape[0] * self.in_shape[1]), "With anchor init 'grid', NZ should be 0/None or correct (in_shape^2)"
        Z = grid.view(-1, 2).clone()

        if free_anchor_loc:
            self.Z = nn.Parameter(Z)
        else:
            self.register_buffer('Z', Z)

        # anchor point values
        weight = torch.randn(out_channels, in_channels*len(Z), device=device)
        self.weight = nn.Parameter(weight)

        self.regression = Regression(2, kernel, noise_std=noise_std, solver=solver, learn_noise=learn_noise)
        self.regression.update(Z=self.Z)

        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def _u(self): # weight2u
        return self.weight.view(self.out_channels, self.in_channels, -1)

    def _set_weight(self, u): # u2weight
        self.weight.data = u.reshape(self.out_channels, -1).contiguous()

    def _set_u_from_weight(self, weight, bias=None, ignore_noise=False): # TODO: size check
        """ Set anchor point values from conv2d weights (and bias). """

        u = self.weight.view(self.out_channels, self.in_channels, -1)

        assert len(u.shape) == 4, f"Weight should be of dimension 4. Got {u.shape} (dim={len(u.shape)})."
        assert u.shape[0] == self.out_channels
        assert u.shape[1] == self.in_channels
        assert u.shape[2] == self.in_shape[0]
        assert u.shape[3] == self.in_shape[1]

        if ignore_noise:
            u_new = u.view(self.out_channels, self.in_channels, self.in_shape[0]*self.in_shape[1])
            self._set_weight(u_new)
        else:
            old_f = u.view(self.out_channels, self.in_channels, self.in_shape[0]*self.in_shape[1])

            # find new_u using closed-form least-squares solution
            Kzz = self.regression._kernel(self.Z, self.Z)
            
            diag_noise = torch.eye(len(Kzz), device=Kzz.device) * (self.regression.noise_std ** 2)
            A = torch.linalg.solve((Kzz + diag_noise).T, Kzz.T).T

            # inverse formula
            convert_matrix = torch.pinverse(A)
            u_new = old_f @ convert_matrix.T # u=(klj), convert_matrix=(ij) -> u_new=(kli)

            self._set_weight(u_new)

        if bias is not None:
            self.bias.data = bias
        
    def reduce(self, factor, dist='uniform', new_std=None,
               method='lstsq', deterministic=False, always_zero=False):
        """ Inference q_grid(. | Z, u),
                where T has shape (N, 2) """
        
        u = self._u()

        device = u.device

        # create new anchor points
        n_anchor = max(round(factor * len(self.Z)), 1)
        #n_anchor = int(min(len(self.Z), max(1, round(factor * math.sqrt(len(self.Z))))))

        # new noise
        std_new = self.regression.noise_std if new_std is None else new_std

        if deterministic:
            torch.manual_seed(1)

        s = ((self.in_shape[0] - 1) / 2, (self.in_shape[1] - 1) / 2)
        if dist == 'normal':
            Z_new = torch.randn(n_anchor, 2, device=device) * torch.tensor(s, device=device).view(1, -1) 
            if always_zero:
                Z_new[0].zero_()
        elif dist == 'uniform':
            Z_new = (torch.rand(n_anchor, 2, device=device) * 2 - 1) * torch.tensor(s, device=device).view(1, -1) 
            if always_zero:
                Z_new[0].zero_()
        else:
            raise NotImplementedError(f'Unknown reduce distribution: {dist}')

        u = self._u()

        old_f = self.weight_fn().view(*u.shape).detach()
        self.old_f = old_f

        if method == 'expectation':
            """ Set new points by simply directly predicting expected value at these points.
                This is a naive approach that does not find the minimising solution (not recommended). """
            # posterior mean inference for new anchor point values
            u_new = self.regression(self.Z, old_f, Z_new)
        elif method == 'formula':
            # solve minimum norm problem for new anchor point values in closed-form
            Kzq = self.regression._kernel(self.Z, Z_new)
            Kqq = self.regression._kernel(Z_new, Z_new)

            diag_noise = torch.eye(len(Kqq), device=Kqq.device) * (std_new ** 2)
            A = Kzq @ torch.inverse(Kqq + diag_noise)

            convert_matrix = torch.pinverse(A)

            u_new = old_f @ convert_matrix.T # u=(klj), convert_matrix=(ij) -> u_new=(kli)
        elif method == 'lstsq':
            # solve minimum norm problem for new anchor point values in closed-form
            Kzq = self.regression._kernel(self.Z, Z_new)
            Kqq = self.regression._kernel(Z_new, Z_new)

            diag_noise = torch.eye(len(Kqq), device=Kqq.device) * (std_new ** 2)

            A = torch.linalg.solve((Kqq + diag_noise).T, Kzq.T).T

            C = A.unsqueeze(0) # (1, i, j)
            u = old_f.view(-1, old_f.shape[-1]).unsqueeze(2) # (kl, i, 1)

            u_new = torch.linalg.lstsq(C, u).solution # (kl, j, 1)
            u_new = u_new.view(*old_f.shape[:2], -1)

            self._set_weight(u_new)
        else:
            raise NotImplementedError(f'Unknown method for anchor point reduction: {method}')

        with torch.no_grad():
            # replace new anchor points
            if self.free_anchor_loc:
                self.Z = nn.Parameter(Z_new)
            else:
                self.register_buffer('Z', Z_new)

            self._set_weight(u_new)
            self.weight = nn.Parameter(self.weight)
                
            if new_std is None:
                self.regression.update(Z=self.Z)
            else:
                self.regression.update(Z=self.Z, noise_std=new_std)

    
    def weight_fn(self, in_shape=None, resolution=1.0):
        """ Inference q_grid(. | Z, u),
                where T has shape (N, 2) """

        u = self._u()

        device = u.device

        # create discrete sampling grid
        sample_size = self.in_shape if in_shape is None else _pair(in_shape)

        if (sample_size == self.in_shape) and (resolution == 1.0):
            grid = self.grid # (HW, 2), use sampling grid buffer for efficiency
        else:
            s = ((sample_size[0] - 1) / 2, (sample_size[1] - 1) / 2)
            s = (s[0] / resolution, s[1] / resolution)
            
            ys = torch.linspace(-s[0], s[0], steps=sample_size[0], device=device)
            xs = torch.linspace(-s[1], s[1], steps=sample_size[1], device=device)

            grid = torch.stack(torch.meshgrid(ys, xs, indexing='xy'), 2) 
            grid = grid.view(-1, 2) # (HW, 2)

        # posterior mean inference
        m = self.regression.predict(self.Z, u, grid) # c', c, s
            
        # convert to SLinearU dimensions
        m = m.view(self.out_features, self.in_features) # c'c, s

        return m
    
    def forward(self, x):
        # Infer q(grid | Z, u)
        weight = self.weight_fn()

        # Apply fast conv2d
        return F.linear(x, weight, self.bias)



class SLinearV(nn.Linear):
    # (out_channels*in_channels)x(out_size) linear layer
    # sparsified spatially within the 'out_size' subdimension

    def __init__(self, out_channels, in_channels, out_size, bias=True, device=None, dtype=None,
                 free_anchor_loc=False, n_anchor=None, order=0,
                 scale=1.0, omega=1.0, noise_std=0.01, kernel_type='rbf',
                 learn_omega=False, learn_scale=False, learn_noise=False,
                 solver='solve'):
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = out_channels * in_channels
        self.out_features = out_size

        super().__init__(self.in_features, self.out_features)

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.out_shape = (int(math.sqrt(out_size)), int(math.sqrt(out_size)))
        self.free_anchor_loc = free_anchor_loc

        # set kernel
        if kernel_type == 'rbf':
            kernel = RBF(scale=scale, omega=omega, learn_scale=learn_scale, learn_omega=learn_omega, device=device)
        else:
            raise NotImplementedError(f"Unknown kernel: {kernel_type}")
        
        # sample grid
        s = ((self.out_shape[0] - 1) / 2, (self.out_shape[1] - 1) / 2)
        
        ys = torch.linspace(-s[0], s[0], steps=self.out_shape[0], device=device)
        xs = torch.linspace(-s[1], s[1], steps=self.out_shape[1], device=device)

        grid = torch.stack(torch.meshgrid(ys, xs, indexing='xy'), 2) 
        grid = grid.view(-1, 2) # (HW, 2)
        self.register_buffer('grid', grid)
        
        # init anchor point locations at grid
        assert n_anchor in (0, None, self.out_shape[0] * self.out_shape[1]), "With anchor init 'grid', NZ should be 0/None or correct (in_shape^2)"
        Z = grid.view(-1, 2).clone()

        if free_anchor_loc:
            self.Z = nn.Parameter(Z)
        else:
            self.register_buffer('Z', Z)

        # anchor point values
                            #out_features x in_features
                            #(sparse)     x in_features 
                            #(sparse)     x out_channels*in_channels

        weight = torch.randn(len(Z), out_channels*in_channels, device=device)
        self.weight = nn.Parameter(weight)

        self.regression = Regression(2, kernel, noise_std=noise_std, solver=solver, learn_noise=learn_noise)
        self.regression.update(Z=self.Z)

        # TODO TODO TODO
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def _u(self): # weight2u
        return self.weight.view(-1, self.out_channels, self.in_channels).permute(1, 2, 0)

    def _set_weight(self, u): # u2weight
        self.weight.data = u.view(self.out_channels*self.in_channels, -1).T.contiguous()

    def _set_u_from_weight(self, weight, bias=None, ignore_noise=False): # TODO: size check
        """ Set anchor point values from conv2d weights (and bias). """

        u = self.weight.view(-1, self.out_channels, self.in_channels).permute(1, 2, 0)

        assert len(u.shape) == 4, f"Weight should be of dimension 4. Got {u.shape} (dim={len(u.shape)})."
        assert u.shape[0] == self.out_channels
        assert u.shape[1] == self.in_channels
        assert u.shape[2] == self.out_shape[0]
        assert u.shape[3] == self.out_shape[1]

        if ignore_noise:
            u_new = u.view(self.out_channels, self.in_channels, self.out_shape[0]*self.out_shape[1])
            self._set_weight(u_new)
        else:
            old_f = u.view(self.out_channels, self.in_channels, self.out_shape[0]*self.out_shape[1])

            # find new_u using closed-form least-squares solution
            Kzz = self.regression._kernel(self.Z, self.Z)
            
            diag_noise = torch.eye(len(Kzz), device=Kzz.device) * (self.regression.noise_std ** 2)
            A = torch.linalg.solve((Kzz + diag_noise).T, Kzz.T).T

            # inverse formula
            convert_matrix = torch.pinverse(A)
            u_new = old_f @ convert_matrix.T # u=(klj), convert_matrix=(ij) -> u_new=(kli)

            self._set_weight(u_new)

        if bias is not None:
            self.bias.data = bias
        
    def reduce(self, factor, dist='uniform', new_std=None,
               method='lstsq', deterministic=False, always_zero=False):
        """ Inference q_grid(. | Z, u),
                where T has shape (N, 2) """
        
        u = self._u()

        device = u.device

        # create new anchor points
        n_anchor = max(round(factor * len(self.Z)), 1)
        # n_anchor = int(min(len(self.Z), max(1, round(factor * math.sqrt(len(self.Z))))))

        # new noise
        std_new = self.regression.noise_std if new_std is None else new_std

        if deterministic:
            torch.manual_seed(1)

        s = ((self.out_shape[0] - 1) / 2, (self.out_shape[1] - 1) / 2)
        if dist == 'normal':
            Z_new = torch.randn(n_anchor, 2, device=device) * torch.tensor(s, device=device).view(1, -1) 
            if always_zero:
                Z_new[0].zero_()
        elif dist == 'uniform':
            Z_new = (torch.rand(n_anchor, 2, device=device) * 2 - 1) * torch.tensor(s, device=device).view(1, -1) 
            if always_zero:
                Z_new[0].zero_()
        else:
            raise NotImplementedError(f'Unknown reduce distribution: {dist}')

        u = self._u()

        old_f = self.weight_fn().reshape(*u.shape).detach()
        self.old_f = old_f

        if method == 'expectation':
            """ Set new points by simply directly predicting expected value at these points.
                This is a naive approach that does not find the minimising solution (not recommended). """
            # posterior mean inference for new anchor point values
            u_new = self.regression(self.Z, old_f, Z_new)
        elif method == 'formula':
            # solve minimum norm problem for new anchor point values in closed-form
            Kzq = self.regression._kernel(self.Z, Z_new)
            Kqq = self.regression._kernel(Z_new, Z_new)

            diag_noise = torch.eye(len(Kqq), device=Kqq.device) * (std_new ** 2)
            A = Kzq @ torch.inverse(Kqq + diag_noise)

            convert_matrix = torch.pinverse(A)

            u_new = old_f @ convert_matrix.T # u=(klj), convert_matrix=(ij) -> u_new=(kli)
        elif method == 'lstsq':
            # solve minimum norm problem for new anchor point values in closed-form
            Kzq = self.regression._kernel(self.Z, Z_new)
            Kqq = self.regression._kernel(Z_new, Z_new)

            diag_noise = torch.eye(len(Kqq), device=Kqq.device) * (std_new ** 2)

            A = torch.linalg.solve((Kqq + diag_noise).T, Kzq.T).T

            C = A.unsqueeze(0) # (1, i, j)
            u = old_f.view(-1, old_f.shape[-1]).unsqueeze(2) # (kl, i, 1)

            u_new = torch.linalg.lstsq(C, u).solution # (kl, j, 1)
            u_new = u_new.view(*old_f.shape[:2], -1)

            self._set_weight(u_new)
        else:
            raise NotImplementedError(f'Unknown method for anchor point reduction: {method}')

        with torch.no_grad():
            # replace new anchor points
            if self.free_anchor_loc:
                self.Z = nn.Parameter(Z_new)
            else:
                self.register_buffer('Z', Z_new)

            self._set_weight(u_new)
            self.weight = nn.Parameter(self.weight)
                
            if new_std is None:
                self.regression.update(Z=self.Z)
            else:
                self.regression.update(Z=self.Z, noise_std=new_std)

    
    def weight_fn(self, out_shape=None, resolution=1.0):
        """ Inference q_grid(. | Z, u),
                where T has shape (N, 2) """

        u = self._u()

        device = u.device

        # create discrete sampling grid
        sample_size = self.out_shape if out_shape is None else _pair(out_shape)

        if (sample_size == self.out_shape) and (resolution == 1.0):
            grid = self.grid # (HW, 2), use sampling grid buffer for efficiency
        else:
            s = ((sample_size[0] - 1) / 2, (sample_size[1] - 1) / 2)
            s = (s[0] / resolution, s[1] / resolution)
            
            ys = torch.linspace(-s[0], s[0], steps=sample_size[0], device=device)
            xs = torch.linspace(-s[1], s[1], steps=sample_size[1], device=device)

            grid = torch.stack(torch.meshgrid(ys, xs, indexing='xy'), 2) 
            grid = grid.view(-1, 2) # (HW, 2)

        # posterior mean inference
        m = self.regression.predict(self.Z, u, grid) # c', c, s
            
        # convert to SLinearV dimensions
        m = m.permute(2, 0, 1).view(self.out_features, self.in_features)  # s, c'c

        return m
    
    def forward(self, x):
        # Infer q(grid | Z, u)
        weight = self.weight_fn() 

        # Apply fast conv2d
        return F.linear(x, weight, self.bias)


class SLinear2d(nn.Module):
    def __init__(self, in_channels, out_channels, in_shape=(28, 28), out_shape=(14, 14), bias=True, learn_omega=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_size = in_shape[0] * in_shape[1]
        self.out_size = out_shape[0] * out_shape[1]
        self.lin_U = SLinearU(in_channels, self.in_size, out_channels, bias=True, learn_omega=learn_omega)
        self.lin_V = SLinearV(out_channels, in_channels, self.out_size, bias=False, learn_omega=learn_omega) # <--- TODO: remove hardcoded bias

        self.reset_parameters()

    def reduce(self, factor, dist='uniform', new_std=None,
               method='lstsq', deterministic=False, always_zero=False):
        self.lin_U.reduce(factor, dist=dist, new_std=new_std, method=method, deterministic=deterministic, always_zero=always_zero)
        self.lin_V.reduce(factor, dist=dist, new_std=new_std, method=method, deterministic=deterministic, always_zero=always_zero)

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

        #self.lin_U.weight.data = torch.sign(self.lin_U.weight.data) * torch.sqrt(torch.abs(self.lin_U.weight.data)) * math.sqrt(2)
        #self.lin_V.weight.data = torch.sign(self.lin_V.weight.data) * torch.sqrt(torch.abs(self.lin_V.weight.data)) * math.sqrt(2)

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

