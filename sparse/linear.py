import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from sparse.regression import Regression
from sparse.kernels import RBF

class SLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 free_anchor_loc=False, n_anchor=None, order=0,
                 scale=1.0, omega=1.0, device=None, noise_std=0.01, kernel_type='rbf',
                 learn_omega=False, learn_scale=False, learn_noise=False,
                 solver='solve'):
        super().__init__()

        print(learn_omega, learn_scale, learn_noise)

        self.out_features = out_features
        self.in_features = in_features
        self.shape_in = _pair(math.sqrt(in_features))
        self.shape_out = _pair(math.sqrt(out_features))

        assert len(self.shape_in) == 2, f"Only supports 2d images for now"
        assert len(self.shape_out) == 2, f"Only supports 2d images for now"

        self.free_anchor_loc = free_anchor_loc

        # set kernel
        if kernel_type == 'rbf':
            kernel = RBF(scale=scale, omega=omega, learn_scale=learn_scale, learn_omega=learn_omega, device=device)
        else:
            raise NotImplementedError(f"Unknown kernel: {kernel_type}")
        
        # sample grid
        ys = torch.linspace(0, self.shape_in[0], steps=len(self.shape_in[0]), device=device)
        xs = torch.linspace(0, self.shape_in[1], steps=len(self.shape_in[0]), device=device)

        grid = torch.stack(torch.meshgrid(ys, xs, indexing='xy'), 2) 
        grid = grid.view(-1, 2) # (HW, 2)
        self.register_buffer('grid', grid)
        
        # init anchor point locations at grid
        assert n_anchor in (0, None, self.in_features), f"anchor points ({n_anchor}) should be (0, None, in_features={self.in_features}."
        Z = grid.view(-1, 2).clone()

        if free_anchor_loc:
            self.Z = nn.Parameter(Z)
        else:
            self.register_buffer('Z', Z)

        # anchor point values
        u = torch.randn(out_features, len(Z), device=device)
        self.u = nn.Parameter(u)

        self.regression = Regression(2, kernel, noise_std=noise_std, solver=solver, learn_noise=learn_noise)
        self.regression.update(Z=self.Z)
        
        # other
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.bias = None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # TODO implement reset_parameters that corresponds regular Linear layer
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        bound = 1. / math.sqrt(self.in_features)
        self.u.data.uniform_(-bound, bound)
        
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)

    def _set_u_from_weight(self, weight, bias=None, ignore_noise=False): # TODO: size check
        """ Set anchor point values from conv2d weights (and bias). """

        assert len(weight.shape) == 2, f"Weight should be of dimension 4. Got {weight.shape} (dim={len(weight.shape)})."
        assert weight.shape[0] == self.out_features
        assert weight.shape[1] == self.in_features

        if ignore_noise:
            self.u.data = weight.view(self.out_features, self.in_features)
        else:
            old_f = weight.view(self.out_features, self.in_features)

            # find new_u using closed-form least-squares solution
            Kzz = self.regression._kernel(self.Z, self.Z)
            
            diag_noise = torch.eye(len(Kzz), device=Kzz.device) * (self.regression.noise_std ** 2)
            A = torch.linalg.solve((Kzz + diag_noise).T, Kzz.T).T

            # inverse formula
            convert_matrix = torch.pinverse(A)
            u_new = old_f @ convert_matrix.T # u=(klj), convert_matrix=(ij) -> u_new=(kli)

            self.u.data = u_new

        self.bias = bias
        
    def reduce(self, factor, dist='uniform', new_std=None,
               method='lstsq', deterministic=False, always_zero=False):
        """ Inference q_grid(. | Z, u),
                where T has shape (N, 2) """
        device = self.u.device

        # create new anchor points
        n_anchor = max(round(factor * len(self.Z)), 1)

        # new noise
        std_new = self.regression.noise_std if new_std is None else new_std

        if deterministic:
            torch.manual_seed(1)

        s = ((self.kernel_size[0] - 1) / 2, (self.kernel_size[1] - 1) / 2)
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

        old_f = self.filter().view(*self.u.shape).detach()
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
            self.u_new = u_new
        else:
            raise NotImplementedError(f'Unknown method for anchor point reduction: {method}')

        with torch.no_grad():
            # replace new anchor points
            if self.free_anchor_loc:
                self.Z = nn.Parameter(Z_new)
            else:
                self.register_buffer('Z', Z_new)

            self.u = nn.Parameter(u_new)
                
            if new_std is None:
                self.regression.update(Z=self.Z)
            else:
                self.regression.update(Z=self.Z, noise_std=new_std)

    
    def filter(self, kernel_size=None, resolution=1.0):
        """ Inference q_grid(. | Z, u),
                where T has shape (N, 2) """

        device = self.u.device

        # create discrete sampling grid
        sample_size = self.kernel_size if kernel_size is None else _pair(kernel_size)

        if (sample_size == self.kernel_size) and (resolution == 1.0):
            grid = self.grid # (HW, 2), use sampling grid buffer for efficiency
        else:
            s = ((sample_size[0] - 1) / 2, (sample_size[1] - 1) / 2)
            s = (s[0] / resolution, s[1] / resolution)
            
            ys = torch.linspace(-s[0], s[0], steps=sample_size[0], device=device)
            xs = torch.linspace(-s[1], s[1], steps=sample_size[1], device=device)

            grid = torch.stack(torch.meshgrid(ys, xs, indexing='xy'), 2) 
            grid = grid.view(-1, 2) # (HW, 2)

        # posterior mean inference
        m = self.regression.predict(self.Z, self.u, grid)
            
        # convert to predicted values to conv2d convention (out_features, in_features)
        m = m.view(self.out_features, self.in_features)

        return m
    
    def forward(self, x):
        # Infer q(grid | Z, u)
        sampled_filter = self.filter()

        # Apply fast conv2d
        return F.conv2d(x, sampled_filter, bias=self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


