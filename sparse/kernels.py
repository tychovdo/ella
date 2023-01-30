import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class RBF(nn.Module):
    def __init__(self, dim=None, scale=1.0, omega=1.0, learn_scale=False, learn_omega=False, device=None, scale_jitter=1e-6):
        super().__init__()

        if learn_scale:
            self.scale = nn.Parameter(torch.full((1, ), scale, device=device))
        else:
            self.register_buffer('scale', torch.full((1, ), scale, device=device))

        if learn_omega:
            self.omega = nn.Parameter(torch.full((1, ), omega, device=device))
        else:
            self.register_buffer('omega', torch.full((1, ), omega, device=device))

        self.dim = dim
        self.scale_jitter = scale_jitter
            
    def __repr__(self):
        return f"RBF({self.dim}, omega={self.omega.item():.2f}, scale={self.scale.item():.2f}, learn_omega={self.omega.requires_grad}, learn_scale={self.scale.requires_grad})"

    def forward(self, x1, x2):
        assert x1.shape[1] == x2.shape[1] # check dimension
        assert x1.numel() >= 0 # sanity check
        assert x2.numel() >= 0 # sanity check
        if self.dim is not None:
            assert x1.shape[1] == self.dim, f"Got dimension {x1.shape[1]}, but expected dim={self.dim}."

        dist = torch.cdist(x1, x2, p=2.0)

        filter_matrix = (self.scale_jitter + (self.scale ** 2)) * torch.exp(-0.5 * (self.omega ** 2) * (dist ** 2))
                                  
        return filter_matrix

