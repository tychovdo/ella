import torch

from torch import nn

class Regression(nn.Module):
    """ Implementation of simple basis function regression.

    Currently, the design of the class is functional and (hyper)parameters are
    always given to functions as arguments rather than stored as objects.
    An exception is caching mechanism for inverse(Kzz), which is stored internally. """
    def __init__(self, dim, kernel, noise_std=0.01, solver='naive', learn_noise=False):
        super().__init__()
        self.dim = dim
        self.solver = solver
        self.learn_noise = learn_noise

        if learn_noise:
            self.noise_std = nn.Parameter(torch.full((1,), noise_std))
        else:
            self.noise_std = noise_std

        self._kernel = kernel
        self._deriv = None

    def update(self, Z=None, u=None, noise_std=None):
        if Z is not None:
            self.register_buffer('Z_idt', torch.eye(len(Z), device=Z.device))

        if noise_std is not None:
            if learn_noise:
                self.noise_std = nn.Parameter(torch.full((1,), noise_std))
            else:
                self.noise_std = noise_std

    def predict(self, Z, u, T):
        """ Given points (Z, u) predict at new test locations (T).

            Return expectation. """

        if self.solver == 'naive':
            print('[warning] using naive implementation')
            Kzz = self._kernel(Z, Z)

            diag_noise = self.Z_idt * (self.noise_std ** 2)
            Kzz_inv = torch.inverse(Kzz + diag_noise)

            Ktz = self._kernel(T, Z)

            Psi = Ktz @ Kzz_inv

            m = u @ Psi.T # u=(klj), Psi=(ij) -> m=(klj)

            return m
        elif self.solver == 'solve': # recommended implementaion
            Kzz = self._kernel(Z, Z) # todo: cache
        
            Kzt = self._kernel(Z, T)

            if self.learn_noise:
                diag_noise = self.Z_idt * (1e-6 + self.noise_std ** 2)
            else:
                diag_noise = self.Z_idt * (self.noise_std ** 2)

            linmap = torch.linalg.solve(Kzz + self.Z_idt * diag_noise, Kzt)
            self._deriv = linmap.detach()

            m = u @ linmap

            return m
        else:
            raise NotImplementedError(f"Unknown regression solve method: {self.solver}")


