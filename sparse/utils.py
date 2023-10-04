import numpy as np
import wandb

import torch
from torch import nn
from torch.nn.utils.convert_parameters import parameters_to_vector

from sparse.sconv2d import SConv2d
# from sparse.slinear import SLinear2d

from laplace import FullLaplace, KronLaplace, DiagLaplace, FunctionalLaplace


class GroupIdentity(nn.Module):
    def __init__(self, module, group_dim=False):
        super().__init__()
        
        self.module = module
        self.group_dim = group_dim

    def forward(self, x):
        assert type(x) == tuple
        lenx = len(x)
        if lenx == 2:
            g, x = x
            lenx = 2
        elif lenx == 3:
            g1, g2, x = x
            lenx = 3
        else:
            raise NotImplementedError(f"HUh?")

        if self.group_dim:
            y = self.module(x.reshape(-1, x.shape[2], *x.shape[3:]))
            y = y.view(*x.shape[:2], *y.shape[1:])
        else:
            y = self.module(x)

        if lenx == 3:
            return g1, g2, y
        elif lenx == 2:
            return g, y
        else:
            raise NotImplementedError(f"HUh?")

class GNorm(nn.Module):
    def __init__(self, num_features, group_stats=True, **kwargs):
        super().__init__()

        self.group_stats = group_stats

        if group_stats:
            self.module = nn.BatchNorm3d(num_features, **kwargs)
        else:
            self.module = nn.BatchNorm2d(num_features, **kwargs)

    def forward(self, x):
        assert type(x) == tuple
        lenx = len(x)
        if lenx == 2:
            g, x = x
            lenx = 2
        elif lenx == 3:
            g1, g2, x = x
            lenx = 3
        else:
            raise NotImplementedError(f"HUh?")

        if self.group_stats:
            y = self.module(x.transpose(1, 2)).transpose(1, 2)
        else:
            y = self.module(x.reshape(-1, x.shape[2], *x.shape[3:]))
            y = y.view(*x.shape[:2], *y.shape[1:])

        if lenx == 3:
            return g1, g2, y
        elif lenx == 2:
            return g, y
        else:
            raise NotImplementedError(f"HUh?")

def convert_model(model, conv_class, reduce_factor=1.0, reduce_args={}, verbose=True,
                  learn_omega=False, learn_scale=False, learn_noise=False, kernel_type='rbf',
                  enlarge=1, independent_rsample=False, free_anchor_loc=True,
                  solver='solve', noise_std=0.01, dt=torch.float64, cholesky='cholesky_ex', mask=False,
                  channel_covariances=False, detach_kl=False, detach_kl2=False, min_convert_size=1,
                  **kwargs):
    if verbose:
        param_count_1 = len(torch.cat([x.reshape(-1) for x in model.parameters()]))

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for mod, module in model._modules.items():
        if isinstance(module, nn.Conv2d) and \
                        (module.kernel_size[0] >= min_convert_size) and \
                        (module.kernel_size[1] >= min_convert_size) and (conv_class in [SConv2d, SConv2dAug]):
            # get original layer properties
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            has_bias = module.bias is not None
            padding_mode = module.padding_mode

            device = module.weight.device

            if len(kernel_size) == 1:
                kernel_size = (kernel_size, kernel_size)

            if len(padding) == 1:
                padding = (padding, padding)

            if enlarge == 2:
                def E(x):
                    out = {1: 1, 3: 7, 5: 11, 7: 15}
                    return out[x]

                def P(x):
                    return (E(x) - x) // 2

                padding = (padding[0] + P(kernel_size[0]), padding[1] + P(kernel_size[1]))
                kernel_size = (E(kernel_size[0]), E(kernel_size[1]))
    
            if conv_class in [SConv2d, SConv2dAug]:
                new_module = conv_class(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=has_bias, padding_mode=padding_mode, device=device,
                                        learn_omega=learn_omega, learn_scale=learn_scale, learn_noise=learn_noise,
                                        kernel_type=kernel_type,
                                        solver=solver, noise_std=noise_std, free_anchor_loc=free_anchor_loc,
                                        **kwargs)
            else:
                raise NotImplementedError(f"Unknown class: {conv_class}")

            # copy weights for grid
            new_module._set_u_from_weight(module.weight, module.bias)

            # reduce, if needed
            if reduce_factor != 1.0:
                new_module.reduce(reduce_factor, **reduce_args)

            # replace object
            setattr(model, mod, new_module)

        # if isinstance(module, nn.Linear) and conv_class in [SLinear, SLinearAug]:
        #     # get original layer properties
        #     in_features = module.in_features
        #     out_features = module.out_features
        #     has_bias = module.bias is not None

        #     device = module.weight.device

        #     if conv_class in [SLinear, SLinearAug]:
        #         new_module = conv_class(in_features, out_features,
        #                                 bias=has_bias, device=device,
        #                                 learn_omega=learn_omega, learn_scale=learn_scale, learn_noise=learn_noise,
        #                                 kernel_type=kernel_type,
        #                                 solver=solver, noise_std=noise_std, free_anchor_loc=free_anchor_loc,
        #                                 **kwargs)
        #     else:
        #         raise NotImplementedError(f"Unknown class: {conv_class}")

        #     # copy weights for grid
        #     new_module._set_u_from_weight(module.weight, module.bias)

        #     # reduce, if needed
        #     if reduce_factor != 1.0:
        #         new_module.reduce(reduce_factor, **reduce_args)

        #     # replace object
        #     setattr(model, mod, new_module)

    for immediate_child_module in model.children():
        convert_model(immediate_child_module, conv_class=conv_class,
                      reduce_args=reduce_args,
                      reduce_factor=reduce_factor, verbose=False, 
                      enlarge=enlarge,
                      learn_omega=learn_omega, learn_scale=learn_scale, learn_noise=learn_noise,
                      kernel_type=kernel_type,
                      independent_rsample=independent_rsample,
                      free_anchor_loc=free_anchor_loc,
                      solver=solver, noise_std=noise_std, dt=dt,
                      mask=mask, cholesky=cholesky, channel_covariances=channel_covariances,
                      detach_kl=detach_kl, detach_kl2=detach_kl2,
                      **kwargs)

    if verbose:
        param_count_2 = len(torch.cat([x.reshape(-1) for x in model.parameters()]))

        diff_percent = 100 * (param_count_2 - param_count_1) / param_count_1
        diff_percent_str = f'+{diff_percent:.2f}%' if diff_percent > 0 else f'{diff_percent:.2f}%'
        print(f'Converted original model with {param_count_1} parameters to inducing point model with {param_count_2} parameters.')
        print(f'Relative change: {diff_percent_str}')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        try:
            from torch import cudnn
            cudnn.deterministic = True
            cudnn.benchmark = True
        except:
            pass


def get_laplace_approximation(structure):
    if structure == 'full':
        return FullLaplace
    elif structure == 'kron':
        return KronLaplace
    elif structure == 'diag':
        return DiagLaplace
    elif structure == 'kernel' or structure == 'kernel-stochastic':
        return FunctionalLaplace


def wandb_log_parameter_hist(model):
    for name, param in model.named_parameters():
        hist, edges = param.data.cpu().histogram(bins=64)
        wandb.log({f'params/{name}': wandb.Histogram(
            np_histogram=(hist.numpy().tolist(), edges.numpy().tolist())
        )}, commit=False)


def wandb_log_parameter_norm(model):
    for name, param in model.named_parameters():
        avg_norm = (param.data.flatten() ** 2).sum().item() / np.prod(param.data.shape)
        wandb.log({f'params/{name}': avg_norm}, commit=False)


def wandb_log_invariance(augmenter):
    aug_params = np.abs(
        parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
    ).tolist()
    if len(aug_params) == 6:
        names = ['Tx', 'Ty', 'R', 'Sx', 'Sy', 'H']
    else:
        names = [f'aug_{i}' for i in range(6)]
    log = {f'invariances/{n}': p for n, p in zip(names, aug_params)}
    wandb.log(log, commit=False)


def wandb_log_prior(prior_prec, prior_structure, model):
    prior_prec = prior_prec.detach().cpu().numpy().tolist()
    if prior_structure == 'scalar':
        wandb.log({'hyperparams/prior_prec': prior_prec[0]}, commit=False)
    elif prior_structure == 'layerwise':
        log = {f'hyperparams/prior_prec_{n}': p for p, (n, _) in
               zip(prior_prec, model.named_parameters())}
        wandb.log(log, commit=False)
    elif prior_structure == 'diagonal':
        hist, edges = prior_prec.data.cpu().histogram(bins=64)
        log = {f'hyperparams/prior_prec': wandb.Histogram(
            np_histogram=(hist.numpy().tolist(), edges.numpy().tolist())
        )}
        wandb.log(log, commit=False)

def wandb_log_effective_dimensionality(effect_dim, prior_structure, model):
    if prior_structure == 'scalar':
        wandb.log({'hyperparams/effect_dim': effect_dim[0]}, commit=False)
    elif prior_structure == 'layerwise':
        log = {f'hyperparams/effect_dim_{n}': p for p, (n, _) in
               zip(effect_dim, model.named_parameters())}
        wandb.log(log, commit=False)
    elif prior_structure == 'diagonal':
        hist, edges = effect_dim.data.cpu().histogram(bins=64)
        log = {f'hyperparams/effect_dim': wandb.Histogram(
            np_histogram=(hist.numpy().tolist(), edges.numpy().tolist())
        )}
        wandb.log(log, commit=False)

def wandb_log_normalised_effective_dimensionality(effect_dim, prior_structure, model):
    if prior_structure == 'scalar':
        wandb.log({'hyperparams/norm_effect_dim': effect_dim[0] / len(parameters_to_vector(model.parameters()))}, commit=False)
    elif prior_structure == 'layerwise':
        log = {f'hyperparams/norm_effect_dim_{n}': e / len(p.reshape(-1)) for e, (n, p) in
               zip(effect_dim, model.named_parameters())}
        wandb.log(log, commit=False)
    elif prior_structure == 'diagonal':
        raise NotImplementedError(f"TODO:")
        hist, edges = effect_dim.data.cpu().histogram(bins=64)
        log = {f'hyperparams/norm_effect_dim': wandb.Histogram(
            np_histogram=(hist.numpy().tolist(), edges.numpy().tolist())
        )}
        wandb.log(log, commit=False)

def wandb_log_len(model):
    log = {f'hyperparams/len_{n}': len(p.view(-1)) for n, p in model.named_parameters()}
    wandb.log(log, commit=False)


