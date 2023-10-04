import logging
import numpy as np
import torch
from torchvision import transforms
import pickle
import wandb
from pathlib import Path
from dotenv import load_dotenv
from torch.nn.utils.convert_parameters import parameters_to_vector
logging.basicConfig(format='[%(filename)s:%(lineno)s]%(levelname)s: %(message)s', level=logging.INFO)

from laplace.curvature.asdl import AsdlGGN, AsdlEF
from laplace.curvature.augmented_asdl import AugAsdlGGN, AugAsdlEF

from sparse.marglik import marglik_optimization
from sparse.invariances import AffineLayer2d
from sparse.utils import get_laplace_approximation, set_seed

from data_utils.datasets import (
    RotatedMNIST, TranslatedMNIST, ScaledMNIST, RotatedFashionMNIST, TranslatedFashionMNIST,
    ScaledFashionMNIST, RotatedCIFAR10, TranslatedCIFAR10, ScaledCIFAR10, RotatedCIFAR100,
    TranslatedCIFAR100, ScaledCIFAR100, QuadrantMNIST, DigitMNIST
)
from data_utils.utils import (
    TensorDataLoader, SubsetTensorDataLoader, GroupedSubsetTensorDataLoader, dataset_to_tensors
)

from data_utils.augmentation import CIFAR10_augment

from sparse.sconv2d import SConv2d

from sparse.utils import convert_model

from sparse.models import D_Both, D_Conv, D_FC
from sparse.models import F_Both, F_FC
from sparse.models import S_Both, S_Conv, S_FC

from sparse.models_group import D_GConv 
from sparse.models_group import D_GConvFull 
from sparse.models_group import D_GConvGSeparable
from sparse.models_group import D_GConvDGSeparable
from sparse.models_group import D_GConvPointwise

from sparse.models_mixedgroup import Mixer1 

def get_dataset(dataset, data_root, download_data, transform):
    if dataset == 'mnist':
        train_dataset = RotatedMNIST(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedMNIST(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'mnist_r90':
        train_dataset = RotatedMNIST(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedMNIST(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'mnist_r180':
        train_dataset = RotatedMNIST(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedMNIST(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_mnist':
        train_dataset = TranslatedMNIST(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedMNIST(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_mnist':
        train_dataset = ScaledMNIST(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledMNIST(data_root, np.log(2), train=False, download=download_data, transform=transform)
    elif dataset == 'fmnist':
        train_dataset = RotatedFashionMNIST(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedFashionMNIST(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'fmnist_r90':
        train_dataset = RotatedFashionMNIST(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedFashionMNIST(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'fmnist_r180':
        train_dataset = RotatedFashionMNIST(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedFashionMNIST(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_fmnist':
        train_dataset = TranslatedFashionMNIST(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedFashionMNIST(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_fmnist':
        train_dataset = ScaledFashionMNIST(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledFashionMNIST(data_root, np.log(2), train=False, download=download_data, transform=transform)
    elif dataset == 'digit_mnist':
        train_dataset = DigitMNIST(data_root, train=True, download=download_data, transform=transform)
        test_dataset = DigitMNIST(data_root, train=False, download=download_data, transform=transform)
    elif dataset == 'quadrant_mnist':
        train_dataset = QuadrantMNIST(data_root, train=True, download=download_data, transform=transform)
        test_dataset = QuadrantMNIST(data_root, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar10':
        train_dataset = RotatedCIFAR10(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR10(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar10_r90':
        train_dataset = RotatedCIFAR10(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR10(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar10_r180':
        train_dataset = RotatedCIFAR10(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR10(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_cifar10':
        train_dataset = TranslatedCIFAR10(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedCIFAR10(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_cifar10':
        train_dataset = ScaledCIFAR10(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledCIFAR10(data_root, np.log(2), train=False, download=download_data, transform=transform)
    elif dataset == 'cifar100':
        train_dataset = RotatedCIFAR100(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR100(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar100_r90':
        train_dataset = RotatedCIFAR100(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR100(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar100_r180':
        train_dataset = RotatedCIFAR100(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR100(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_cifar100':
        train_dataset = TranslatedCIFAR100(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedCIFAR100(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_cifar100':
        train_dataset = ScaledCIFAR100(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledCIFAR100(data_root, np.log(2), train=False, download=download_data, transform=transform)
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset}')

    return train_dataset, test_dataset


def main(
    seed, method, approx, curv, dataset, model, n_epochs, batch_size, marglik_batch_size, partial_batch_size, bound,
    subset_size, n_samples_aug, softplus, init_aug, lr, lr_min, lr_hyp, lr_hyp_min, lr_aug, lr_aug_min, grouped_loader,
    prior_prec_init, n_epochs_burnin, marglik_frequency, n_hypersteps, n_hypersteps_prior, random_flip, sam, sam_with_prior,
    last_logit_constant, save, device, download_data, data_root, use_wandb, independent_outputs, kron_jac, single_output,
    data_augmentation, data_augmentation_marglik, sparse, curvature, track_kron_la, scheduler, alpha, shared_uv, omega_hyper,
    log_weights
):
    # dataset-specific static transforms (preprocessing)
    if 'mnist' in dataset:
        mean = 0.1307
        std = 0.3081
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), transforms.Resize((32, 32))])
    elif 'cifar' in dataset:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        raise NotImplementedError(f'Transform for {dataset} unavailable.')

    train_dataset, test_dataset = get_dataset(dataset, data_root, download_data, transform)

    # dataset-specific number of classes
    if 'cifar100' in dataset:
        n_classes = 100
    elif 'quadrant' in dataset:
        n_classes = 40
    else:
        n_classes = 10

    # Load data
    set_seed(seed)

    # Subset the data if subset_size is given.
    subset_size = len(train_dataset) if subset_size <= 0 else subset_size
    if subset_size < len(train_dataset):
        subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    else:
        subset_indices = None
    X_train, y_train = dataset_to_tensors(train_dataset, subset_indices, device)
    X_test, y_test = dataset_to_tensors(test_dataset, None, device)

    if method == 'lila':
        assert not data_augmentation
        assert not data_augmentation_marglik
        augmenter = AffineLayer2d(n_samples=n_samples_aug, init_value=init_aug,
                                  softplus=softplus, random_flip=random_flip).to(device)
        augmenter_valid = augmenter_marglik = augmenter
        augmenter.rot_factor.requires_grad = True
    elif data_augmentation:
        assert 'cifar' in dataset
        augmenter = CIFAR10_augment
        augmenter_valid = None
        augmenter_marglik = augmenter if data_augmentation_marglik else None
    else:
        augmenter = augmenter_valid = augmenter_marglik = None
    optimize_aug = (method == 'lila')

    temperature = 1.0
    batch_size = subset_size if batch_size <= 0 else min(batch_size, subset_size)
    ml_batch_size = subset_size if marglik_batch_size <= 0 else min(marglik_batch_size, subset_size)
    pl_batch_size = subset_size if partial_batch_size <= 0 else min(partial_batch_size, subset_size)
    train_loader = TensorDataLoader(X_train, y_train, transform=augmenter, batch_size=batch_size, shuffle=True, detach=True)
    valid_loader = TensorDataLoader(X_test, y_test, transform=augmenter, batch_size=batch_size, detach=True)
    if bound == 'None':
        stochastic_grad = False
        marglik_loader = TensorDataLoader(X_train, y_train, transform=augmenter_marglik, batch_size=ml_batch_size, shuffle=True, detach=True)
        partial_loader = TensorDataLoader(X_train, y_train, transform=augmenter_marglik, batch_size=pl_batch_size, shuffle=True, detach=False)
    else:
        stochastic_grad = True
        data_factor = 1.0
        if bound == 'lower':  # lower bound to the linearized laplace marglik
            data_factor = len(X_train) / ml_batch_size
        elif bound == 'independent_upper':  # just scale up the whole log marglik
            pass
        elif bound == 'hescaling_upper':  # just scale up the temperature (log lik and it's Hessian)
            temperature = ml_batch_size / len(X_train)
        else:
            raise ValueError('Invalid bound', bound)
        
        DataLoaderCls = GroupedSubsetTensorDataLoader if grouped_loader else SubsetTensorDataLoader
        marglik_loader = DataLoaderCls(X_train, y_train, transform=augmenter_marglik, subset_size=ml_batch_size,
                                       detach=False, data_factor=data_factor)
        partial_loader = None

    # model
    optimizer = 'SGD'
    prior_structure = 'layerwise'

    if 'cifar10' in dataset:
        in_channels = 3
    else:
        in_channels = 1

    if model == 'd_conv':
        optimizer = 'SGD'
        net = D_Conv(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_fc':
        optimizer = 'SGD'
        net = D_FC(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_both':
        optimizer = 'SGD'
        net = D_Both(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'f_fc':
        optimizer = 'SGD'
        net = F_FC(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'f_both':
        optimizer = 'SGD'
        net = F_Both(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 's_fc':
        optimizer = 'SGD'
        net = S_FC(in_ch=in_channels, num_classes=n_classes, alpha=alpha, learn_omega=omega_hyper) 
    elif model == 's_conv':
        optimizer = 'SGD'
        net = S_Conv(in_ch=in_channels, num_classes=n_classes, alpha=alpha, learn_omega=omega_hyper)
    elif model == 's_both':
        optimizer = 'SGD'
        net = S_Both(in_ch=in_channels, num_classes=n_classes, alpha=alpha, learn_omega=omega_hyper) 
    elif model == 'd_gconv':
        optimizer = 'SGD'
        net = D_GConv(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_gconv_full':
        optimizer = 'SGD'
        net = D_GConvFull(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_gconv_gseparable':
        optimizer = 'SGD'
        net = D_GConvGSeparable(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_gconv_dgseparable':
        optimizer = 'SGD'
        net = D_GConvDGSeparable(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_gconv_pointwise':
        optimizer = 'SGD'
        net = D_GConvPointwise(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_gfc':
        optimizer = 'SGD'
        net = D_GFC(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'd_gboth':
        optimizer = 'SGD'
        net = D_GBoth(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'f_gfc':
        optimizer = 'SGD'
        net = F_GFC(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'f_gboth':
        optimizer = 'SGD'
        net = F_GBoth(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 's_gfc':
        optimizer = 'SGD'
        net = S_GFC(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 's_gconv':
        optimizer = 'SGD'
        net = S_GConv(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 's_gboth':
        optimizer = 'SGD'
        net = S_GBoth(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    elif model == 'mixer1':
        optimizer = 'SGD'
        net = Mixer1(in_ch=in_channels, num_classes=n_classes, alpha=alpha) 
    else:
        raise ValueError('Unavailable model for cifar10')

    if False: # Other way of initialising / Not being used.
        for name, param in net.named_parameters():
            with torch.no_grad():
                if 'bias' in name:
                    param.data.zero_()
                elif 'weight' in name:
                    param.data = param.data * 0.0
                    #torch.nn.init.xavier_uniform_(param, gain=1.4142)
                else:
                    raise NotImplementedError(f"Unknown initialisation for param with name '{name}'.")

    if sparse:
        if optimize_aug:
            convert_model(net,
                                free_anchor_loc=False,
                                reduce_factor=0.5,
                                omega=1.0, scale=1.0,
                                learn_omega=False, learn_scale=False, learn_noise=False,
                                conv_class=SConv2dAug,
                                kernel_type='rbf',
                                solver='solve', noise_std=0.001)
            raise NotImplementedError(f"SLinearAug not implemented yet...")
        else:
            convert_model(net,
                                free_anchor_loc=False,
                                reduce_factor=0.5,
                                omega=1.0, scale=1.0,
                                learn_omega=False, learn_scale=False, learn_noise=False,
                                conv_class=SConv2d,
                                kernel_type='rbf',
                                solver='solve', noise_std=0.001)

    net.to(device)

    result = dict()
    laplace = get_laplace_approximation(approx)
    if optimize_aug:
        backend = AugAsdlGGN if curv == 'ggn' else AugAsdlEF
    else:
        backend = AsdlGGN if curv == 'ggn' else AsdlEF

    la, net, margliks, valid_perfs, aug_history, weights = marglik_optimization(
        net, train_loader, marglik_loader, valid_loader, partial_loader, likelihood='classification',
        lr=lr, lr_hyp=lr_hyp, lr_hyp_min=lr_hyp_min, lr_aug=lr_aug, n_epochs=n_epochs, 
        sam=sam, sam_with_prior=sam_with_prior,
        n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency, laplace=laplace,
        prior_structure=prior_structure, backend=backend, n_epochs_burnin=n_epochs_burnin,
        method=method, augmenter=augmenter_valid, lr_min=lr_min, scheduler=scheduler, optimizer=optimizer,
        n_hypersteps_prior=n_hypersteps_prior, temperature=temperature, lr_aug_min=lr_aug_min,
        prior_prec_init=prior_prec_init, stochastic_grad=stochastic_grad, use_wandb=use_wandb,
        track_kron_la=track_kron_la, independent=independent_outputs, kron_jac=kron_jac, single_output=single_output,
        curvature=curvature, shared_uv=shared_uv, omega_hyper=omega_hyper, log_weights=log_weights
    )

    prior_prec = la.prior_precision.mean().item()
    opt_marglik = 0 if len(margliks) == 0 else np.min(margliks)
    if optimize_aug:
        aug_params = parameters_to_vector(augmenter.parameters()).detach().cpu().numpy()
        logging.info(f'prior prec: {prior_prec:.2f}, aug params: {aug_params}, margLik: {opt_marglik:.2f}.')
    else:
        logging.info(f'prior prec: {prior_prec:.2f}, margLik: {opt_marglik:.2f}.')

    result['marglik'] = opt_marglik
    result['margliks'] = [] if opt_marglik == 0 else margliks

    if save:
        result_path = Path(f'results/{dataset}/{model}/')
        if optimize_aug:
            result['aug_optimum'] = aug_params
            result['aug_history'] = aug_history
        result['valid_perfs'] = valid_perfs
        choice_type = 'optaug' if optimize_aug else 'fixedaug'
        if softplus:
            if np.abs(init_aug - np.log(2)) > 0.01: # init parameter to non-zero (for softplus)
                choice_type += 'softplus=' + str(init_aug)
            else:
                choice_type += 'softplus'
        elif np.abs(init_aug) > 0.01: # init parameter non-zero
            choice_type += '=' + str(init_aug)
        if 'kernel' in approx:
            approx_type = f'_approx={approx}-{marglik_batch_size}'
        else:
            approx_type = '' if method == 'augerino' else f'_approx={approx}'
        file_name = f'{method}{approx_type}_{choice_type}_E={n_epochs}_N={subset_size}_S={n_samples_aug}_seed={seed}.pkl'

        result_path.mkdir(parents=True, exist_ok=True)
        with open(result_path / file_name, 'wb') as f:
            pickle.dump(result, f)

        print('saved.')

                
                


if __name__ == '__main__':
    import sys
    import argparse
    from arg_utils import set_defaults_with_yaml_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--method', default='baseline', choices=['baseline', 'lila'])
    parser.add_argument('--approx', default='full', choices=['full', 'kron', 'diag', 'kernel'])
    parser.add_argument('--curv', default='ggn', choices=['ggn', 'ef'])
    parser.add_argument('--dataset', default='mnist', choices=[
        'mnist', 'mnist_r90', 'mnist_r180', 'translated_mnist', 'scaled_mnist', 'scaled_mnist2',
        'fmnist', 'fmnist_r90', 'fmnist_r180', 'translated_fmnist', 'scaled_fmnist', 'scaled_fmnist2',
        'cifar10', 'cifar10_r90', 'cifar10_r180', 'translated_cifar10', 'scaled_cifar10',
        'cifar100', 'cifar100_r90', 'cifar100_r180', 'translated_cifar100', 'scaled_cifar100',
        'digit_mnist', 'quadrant_mnist'
    ])
    parser.add_argument('--model', default='d_fc', choices=['d_conv', 'd_fc', 'd_both', 'f_fc', 'f_both', 's_fc', 's_conv', 's_both', 'd_gconv', 'd_gconv_full', 'd_gconv_gseparable', 'd_gconv_dgseparable', 'd_gconv_pointwise', 'd_gfc', 'd_gboth', 'f_gfc', 'f_gboth', 's_gfc', 's_gconv', 's_gboth', 'mixer1'])
    parser.add_argument('--n_epochs', default=500, type=int)
    parser.add_argument('--n_epochs_burnin', default=10, type=int)
    parser.add_argument('--independent_outputs', default=False, action=argparse.BooleanOptionalAction,
                        help='Independent outputs (currently only supported for FunctionalLaplace)')
    parser.add_argument('--single_output', default=False, action=argparse.BooleanOptionalAction,
                        help='Sample only single output (currently only supported for FunctionalLaplace)')
    parser.add_argument('--kron_jac', default=True, action=argparse.BooleanOptionalAction,
                        help='Use Kronecker (approximation) for Jacobians (only used within kernel).')
    parser.add_argument('--subset_size', default=-1, type=int, help='Observations in generated data.')
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--marglik_frequency', default=1, type=int)
    parser.add_argument('--marglik_batch_size', default=-1, type=int, help='Used for fitting laplace.')
    parser.add_argument('--partial_batch_size', default=-1, type=int, help='Used for JVPs when necessary.')
    parser.add_argument('--grouped_loader', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--bound', default='None', choices=['None', 'independent_upper', 'hescaling_upper', 'lower'],
                        help='None: use JVP and parametric, indep: fit entirely on batch, hescaling: scale hessian, lower: derived')
    parser.add_argument('--n_hypersteps', default=1, help='Number of steps on every marglik estimate (partial grad accumulation)', type=int)
    parser.add_argument('--n_hypersteps_prior', default=1, help='Same as n_hypersteps but for the prior precision.', type=int)
    parser.add_argument('--n_samples_aug', default=31, type=int)
    parser.add_argument('--softplus', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--random_flip', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--sam', default=False, action=argparse.BooleanOptionalAction, help='Use SAM optimizer.')
    parser.add_argument('--sam_with_prior', default=False, action=argparse.BooleanOptionalAction, help='Use log prior also in the opposite step of sam.')
    parser.add_argument('--last_logit_constant', default=False, action=argparse.BooleanOptionalAction, help='Add last logit constant 0.')
    parser.add_argument('--prior_prec_init', default=1.0, type=float)
    parser.add_argument('--init_aug', default=0.0, type=float)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_min', default=0.1, type=float)
    parser.add_argument('--lr_hyp', default=0.1, type=float)
    parser.add_argument('--lr_hyp_min', default=0.1, type=float)
    parser.add_argument('--lr_aug', default=0.005, type=float)
    parser.add_argument('--lr_aug_min', default=0.00001, type=float)
    parser.add_argument('--data_augmentation', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_augmentation_marglik', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--save', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--download_data', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--sparse', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--use_wandb', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--config', nargs='+')
    parser.add_argument('--curvature', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--track_kron_la', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--scheduler', default='cos')
    parser.add_argument('--alpha', default=10, type=int)
    parser.add_argument('--shared_uv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--omega_hyper', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--log_weights', default=False, action=argparse.BooleanOptionalAction)
    set_defaults_with_yaml_config(parser, sys.argv)
    args = vars(parser.parse_args())
    args.pop('config')
    if args['use_wandb']:
        import uuid
        import copy
        bound_tag = args['bound'].split('_')[0] if args['bound'] != 'None' else 'nobound'
        tags = [args['dataset'], args['model'], args['approx'], bound_tag]
        config = copy.deepcopy(args)
        config['map'] = (args['n_epochs_burnin'] > args['n_epochs'])
        if config['map']:  # MAP
            tags = [args['dataset'], args['model'], 'map']
        run_name = '-'.join(tags)
        if args['method'] == 'lila':
            run_name += '-lila'
        run_name += '-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(args)))[:4]
        load_dotenv()
        wandb.init(project='sparse', config=config, name=run_name, tags=tags,
                   dir='tmp')
    main(**args)
