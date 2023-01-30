import logging
# from copy import deepcopy
# import warnings
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
import wandb

from laplace import KronLaplace, FunctionalLaplace
from laplace.curvature import AsdlGGN

from sparse.sam import SAM
from sparse.utils import (
    wandb_log_invariance, wandb_log_prior, wandb_log_parameter_norm
)

GB_FACTOR = 1024 ** 3


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device):
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    return log_prior_prec


def valid_performance(model, test_loader, likelihood, criterion, method, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        with torch.no_grad():
            if method in ['lila', 'augerino']:
                f = model(X).mean(dim=1)
            else:
                f = model(X)
        if likelihood == 'classification':
            perf += (torch.argmax(f, dim=-1) == y).sum() / N
        else:
            perf += (f - y).square().sum() / N
        nll += criterion(f, y) / len(test_loader)
    return perf.item(), nll


def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')


def get_model_optimizer(optimizer, model, lr, weight_decay=0, sam=False):
    # NOTE: with adaptive=True sam use rho=2.0
    if optimizer == 'Adam':
        if sam:
            return SAM(model.parameters(), Adam, lr=lr, weight_decay=weight_decay)
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        if sam:
            return SAM(params, SGD, lr=lr, momentum=0.9, weight_decay=weight_decay)
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')


def gradient_to_vector(parameters):
    return parameters_to_vector([e.grad for e in parameters])


def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [e.grad for e in parameters])


def marglik_optimization(model,
                         train_loader,
                         marglik_loader=None,
                         valid_loader=None,
                         partial_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='exp',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         n_hypersteps_prior=1,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         lr_hyp_min=1e-1,
                         lr_aug=1e-2,
                         lr_aug_min=1e-2,
                         laplace=KronLaplace,
                         backend=AsdlGGN,
                         independent=False,
                         single_output=False,
                         kron_jac=True,
                         method='baseline',
                         augmenter=None,
                         stochastic_grad=False,
                         sam=False,
                         sam_with_prior=False,
                         track_kron_la=False,
                         track_bound=False,
                         use_wandb=False):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    likelihood : str
        'classification' or 'regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    method : augmentation strategy, one of ['baseline'] -> no change
        or ['lila'] -> change in protocol.
    augmenter : torch.nn.Module with differentiable parameter
    stochastic_grad : bool
    independent : bool
        whether to use independent functional laplace
    single_output : bool
        whether to use single random output for functional laplace
    kron_jac : bool
        whether to use kron_jac in the backend

    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    if marglik_loader is None:
        marglik_loader = train_loader
    if partial_loader is None:
        partial_loader = marglik_loader
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    optimize_aug = augmenter is not None and parameters_to_vector(augmenter.parameters()).requires_grad
    backend_kwargs = dict(differentiable=(stochastic_grad and optimize_aug) or laplace is FunctionalLaplace, 
                          kron_jac=kron_jac)
    la_kwargs = dict(sod=stochastic_grad, single_output=single_output)
    if laplace is FunctionalLaplace:
        la_kwargs['independent'] = independent
    if use_wandb:
        wandb.config.update(dict(n_params=P, n_param_groups=H, n_data=N))

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr, sam=sam)
    scheduler = get_scheduler(scheduler, optimizer.base_optimizer if sam else optimizer,
                              train_loader, n_epochs, lr, lr_min)

    n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * n_hypersteps_prior
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)
    if optimize_aug:
        logging.info('MARGLIK: optimize augmentation.')
        aug_optimizer = Adam(augmenter.parameters(), lr=lr_aug)
        n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * (n_hypersteps if stochastic_grad else 1)
        aug_scheduler = CosineAnnealingLR(aug_optimizer, n_steps, eta_min=lr_aug_min)
        aug_history = [parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy()]

    best_marglik = np.inf
    # best_model_dict = None
    best_precision = None
    losses = list()
    valid_perfs = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict()

        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / temperature / (2 * sigma_noise.square())
            else:
                crit_factor = 1 / temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)
            if method == 'lila':
                f = model(X).mean(dim=1)
            else:
                f = model(X)

            if sam:
                # 1. step
                loss = criterion(f, y)
                if sam_with_prior:
                    theta = parameters_to_vector(model.parameters())
                    loss += (0.5 * (delta * theta) @ theta) / N / crit_factor
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # 2. step
                if method == 'lila':
                    f = model(X).mean(dim=1)
                else:
                    f = model(X)
                theta = parameters_to_vector(model.parameters())
                loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                theta = parameters_to_vector(model.parameters())
                loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
                loss.backward()
                optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            epoch_nll += criterion(f.detach(), y).item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()

        losses.append(epoch_loss)
        logging.info('MAP memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/GB_FACTOR) + ' Gb.')
        logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf*100:.2f}%; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        if use_wandb and ((epoch % 5) == 0):
            wandb_log_parameter_norm(model)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                val_perf, val_nll = valid_performance(model, valid_loader, likelihood, criterion, method, device)
                valid_perfs.append(val_perf)
                logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf*100:.2f}%; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})

        # track kfac laplace marglik
        if track_kron_la and (epoch % 10 == 0):
            sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise.detach())
            prior_prec = torch.exp(log_prior_prec.detach())
            lap = KronLaplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                              temperature=1.0, backend=backend, backend_kwargs=dict(differentiable=False))
            lap.fit(train_loader)
            kron_marglik = -lap.log_marginal_likelihood().item() / N
            logging.info(f'MARGLIK[epoch={epoch}]: KFAC marglik {kron_marglik:.5f}.')
            epoch_log['train/marglik_kron'] = kron_marglik

        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            if use_wandb:
                wandb.log(epoch_log, step=epoch, commit=((epoch % 10) == 0))
            print(epoch, marglik_frequency, n_epochs_burnin)
            continue

        # optimizer hyperparameters by differentiating marglik
        # 1. fit laplace approximation
        torch.cuda.empty_cache()
        if optimize_aug:
            if stochastic_grad:  # differentiable
                marglik_loader.attach()
            else:  # jvp
                marglik_loader.detach()

        # first optimize prior precision jointly with direct marglik grad
        margliks_local = list()
        n_hyper = max(n_hypersteps_prior, n_hypersteps) if stochastic_grad else n_hypersteps_prior
        for i in range(n_hyper):
            if i == 0 or stochastic_grad:
                sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
                prior_prec = torch.exp(log_prior_prec)
                lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                              temperature=temperature, backend=backend, backend_kwargs=backend_kwargs,
                              **la_kwargs)
                lap.fit(marglik_loader)
            if i < n_hypersteps and optimize_aug and stochastic_grad:
                aug_optimizer.zero_grad()
            if i < n_hypersteps_prior:
                hyper_optimizer.zero_grad()
            if i < n_hypersteps_prior and not stochastic_grad:  # does not fit every it
                sigma_noise = None if likelihood == 'classification' else torch.exp(log_sigma_noise)
                prior_prec = torch.exp(log_prior_prec)
                marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / N
            else:  # fit with updated hparams
                marglik = -lap.log_marginal_likelihood() / N
            marglik.backward()
            margliks_local.append(marglik.item())
            if i < n_hypersteps_prior:
                hyper_optimizer.step()
                hyper_scheduler.step()
            if i < n_hypersteps and optimize_aug and stochastic_grad:
                aug_optimizer.step()
                aug_scheduler.step()

        if stochastic_grad:
            marglik = np.mean(margliks_local)
        else:
            marglik = margliks_local[-1]

        if stochastic_grad and track_bound:
            # TODO: implement as a passed function!
            original_subset_size = marglik_loader.subset_size
            for s in range(10, marglik_loader.subset_size+1, 10):
                np.random.seed(711)
                marglik_loader.subset_size = s
                marglik_loader.data_factor = N / s
                prior_prec = torch.exp(log_prior_prec.detach())
                lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                              temperature=temperature, backend=backend, backend_kwargs=backend_kwargs,
                              **la_kwargs)
                lap.fit(marglik_loader)
                lml = - lap.log_marginal_likelihood()
                logging.info(f'SoD marglik for subset size {s} is {lml.item():.5f}')
            marglik_loader.subset_size = original_subset_size
            marglik_loader.data_factor = N / original_subset_size

        if use_wandb:
            wandb_log_prior(torch.exp(log_prior_prec.detach()), prior_structure, model)
        if likelihood == 'regression':
            epoch_log['hyperparams/sigma_noise'] = torch.exp(log_sigma_noise.detach()).cpu().item()
        epoch_log['train/marglik'] = marglik
        logging.info('LA memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/GB_FACTOR) + ' Gb.')

        # option 2: jvp (not direct_grad)
        torch.cuda.empty_cache()
        if optimize_aug and not stochastic_grad:  # accumulate gradient with JVP
            partial_loader.attach()
            aug_grad = torch.zeros_like(parameters_to_vector(augmenter.parameters()))
            lap.backend.differentiable = True
            if isinstance(lap, KronLaplace):
                # does the inversion internally
                hess_inv = lap.posterior_precision.jvp_logdet()
            else:
                hess_inv = lap.posterior_covariance.flatten()
            for i, (X, y) in zip(range(n_hypersteps), partial_loader):
                lap.loss, H_batch = lap._curv_closure(X, y, N)
                # curv closure creates gradient already, need to zero
                aug_optimizer.zero_grad()
                # compute grad wrt. neg. log-lik
                (- lap.log_likelihood).backward(inputs=list(augmenter.parameters()), retain_graph=True)
                # compute grad wrt. log det = 0.5 vec(P_inv) @ (grad-vec H)
                (0.5 * H_batch.flatten()).backward(gradient=hess_inv, inputs=list(augmenter.parameters()))
                aug_grad = (aug_grad + gradient_to_vector(augmenter.parameters()).data.clone())

            lap.backend.differentiable = False

            vector_to_gradient(aug_grad, augmenter.parameters())
            aug_optimizer.step()
            aug_scheduler.step()

        if optimize_aug:
            aug_history.append(parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy())
            logging.info(f'Augmentation params epoch {epoch}: {aug_history[-1]}')
            if use_wandb:
                wandb_log_invariance(augmenter)

        logging.info('LA memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/GB_FACTOR) + ' Gb.')

        margliks.append(marglik)
        del lap
        if use_wandb:
            if optimize_aug:
                epoch_log['train/lr_aug'] = aug_scheduler.get_last_lr()[0]
            epoch_log['train/lr_hyp'] = hyper_scheduler.get_last_lr()[0]
            wandb.log(epoch_log, step=epoch, commit=((epoch % 10) == 0))

        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            best_marglik = margliks[-1]
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.5f}, prec: {prior_prec.detach().mean().item():.2f}.')
            # model.cpu()
            # best_model_dict = deepcopy(model.state_dict())
            # model.to(device)
            # best_precision = deepcopy(prior_prec.detach())
            # best_sigma = 1 if likelihood == 'classification' else deepcopy(sigma_noise.detach())
            # if optimize_aug:
            #     best_augmenter = deepcopy(augmenter.state_dict())
        else:
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.5f}, prec: {prior_prec.detach().mean().item():.2f}.')

    # if best_model_dict is not None:
    #     model.load_state_dict(best_model_dict)
    #     sigma_noise = best_sigma
    #     prior_prec = best_precision
    #     if optimize_aug:
    #         augmenter.load_state_dict(best_augmenter)
    sigma_noise = 1 if sigma_noise is None else sigma_noise
    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend, backend_kwargs=backend_kwargs,
                  **la_kwargs)
    lap.fit(marglik_loader.detach())
    if optimize_aug:
        return lap, model, margliks, valid_perfs, aug_history
    return lap, model, margliks, valid_perfs, None


# def stochastic_marglik_optimization(model,
#                                     train_loader,
#                                     marglik_loader=None,
#                                     valid_loader=None,
#                                     likelihood='classification',
#                                     prior_structure='layerwise',
#                                     prior_prec_init=1.,
#                                     sigma_noise_init=1.,
#                                     temperature=1.,
#                                     n_epochs=500,
#                                     lr=1e-3,
#                                     lr_min=None,
#                                     optimizer='Adam',
#                                     scheduler='exp',
#                                     n_epochs_burnin=0,
#                                     n_hypersteps=None,
#                                     marglik_frequency=1,
#                                     interleaved=False,
#                                     joint_update=False,
#                                     lr_hyp=1e-1,
#                                     lr_hyp_min=1e-1,
#                                     lr_aug=1e-2,
#                                     lr_aug_min=1e-2,
#                                     backend=AsdlGGN,
#                                     differentiable=True,
#                                     method='baseline',
#                                     augmenter=None,
#                                     sam=False,
#                                     sam_with_prior=False):
#     """
#     if interleaved, optimize marginal likelihood together with training on train_loader, otherwise as
#     in `marglik_optimization` after every `marglik_frequency_epoch` epoch.
#     if joint_update, then network parameters and hyperparameters are updated jointly."""
#     # Most code just copied from `marglik_optimization`, should make this modular in the future
#     if lr_min is None:  # don't decay lr
#         lr_min = lr
#     if marglik_loader is None:
#         marglik_loader = train_loader
#     else:
#         if interleaved:
#             raise ValueError('Cannot do interleaved optimization with custom loader.')
#     if n_hypersteps is None or n_hypersteps == -1:  # default to all batches of marglik_loader
#         n_hypersteps = len(marglik_loader)
#     if n_hypersteps > len(marglik_loader):
#         warnings.warn(f'Reducing n_hypersteps {n_hypersteps} to maximum {len(marglik_loader)}.')

#     device = parameters_to_vector(model.parameters()).device
#     N = len(train_loader.dataset)
#     H = len(list(model.parameters()))
#     P = len(parameters_to_vector(model.parameters()))
#     backend_kwargs = dict(differentiable=differentiable)

#     # differentiable hyperparameters
#     hyperparameters = list()
#     # prior precision
#     log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
#     hyperparameters.append(log_prior_prec)

#     # set up loss (and observation noise hyperparam)
#     if likelihood == 'classification':
#         criterion = CrossEntropyLoss(reduction='mean')
#         sigma_noise = 1.
#     elif likelihood == 'regression':
#         criterion = MSELoss(reduction='mean')
#         log_sigma_noise_init = np.log(sigma_noise_init)
#         log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
#         log_sigma_noise.requires_grad = True
#         hyperparameters.append(log_sigma_noise)

#     # set up model optimizer and scheduler
#     optimizer = get_model_optimizer(optimizer, model, lr, sam)
#     scheduler = get_scheduler(scheduler, optimizer.base_optimizer if sam else optimizer,
#                               train_loader, n_epochs, lr, lr_min)

#     # set up hyperparameter optimizer
#     n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * n_hypersteps
#     hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
#     hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)
#     optimize_aug = augmenter is not None and parameters_to_vector(augmenter.parameters()).requires_grad
#     if optimize_aug:
#         logging.info('MARGLIK: optimize augmentation.')
#         aug_optimizer = Adam(augmenter.parameters(), lr=lr_aug)
#         aug_scheduler = CosineAnnealingLR(aug_optimizer, n_steps, eta_min=lr_aug_min)
#         aug_history = [parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy()]

#     best_marglik = np.inf
#     best_model_dict = None
#     best_precision = None
#     losses = list()
#     margliks = list()
#     valid_perfs = list()

#     # this is the worst code every, I'm ashamed...
#     for epoch in range(1, n_epochs + 1):
#         epoch_loss = 0
#         epoch_perf = 0
#         if interleaved:
#             epoch_marglik = 0

#         compute_marglik = (epoch % marglik_frequency) == 0 and epoch >= n_epochs_burnin
#         # TODO: remove the individual step stuff and just do one after the other.
#         # TODO: add sam update as above
#         # TODO: can we merge this then with above marglik optimization code?

#         for i, (X, y) in enumerate(train_loader):
#             # normalizing loss for stability
#             if likelihood == 'regression':
#                 sigma_noise = torch.exp(log_sigma_noise).detach()
#                 crit_factor = 1 / temperature / (2 * sigma_noise.square())
#             else:
#                 crit_factor = 1 / temperature

#             if interleaved:
#                 if joint_update:
#                     warnings.warn('interleaved joint updated not maintained; might be wrong.')
#                     # jointly update theta and hyperparams, each one step
#                     X, y = X.to(device), y.to(device)
#                     sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
#                     prior_prec = torch.exp(log_prior_prec)
#                     lap = FunctionalLaplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
#                                             temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
#                     lap.fit_batch(X, y, N)
#                     optimizer.zero_grad()
#                     hyper_optimizer.zero_grad()
#                     if optimize_aug:
#                         aug_optimizer.zero_grad()

#                     # Take the negative of the following to quantities to backprop
#                     # log_joint = log_lik - 0.5 * scatter
#                     # log_marglik = log_joint - 0.5 * log_det_ratio  see BaseLA
#                     # first compute and backward log-joint
#                     log_joint = (0.5 * lap.scatter - lap.log_likelihood)
#                     log_joint.backward(create_graph=True, retain_graph=True)
#                     grad = gradient_to_vector(model.parameters()).detach().clone() / N / crit_factor
#                     log_det_ratio = 0.5 * lap.log_det_ratio
#                     log_det_ratio.backward()
#                     vector_to_gradient(grad, model.parameters())
#                     optimizer.step()
#                     hyper_optimizer.step()
#                     hyper_scheduler.step()
#                     if optimize_aug:
#                         aug_optimizer.step()
#                         aug_scheduler.step()
#                         aug_history.append(parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy())

#                     epoch_loss += log_joint.cpu().item() / len(train_loader)
#                     epoch_marglik += (log_joint + log_det_ratio).cpu().item() / len(train_loader)
#                 else:  # separate update
#                     # compute and backward log joint first
#                     Xd, y = X.detach().to(device), y.to(device)
#                     optimizer.zero_grad()
#                     prior_prec = torch.exp(log_prior_prec).detach()
#                     theta = parameters_to_vector(model.parameters())
#                     delta = expand_prior_precision(prior_prec, model)
#                     if method == 'avgfunc':
#                         f = model(Xd).mean(dim=1)
#                     else:
#                         f = model(Xd)
#                     loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
#                     loss.backward()
#                     optimizer.step()
#                     epoch_loss += loss.cpu().item() / len(train_loader)

#                     if compute_marglik and i < n_hypersteps:
#                         # now compute log_marglik and backward (after params have been updated)
#                         sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
#                         prior_prec = torch.exp(log_prior_prec)
#                         lap = FunctionalLaplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
#                                                 temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
#                         lap.fit_batch(X.to(device), y, N)
#                         hyper_optimizer.zero_grad()
#                         if optimize_aug:
#                             aug_optimizer.zero_grad()
#                         marglik = -lap.log_marginal_likelihood()
#                         marglik.backward()
#                         hyper_optimizer.step()
#                         hyper_scheduler.step()
#                         if optimize_aug:
#                             aug_optimizer.step()
#                             aug_scheduler.step()
#                             aug_history.append(parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy())

#                         epoch_marglik += marglik.cpu().item() / n_hypersteps

#             else:  # not interleaved
#                 # just update NN parameters, nothing else
#                 X, y = X.detach().to(device), y.to(device)
#                 optimizer.zero_grad()
#                 prior_prec = torch.exp(log_prior_prec).detach()
#                 theta = parameters_to_vector(model.parameters())
#                 delta = expand_prior_precision(prior_prec, model)
#                 if method == 'avgfunc':
#                     f = model(X).mean(dim=1)
#                 else:
#                     f = model(X)
#                 loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.cpu().item() / len(train_loader)

#             # aggregate val loss per batch
#             if likelihood == 'regression':
#                 epoch_perf += (f.detach() - y).square().sum() / N
#             else:
#                 epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
#             scheduler.step()

#         logging.info(f'MARGLIK[epoch={epoch}]: training performance {epoch_perf*100:.2f}%.')
#         gb_factor = 1024 ** 3
#         logging.info('Max memory allocated: ' + str(torch.cuda.max_memory_allocated(loss.device)/gb_factor) + ' Gb.')
#         optimizer.zero_grad(set_to_none=True)
#         torch.cuda.empty_cache()

#         if not interleaved and compute_marglik:  # compute marglik after epoch of training
#             epoch_marglik = 0
#             for i, (X, y) in enumerate(marglik_loader):
#                 if i >= n_hypersteps:
#                     break
#                 X, y = X.to(device), y.to(device)
#                 prior_prec = torch.exp(log_prior_prec)
#                 sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
#                 lap = FunctionalLaplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
#                                         temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
#                 lap.fit_batch(X, y, N)
#                 hyper_optimizer.zero_grad()
#                 if optimize_aug:
#                     aug_optimizer.zero_grad()
#                 marglik = -lap.log_marginal_likelihood()
#                 marglik.backward()
#                 hyper_optimizer.step()
#                 hyper_scheduler.step()
#                 if optimize_aug:
#                     aug_optimizer.step()
#                     aug_scheduler.step()
#                     aug_history.append(parameters_to_vector(augmenter.parameters()).squeeze().detach().cpu().numpy())
#                 epoch_marglik += marglik.cpu().item() / n_hypersteps

#             if optimize_aug:
#                 logging.info(f'Augmentation params epoch {epoch}: {aug_history[-1]}')

#         # compute validation error to report during training
#         if valid_loader is not None:
#             with torch.no_grad():
#                 valid_perf = valid_performance(model, valid_loader, likelihood, method, device)
#                 valid_perfs.append(valid_perf)
#                 logging.info(f'Validation performance {valid_perf*100:.2f}.%')

#         losses.append(epoch_loss)
#         if compute_marglik:
#             margliks.append(epoch_marglik)
#         else:
#             continue

#         # early stopping on marginal likelihood
#         if margliks[-1] < best_marglik:
#             best_model_dict = deepcopy(model.state_dict())
#             best_precision = deepcopy(prior_prec.detach())
#             best_sigma = 1 if likelihood == 'classification' else deepcopy(sigma_noise.detach())
#             best_marglik = margliks[-1]
#             if optimize_aug:
#                 best_augmenter = deepcopy(augmenter.state_dict())
#             logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}, prec: {best_precision.mean().item():.2f}. '
#                          + 'Saving new best model.')
#         else:
#             logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}, prec: {prior_prec.mean().item():.2f}. '
#                          + f'No improvement over {best_marglik:.2f}')

#     logging.info(f'MARGLIK: finished training. Recover best model and fit Lapras. Marglik={best_marglik:.2f}')
#     if best_model_dict is not None:
#         model.load_state_dict(best_model_dict)
#         sigma_noise = best_sigma
#         prior_prec = best_precision
#         if optimize_aug:
#             augmenter.load_state_dict(best_augmenter)
#     if P <= 10000:
#         lap = FullLaplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
#                           temperature=temperature, backend=backend, backend_kwargs=dict(differentiable=False))
#     else:
#         lap = KronLaplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
#                           temperature=temperature, backend=backend, backend_kwargs=dict(differentiable=False))
#     lap.fit(marglik_loader.detach())
#     if optimize_aug:
#         return lap, model, margliks, valid_perfs, aug_history
#     return lap, model, margliks, valid_perfs, None
