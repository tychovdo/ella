# ELLA (Equivariance Learning with Laplace Approximations)

This code accompanies our NeurIPS 2023 spotlight paper:

[Learning Layer-wise Equivariances Automatically using Gradients](https://openreview.net/pdf?id=bNIHdyunFC) </br>
Tycho F. A. van der Ouderaa, Alexander Immer, and Mark van der Wilk.

### Example Usage

For example, to learn layer-wise translational equivariances with sparsified parameterisation (S-FC + S-CONV) we can run:

```
python classification_image.py --config configs/cifar10.yaml --model s_both --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 4000 --data_augmentation --download
```

The code uses `wandb` to log experiments (which requires setting up a [weights and biases](https://wandb.ai/site) account)

### Reproducing experiments

Scripts for experiments performed in the paper can be found here:

| Experiment  | Script location |
| ------------- | ------------- |
| Toy problem (Sec. 6.1)  | `scripts/toy_problem`  |
| Main CIFAR-10 (Sec. 6.2)  | `scripts/main_cifar10` |
| Multiple groups (Sec. 6.3)  | `scripts/multiple_groups_mnist`, `scripts/multiple_groups_mnist` |

Please get in touch if there are any questions about the code.

### Setup and dependencies

This project uses patched versions of [Laplace](https://github.com/aleximmer/Laplace/tree/main) and [ASDL](https://github.com/kazukiosawa/asdl) libraries. For experiments, we use python 3.9. Dependencies can be installed by running:

```
pip install -r requirements.txt
cd dependencies/asdl && pip install . && cd ../..
cd dependencies/laplace && pip install . && cd ../..
```

###### Note on practical performance

The Differentiable Laplace objective used in this repository performs better than MAP but can be slower in training. Linearised Laplace approximations are an active research topic and are actively being improved. For instance, concurrent work ([Immer, Alexander, et al. "Stochastic marginal likelihood gradients using neural tangent kernels." ICML 2023.](https://github.com/aleximmer/ntk-marglik)) shows Laplace approximations that give a 10x improvement and should be readily applicable to our setting.

### References

The code uses elements from [LILA](https://github.com/tychovdo/lila) (Immer, Alexander, et al. "Invariance learning in deep neural networks with differentiable Laplace approximations." NeurIPS 2022) and [RPP](https://github.com/mfinzi/residual-pathway-priors) (Finzi, Marc, Gregory Benton, and Andrew G. Wilson. "Residual pathway priors for soft equivariance constraints." NeurIPS 2021) repositories.

