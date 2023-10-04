# Learning layer-wise equivariances automatically using gradients

Code for NeurIPS 2023 paper: TODO

### Example usage

```
python classification_image.py 
```

### Reproducing experiments

| Experiment  | Script location |
| ------------- | ------------- |
| Toy problem (Sec. 6.1)  | `scripts/toy_problem`  |
| Main CIFAR-10 (Sec. 6.2)  | `scripts/main_cifar10` |
| Multiple groups (Sec. 6.3)  | `scripts/multiple_groups_mnist`, `scripts/multiple_groups_mnist` |

Please get in touch if there are any issues replicating the experiments.

### Note on practical performance

The Differentiable Laplace objective used in this repository performs better than MAP but can be slower in training. Linearised Laplace approximations are an active research topic and are actively being improved. For instance, concurrent work ([Immer, Alexander, et al. "Stochastic marginal likelihood gradients using neural tangent kernels." ICML 2023.](https://github.com/aleximmer/ntk-marglik)) shows Laplace approximations that give a 10x improvement and should be readily applicable to our setting.

### References

The code uses elements from [LILA](https://github.com/tychovdo/lila) (Immer, Alexander, et al. "Invariance learning in deep neural networks with differentiable Laplace approximations." NeurIPS 2022) and [RPP](https://github.com/mfinzi/residual-pathway-priors) (Finzi, Marc, Gregory Benton, and Andrew G. Wilson. "Residual pathway priors for soft equivariance constraints." NeurIPS 2021) repositories.

### Citation

If you build upon this work, please cite us:

```

```
