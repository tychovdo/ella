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

### Performance

The Differentiable Laplace objective used in this repository performs better than MAP but is ~1.2-2x slower in training. We would like to mention that we rely on modern linearised Laplace approximations, which are currently actively being studied and improved. For instance, concurrent work ([Immer, Alexander, et al. "Stochastic marginal likelihood gradients using neural tangent kernels." ICML 2023.](https://github.com/aleximmer/ntk-marglik)) shows a 10x improvement of the Laplace approximation and should also readily be applicable to our setting.

### References

The code uses elements from [LILA](https://github.com/tychovdo/lila) (Immer, Alexander, et al. "Invariance learning in deep neural networks with differentiable Laplace approximations." NeurIPS 2022) and [RPP](https://github.com/mfinzi/residual-pathway-priors) (Finzi, Marc, Gregory Benton, and Andrew G. Wilson. "Residual pathway priors for soft equivariance constraints." NeurIPS 2021) repositories.

### Citation

If you build upon this work, please cite us:

```

```
