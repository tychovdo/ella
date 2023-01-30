import torch
from torchvision import transforms

cifar_mean = (0.49139968, 0.48215841, 0.44653091)
cifar_std = (0.24703223, 0.24348513, 0.26158784)

CIFAR10_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ]
)

CIFAR10_augment = torch.jit.script(
    torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=[4, ],)
    )
)

MNIST_transform = transforms.ToTensor()
