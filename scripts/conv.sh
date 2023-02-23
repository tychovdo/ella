#!/bin/bash

cd ..

# python classification_image.py --config configs/cifar10.yaml --model d_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 201 --prior_prec 1.0 --n_epochs 200
# python classification_image.py --config configs/cifar10.yaml --model d_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 201 --prior_prec 1.0 --n_epochs 200 --data_augmentation

python classification_image.py --config configs/cifar10.yaml --model d_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 200
python classification_image.py --config configs/cifar10.yaml --model d_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 200 --data_augmentation

#--

# python classification_image.py --config configs/cifar10.yaml --model d_conv_b --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 201 --prior_prec 1.0 --n_epochs 200
# python classification_image.py --config configs/cifar10.yaml --model d_conv_b --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 201 --prior_prec 1.0 --n_epochs 200 --data_augmentation

python classification_image.py --config configs/cifar10.yaml --model d_conv_b --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 200
python classification_image.py --config configs/cifar10.yaml --model d_conv_b --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 200 --data_augmentation


