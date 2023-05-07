#!/bin/bash

cd ../..

# MAP
# python classification_image.py --dataset quadrant_mnist --config configs/cifar10.yaml --model d_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 1001 --prior_prec 1.0 --n_epochs 999 --download
# python classification_image.py --dataset quadrant_mnist --config configs/cifar10.yaml --model f_rpp --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 1001 --prior_prec 1.0 --n_epochs 999
# python classification_image.py --dataset quadrant_mnist --config configs/cifar10.yaml --model d_fc --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 1001 --prior_prec 1.0 --n_epochs 999 --download

# Diff
# python classification_image.py --dataset quadrant_mnist --config configs/cifar10.yaml --model d_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 999
python classification_image.py --dataset quadrant_mnist --config configs/cifar10.yaml --model f_rpp --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 999
python classification_image.py --dataset quadrant_mnist --config configs/cifar10.yaml --model d_fc --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 999


