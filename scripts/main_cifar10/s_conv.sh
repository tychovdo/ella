#!/bin/bash

cd ../..

python classification_image.py --config configs/cifar10.yaml --model s_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 4001 --prior_prec 1.0 --n_epochs 4000 --download
python classification_image.py --config configs/cifar10.yaml --model s_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 4001 --prior_prec 1.0 --n_epochs 4000 --data_augmentation --download

python classification_image.py --config configs/cifar10.yaml --model s_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 4000 --download
python classification_image.py --config configs/cifar10.yaml --model s_conv --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 10 --prior_prec 1.0 --n_epochs 4000 --data_augmentation --download

