#!/bin/bash

cd ../..

python classification_image.py --config configs/cifar10.yaml --model mixer1 --batch_size 128 --marglik_batch_size 150 --approx kron --lr 0.01 --n_epochs_burnin 1001 --prior_prec 1.0 --n_epochs 1000 --alpha 2 --dataset mnist --download

