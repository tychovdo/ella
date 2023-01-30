#!/bin/bash

cd ..

python classification_image.py --config configs/cifar10.yaml --dataset cifar10_r180 --model mlp --batch_size 250 --marglik_batch_size 125 --partial_batch_size 50 --approx kron --lr 0.01 --method lila --n_samples_aug 20 --download --n_hypersteps_prior 4 --n_hypersteps 100 --sparse --lr_aug 0.01


