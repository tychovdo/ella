#!/bin/bash

cd ..

python classification_image.py --config configs/cifar10.yaml  --model resnet --batch_size 128 --marglik_batch_size 250 --approx kron --lr 0.01

