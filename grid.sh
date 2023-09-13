#!/usr/bin/zsh

for lr in 1 2
do
    python train.py -c $lr
done