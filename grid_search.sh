#!/usr/bin/zsh


for lr in 5e-3 3e-3 5e-4 1e-4
do
    for weight_decay in 1e-4 1e-5 1e-6
    do 
        python train.py --lr $lr --weight_decay $weight_decay
    done
done 