#!/usr/bin/zsh

for lr in 1e-3 3e-3 5e-3 7e-3 1e-4 3e-4 5e-4 7e-4
do
echo "Current LR: $lr"
python simCLR.py --lr $lr -b 128 -e 100 --num_workers 12 --save_dir $lr --device 3
done