#!/usr/bin/zsh

lr=1e-3
epoch=2000
batch=128
num_workers=8
device=1

echo "Current LR: $lr"
echo "Current epoch: $epoch"
echo "Current batch: $batch"
echo "Current device: $device"

python simCLR.py --lr $lr -b $batch -e $epoch --num_workers $num_workers --save_dir $lr --device $device
