from argparser import arg_parse
import numpy as np
import utils
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import torch.nn as nn
from loader import Loader
import torchvision.transforms as T
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import sleep
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torch.backends.cudnn as cudnn
from model import Net
import utils
import os




model = Net()
train_data = Loader(root='data/train.txt', num_classes=4)
train_dataloader = DataLoader(train_data, batch_size=64,
                              shuffle=False, pin_memory=True, num_workers=8, drop_last=True)
for imgs, labels in train_dataloader:
    print(imgs.shape)
    print(labels.shape)

