# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from argparser import arg_parse
import numpy as np
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
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torch.backends.cudnn as cudnn
from model import Net
import os
import torch.nn.functional as F


def main(opt):
    loss_dir = f'{opt.save_dir}/loss'
    image_dir = f'{opt.save_dir}/image'
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    train_data = Loader(root='data/train.txt', num_classes=4)

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size,
                                  shuffle=True, pin_memory=True, num_workers=opt.num_workers, drop_last=True)
    device = f'cuda:{opt.device}'
    model = Net().to(device)
    
    print(f'Train from scratch')
    optimizer = Adam(params=model.parameters(), lr=opt.lr, weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
    loss_list = []
    to_softmax = nn.Softmax(dim=1)

    min_err = 100

    start_epoch = 0
    patience = 0

    # Resume

    # old_state = torch.load('result/5e-4/min_err.pt')
    # print(old_state.keys())
    
    # start_epoch = old_state['Epoch']
    # optimizer.load_state_dict(old_state['optimizer'])ss
    # model.load_state_dict(old_state['model'])

    print(model)

    for e in range(start_epoch, opt.epoch + 1):

        print(f'Start {e} epoch')
        epoch_loss = list()

        for imgs1, imgs2, labels in tqdm(train_dataloader):
            sleep(1e-2)
            optimizer.zero_grad()
            model.train()

            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            labels = labels.to(device)

            pred_vec1 = model(imgs1)
            pred_vec2 = model(imgs2)
            
            normalized_pred_vec1 = F.normalize(pred_vec1, p=2, dim=1)
            normalized_pred_vec2 = F.normalize(pred_vec2, p=2, dim=1)
            similarity = torch.matmul(normalized_pred_vec1, normalized_pred_vec2.T)

            similarity /= 0.1 # temperature
            similarity = to_softmax(similarity)

            labels = torch.mm(labels, labels.T) # positive pairs: same class
            loss_infonce_s = -torch.sum(labels * torch.log(similarity))

            normalized_labels = torch.sum(labels)

            loss_infonce = loss_infonce_s / normalized_labels
            
            epoch_loss.append(loss_infonce.item())

            loss_infonce.backward()

            optimizer.step()
        # scheduler.step()
        loss_list.append((sum(epoch_loss)/len(epoch_loss)))

        print(f'epoch:{e:03d}, loss:{loss_list[-1]:03f}')
        
        if sum(epoch_loss)/len(epoch_loss) < min_err:
            patience = 0
            print("Renew")
            min_err = sum(epoch_loss)/len(epoch_loss)
            state = {
                'Epoch' : e,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
            torch.save(state, osp.join(opt.save_dir, 'min_err.pt'))
        
        else:
            patience += 1
            if patience > 100:
                break
        
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')
    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    opt = arg_parse()
    os.makedirs('result', exist_ok=True)
    opt.save_dir = osp.join('result', opt.save_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    main(opt)
