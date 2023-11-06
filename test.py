from ultralytics import YOLO
import torch
import torch.nn as nn


# Load a model
# img = torch.ones_like(torch.Tensor(1, 3, 640, 640))
# print(f'Input shape : {img.shape}')

param = torch.load('model/3_95.pt')

from collections import OrderedDict

new_state_dict = OrderedDict()

for k, v in param['model'].items():
    if 'backbone' in k:
        new_state_dict[k[9:]] = v

model = YOLO('best.pt') # build from YAML and transfer weights

cnt = 0

for name, param in model.state_dict().items():
    # name : 'model.model.' ~
    # param.keys() : 'backbone.' ~
    if name[12:] in new_state_dict.keys():
        print(f'before: {torch.sum(torch.ne(param, new_state_dict[name[12:]].cpu())).item()}')
        param = new_state_dict[name[12:]]
        print(f'after: {torch.sum(torch.ne(param, new_state_dict[name[12:]])).item()}')
        cnt += 1

print(cnt)
print(len(new_state_dict.keys()))

exit()
