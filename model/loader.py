from torch.utils.data import Dataset
import os.path as osp
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn.functional as F


color_jitter = T.ColorJitter(brightness=(0.4, 1.4))
data_transforms = T.Compose([   T.Resize((640, 640), antialias=True),
                                T.RandomHorizontalFlip(),
                                T.RandomVerticalFlip(),
                                T.RandomApply([color_jitter], p=0.8),
                                T.RandomApply([T.GaussianBlur(kernel_size=(15,15), sigma=(1.0, 3.0))], p=0.5),
                                T.ToTensor()])


class Loader(Dataset):
    def __init__(self, root, num_classes):
        assert osp.exists(root) == True
        super().__init__()
        self.root = root
        info = open(root, 'r')
        self.info = info.readlines()
        info.close()
        self.num_classes = num_classes
        self.transforms = data_transforms

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_path, label_origin = self.info[idx].strip().split(' ')
        img = Image.open(osp.join(osp.dirname(osp.abspath(self.root)), img_path))
        label_origin = torch.tensor(int(label_origin))
        label = F.one_hot(label_origin, num_classes = self.num_classes).type(torch.float64)
        img1 = self.transforms(img)
        img2 = self.transforms(img)
        return img1, img2, label
    
