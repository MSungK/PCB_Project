import torch.nn as nn
import torch
from ultralytics import YOLO
import torchvision.transforms as T


class Net(nn.Module):
    
    def __init__(self, input_shape=(640, 640)):
        super().__init__()
        origin = YOLO('/workspace/Minsung/PCB_Project/yolov8m.pt') # build from YAML and transfer weights
        model_children_list = list(origin.model.children())
        self.backbone = model_children_list[0][:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.project = nn.Sequential(nn.Linear(576, 288, bias=True), nn.ReLU(), nn.Linear(288, 144, bias=True))
    
    def forward(self, img):
        h = self.backbone(img)

        h_flatten = self.flatten(self.avg_pool(h))

        pred_vec = self.project(h_flatten)

        return pred_vec