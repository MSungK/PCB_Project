from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Load a model

    model = YOLO('runs/detect/train/weights/best.pt') # build from YAML and transfer weights
    model.val()
