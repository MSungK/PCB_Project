import os
import os.path as osp


def normalize(norm, x1, y1, x2, y2):
    x_center = ((x2 + x1) / 2) / norm
    y_center = ((y1 + y2) / 2) / norm
    width = (x2 - x1) / norm
    height = (y2 - y1) / norm
    return x_center, y_center, width, height



