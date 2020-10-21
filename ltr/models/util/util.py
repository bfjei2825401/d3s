import torch
import numpy as np


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0] - 1) and \
            all(roi[:, 4] <= image_size[1] - 1)
    return valid


def normalize_vis_img(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)
