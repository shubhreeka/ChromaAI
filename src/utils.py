import warnings

import numpy as np
import torch
from skimage.color import lab2rgb, rgb2lab

warnings.filterwarnings("ignore", category=UserWarning)


def rgb_to_lab_tensors(rgb):
    lab = rgb2lab(rgb).astype(np.float32)
    L = lab[:, :, 0] / 50.0 - 1.0
    ab = lab[:, :, 1:] / 110.0
    return torch.from_numpy(L).unsqueeze(0), torch.from_numpy(ab.transpose(2, 0, 1).copy())


def lab_tensors_to_rgb(L, ab):
    L = L.detach().float().cpu().squeeze().numpy()
    ab = np.clip(ab.detach().float().cpu().numpy(), -1.0, 1.0)
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    lab = np.concatenate([L[None], ab], axis=0).transpose(1, 2, 0).astype(np.float64)
    return np.clip(lab2rgb(lab), 0, 1)
