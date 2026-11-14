from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import rgb_to_lab_tensors


class ColorizationDataset(Dataset):
    def __init__(self, split_file, image_size=128, augment=False):
        text = Path(split_file).read_text(encoding="utf-8")
        self.paths = [Path(p) for p in text.splitlines() if p.strip()]
        steps = [transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC)]
        if augment:
            steps.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        return rgb_to_lab_tensors(rgb)
