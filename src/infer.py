import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.color import lab2rgb, rgb2lab

from src.models import MODEL_NAMES, build_model

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def best_info():
    path = CKPT / "best.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def available_models():
    return [n for n in MODEL_NAMES + ["resnet18_gan", "resnet18_class"] if (CKPT / f"{n}.pt").exists()]


def load_model(name, size, device):
    model = build_model(name, size, pretrained=False)
    model.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_ab(image, model, size, device, max_side=1024):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    w, h = image.size
    full = np.asarray(image).astype(np.float32) / 255.0
    L_full = rgb2lab(full)[:, :, 0]
    small = image.convert("L").convert("RGB").resize((size, size), Image.BICUBIC)
    L_small = rgb2lab(np.asarray(small).astype(np.float32) / 255.0)[:, :, 0]
    x = torch.from_numpy((L_small / 50.0 - 1.0).astype(np.float32))[None, None].to(device)
    ab = model(x)
    ab = F.interpolate(ab, size=(h, w), mode="bilinear", align_corners=False)[0].cpu().numpy() * 110.0
    return L_full, ab


def compose(L_full, ab, vibrance=1.0):
    ab = np.clip(ab * vibrance, -110.0, 110.0)
    lab = np.concatenate([L_full[None], ab], axis=0).transpose(1, 2, 0).astype(np.float64)
    rgb = np.clip(lab2rgb(lab), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def timed_predict(image, model, size, device, max_side=1024):
    start = time.time()
    L, ab = predict_ab(image, model, size, device, max_side)
    return L, ab, time.time() - start
