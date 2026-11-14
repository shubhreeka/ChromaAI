from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ColorizationDataset
from src.utils import lab_tensors_to_rgb

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(image_size, batch_size, num_workers):
    sets = {
        k: ColorizationDataset(SPLITS / f"{k}.txt", image_size, augment=(k == "train"))
        for k in ("train", "val", "test")
    }
    print({k: len(v) for k, v in sets.items()}, flush=True)
    kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train = DataLoader(sets["train"], batch_size, shuffle=True, drop_last=True, **kw)
    val = DataLoader(sets["val"], batch_size, shuffle=False, **kw)
    test = DataLoader(sets["test"], batch_size, shuffle=False, **kw)
    return train, val, test


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for L, ab in tqdm(loader, desc=desc, leave=False):
            L = L.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(L)
            loss = criterion(pred.float(), ab)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * L.size(0)
            count += L.size(0)
    return total / count


def fit(model, train_loader, val_loader, epochs, lr, device, ckpt_path, best_val=float("inf")):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, f"train {epoch}/{epochs}")
        va = run_epoch(model, val_loader, criterion, device, desc=f"val {epoch}/{epochs}")
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        mark = ""
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), ckpt_path)
            mark = "  *"
        print(f"epoch {epoch}/{epochs}  train={tr:.4f}  val={va:.4f}{mark}", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return history, best_val


@torch.no_grad()
def evaluate(model, loader, device, max_images=300):
    model.eval()
    l1, psnr, ssim, count = 0.0, [], [], 0
    for L, ab in loader:
        L = L.to(device)
        ab = ab.to(device)
        pred = model(L)
        l1 += torch.abs(pred - ab).mean().item() * L.size(0)
        for i in range(L.size(0)):
            true_rgb = lab_tensors_to_rgb(L[i], ab[i])
            pred_rgb = lab_tensors_to_rgb(L[i], pred[i])
            psnr.append(peak_signal_noise_ratio(true_rgb, pred_rgb, data_range=1.0))
            ssim.append(structural_similarity(true_rgb, pred_rgb, channel_axis=2, data_range=1.0))
        count += L.size(0)
        if count >= max_images:
            break
    return {"test_l1": l1 / count, "psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}
