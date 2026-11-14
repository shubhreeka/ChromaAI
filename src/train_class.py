import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.engine import evaluate, get_device, make_loaders
from src.models import BINS_PER_AXIS, NUM_BINS, ColorClassifier

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def to_classes(ab):
    idx = torch.clamp(torch.floor((ab * 110.0 + 110.0) / 10.0), 0, BINS_PER_AXIS - 1).long()
    return idx[:, 0] * BINS_PER_AXIS + idx[:, 1]


def class_weights(loader):
    counts = torch.zeros(NUM_BINS, dtype=torch.long)
    for _, ab in tqdm(loader, desc="color prior", leave=False):
        counts += torch.bincount(to_classes(ab).flatten(), minlength=NUM_BINS)
    p = counts.numpy().astype(np.float64)
    p /= p.sum()
    p = gaussian_filter(p.reshape(BINS_PER_AXIS, BINS_PER_AXIS), sigma=2).reshape(-1)
    p /= p.sum()
    w = 1.0 / (0.5 * p + 0.5 / NUM_BINS)
    w /= (p * w).sum()
    return torch.tensor(w, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-images", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    train_loader, val_loader, test_loader = make_loaders(args.image_size, args.batch_size, args.num_workers)

    weights = class_weights(train_loader).to(device)
    model = ColorClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    CKPT.mkdir(exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for L, ab in tqdm(train_loader, desc=f"class {epoch}/{args.epochs}", leave=False):
            L = L.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model.net(L)
            loss = criterion(logits.float(), to_classes(ab))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * L.size(0)
            n += L.size(0)
        scheduler.step()

        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for L, ab in val_loader:
                L = L.to(device)
                ab = ab.to(device)
                val_total += criterion(model.net(L).float(), to_classes(ab)).item() * L.size(0)
                val_n += L.size(0)
        val_loss = val_total / val_n
        mark = ""
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), CKPT / "resnet18_class.pt")
            mark = "  *"
        print(f"epoch {epoch}/{args.epochs}  train={total / n:.3f}  val={val_loss:.3f}{mark}", flush=True)

    model.load_state_dict(torch.load(CKPT / "resnet18_class.pt", map_location=device))
    print(evaluate(model, test_loader, device, args.eval_images))


if __name__ == "__main__":
    main()
