import argparse
import json
from pathlib import Path

import torch

import torch.nn as nn

from src.engine import evaluate, fit, get_device, make_loaders, run_epoch
from src.models import build_model

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-images", type=int, default=500)
    args = parser.parse_args()

    device = get_device()
    info = json.loads((CKPT / "best.json").read_text(encoding="utf-8"))
    train_loader, val_loader, test_loader = make_loaders(info["image_size"], args.batch_size, args.num_workers)

    model = build_model(info["name"], info["image_size"]).to(device)
    model.load_state_dict(torch.load(CKPT / "best.pt", map_location=device))

    start_val = run_epoch(model, val_loader, nn.L1Loss(), device)
    print("starting val loss:", round(start_val, 4), flush=True)

    fit(model, train_loader, val_loader, args.epochs, args.lr, device, CKPT / "best.pt", best_val=start_val)
    metrics = evaluate(model, test_loader, device, args.eval_images)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "final_metrics.json").write_text(json.dumps({"model": info["name"], **metrics}, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
