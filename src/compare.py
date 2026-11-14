import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.engine import evaluate, fit, get_device, make_loaders
from src.models import DISPLAY_NAMES, MODEL_NAMES, build_model

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print("device:", device, flush=True)
    train_loader, val_loader, test_loader = make_loaders(args.image_size, args.batch_size, args.num_workers)
    CKPT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    rows, histories = [], {}
    for name in MODEL_NAMES:
        print(f"\n=== {DISPLAY_NAMES[name]} ===", flush=True)
        torch.manual_seed(args.seed)
        model = build_model(name, args.image_size).to(device)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        start = time.time()
        history, best_val = fit(model, train_loader, val_loader, args.epochs, args.lr, device, CKPT / f"{name}.pt")
        minutes = (time.time() - start) / 60
        metrics = evaluate(model, test_loader, device, args.eval_images)
        rows.append(
            {
                "model": name,
                "display_name": DISPLAY_NAMES[name],
                "params_m": round(params, 2),
                "train_minutes": round(minutes, 2),
                "best_val_loss": round(best_val, 4),
                "test_l1": round(metrics["test_l1"], 4),
                "psnr": round(metrics["psnr"], 3),
                "ssim": round(metrics["ssim"], 4),
            }
        )
        histories[name] = history
        print(rows[-1], flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df["rank_psnr"] = df["psnr"].rank(ascending=False)
    df["rank_ssim"] = df["ssim"].rank(ascending=False)
    df["rank_l1"] = df["test_l1"].rank(ascending=True)
    df["avg_rank"] = (df["rank_psnr"] + df["rank_ssim"] + df["rank_l1"]) / 3
    df = df.sort_values(["avg_rank", "ssim"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(RESULTS / "comparison.csv", index=False)
    (RESULTS / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")

    best = df.iloc[0]["model"]
    shutil.copy(CKPT / f"{best}.pt", CKPT / "best.pt")
    (CKPT / "best.json").write_text(json.dumps({"name": best, "image_size": args.image_size}), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        ax.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], marker="o", label=DISPLAY_NAMES[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation L1 loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(axes, ["psnr", "ssim", "test_l1"], ["PSNR (higher is better)", "SSIM (higher is better)", "L1 on ab (lower is better)"]):
        ax.bar(df["display_name"], df[col], color="#6a8cff")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(RESULTS / "metrics.png", dpi=140)
    plt.close(fig)

    print("\n", df[["display_name", "params_m", "test_l1", "psnr", "ssim", "avg_rank"]].to_string(index=False))
    print("\nBest model:", best)


if __name__ == "__main__":
    main()
