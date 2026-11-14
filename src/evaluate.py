import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.engine import evaluate, get_device, make_loaders
from src.infer import available_models, best_info, load_model
from src.models import DISPLAY_NAMES

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CLASSES = ["Neutral", "Red", "Yellow", "Lime", "Green", "Blue", "Purple"]
K = len(CLASSES)


def classify(ab):
    a = ab[:, 0] * 110.0
    b = ab[:, 1] * 110.0
    chroma = torch.sqrt(a * a + b * b)
    angle = torch.rad2deg(torch.atan2(b, a))
    sector = torch.floor(((angle + 30.0) % 360.0) / 60.0).long().clamp(0, 5) + 1
    return torch.where(chroma < 12.0, torch.zeros_like(sector), sector)


@torch.no_grad()
def confusion(model, loader, device, max_images):
    model.eval()
    cm = torch.zeros(K, K, dtype=torch.long)
    pred_chroma, true_chroma, pixels, seen = 0.0, 0.0, 0, 0
    for L, ab in loader:
        L = L.to(device)
        ab = ab.to(device)
        pred = model(L)
        t = classify(ab).flatten().cpu()
        p = classify(pred).flatten().cpu()
        cm += torch.bincount(t * K + p, minlength=K * K).view(K, K)
        pred_chroma += torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2).sum().item() * 110.0
        true_chroma += torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2).sum().item() * 110.0
        pixels += pred[:, 0].numel()
        seen += L.size(0)
        if seen >= max_images:
            break
    return cm.numpy(), pred_chroma / pixels, true_chroma / pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-images", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    info = best_info()
    _, _, test = make_loaders(info["image_size"], args.batch_size, args.num_workers)

    rows, matrices = [], {}
    for name in available_models():
        print("evaluating", name, flush=True)
        model = load_model(name, info["image_size"], device)
        base = evaluate(model, test, device, args.eval_images)
        cm, pred_c, true_c = confusion(model, test, device, args.eval_images)
        recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        present = cm.sum(axis=1) > 0
        rows.append(
            {
                "model": name,
                "display_name": DISPLAY_NAMES[name],
                "accuracy": round(float(np.trace(cm) / cm.sum()), 4),
                "balanced_accuracy": round(float(recalls[present].mean()), 4),
                "pred_chroma": round(pred_c, 2),
                "true_chroma": round(true_c, 2),
                "test_l1": round(base["test_l1"], 4),
                "psnr": round(base["psnr"], 3),
                "ssim": round(base["ssim"], 4),
            }
        )
        matrices[name] = cm
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "full_comparison.csv", index=False)

    n = len(df)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 5.4 * rows_n))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, r) in zip(axes, df.iterrows()):
        cm = matrices[r["model"]]
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{r['display_name']}\nacc {r['accuracy'] * 100:.1f}%  balanced {r['balanced_accuracy'] * 100:.1f}%", fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=140)
    plt.close(fig)
    print(df.drop(columns=["model"]).to_string(index=False))


if __name__ == "__main__":
    main()
