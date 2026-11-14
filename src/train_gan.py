import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.engine import evaluate, get_device, make_loaders, run_epoch
from src.models import build_model

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


class PatchDiscriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()

        def block(i, o, stride, norm=True):
            layers = [nn.Conv2d(i, o, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(3, ch, 2, False),
            *block(ch, ch * 2, 2),
            *block(ch * 2, ch * 4, 2),
            *block(ch * 4, ch * 8, 1),
            nn.Conv2d(ch * 8, 1, 4, 1, 1),
        )

    def forward(self, L, ab):
        return self.net(torch.cat([L, ab], dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-images", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    train_loader, val_loader, test_loader = make_loaders(args.image_size, args.batch_size, args.num_workers)

    gen = build_model("resnet18_gan", args.image_size, pretrained=False)
    gen.load_state_dict(torch.load(CKPT / "resnet18_unet.pt", map_location=device))
    gen.to(device)
    disc = PatchDiscriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr / 2, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        g_tot, d_tot, n = 0.0, 0.0, 0
        for L, ab in tqdm(train_loader, desc=f"gan {epoch}/{args.epochs}", leave=False):
            L = L.to(device)
            ab = ab.to(device)
            fake = gen(L)

            opt_d.zero_grad(set_to_none=True)
            real_out = disc(L, ab)
            fake_out = disc(L, fake.detach())
            d_loss = 0.5 * (bce(real_out, torch.ones_like(real_out)) + bce(fake_out, torch.zeros_like(fake_out)))
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_out = disc(L, fake)
            g_loss = bce(fake_out, torch.ones_like(fake_out)) + args.lambda_l1 * l1(fake, ab)
            g_loss.backward()
            opt_g.step()

            g_tot += g_loss.item() * L.size(0)
            d_tot += d_loss.item() * L.size(0)
            n += L.size(0)

        val_l1 = run_epoch(gen, val_loader, l1, device)
        torch.save(gen.state_dict(), CKPT / "resnet18_gan.pt")
        print(f"epoch {epoch}/{args.epochs}  G={g_tot / n:.3f}  D={d_tot / n:.3f}  val_l1={val_l1:.4f}", flush=True)

    print(evaluate(gen, test_loader, device, args.eval_images))


if __name__ == "__main__":
    main()
