import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

BINS_PER_AXIS = 22
NUM_BINS = BINS_PER_AXIS * BINS_PER_AXIS

MODEL_NAMES = ["autoencoder", "classic_unet", "attention_unet", "pix2pix_unet", "resnet18_unet"]

DISPLAY_NAMES = {
    "autoencoder": "Conv Autoencoder",
    "classic_unet": "Classic U-Net",
    "attention_unet": "Attention U-Net",
    "pix2pix_unet": "Pix2Pix U-Net",
    "resnet18_unet": "ResNet18 U-Net",
    "resnet18_gan": "ResNet18 U-Net + GAN",
    "resnet18_class": "ResNet18 U-Net + Color Classification",
}


def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 512)):
        super().__init__()
        layers, i = [], 1
        for c in channels:
            layers += [nn.Conv2d(i, c, 4, 2, 1, bias=False), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True)]
            i = c
        self.encoder = nn.Sequential(*layers)
        layers = []
        for c in list(channels[::-1])[1:] + [64]:
            layers += [nn.ConvTranspose2d(i, c, 4, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            i = c
        layers += [nn.Conv2d(i, 2, 3, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AttGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        m = max(c // 2, 8)
        self.wg = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m))
        self.wx = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m))
        self.psi = nn.Sequential(nn.Conv2d(m, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.wg(g) + self.wx(x)))


class ClassicUNet(nn.Module):
    def __init__(self, base=32, attention=False):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([conv_block(1, c[0])] + [conv_block(c[i], c[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, 2) for i in range(4)])
        self.dec = nn.ModuleList([conv_block(c[i] * 2, c[i]) for i in range(4)])
        self.gates = nn.ModuleList([AttGate(c[i]) for i in range(4)]) if attention else None
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(c[0], 2, 1)

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            feats.append(x)
        x = feats[-1]
        for i in range(3, -1, -1):
            u = self.ups[i](x)
            s = feats[i]
            if self.gates is not None:
                s = self.gates[i](u, s)
            x = self.dec[i](torch.cat([u, s], dim=1))
        return torch.tanh(self.head(x))


class SkipBlock(nn.Module):
    def __init__(self, outer, inner, in_ch=None, sub=None, outermost=False, innermost=False, dropout=False):
        super().__init__()
        self.outermost = outermost
        in_ch = in_ch or outer
        down_conv = nn.Conv2d(in_ch, inner, 4, 2, 1, bias=False)
        down_relu = nn.LeakyReLU(0.2, inplace=True)
        down_norm = nn.BatchNorm2d(inner)
        up_relu = nn.ReLU(inplace=True)
        up_norm = nn.BatchNorm2d(outer)
        if outermost:
            up_conv = nn.ConvTranspose2d(inner * 2, outer, 4, 2, 1)
            layers = [down_conv, sub, up_relu, up_conv, nn.Tanh()]
        elif innermost:
            up_conv = nn.ConvTranspose2d(inner, outer, 4, 2, 1, bias=False)
            layers = [down_relu, down_conv, up_relu, up_conv, up_norm]
        else:
            up_conv = nn.ConvTranspose2d(inner * 2, outer, 4, 2, 1, bias=False)
            layers = [down_relu, down_conv, down_norm, sub, up_relu, up_conv, up_norm]
            if dropout:
                layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class Pix2PixUNet(nn.Module):
    def __init__(self, size=128, ngf=64):
        super().__init__()
        downs = int(math.log2(size))
        block = SkipBlock(ngf * 8, ngf * 8, innermost=True)
        for _ in range(downs - 5):
            block = SkipBlock(ngf * 8, ngf * 8, sub=block, dropout=True)
        block = SkipBlock(ngf * 4, ngf * 8, sub=block)
        block = SkipBlock(ngf * 2, ngf * 4, sub=block)
        block = SkipBlock(ngf, ngf * 2, sub=block)
        self.model = SkipBlock(2, ngf, in_ch=1, sub=block, outermost=True)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, i, s, o):
        super().__init__()
        self.conv = conv_block(i + s, o)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet18UNet(nn.Module):
    def __init__(self, pretrained=True, out_ch=2, activate=True):
        super().__init__()
        try:
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            net = tvm.resnet18(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.u3 = UpBlock(512, 256, 256)
        self.u2 = UpBlock(256, 128, 128)
        self.u1 = UpBlock(128, 64, 64)
        self.u0 = UpBlock(64, 64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, out_ch, 1), *([nn.Tanh()] if activate else []))

    def forward(self, x):
        h, w = x.shape[2:]
        x3 = ((x + 1.0) / 2.0).repeat(1, 3, 1, 1)
        x3 = (x3 - self.mean) / self.std
        f0 = self.stem(x3)
        f1 = self.l1(self.pool(f0))
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        d = self.u3(f4, f3)
        d = self.u2(d, f2)
        d = self.u1(d, f1)
        d = self.u0(d, f0)
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(d)


class ColorClassifier(nn.Module):
    def __init__(self, pretrained=True, temperature=0.38):
        super().__init__()
        self.net = ResNet18UNet(pretrained=pretrained, out_ch=NUM_BINS, activate=False)
        grid = torch.arange(BINS_PER_AXIS).float() * 10.0 - 105.0
        a, b = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("centers", torch.stack([a, b], dim=-1).reshape(-1, 2) / 110.0)
        self.temperature = temperature

    def forward(self, x):
        logits = self.net(x)
        if self.training:
            return logits
        probs = torch.softmax(logits / self.temperature, dim=1)
        return torch.einsum("bkhw,kc->bchw", probs, self.centers)


def build_model(name, size=128, pretrained=True):
    if name == "autoencoder":
        return ConvAutoencoder()
    if name == "classic_unet":
        return ClassicUNet()
    if name == "attention_unet":
        return ClassicUNet(attention=True)
    if name == "pix2pix_unet":
        return Pix2PixUNet(size=size)
    if name in ("resnet18_unet", "resnet18_gan"):
        return ResNet18UNet(pretrained=pretrained)
    if name == "resnet18_class":
        return ColorClassifier(pretrained=pretrained)
    raise ValueError(name)
