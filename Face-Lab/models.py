"""The expression classifiers: a pretrained torchvision backbone + a 7-class head, with the preprocessing inside the model.

    FerNet.forward(x)      x = uint8-valued float [N,1,48,48] (a grayscale face crop)  ->  logits [N,7]
                           = bilinear 48 -> res, /255, gray -> 3 channels, ImageNet mean/std, backbone, head   (this is what is exported)
    FerNet.forward_res(x)  x = [N,1,res,res] in 0..1 (the training path: the augmentation already resampled to `res`)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ARCHS = {"mobilenet_v3_large": "MobileNet-V3 Large (ImageNet-1K V2)", "efficientnet_b0": "EfficientNet-B0 (ImageNet-1K V1)", "resnet18": "ResNet-18 (ImageNet-1K V1)"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class FerNet(nn.Module):
    def __init__(self, arch="mobilenet_v3_large", res=96, n_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        self.arch, self.res = arch, res
        if arch == "mobilenet_v3_large":
            m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
            self.features, self.pool = m.features, m.avgpool
            self.head = nn.Sequential(m.classifier[0], m.classifier[1], nn.Dropout(dropout), nn.Linear(m.classifier[3].in_features, n_classes))
        elif arch == "efficientnet_b0":
            m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features, self.pool = m.features, m.avgpool
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, n_classes))
        elif arch == "resnet18":
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = nn.Sequential(nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool), m.layer1, m.layer2, m.layer3, m.layer4)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, n_classes))
        else:
            raise ValueError(arch)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward_res(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.head(torch.flatten(self.pool(self.features(x)), 1))

    def forward(self, x):
        return self.forward_res(F.interpolate(x / 255.0, size=(self.res, self.res), mode="bilinear", align_corners=False))

    # --- two-stage fine-tuning helpers
    def set_trainable(self, from_frac):
        """Backbone blocks with index >= from_frac * n_blocks train; earlier blocks stay frozen (and their BatchNorm in eval mode).
        from_frac 1.0 = head only, 0.0 = everything."""
        n = len(self.features)
        k = int(round(from_frac * n))
        for i, b in enumerate(self.features):
            for p in b.parameters():
                p.requires_grad = i >= k
        for p in self.head.parameters():
            p.requires_grad = True
        self._frozen = k

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for i, b in enumerate(self.features):
                if i < getattr(self, "_frozen", 0):
                    b.eval()
        return self


def export_onnx(model, path, opset=17):
    """Deployment graph: input `face` [N,1,48,48] float 0..255 -> `logits` [N,7] (dynamic batch). Preprocessing is inside."""
    m = model.eval().cpu().float()
    x = torch.randint(0, 256, (2, 1, 48, 48)).float()
    torch.onnx.export(m, x, str(path), input_names=["face"], output_names=["logits"], dynamic_axes={"face": {0: "n"}, "logits": {0: "n"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
