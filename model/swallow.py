from __future__ import annotations

import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class MLPHead(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 512, projection_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwallowNetwork(nn.Module):
    """Official Swallow ResNet backbone followed by its projection head."""

    def __init__(self, config: dict):
        super().__init__()
        name = str(config.get("name", "resnet18"))
        constructors = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }
        if name not in constructors:
            raise ValueError(f"unsupported Swallow backbone: {name}")
        resnet = constructors[name](weights=None)
        resnet.conv1 = nn.Conv2d(
            1,
            resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=resnet.conv1.bias,
        )
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(
            resnet.fc.in_features,
            int(config.get("projection_hidden_dim", 512)),
            int(config.get("projection_dim", 128)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != 2:
            raise ValueError(f"expected CIF shape [batch, 2, length] or [batch, 1, 2, length], got {tuple(x.shape)}")
        features = self.encoder(x.float()).flatten(1)
        return self.projection(features)


class SwallowBYOL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        projection_dim = int(config.get("projection_dim", 128))
        self.online_network = SwallowNetwork(config)
        self.target_network = copy.deepcopy(self.online_network)
        self.target_network.requires_grad_(False)
        self.predictor = MLPHead(
            projection_dim,
            int(config.get("predictor_hidden_dim", 512)),
            projection_dim,
        )

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        p1 = self.predictor(self.online_network(view1))
        p2 = self.predictor(self.online_network(view2))
        with torch.no_grad():
            z1 = self.target_network(view1)
            z2 = self.target_network(view2)
        return self.regression_loss(p1, z2) + self.regression_loss(p2, z1)

    @staticmethod
    def regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (2 - 2 * (F.normalize(prediction, dim=-1) * F.normalize(target, dim=-1)).sum(-1)).mean()

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for source, destination in zip(self.online_network.parameters(), self.target_network.parameters()):
            destination.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)

    @property
    def encoder(self) -> nn.Module:
        return self.online_network.encoder
