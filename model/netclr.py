from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class DFEncoder(nn.Module):
    """DF backbone used by NetCLR, adapted from WFlib for 10,000 cells."""

    def __init__(self, config: dict):
        super().__init__()
        channels = [int(value) for value in config.get("channels")]

        self.input_length = int(config.get("input_length"))
        self.kernel_size = int(config.get("kernel_size"))
        self.pool_size = int(config.get("pool_size"))
        self.pool_stride = int(config.get("pool_stride"))
        conv_stride = int(config.get("conv_stride"))
        dropout = float(config.get("dropout"))
        embedding_dim = int(config.get("embedding_dim"))

        c1, c2, c3, c4 = channels
        self.conv1 = nn.Conv1d(1, c1, self.kernel_size, stride=conv_stride)
        self.conv1_1 = nn.Conv1d(c1, c1, self.kernel_size, stride=conv_stride)
        self.conv2 = nn.Conv1d(c1, c2, self.kernel_size, stride=conv_stride)
        self.conv2_2 = nn.Conv1d(c2, c2, self.kernel_size, stride=conv_stride)
        self.conv3 = nn.Conv1d(c2, c3, self.kernel_size, stride=conv_stride)
        self.conv3_3 = nn.Conv1d(c3, c3, self.kernel_size, stride=conv_stride)
        self.conv4 = nn.Conv1d(c3, c4, self.kernel_size, stride=conv_stride)
        self.conv4_4 = nn.Conv1d(c4, c4, self.kernel_size, stride=conv_stride)

        self.batch_norm1 = nn.BatchNorm1d(c1)
        self.batch_norm2 = nn.BatchNorm1d(c2)
        self.batch_norm3 = nn.BatchNorm1d(c3)
        self.batch_norm4 = nn.BatchNorm1d(c4)
        self.pools = nn.ModuleList(
            nn.MaxPool1d(kernel_size=self.pool_size, stride=self.pool_stride)
            for _ in range(4)
        )
        self.dropouts = nn.ModuleList(nn.Dropout(p=dropout) for _ in range(4))

        output_length = self.input_length
        for _ in range(4):
            output_length = math.floor(
                (output_length + 7 - self.pool_size) / self.pool_stride + 1
            )

        self.fc = nn.Linear(c4 * output_length, embedding_dim)
        self.output_dim = embedding_dim

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected input shape [batch, length] or [batch, 1, length], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_length:
            raise ValueError(f"expected direction sequence length {self.input_length}, got {x.shape[-1]}")
        x = x.float()

        x = F.elu(self.conv1(self._pad(x)))
        x = F.elu(self.batch_norm1(self.conv1_1(self._pad(x))))
        x = self.dropouts[0](self.pools[0](self._pad(x)))

        x = F.relu(self.conv2(self._pad(x)))
        x = F.relu(self.batch_norm2(self.conv2_2(self._pad(x))))
        x = self.dropouts[1](self.pools[1](self._pad(x)))

        x = F.relu(self.conv3(self._pad(x)))
        x = F.relu(self.batch_norm3(self.conv3_3(self._pad(x))))
        x = self.dropouts[2](self.pools[2](self._pad(x)))

        x = F.relu(self.conv4(self._pad(x)))
        x = F.relu(self.batch_norm4(self.conv4_4(self._pad(x))))
        x = self.dropouts[3](self.pools[3](self._pad(x)))

        x = x.flatten(1)
        x = self.fc(x)
        return x


class NetCLR(nn.Module):
    """NetCLR pre-training network: DF backbone plus SimCLR projection head."""

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = DFEncoder(config)
        projection_dim = int(config.get("projection_dim"))
        feature_dim = self.encoder.output_dim
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, projection_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))
