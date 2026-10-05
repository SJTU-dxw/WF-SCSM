from __future__ import annotations

import math
import numpy as np
import torch
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from torch import nn
from timm.layers import DropPath


def _position_embedding(dimension: int, length: int) -> torch.Tensor:
    positions = np.arange(length, dtype=np.float32)
    frequencies = 1.0 / 10000 ** (np.arange(dimension // 2, dtype=np.float64) / (dimension / 2.0))
    values = np.einsum("m,d->md", positions, frequencies)
    return torch.from_numpy(np.concatenate((np.sin(values), np.cos(values)), axis=1)).float()


class Mamba2(nn.Module):
    def __init__(self, layer_idx: int, d_model: int = 256, d_state: int = 128,
                 d_conv: int = 4, expand: int = 2, headdim: int = 64, chunk_size: int = 256):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.chunk_size = chunk_size
        projection_size = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, projection_size, bias=False)
        convolution_size = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(convolution_size, convolution_size, d_conv,
                                groups=convolution_size, padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = dt.clamp_min(0.0001)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(1, 1.1).log())
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False,
                                 group_size=self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mamba_split_conv1d_scan_combined(
            self.in_proj(x),
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            -torch.exp(self.A_log.float()),
            D=self.D,
            chunk_size=self.chunk_size,
            activation="silu",
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.headdim,
            ngroups=1,
            norm_before_gate=False,
        )


class SimpleConvBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size - 1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x)[:, :, :-(self.kernel_size - 1)]))


class ConvBlock1d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size - 1)
        self.norm1 = nn.BatchNorm1d(output_channels)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=kernel_size - 1)
        self.norm2 = nn.BatchNorm1d(output_channels)
        self.downsample = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv1(x)[:, :, :-(self.kernel_size - 1)]
        output = torch.relu(self.norm1(output))
        output = self.conv2(output)[:, :, :-(self.kernel_size - 1)]
        output = torch.relu(self.norm2(output))
        residual = x if self.downsample is None else self.downsample(x)
        return torch.relu(output + residual)


class CausalCNN(nn.Module):
    def __init__(self, dimension: int, simple_mode: bool, kernel_size: int):
        super().__init__()
        if simple_mode:
            self.conv1 = SimpleConvBlock1d(dimension, kernel_size)
            self.conv2 = SimpleConvBlock1d(dimension, kernel_size)
            self.conv3 = SimpleConvBlock1d(dimension, kernel_size)
            self.conv4 = SimpleConvBlock1d(dimension, kernel_size)
        else:
            self.conv1 = ConvBlock1d(dimension, dimension // 8, kernel_size)
            self.conv2 = ConvBlock1d(dimension // 8, dimension // 4, kernel_size)
            self.conv3 = ConvBlock1d(dimension // 4, dimension // 2, kernel_size)
            self.conv4 = ConvBlock1d(dimension // 2, dimension, kernel_size)
        self.pool1 = nn.MaxPool1d(3)
        self.dropout1 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(2)
        x = self.dropout1(self.pool1(self.conv2(self.conv1(x))))
        x = self.dropout2(self.pool2(self.conv4(self.conv3(x))))
        return x.transpose(1, 2)


class SCSM(nn.Module):
    """Uploaded SCSM pre-training encoder: WTCM patching, local CNN and Mamba2."""

    def __init__(self, config: dict):
        super().__init__()
        matrix_length = int(config.get("max_matrix_length"))
        dimension = int(config.get("embedding_dim"))
        depth = int(config.get("depth"))
        simple_mode = bool(config.get("simple_mode"))
        kernel_size = int(config.get("kernel_size"))
        if matrix_length % 6 or dimension % 8:
            raise ValueError("max_matrix_length must be divisible by 6 and embedding_dim by 8")
        patches = matrix_length // 6
        self.patch_embed = nn.Conv2d(1, dimension, kernel_size=(8, 1), stride=(1, 1))
        self.local_model = CausalCNN(dimension, simple_mode, kernel_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        position = torch.cat((torch.zeros(1, dimension), _position_embedding(dimension, patches)), dim=0)
        self.register_buffer("pos_embed", position.unsqueeze(0), persistent=True)
        self.blocks = nn.ModuleList(Mamba2(i, d_model=dimension, headdim=dimension // 4) for i in range(depth))
        probabilities = torch.linspace(0, float(config.get("drop_path_rate", 0.2)), depth)
        self.drop_paths = nn.ModuleList(DropPath(float(value)) for value in probabilities)
        self.output_dim = dimension
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, matrix: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        x = self.local_model(self.patch_embed(matrix.float()))
        x = x + self.pos_embed[:, 1:1 + x.shape[1]]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for block, drop_path in zip(self.blocks, self.drop_paths):
            x = drop_path(block(x)) + x
        x = x[:, 1:]
        aggregate = torch.floor(last_index / 6).long().clamp(0, x.shape[1] - 1)
        return torch.stack([row[: int(end) + 1].mean(0) for row, end in zip(x, aggregate)])

    @property
    def encoder(self) -> nn.Module:
        return self
