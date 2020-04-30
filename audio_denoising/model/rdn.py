import torch
import argparse
from torch import nn
from torch import optim

# Adapted from the PyTorch implementation of Residual Dense Network
# for Image Super-Resolution by @yjn870,
# https://github.com/yjn870/RDN-pytorch


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.dense_layers = nn.Sequential(
            *[
                DenseLayer(in_channels + growth_rate * layer_idx, growth_rate)
                for layer_idx in range(num_layers)
            ]
        )

        self.local_feature_fusion = nn.Conv2d(
            in_channels + growth_rate * num_layers, in_channels, kernel_size=1
        )

    def forward(self, x):
        return x + self.local_feature_fusion(self.dense_layers(x))


class ResidualDenseNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.kernel_size = args.kernel_size
        self.num_channels = args.num_channels
        self.growth_rate = args.growth_rate
        self.num_features = args.num_features
        self.num_blocks = args.num_blocks
        self.num_layers = args.num_layers

        self.outer_shallow_features = nn.Conv2d(
            self.num_channels,
            self.num_features,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )
        self.inner_shallow_features = nn.Conv2d(
            self.num_features,
            self.num_features,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )

        self.residual_dense_blocks = [
            ResidualDenseBlock(self.num_features, self.growth_rate, self.num_layers)
            for _ in range(self.num_blocks)
        ]

        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(
                self.num_features * self.num_blocks, self.num_features, kernel_size=1,
            ),
            nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1,),
        )
        self.output = nn.Conv2d(
            self.num_features,
            self.num_channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )

    def forward(self, x):
        outer_shallow_features = self.outer_shallow_features(x)
        x = self.inner_shallow_features(outer_shallow_features)

        local_features = []
        for block_idx in range(self.num_blocks):
            x = self.residual_dense_blocks[block_idx](x)
            local_features.append(x)

        x = (
            self.global_feature_fusion(torch.cat(local_features, 1))
            + outer_shallow_features
        )
        x = self.output(x)
        return x
