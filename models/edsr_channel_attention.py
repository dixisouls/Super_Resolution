import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_ratio, num_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EDSR(nn.Module):
    def __init__(
        self,
        scale_factor,
        num_channels=3,
        num_features=64,
        num_residual_blocks=16,
        reduction_ratio=16,
    ):
        super(EDSR, self).__init__()

        # feature extraction
        self.conv_first = nn.Conv2d(
            num_channels, num_features, kernel_size=3, padding=1
        )

        # residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                nn.Sequential(
                    ResidualBlock(num_features),
                    ChannelAttention(num_features, reduction_ratio),
                )
                for _ in range(num_residual_blocks)
            ]
        )

        # upscaling
        self.conv_up = nn.Conv2d(
            num_features, num_features * (scale_factor**2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # reconstruction
        self.conv_last = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

        # activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv_first(x))
        res = self.residual_blocks(x)
        res += x

        x = self.pixel_shuffle(self.conv_up(res))
        x = self.conv_last(x)

        return x
