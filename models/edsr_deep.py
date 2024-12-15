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
        out += residual  # Add the input (residual connection)
        return out


class EDSR(nn.Module):
    def __init__(
        self, scale_factor, num_channels=3, num_features=64, num_res_blocks=16
    ):
        super(EDSR, self).__init__()

        # Feature extraction
        self.conv_first = nn.Conv2d(
            num_channels, num_features, kernel_size=3, padding=1
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_res_blocks)]
        )

        # Upscaling
        self.conv_up = nn.Conv2d(
            num_features, num_features * (scale_factor**2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # Reconstruction
        self.conv_last = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv_first(x))

        res = self.res_blocks(x)
        res += x  # Add the input of the first res_block

        x = self.pixel_shuffle(self.conv_up(res))
        x = self.conv_last(x)

        return x
