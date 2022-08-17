import torch
import torch.nn as nn


class GAPFlatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.mean(dim=-1)  # b, c, h
        x = x.mean(dim=-1)  # b, c
        return x  # b, c


class GAPUnflatten(nn.Module):

    def __init__(self, output_shape=(4, 4)):
        super().__init__()
        self.h, self.w = output_shape
        self.linear = nn.Linear(1, self.h * self.w)
    
    def forward(self, x):
        b, c = x.shape[:2]
        x = x.reshape(b, c, 1)  # (b, c, 1)
        x = self.linear(x)  # (b, c, h*w)
        x = x.reshape(b, c, self.h, self.w)  # (b, c, h, w)
        return x


class TinyResNetAE(nn.Module):

    def __init__(self, in_channels, hidden_channels, expansion_factor=2, latent_image_size=(4, 4)):
        super().__init__()
        exf = expansion_factor

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),  # 32
            nn.MaxPool2d(kernel_size=exf, stride=exf),  # 16

            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels * exf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels * exf),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=exf, stride=exf),  # 8

            nn.Conv2d(in_channels=hidden_channels * exf, out_channels=hidden_channels * (exf**2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels * (exf**2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=exf, stride=exf),  # 4

            nn.Conv2d(in_channels=hidden_channels * (exf**2), out_channels=hidden_channels * (exf**2), kernel_size=3, stride=1, padding=1),  # 4
            GAPFlatten(),
        )

        self.decoder = nn.Sequential(
            GAPUnflatten(output_shape=latent_image_size),

            nn.Conv2d(in_channels=hidden_channels * (exf**2), out_channels=hidden_channels * exf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels * exf),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 8

            nn.Conv2d(in_channels=hidden_channels * exf, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 16

            nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 32

            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Images will be normalized between [-1, 1]
        )

    def forward(self, x):
        encoding = self.encoder(x)
        decoded_image = self.decoder(encoding)
        return encoding, decoded_image
