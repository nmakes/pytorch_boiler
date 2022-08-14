import torch
import torch.nn as nn


class TinyResNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, output_channels, num_layers, expansion_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels * (2**i), hidden_channels * (2**i), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_channels * (2**i)),
                nn.ReLU(),
                nn.Conv2d(hidden_channels * (2**i), hidden_channels * (2**(i+1)), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_channels * (2**(i+1))),
                nn.ReLU(),
                nn.Conv2d(hidden_channels * (2**(i+1)), hidden_channels * (2**(i+1)), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_channels * (2**(i+1))),
                nn.ReLU(),
            )
            for i in range(num_layers)
        ])

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.skip_connections = nn.ModuleList([
            nn.Conv2d(hidden_channels * (2**i), hidden_channels * (2**(i+1)), kernel_size=3, stride=1, padding=1)
            for i in range(num_layers)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * (2**(num_layers)), hidden_channels * (2**(num_layers))), 
            nn.ReLU(),
            nn.Linear(hidden_channels * (2**(num_layers)), output_channels)
        )

    def forward(self, x):
        x = self.conv1(x)  # (B, C, H, W)

        for conv_block, skip_conn in zip(self.conv_blocks, self.skip_connections):
            block_out = conv_block(x)
            skip_out = skip_conn(x)
            x = block_out + skip_out
            x = self.max_pool(x)

        x = self.global_avg_pool(x)  # (B, C, 1, 1)
        x = x.squeeze(-1).squeeze(-1)

        x = self.fc(x)
        return x
