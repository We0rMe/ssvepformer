"""SSVEPformer：通道组合、两个残差编码器、MLP 分类头。"""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels, features, cfg):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.LayerNorm(features),
            nn.Conv1d(channels, channels, cfg.kernel_size, padding=cfg.kernel_size // 2),
            nn.LayerNorm(features), 
            nn.GELU(), 
            nn.Dropout(cfg.dropout),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(features), 
            nn.Linear(features, features), 
            nn.GELU(), 
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.cnn(x)
        return x + self.mlp(x)


class SSVEPformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = len(cfg.channels) * cfg.spatial_multiplier
        features, hidden = cfg.features, cfg.head_multiplier * cfg.classes
        self.channel_combination = nn.Sequential(
            nn.Conv1d(len(cfg.channels), channels, 1),
            nn.LayerNorm(features),
            nn.GELU(), 
            nn.Dropout(cfg.dropout),
        )
        self.encoder = nn.Sequential(*(Encoder(channels, features, cfg) for _ in range(cfg.encoder_blocks)))
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(cfg.dropout), 
            nn.Linear(channels * features, hidden),
            nn.LayerNorm(hidden), 
            nn.GELU(), 
            nn.Dropout(cfg.dropout), 
            nn.Linear(hidden, cfg.classes),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0, std=cfg.init_std)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.encoder(self.channel_combination(x)))
