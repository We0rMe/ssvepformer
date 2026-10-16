"""多个独立 SSVEPformer 提取子带证据，卷积层学习怎样融合它们。"""

import torch
from torch import nn

from model import SSVEPformer


class FBSSVEPformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.subnetworks = nn.ModuleList(SSVEPformer(cfg) for _ in range(cfg.subbands))
        # 同一组子带权重应用于所有类别；直接融合 logits，不先 softmax。
        self.fusion = nn.Conv1d(cfg.subbands, 1, kernel_size=1, bias=False)
        nn.init.normal_(self.fusion.weight, mean=0, std=cfg.init_std)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != len(self.subnetworks):
            raise ValueError("FB 输入须为 [试次, 子带, 通道, 复频谱特征]")
        scores = torch.stack([network(x[:, band]) for band, network in enumerate(self.subnetworks)], dim=1)
        return self.fusion(scores).squeeze(1)
