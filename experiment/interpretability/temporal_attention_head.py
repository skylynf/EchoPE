from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttentionPoolingHead(nn.Module):
    """
    预留给第二阶段 ablation 的轻量 temporal pooling 头。

    输入是逐帧特征 `(B, T, D)`，输出：
    - `pooled`: `(B, D)` 的加权池化特征
    - `weights`: `(B, T)` 的时间权重

    该模块本身不依赖 EchoPrime 结构，因此可以在后续任何
    “暴露逐帧特征”的分支上复用。
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frame_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if frame_features.ndim != 3:
            raise ValueError(f"Expected (B, T, D) input, got {tuple(frame_features.shape)}")
        scores = self.scorer(frame_features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("btd,bt->bd", frame_features, weights)
        return {"pooled": pooled, "weights": weights}


class TemporalAttentionClassifier(nn.Module):
    """
    一个最小可训练对照头：先做时间加权，再输出二分类 logits。

    设计目的不是替代现有主线，而是在第一阶段可解释性信号
    不够稳定时，快速接入一个“显式 temporal weights”分支。
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.temporal_pool = TemporalAttentionPoolingHead(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, frame_features: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.temporal_pool(frame_features)
        logits = self.classifier(pooled["pooled"])
        return {
            "logits": logits,
            "temporal_weights": pooled["weights"],
            "pooled": pooled["pooled"],
        }
