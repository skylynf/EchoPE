#!/usr/bin/env python3
"""
视角辅助监督模块。

提供两种互补头（可同时启用）：

1. ViewAuxHead   : encoder 嵌入 → 2 类 (A4 / PSS)，与数据集自带 view 标签做 CE
2. ViewKDHead    : encoder 嵌入 → 11 类 (与 EchoPrime view_classifier 一致)，
                   与 view_classifier(第一帧) 的 softmax 做 KL 蒸馏

view_classifier 的 11 个类别（来自 utils.COARSE_VIEWS）：
    0:A2C  1:A3C  2:A4C  3:A5C  4:Apical_Doppler
    5:Doppler_Parasternal_Long  6:Doppler_Parasternal_Short
    7:Parasternal_Long  8:Parasternal_Short  9:SSN  10:Subcostal
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

VIEW_NAMES = ["A4", "PSS"]   # 与 train_lora.py 的 VIEWS 对齐
VIEW_TO_IDX = {v: i for i, v in enumerate(VIEW_NAMES)}
NUM_PRETRAINED_VIEWS = 11    # COARSE_VIEWS 的长度


# ═══════════════════════════════════════════════════════════════════════════════
#  辅助头
# ═══════════════════════════════════════════════════════════════════════════════

class ViewAuxHead(nn.Module):
    """从 encoder 嵌入预测数据集 view 标签 (A4 / PSS)。"""

    def __init__(self, in_dim: int = 512, hidden: int = 128, dropout: float = 0.2,
                 num_classes: int = len(VIEW_NAMES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class ViewKDHead(nn.Module):
    """从 encoder 嵌入预测 11 类（与预训练 view_classifier 对齐），用于 KL 蒸馏。"""

    def __init__(self, in_dim: int = 512, hidden: int = 256, dropout: float = 0.2,
                 num_classes: int = NUM_PRETRAINED_VIEWS):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


# ═══════════════════════════════════════════════════════════════════════════════
#  Teacher 加载（沿用 EchoPrime 的 view_classifier）
# ═══════════════════════════════════════════════════════════════════════════════

def load_view_teacher(weights_path: str, device: torch.device) -> nn.Module:
    """加载冻结的 ConvNeXt view_classifier 作为 view-KD teacher。

    输入：(B, 3, H, W) 的视频第一帧
    输出：(B, 11) logits
    """
    state = torch.load(str(weights_path), map_location="cpu")
    teacher = torchvision.models.convnext_base()
    teacher.classifier[-1] = nn.Linear(
        teacher.classifier[-1].in_features, NUM_PRETRAINED_VIEWS,
    )
    teacher.load_state_dict(state)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ═══════════════════════════════════════════════════════════════════════════════
#  损失
# ═══════════════════════════════════════════════════════════════════════════════

def view_ce_loss(logits: torch.Tensor, view_labels: torch.Tensor) -> torch.Tensor:
    """A4/PSS 二类 CE。"""
    return F.cross_entropy(logits, view_labels)


def view_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 temperature: float = 2.0) -> torch.Tensor:
    """soft KL 蒸馏，T 平方校正梯度量级（标准 Hinton KD 公式）。"""
    teacher_logits = teacher_logits.detach()
    s_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  工具：把字符串 view 列表 → tensor
# ═══════════════════════════════════════════════════════════════════════════════

def views_to_tensor(views: list[str], device: torch.device | None = None) -> torch.Tensor:
    idx = [VIEW_TO_IDX[v] for v in views]
    t = torch.tensor(idx, dtype=torch.long)
    return t if device is None else t.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
#  自检
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    emb = torch.randn(4, 512)
    aux = ViewAuxHead()
    kd = ViewKDHead()
    logits_a = aux(emb)
    logits_k = kd(emb)
    print("ViewAuxHead 输出:", logits_a.shape)
    print("ViewKDHead 输出:", logits_k.shape)

    y = torch.tensor([0, 1, 0, 1])
    print("CE loss:", view_ce_loss(logits_a, y).item())

    teacher_logits = torch.randn(4, 11)
    print("KD loss:", view_kd_loss(logits_k, teacher_logits).item())

    # student == teacher 时 KD 应该 ≈ 0
    print("KD self loss:", view_kd_loss(teacher_logits, teacher_logits).item())

    print("views_to_tensor(['A4','PSS','A4']):",
          views_to_tensor(["A4", "PSS", "A4"]).tolist())
