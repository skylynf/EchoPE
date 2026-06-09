#!/usr/bin/env python3
"""
特征 / 关系蒸馏损失，以及 frozen EchoPrime teacher 加载器。

用法（在 train_lora_distill.py 内）：
    from distill import kd_loss, load_frozen_teacher

    teacher = load_frozen_teacher(weights_path, device)
    with torch.no_grad():
        t_emb = teacher(videos)
    s_emb = student.encoder(videos)
    loss_kd = kd_loss(s_emb, t_emb, mode="cos")

支持模式：
    cos      : 1 - cos(s, t)         逐样本平均
    l2       : ||s - t||_2           逐样本平均（输入会先 L2 normalize）
    rkd-d    : 保 batch 内 pair-wise 距离 (Park et al. 2019, RKD-D)
    rkd-a    : 保 batch 内三元组角度 (Park et al. 2019, RKD-A)
    combo    : cos + lam_rkd * rkd-d   （推荐组合，鲁棒）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

KD_MODES = ("none", "cos", "l2", "rkd-d", "rkd-a", "combo")


# ═══════════════════════════════════════════════════════════════════════════════
#  Teacher 加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_frozen_teacher(weights_path: str, device: torch.device) -> nn.Module:
    """加载一份完全冻结的 EchoPrime MViT 编码器作为 teacher。

    Args:
        weights_path: echo_prime_encoder.pt 的绝对路径
        device:       teacher 推理设备

    Returns:
        nn.Module，已 .eval()，所有参数 requires_grad=False
    """
    ckpt = torch.load(str(weights_path), map_location="cpu")
    teacher = torchvision.models.video.mvit_v2_s()
    teacher.head[-1] = nn.Linear(teacher.head[-1].in_features, 512)
    teacher.load_state_dict(ckpt)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ═══════════════════════════════════════════════════════════════════════════════
#  原子损失
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_kd(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity，逐样本平均。"""
    return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()


def _l2_kd(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """L2 normalize 后的均方距离（与 cos 在量纲上接近）。"""
    s_n = F.normalize(s, dim=-1)
    t_n = F.normalize(t, dim=-1)
    return (s_n - t_n).pow(2).sum(dim=-1).mean()


def _pdist(e: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """成对欧氏距离矩阵 (B, B)。"""
    e_sq = (e * e).sum(dim=1)
    prod = e @ e.t()
    dist = (e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2 * prod).clamp(min=eps).sqrt()
    # 把对角线置 0，避免 0/0 影响均值
    dist = dist - torch.diag(dist.diag())
    return dist


def _rkd_distance(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """RKD-D：保持 batch 内成对距离的相对结构（除以各自均值后做 SmoothL1）。"""
    if s.size(0) < 2:
        return s.new_zeros(())
    with torch.no_grad():
        t_d = _pdist(t)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / (mean_td + 1e-12)

    s_d = _pdist(s)
    mean_sd = s_d[s_d > 0].mean().detach()
    s_d = s_d / (mean_sd + 1e-12)

    return F.smooth_l1_loss(s_d, t_d)


def _rkd_angle(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """RKD-A：三元组 (i, j, k) 的夹角余弦对齐。"""
    if s.size(0) < 3:
        return s.new_zeros(())

    def _angle(e: torch.Tensor) -> torch.Tensor:
        # e: (B, D) → 对每对 (i, j) 算 e_i - e_j，再两两点乘 → (B, B, B) 的 cos
        diff = e.unsqueeze(0) - e.unsqueeze(1)         # (B, B, D)  diff[j,i] = e_i - e_j
        norm = F.normalize(diff, p=2, dim=2)
        # angle[i,j,k] = <e_i - e_j, e_k - e_j> / (||·||·||·||)
        return torch.einsum("jid,jkd->jik", norm, norm)

    with torch.no_grad():
        t_ang = _angle(t)
    s_ang = _angle(s)
    return F.smooth_l1_loss(s_ang, t_ang)


# ═══════════════════════════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════════════════════════

def kd_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    mode: str = "cos",
    lam_rkd: float = 1.0,
) -> torch.Tensor:
    """计算 student 与 teacher 嵌入之间的蒸馏损失。

    Args:
        student_emb: (B, D) student 嵌入（带梯度）
        teacher_emb: (B, D) teacher 嵌入（应已 detach / no_grad）
        mode:        见 KD_MODES
        lam_rkd:     `combo` 模式中 rkd-d 的系数

    Returns:
        标量 loss
    """
    if mode == "none":
        return student_emb.new_zeros(())

    # 防御：teacher 必须无梯度
    teacher_emb = teacher_emb.detach()

    if mode == "cos":
        return _cosine_kd(student_emb, teacher_emb)
    if mode == "l2":
        return _l2_kd(student_emb, teacher_emb)
    if mode == "rkd-d":
        return _rkd_distance(student_emb, teacher_emb)
    if mode == "rkd-a":
        return _rkd_angle(student_emb, teacher_emb)
    if mode == "combo":
        return _cosine_kd(student_emb, teacher_emb) + lam_rkd * _rkd_distance(
            student_emb, teacher_emb
        )

    raise ValueError(f"未知 kd mode: {mode}，可选: {KD_MODES}")


# ═══════════════════════════════════════════════════════════════════════════════
#  自检
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    s = torch.randn(8, 16, requires_grad=True)
    t = torch.randn(8, 16)

    print("距离矩阵 (前 3x3):")
    print(_pdist(t)[:3, :3])

    for m in ("cos", "l2", "rkd-d", "rkd-a", "combo"):
        loss = kd_loss(s, t, mode=m)
        loss.backward(retain_graph=True)
        s.grad = None
        print(f"  {m:8s}  loss = {loss.item():.6f}")

    # student == teacher 时的 loss 应该接近 0
    s2 = t.clone().requires_grad_(True)
    print("\nstudent == teacher 时 (应≈0):")
    for m in ("cos", "l2", "rkd-d", "rkd-a", "combo"):
        loss = kd_loss(s2, t, mode=m)
        print(f"  {m:8s}  loss = {loss.item():.2e}")
