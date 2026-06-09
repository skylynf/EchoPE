#!/usr/bin/env python3
"""
PE 分类实验脚本
策略：冻结 EchoPrime 视频编码器 → 提取 512 维特征 → 训练分类头

数据集默认根目录为 EchoPrime 上一级下的 hope_dataset，例如：
  /path/to/echoprime/hope_dataset/
  ├── Normal/{A4,PSS}/*.mkv
  └── PE/{A4,PSS}/*.mkv

用法（在 baseline 目录下）：
  ../.venv/bin/python train_pe_classifier.py                          # 默认 MLP 头
  ../.venv/bin/python train_pe_classifier.py --head residual          # 残差 MLP
  ../.venv/bin/python train_pe_classifier.py --head attention         # Transformer 注意力头
  ../.venv/bin/python train_pe_classifier.py --force-extract          # 强制重新提取特征
  ../.venv/bin/python train_pe_classifier.py --epochs 100 --lr 3e-4

可选分类头 (--head):
  mlp       : 基础 MLP（LayerNorm → Linear → ReLU → Dropout → Linear）
  residual  : 深层残差 MLP（多个跳跃连接块 + GELU，更强的特征变换）
  attention : Transformer 注意力头（将 512-d 特征切分为 token 序列 + [CLS] token）
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ── 路径设置：工作目录为 EchoPrime 仓库根，以便 utils / model_data / assets 相对路径有效 ──
HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
from echo_paths import DEFAULT_HOPE_DATASET, setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()
import utils  # noqa: E402  (EchoPrime 的 utils 包)

# ── 默认常量 ────────────────────────────────────────────────────────────────────
DATASET_ROOT = DEFAULT_HOPE_DATASET
CACHE_FILE = HERE / "pe_features_cache.pt"

FRAMES_TO_TAKE = 32
FRAME_STRIDE = 2
VIDEO_SIZE = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

LABEL_MAP = {"Normal": 0, "PE": 1}
VIEWS = ["A4", "PSS"]


# ── 视频预处理（复用 EchoPrime 逻辑）─────────────────────────────────────────────
def preprocess_single_video(path: str) -> torch.Tensor:
    """
    读取单个视频文件并返回 (3, 16, 224, 224) 的预处理张量，
    与 EchoPrime.process_mp4s 保持完全一致。
    """
    pixels = utils.read_video_rgb_numpy(path)  # (T, H, W, 3) uint8

    x = np.zeros((len(pixels), VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.float32)
    for i in range(len(pixels)):
        x[i] = utils.crop_and_scale(pixels[i])

    x = torch.as_tensor(x).permute(3, 0, 1, 2)  # (3, T, H, W)
    x = x.sub(MEAN).div(STD)

    if x.shape[1] < FRAMES_TO_TAKE:
        pad = torch.zeros(3, FRAMES_TO_TAKE - x.shape[1], VIDEO_SIZE, VIDEO_SIZE)
        x = torch.cat([x, pad], dim=1)

    # 取前 FRAMES_TO_TAKE 帧，步长 FRAME_STRIDE → 时间维 = 16
    x = x[:, 0:FRAMES_TO_TAKE:FRAME_STRIDE, :, :]  # (3, 16, 224, 224)
    return x


# ── 编码器加载 ────────────────────────────────────────────────────────────────────
def load_frozen_encoder(device: torch.device) -> nn.Module:
    """加载并冻结 EchoPrime 视频编码器（MViT-v2-s → 512 维）。"""
    ckpt = torch.load(
        "model_data/weights/echo_prime_encoder.pt", map_location=device
    )
    encoder = torchvision.models.video.mvit_v2_s()
    encoder.head[-1] = nn.Linear(encoder.head[-1].in_features, 512)
    encoder.load_state_dict(ckpt)
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


# ── 数据收集 ──────────────────────────────────────────────────────────────────────
def collect_samples(root: Path) -> list[tuple[str, int, str]]:
    """
    遍历 root/{Normal,PE}/{A4,PSS}/*.mkv
    返回 [(path, label_int, view_str), ...]
    """
    samples: list[tuple[str, int, str]] = []
    for cls_name, label in LABEL_MAP.items():
        for view in VIEWS:
            view_dir = root / cls_name / view
            if not view_dir.is_dir():
                print(f"[WARN] 目录不存在，跳过: {view_dir}")
                continue
            files = sorted(view_dir.glob("*.mkv"))
            for p in files:
                samples.append((str(p), label, view))
    return samples


# ── 特征提取与缓存 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_and_cache(
    samples: list[tuple[str, int, str]],
    encoder: nn.Module,
    device: torch.device,
    cache_path: Path,
) -> dict:
    """
    对每个视频提取 512 维嵌入并保存到磁盘。
    每段视频可能有多个 clip（模型输入步长决定），
    此处直接取单 clip（每个 .mkv 文件 = 一个 clip）。
    """
    embeddings_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    views_list: list[str] = []
    paths_list: list[str] = []
    skipped = 0

    for path, label, view in tqdm(samples, desc="提取特征"):
        try:
            x = preprocess_single_video(path)       # (3, 16, 224, 224)
            x = x.unsqueeze(0).to(device)           # (1, 3, 16, 224, 224)
            emb = encoder(x).squeeze(0).cpu()       # (512,)
            embeddings_list.append(emb)
            labels_list.append(label)
            views_list.append(view)
            paths_list.append(str(path))
        except Exception as e:
            print(f"  [WARN] 跳过 {path}: {e}")
            skipped += 1

    if skipped:
        print(f"  共跳过 {skipped} 个损坏文件")

    cache = {
        "embeddings": torch.stack(embeddings_list),          # (N, 512)
        "labels": torch.tensor(labels_list, dtype=torch.long),  # (N,)
        "views": views_list,
        "paths": paths_list,  # 与上列一一对应，供 finetuning 等与缓存对齐的视频路径
    }
    torch.save(cache, cache_path)
    print(f"特征缓存已保存 → {cache_path}  ({len(labels_list)} 个样本)")
    return cache


# ── Dataset ───────────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


# ── 分类头 ────────────────────────────────────────────────────────────────────────

class PEClassifier(nn.Module):
    """基础 MLP：LayerNorm → Linear → ReLU → Dropout → Linear（原版）"""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualBlock(nn.Module):
    """Pre-LN 残差块：LayerNorm → Linear(dim→dim*2) → GELU → Dropout → Linear(dim*2→dim)"""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PEClassifierResidual(nn.Module):
    """
    深层残差 MLP。

    结构：
      投影层 (in_dim → hidden_dim)
      → N 个 Pre-LN 残差块（每块内 hidden_dim → hidden_dim*2 → hidden_dim）
      → 分类线性层

    相比基础 MLP，提供更深的非线性变换能力和稳定的梯度流。
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class PEClassifierAttention(nn.Module):
    """
    Transformer 注意力分类头。

    结构：
      将 512-d 特征重塑为 num_tokens 个子向量（默认 16 × 32）
      → 拼接可学习 [CLS] token → Transformer 编码器（Pre-LN, GELU）
      → 取 [CLS] 输出 → 分类线性层

    注意力机制让模型能够学习特征维度之间的交互，
    适合捕捉 EchoPrime 嵌入内部的结构信息。
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_tokens: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if in_dim % num_tokens != 0:
            raise ValueError(f"in_dim ({in_dim}) 必须能被 num_tokens ({num_tokens}) 整除")
        self.num_tokens = num_tokens
        self.token_dim = in_dim // num_tokens  # 默认 32

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=self.token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN 更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.token_dim)
        self.head = nn.Linear(self.token_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = x.view(B, self.num_tokens, self.token_dim)       # (B, T, D)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                   # (B, T+1, D)
        tokens = self.transformer(tokens)
        cls_out = self.norm(tokens[:, 0])                          # (B, D)
        return self.head(cls_out)


# 分类头名称 → 类型映射（供外部脚本导入）
HEAD_REGISTRY: dict[str, type] = {
    "mlp": PEClassifier,
    "residual": PEClassifierResidual,
    "attention": PEClassifierAttention,
}


def build_classifier(head: str, args: argparse.Namespace) -> nn.Module:
    """根据 --head 参数实例化对应的分类头。"""
    if head == "mlp":
        return PEClassifier(in_dim=512, hidden_dim=args.hidden, dropout=args.dropout)
    elif head == "residual":
        return PEClassifierResidual(
            in_dim=512,
            hidden_dim=args.hidden,
            dropout=args.dropout,
            num_blocks=args.num_blocks,
        )
    elif head == "attention":
        return PEClassifierAttention(
            in_dim=512,
            num_tokens=args.attn_tokens,
            num_heads=args.attn_heads,
            num_layers=args.attn_layers,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"未知分类头: {head}，可选: {list(HEAD_REGISTRY)}")


# ── 训练 / 评估函数 ───────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(y.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── 打印指标 ──────────────────────────────────────────────────────────────────────
def print_metrics(
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    views: list[str] | None = None,
) -> None:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*52}")
    print(f"  {split_name} 评估结果")
    print(f"{'='*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  混淆矩阵 (行=真实, 列=预测):")
    print(f"             Normal    PE")
    print(f"  Normal  {cm[0, 0]:8d}  {cm[0, 1]:6d}")
    print(f"  PE      {cm[1, 0]:8d}  {cm[1, 1]:6d}")

    if views is not None:
        print(f"\n  按视角分类准确率:")
        for view in VIEWS:
            mask = [i for i, v in enumerate(views) if v == view]
            if mask:
                v_acc = accuracy_score(y_true[mask], y_pred[mask])
                print(f"    {view:4s}: {v_acc:.4f}  ({len(mask)} 个样本)")


# ── main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PE 二分类：EchoPrime 冻结编码器 + 可选分类头"
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_ROOT),
        help="hope_dataset 根目录（默认：EchoPrime 同级目录下的 hope_dataset）",
    )
    parser.add_argument(
        "--cache", default=str(CACHE_FILE), help="特征缓存文件路径（.pt）"
    )
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 比例")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="强制重新提取特征，忽略已有缓存",
    )
    # 分类头选择
    parser.add_argument(
        "--head",
        default="mlp",
        choices=list(HEAD_REGISTRY),
        help="分类头类型: mlp（基础）| residual（残差MLP）| attention（Transformer）",
    )
    # mlp / residual 共用参数
    parser.add_argument("--hidden", type=int, default=256, help="MLP/残差块隐层维度")
    # residual 专用参数
    parser.add_argument("--num-blocks", type=int, default=3, dest="num_blocks",
                        help="[residual] 残差块数量")
    # attention 专用参数
    parser.add_argument("--attn-tokens", type=int, default=16, dest="attn_tokens",
                        help="[attention] 将 512-d 特征切分为多少个 token（须整除 512）")
    parser.add_argument("--attn-heads", type=int, default=4, dest="attn_heads",
                        help="[attention] 多头注意力头数")
    parser.add_argument("--attn-layers", type=int, default=2, dest="attn_layers",
                        help="[attention] Transformer 层数")
    args = parser.parse_args()

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 阶段1：特征提取 ──────────────────────────────────────────────────────────
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.force_extract:
        print(f"加载已有特征缓存: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
    else:
        samples = collect_samples(Path(args.dataset))
        print(f"\n发现 {len(samples)} 个视频文件")
        for cls_name in LABEL_MAP:
            n = sum(1 for _, lbl, _ in samples if lbl == LABEL_MAP[cls_name])
            print(f"  {cls_name}: {n}")

        print("\n加载编码器...")
        encoder = load_frozen_encoder(device)
        print("开始提取特征（首次运行，约需几分钟）...")
        cache = extract_and_cache(samples, encoder, device, cache_path)
        del encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    embeddings: torch.Tensor = cache["embeddings"]  # (N, 512)
    labels: torch.Tensor = cache["labels"]           # (N,)
    all_views: list[str] = cache["views"]

    n_total = len(labels)
    n_normal = (labels == 0).sum().item()
    n_pe = (labels == 1).sum().item()
    print(f"\n特征缓存统计: 总计={n_total}, Normal={n_normal}, PE={n_pe}")

    # ── 阶段2：划分数据集 ────────────────────────────────────────────────────────
    indices = list(range(n_total))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels.numpy(),
        random_state=args.seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=labels[temp_idx].numpy(),
        random_state=args.seed,
    )
    print(f"数据划分: 训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")

    def make_ds(idx: list[int]) -> EmbeddingDataset:
        return EmbeddingDataset(embeddings[idx], labels[idx])

    # 加权随机采样处理类别不均衡
    train_labels = labels[train_idx]
    class_counts = torch.bincount(train_labels)
    sample_weights = (1.0 / class_counts.float())[train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_labels), replacement=True
    )

    train_loader = DataLoader(make_ds(train_idx), batch_size=args.batch, sampler=sampler)
    val_loader = DataLoader(make_ds(val_idx), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(make_ds(test_idx), batch_size=args.batch, shuffle=False)

    # ── 阶段3：训练分类头 ────────────────────────────────────────────────────────
    model = build_classifier(args.head, args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n分类头: {args.head}  (可训练参数: {n_params:,})")

    class_weight = (n_total / (2.0 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0

    print(f"\n开始训练 (epochs={args.epochs}, lr={args.lr}, batch={args.batch})")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValAcc':>8}  {'BestAcc':>8}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_y, val_pred, _ = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_y, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"{epoch:6d}  {train_loss:10.4f}  {val_acc:8.4f}  {best_val_acc:8.4f}"
            )

        if no_improve >= args.patience:
            print(f"\nEarly stopping（连续 {args.patience} epoch 验证集无提升，停在 epoch {epoch}）")
            break

    # ── 阶段4：测试集评估 ────────────────────────────────────────────────────────
    assert best_state is not None
    model.load_state_dict(best_state)

    # 验证集最终结果
    val_y, val_pred, val_prob = evaluate(model, val_loader, device)
    val_views = [all_views[i] for i in val_idx]
    print_metrics("验证集", val_y, val_pred, val_prob, val_views)

    # 测试集结果
    test_y, test_pred, test_prob = evaluate(model, test_loader, device)
    test_views = [all_views[i] for i in test_idx]
    print_metrics("测试集", test_y, test_pred, test_prob, test_views)

    # 保存最优分类头权重（文件名含头类型，避免覆盖）
    save_path = HERE / f"pe_classifier_{args.head}_best.pt"
    torch.save(
        {
            "model_state": best_state,
            "head": args.head,
            "val_acc": best_val_acc,
            "args": vars(args),
        },
        save_path,
    )
    print(f"\n最优模型已保存 → {save_path}")


if __name__ == "__main__":
    main()
