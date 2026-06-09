#!/usr/bin/env python3
"""
LoRA + 蒸馏 + view 辅助监督  联合微调 EchoPrime 视频编码器进行 PE 分类。

总损失：
    L = CE_pe  +  λ_kd · L_distill  +  λ_view · L_view_aux  +  λ_view_kd · L_view_kd

支持的消融组合（通过 flag 控制）：
  - --lora on/off                              是否注入 LoRA
  - --ft-mode {lora,full,partial-stage4,frozen}  当 --lora off 时控制 encoder 可训练范围
  - --kd-mode {none,cos,l2,rkd-d,rkd-a,combo}  与 frozen EchoPrime teacher 的特征蒸馏
  - --view-aux {none,ce,kd,both}               视角辅助监督模式

用法（在 finetuning 目录下）：
  # 1) 现有 LoRA only baseline（与 train_lora.py 等价，作为对照）
  ../../.venv/bin/python train_lora_distill.py

  # 2) LoRA + cos KD（推荐起步）
  ../../.venv/bin/python train_lora_distill.py --kd-mode cos --kd-weight 0.5

  # 3) LoRA + KD + view 辅助 CE
  ../../.venv/bin/python train_lora_distill.py --kd-mode combo --view-aux ce

  # 4) "KD 单独用"：full-FT + KD（不用 LoRA）
  ../../.venv/bin/python train_lora_distill.py --lora off --ft-mode full --kd-mode combo

  # 5) Smoke test
  ../../.venv/bin/python train_lora_distill.py --epochs 2 --smoke
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
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from echo_paths import DEFAULT_HOPE_DATASET, ECHO_ROOT, setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

# 复用 train_lora.py 的工具
from train_lora import (  # noqa: E402
    FRAMES_TO_TAKE,
    FRAME_STRIDE,
    AugmentationConfig,
    HEAD_REGISTRY,
    LABEL_MAP,
    MEAN,
    STD,
    TARGET_MODULE_PATTERNS,
    VIDEO_SIZE,
    VIEWS,
    PEVideoDataset,
    add_augmentation_args,
    build_classifier,
    build_augmentation_config,
    collect_samples,
    count_parameters,
    inject_lora,
    samples_from_feature_cache,
)

from distill import KD_MODES, kd_loss, load_frozen_teacher  # noqa: E402
from aux_heads import (  # noqa: E402
    ViewAuxHead,
    ViewKDHead,
    load_view_teacher,
    view_ce_loss,
    view_kd_loss,
    views_to_tensor,
)

DATASET_ROOT = DEFAULT_HOPE_DATASET
VIEW_AUX_MODES = ("none", "ce", "kd", "both")
FT_MODES = ("lora", "full", "partial-stage4", "frozen")
HEAD_VIEW_MODES = ("none", "coarse4")
NUM_COARSE_VIEWS = 4


# ═══════════════════════════════════════════════════════════════════════════════
#  数据集（带 view 标签）
# ═══════════════════════════════════════════════════════════════════════════════

class PEVideoDatasetWithView(Dataset):
    """与 PEVideoDataset 相同，但额外返回 view 字符串与可选 coarse-view 索引。"""

    def __init__(
        self,
        samples: list[tuple[str, int, str]],
        training: bool = False,
        augmentation: AugmentationConfig | None = None,
    ):
        self.samples = samples
        self._inner = PEVideoDataset(
            [(sample[0], sample[1], sample[2]) for sample in samples],
            training=training,
            augmentation=augmentation,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video, label = self._inner[idx]
        view = self.samples[idx][2]
        if len(self.samples[idx]) >= 4:
            coarse_view_idx = int(self.samples[idx][3])
            return video, label, view, coarse_view_idx
        return video, label, view


def view_collate(batch):
    videos = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    views = [b[2] for b in batch]
    if len(batch[0]) >= 4:
        coarse_view_idx = torch.tensor([int(b[3]) for b in batch], dtype=torch.long)
        return videos, labels, views, coarse_view_idx
    return videos, labels, views


def unpack_view_batch(
    batch,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor | None]:
    videos = batch[0]
    labels = batch[1]
    views = batch[2] if len(batch) >= 3 else []
    coarse_view_idx = batch[3] if len(batch) >= 4 else None
    return videos, labels, views, coarse_view_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  联合模型
# ═══════════════════════════════════════════════════════════════════════════════

class LoRAEchoPrimeDistillModel(nn.Module):
    """
    encoder（可选 LoRA） + 主分类头 + (可选) view 辅助头 + (可选) view-KD 头。

    forward 返回 dict:
        {
          "logits"     : (B, 2)         主任务 PE 二分类
          "emb"        : (B, 512)       encoder 嵌入
          "view_ce"    : (B, 2) | None  view A4/PSS 辅助 logits
          "view_kd"    : (B, 11) | None view-KD 辅助 logits
        }
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        view_aux: str = "none",
        head_view_mode: str = "none",
        num_head_views: int = NUM_COARSE_VIEWS,
        view_aux_hidden: int = 128,
        view_kd_hidden: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.view_aux_mode = view_aux
        self.head_view_mode = head_view_mode
        self.num_head_views = num_head_views
        if self.head_view_mode not in HEAD_VIEW_MODES:
            raise ValueError(f"Unknown head_view_mode: {self.head_view_mode}. Choices: {HEAD_VIEW_MODES}")

        self.view_ce_head = (
            ViewAuxHead(in_dim=512, hidden=view_aux_hidden, dropout=head_dropout)
            if view_aux in ("ce", "both") else None
        )
        self.view_kd_head = (
            ViewKDHead(in_dim=512, hidden=view_kd_hidden, dropout=head_dropout)
            if view_aux in ("kd", "both") else None
        )

    def _build_head_view_features(
        self,
        coarse_view_idx: torch.Tensor | None,
        emb: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.head_view_mode == "none":
            return None
        if coarse_view_idx is None:
            raise ValueError(f"coarse_view_idx is required when head_view_mode={self.head_view_mode!r}")
        return F.one_hot(coarse_view_idx.long(), num_classes=self.num_head_views).to(dtype=emb.dtype, device=emb.device)

    def forward(
        self,
        x: torch.Tensor,
        coarse_view_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        emb = self.encoder(x)
        head_view = self._build_head_view_features(coarse_view_idx, emb)
        head_input = torch.cat([emb, head_view], dim=1) if head_view is not None else emb
        out = {
            "logits": self.classifier(head_input),
            "emb": emb,
            "view_ce": self.view_ce_head(emb) if self.view_ce_head is not None else None,
            "view_kd": self.view_kd_head(emb) if self.view_kd_head is not None else None,
        }
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Encoder 构建：LoRA on/off + ft-mode
# ═══════════════════════════════════════════════════════════════════════════════

def build_encoder(args, weights_path: Path) -> tuple[nn.Module, list[str]]:
    """
    构建 encoder 并按 (--lora, --ft-mode) 配置可训练参数。

    返回 (encoder, injected_lora_paths)。
    """
    ckpt = torch.load(str(weights_path), map_location="cpu")
    encoder = torchvision.models.video.mvit_v2_s()
    encoder.head[-1] = nn.Linear(encoder.head[-1].in_features, 512)
    encoder.load_state_dict(ckpt)

    # 默认全部冻结
    for p in encoder.parameters():
        p.requires_grad = False

    injected: list[str] = []

    if args.lora == "on":
        injected = inject_lora(
            encoder,
            rank=args.rank,
            alpha=args.alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
        # ft-mode 在 lora=on 时强制为 lora（即只 LoRA + head 训练）
        if args.ft_mode != "lora":
            print(f"[WARN] --lora on 时 --ft-mode 强制为 'lora'，忽略 {args.ft_mode!r}")
    else:
        # lora=off 时按 ft-mode 决定哪些层训练
        if args.ft_mode == "full":
            for p in encoder.parameters():
                p.requires_grad = True
        elif args.ft_mode == "partial-stage4":
            # 只解冻最后一组 blocks 与 head
            # MViT 的 blocks 共 16 层（mvit_v2_s），后 4 层视作 stage4
            n_unfreeze_blocks = 4
            total = len(encoder.blocks)
            for i in range(total - n_unfreeze_blocks, total):
                for p in encoder.blocks[i].parameters():
                    p.requires_grad = True
            for p in encoder.head.parameters():
                p.requires_grad = True
        elif args.ft_mode == "frozen":
            pass  # 全冻结（仅训练 classifier / view heads）
        else:
            raise ValueError(f"未知 --ft-mode: {args.ft_mode}（lora off 时不能选 'lora'）")

    return encoder, injected


# ═══════════════════════════════════════════════════════════════════════════════
#  KD warmup 调度
# ═══════════════════════════════════════════════════════════════════════════════

def kd_weight_at_epoch(epoch: int, warmup_epochs: int, target: float) -> float:
    """Epoch 1 起 0，warmup_epochs 内线性升到 target。"""
    if warmup_epochs <= 0:
        return target
    if epoch <= warmup_epochs:
        return target * (epoch / warmup_epochs)
    return target


# ═══════════════════════════════════════════════════════════════════════════════
#  评估
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        x, y, _views, coarse_view_idx = unpack_view_batch(batch)
        x = x.to(device)
        coarse_view_idx = coarse_view_idx.to(device) if coarse_view_idx is not None else None
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x, coarse_view_idx=coarse_view_idx)
            logits = out["logits"]
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(y.numpy() if isinstance(y, torch.Tensor) else list(y))
        all_preds.extend(preds)
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(split: str, y_true, y_pred, y_prob, views: list[str] | None = None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*52}\n  {split} 评估\n{'='*52}")
    print(f"  Accuracy : {acc:.4f}    F1 : {f1:.4f}    AUC : {auc:.4f}")
    print(f"  Confusion Matrix (行=真, 列=预):")
    print(f"             Normal    PE")
    print(f"  Normal  {cm[0, 0]:8d}  {cm[0, 1]:6d}")
    print(f"  PE      {cm[1, 0]:8d}  {cm[1, 1]:6d}")
    if views is not None:
        print(f"\n  按视角 acc:")
        for v in VIEWS:
            mask = [i for i, vv in enumerate(views) if vv == v]
            if mask:
                v_acc = accuracy_score(y_true[mask], y_pred[mask])
                print(f"    {v:4s}: {v_acc:.4f}  ({len(mask)} 样本)")


# ═══════════════════════════════════════════════════════════════════════════════
#  训练循环
# ═══════════════════════════════════════════════════════════════════════════════

def first_frames(videos: torch.Tensor) -> torch.Tensor:
    """videos: (B, 3, T, H, W) → (B, 3, H, W) 取 t=0。"""
    return videos[:, :, 0, :, :]


def train_one_epoch(
    epoch: int,
    model: LoRAEchoPrimeDistillModel,
    loader: DataLoader,
    teacher_kd: nn.Module | None,
    teacher_view: nn.Module | None,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    sums = {"ce": 0.0, "kd": 0.0, "view_ce": 0.0, "view_kd": 0.0, "total": 0.0}
    n_seen = 0

    cur_kd_w = kd_weight_at_epoch(epoch, args.kd_warmup_epochs, args.kd_weight)
    cur_view_w = args.view_weight
    cur_view_kd_w = args.view_kd_weight

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False, ncols=100)

    for step, batch in enumerate(pbar):
        videos, labels, views, coarse_view_idx = unpack_view_batch(batch)
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        coarse_view_idx = coarse_view_idx.to(device, non_blocking=True) if coarse_view_idx is not None else None

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(videos, coarse_view_idx=coarse_view_idx)
            logits = out["logits"]
            emb = out["emb"]

            loss_ce = criterion(logits, labels)
            loss_total = loss_ce
            loss_kd = torch.zeros((), device=device)
            loss_v_ce = torch.zeros((), device=device)
            loss_v_kd = torch.zeros((), device=device)

            # 特征蒸馏
            if teacher_kd is not None and args.kd_mode != "none" and cur_kd_w > 0:
                with torch.no_grad():
                    t_emb = teacher_kd(videos)
                # KD 在 fp32 计算（避免 RKD 的距离矩阵在 fp16 里数值不稳）
                loss_kd = kd_loss(
                    emb.float(), t_emb.float(),
                    mode=args.kd_mode, lam_rkd=args.lam_rkd,
                )
                loss_total = loss_total + cur_kd_w * loss_kd

            # view 辅助 CE
            if out["view_ce"] is not None:
                view_lbl = views_to_tensor(views, device)
                loss_v_ce = view_ce_loss(out["view_ce"], view_lbl)
                loss_total = loss_total + cur_view_w * loss_v_ce

            # view-KD（用预训练 view_classifier 当 teacher）
            if out["view_kd"] is not None and teacher_view is not None:
                ff = first_frames(videos)
                with torch.no_grad():
                    t_view_logits = teacher_view(ff)
                loss_v_kd = view_kd_loss(
                    out["view_kd"].float(), t_view_logits.float(),
                    temperature=args.view_kd_temperature,
                )
                loss_total = loss_total + cur_view_kd_w * loss_v_kd

            loss_step = loss_total / args.grad_accum

        scaler.scale(loss_step).backward()

        if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        bs = labels.size(0)
        sums["ce"] += loss_ce.item() * bs
        sums["kd"] += loss_kd.item() * bs
        sums["view_ce"] += loss_v_ce.item() * bs
        sums["view_kd"] += loss_v_kd.item() * bs
        sums["total"] += loss_total.item() * bs
        n_seen += bs

        pbar.set_postfix({
            "ce": f"{sums['ce']/n_seen:.3f}",
            "kd": f"{sums['kd']/n_seen:.3f}",
            "vce": f"{sums['view_ce']/n_seen:.3f}",
            "vkd": f"{sums['view_kd']/n_seen:.3f}",
            "kdw": f"{cur_kd_w:.2f}",
        })

    return {k: v / max(n_seen, 1) for k, v in sums.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA + 蒸馏 + view 辅助 微调 EchoPrime 进行 PE 二分类",
    )
    # 数据
    p.add_argument("--dataset", default=str(DATASET_ROOT))
    p.add_argument("--feature-cache", default="", dest="feature_cache",
                   help="可选 pe_features_cache.pt（含 paths 时与基线行对齐）")
    # encoder 微调模式
    p.add_argument("--lora", choices=["on", "off"], default="on")
    p.add_argument("--ft-mode", choices=list(FT_MODES), default="lora", dest="ft_mode")
    # LoRA
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    p.add_argument("--target-modules", nargs="+", default=["qkv", "proj"],
                   choices=list(TARGET_MODULE_PATTERNS), dest="target_modules")
    # 分类头
    p.add_argument("--head", default="mlp", choices=list(HEAD_REGISTRY))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.3, dest="head_dropout")
    p.add_argument("--num-blocks", type=int, default=3, dest="num_blocks")
    p.add_argument("--head-view-mode", choices=list(HEAD_VIEW_MODES), default="none",
                   dest="head_view_mode")
    # KD（特征/关系蒸馏）
    p.add_argument("--kd-mode", choices=list(KD_MODES), default="none", dest="kd_mode")
    p.add_argument("--kd-weight", type=float, default=0.5, dest="kd_weight")
    p.add_argument("--kd-warmup-epochs", type=int, default=2, dest="kd_warmup_epochs")
    p.add_argument("--lam-rkd", type=float, default=1.0, dest="lam_rkd",
                   help="combo 模式下 rkd-d 的权重")
    # view 辅助
    p.add_argument("--view-aux", choices=list(VIEW_AUX_MODES), default="none",
                   dest="view_aux")
    p.add_argument("--view-weight", type=float, default=0.3, dest="view_weight",
                   help="view CE 辅助权重")
    p.add_argument("--view-kd-weight", type=float, default=0.3, dest="view_kd_weight",
                   help="view-KD 蒸馏权重")
    p.add_argument("--view-kd-temperature", type=float, default=2.0,
                   dest="view_kd_temperature")
    # 训练
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="LoRA 参数（或 encoder 主体）学习率")
    p.add_argument("--head-lr", type=float, default=1e-3, dest="head_lr",
                   help="分类头 / view 头 学习率")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    p.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    p.add_argument("--warmup-epochs", type=int, default=3, dest="warmup_epochs")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    add_augmentation_args(p)
    # 其他
    p.add_argument("--teacher-device", default="", dest="teacher_device",
                   help="可选: cpu / cuda:N，默认与 student 同 device")
    p.add_argument("--smoke", action="store_true",
                   help="只用 8 条样本 + 2 epoch 跑通，用于测脚本")
    p.add_argument("--save-prefix", default="", dest="save_prefix",
                   help="保存文件名前缀；默认根据 flag 自动生成")
    return p.parse_args()


def auto_save_prefix(args: argparse.Namespace) -> str:
    parts = []
    parts.append("lora" if args.lora == "on" else f"nolora-{args.ft_mode}")
    if args.kd_mode != "none":
        parts.append(f"kd-{args.kd_mode}")
    if args.view_aux != "none":
        parts.append(f"view-{args.view_aux}")
    if args.head_view_mode != "none":
        parts.append(f"headview-{args.head_view_mode}")
    aug_tags = []
    if getattr(args, "aug_intensity", False):
        aug_tags.append("intensity")
    if getattr(args, "aug_affine", False):
        aug_tags.append("affine")
    if getattr(args, "aug_temporal", False):
        aug_tags.append("temporal")
    if aug_tags:
        parts.append("aug-" + "-".join(aug_tags))
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()

    # 一致性检查
    if args.lora == "on" and args.ft_mode != "lora":
        print(f"[INFO] --lora on，--ft-mode 自动忽略（保持 'lora'）")
        args.ft_mode = "lora"
    if args.lora == "off" and args.ft_mode == "lora":
        print(f"[ERROR] --lora off 时 --ft-mode 不能为 'lora'，请选 full/partial-stage4/frozen")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_device = torch.device(args.teacher_device) if args.teacher_device else device
    print(f"设备: student={device}  teacher={teacher_device}")

    # ── 1. 数据 ──────────────────────────────────────────────────────────────────
    samples: list[tuple[str, int, str]]
    if args.feature_cache:
        cpath = Path(args.feature_cache)
        if cpath.is_file():
            fc = torch.load(str(cpath), map_location="cpu")
            from_cache = samples_from_feature_cache(fc)
            samples = from_cache or collect_samples(Path(args.dataset))
        else:
            samples = collect_samples(Path(args.dataset))
    else:
        samples = collect_samples(Path(args.dataset))

    if len(samples) == 0:
        print(f"[ERROR] 找不到样本，dataset={args.dataset}")
        sys.exit(1)

    if args.smoke:
        # 每类 4 条，方便端到端 smoke
        per_cls = {0: [], 1: []}
        for s in samples:
            if len(per_cls[s[1]]) < 4:
                per_cls[s[1]].append(s)
        samples = per_cls[0] + per_cls[1]
        args.epochs = max(args.epochs, 2)
        args.patience = 999
        args.batch = min(args.batch, 2)
        args.grad_accum = 1
        args.warmup_epochs = 0
        args.kd_warmup_epochs = 0
        print(f"[SMOKE] 缩减为 {len(samples)} 条样本，epochs={args.epochs}")

    print(f"\n共 {len(samples)} 个视频")
    for cls_name, lbl in LABEL_MAP.items():
        print(f"  {cls_name}: {sum(1 for _, l, _ in samples if l == lbl)}")

    labels_all = [s[1] for s in samples]
    views_all = [s[2] for s in samples]
    indices = list(range(len(samples)))

    if args.smoke:
        # smoke：交错两类后做 4/2/2 划分，保证三组都两类齐全（让指标有意义）
        cls0 = [i for i in indices if labels_all[i] == 0]
        cls1 = [i for i in indices if labels_all[i] == 1]
        # 交错：c0_0, c1_0, c0_1, c1_1, ...
        interleaved = [v for pair in zip(cls0, cls1) for v in pair]
        train_idx = interleaved[:4]
        val_idx = interleaved[4:6]
        test_idx = interleaved[6:8]
    else:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, stratify=labels_all, random_state=args.seed,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5,
            stratify=[labels_all[i] for i in temp_idx], random_state=args.seed,
        )
    print(f"划分: 训练={len(train_idx)} 验证={len(val_idx)} 测试={len(test_idx)}")

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_labels = torch.tensor([s[1] for s in train_samples])
    class_counts = torch.bincount(train_labels, minlength=2)
    sample_weights = (1.0 / class_counts.float().clamp(min=1))[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
    aug_config = build_augmentation_config(args)

    train_loader = DataLoader(
        PEVideoDatasetWithView(train_samples, training=True, augmentation=aug_config), batch_size=args.batch,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn=view_collate,
    )
    val_loader = DataLoader(
        PEVideoDatasetWithView(val_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        collate_fn=view_collate,
    )
    test_loader = DataLoader(
        PEVideoDatasetWithView(test_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        collate_fn=view_collate,
    )

    # ── 2. Encoder + 主分类头 ────────────────────────────────────────────────────
    print("\n构建 encoder...")
    weights_path = ECHO_ROOT / "model_data" / "weights" / "echo_prime_encoder.pt"
    encoder, injected = build_encoder(args, weights_path)
    if injected:
        print(f"  LoRA 注入 {len(injected)} 层（target={args.target_modules}, "
              f"r={args.rank}, alpha={args.alpha}）")
    else:
        print(f"  无 LoRA；ft-mode={args.ft_mode}")

    classifier_in_dim = 512 + (NUM_COARSE_VIEWS if args.head_view_mode == "coarse4" else 0)
    classifier = build_classifier(
        args.head, args.hidden, args.head_dropout,
        num_blocks=args.num_blocks,
        in_dim=classifier_in_dim,
    )
    model = LoRAEchoPrimeDistillModel(
        encoder, classifier,
        view_aux=args.view_aux,
        head_view_mode=args.head_view_mode,
        head_dropout=args.head_dropout,
    ).to(device)

    trainable, total = count_parameters(model)
    print(f"  参数: 可训练={trainable:,} / 总={total:,}  "
          f"({100 * trainable / total:.2f}%)")

    # ── 3. Teachers ──────────────────────────────────────────────────────────────
    teacher_kd = None
    if args.kd_mode != "none":
        print("\n加载 KD teacher (frozen EchoPrime encoder)...")
        teacher_kd = load_frozen_teacher(weights_path, teacher_device)

    teacher_view = None
    if args.view_aux in ("kd", "both"):
        print("加载 view-KD teacher (frozen view_classifier)...")
        view_w = ECHO_ROOT / "model_data" / "weights" / "view_classifier.pt"
        teacher_view = load_view_teacher(view_w, teacher_device)

    # ── 4. 优化器：分组 lr ───────────────────────────────────────────────────────
    enc_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    aux_params: list[nn.Parameter] = []
    if model.view_ce_head is not None:
        aux_params.extend(model.view_ce_head.parameters())
    if model.view_kd_head is not None:
        aux_params.extend(model.view_kd_head.parameters())

    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.lr,
                             "weight_decay": args.weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr,
                             "weight_decay": 1e-4})
    if aux_params:
        param_groups.append({"params": aux_params, "lr": args.head_lr,
                             "weight_decay": 1e-4})

    if not param_groups:
        print("[ERROR] 没有可训练参数")
        sys.exit(1)

    optimizer = torch.optim.AdamW(param_groups)

    total_steps = max(1, args.epochs * math.ceil(len(train_loader) / args.grad_accum))
    warmup_steps = max(0, args.warmup_epochs *
                       math.ceil(len(train_loader) / args.grad_accum))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_total = len(train_labels)
    class_weight = (n_total / (2.0 * class_counts.clamp(min=1).float())).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── 5. 训练循环 ──────────────────────────────────────────────────────────────
    best_val_auc = -1.0
    best_state: dict | None = None
    no_improve = 0

    print(f"\n开始训练: lora={args.lora} ft={args.ft_mode} "
          f"kd={args.kd_mode}(w={args.kd_weight}) view={args.view_aux}")
    print(f"{'Epoch':>6}  {'CE':>8}  {'KD':>8}  {'VCE':>8}  {'VKD':>8}  "
          f"{'ValAcc':>8}  {'ValAUC':>8}  {'BestAUC':>8}")
    print("-" * 88)

    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(
            epoch, model, train_loader, teacher_kd, teacher_view,
            criterion, optimizer, scheduler, scaler, device, args,
        )

        val_y, val_pred, val_prob = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_y, val_pred)
        try:
            val_auc = roc_auc_score(val_y, val_prob)
        except ValueError:
            val_auc = float("nan")

        improved = (val_auc > best_val_auc) if not math.isnan(val_auc) else False
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            mark = " *"
        else:
            no_improve += 1
            mark = ""

        print(f"{epoch:6d}  {losses['ce']:8.4f}  {losses['kd']:8.4f}  "
              f"{losses['view_ce']:8.4f}  {losses['view_kd']:8.4f}  "
              f"{val_acc:8.4f}  {val_auc:8.4f}  {best_val_auc:8.4f}{mark}")

        if no_improve >= args.patience:
            print(f"\nEarly stop @ epoch {epoch}（{args.patience} 无提升）")
            break

    # ── 6. 测试 ──────────────────────────────────────────────────────────────────
    if best_state is None:
        print("[WARN] 没有 best_state（可能所有 epoch 都 NaN AUC），使用最终权重")
    else:
        model.load_state_dict(best_state)
    model.to(device)

    val_y, val_pred, val_prob = evaluate(model, val_loader, device)
    print_metrics("验证集", val_y, val_pred, val_prob,
                  [views_all[i] for i in val_idx])

    test_y, test_pred, test_prob = evaluate(model, test_loader, device)
    print_metrics("测试集", test_y, test_pred, test_prob,
                  [views_all[i] for i in test_idx])

    # ── 7. 保存 ──────────────────────────────────────────────────────────────────
    save_prefix = args.save_prefix or auto_save_prefix(args)
    save_state = {k: v for k, v in (best_state or model.state_dict()).items()
                  if "lora_A" in k or "lora_B" in k
                  or "classifier" in k
                  or "view_ce_head" in k or "view_kd_head" in k}
    save_path = HERE / f"{save_prefix}.pt"
    torch.save({
        "save_state": save_state,
        "args": vars(args),
        "val_auc": best_val_auc,
    }, save_path)
    print(f"\n保存 → {save_path}  ({save_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
