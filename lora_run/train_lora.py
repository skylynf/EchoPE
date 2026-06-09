#!/usr/bin/env python3
"""
LoRA 微调 EchoPrime 视频编码器进行 PE 分类

策略：对 MViT-v2-s 编码器注入 LoRA 适配器（仅微调低秩增量），
     联合训练分类头，端到端反向传播。

数据集结构（与 pe_run 一致），默认根目录为 EchoPrime 上一级下的 hope_dataset，例如：
  /path/to/echoprime/hope_dataset/
  ├── Normal/{A4,PSS}/*.mkv
  └── PE/{A4,PSS}/*.mkv

用法（在 lora_run 目录下）：
  ../.venv/bin/python train_lora.py                           # 默认参数
  ../.venv/bin/python train_lora.py --rank 8 --alpha 16       # 自定义 LoRA 秩和 alpha
  ../.venv/bin/python train_lora.py --target-modules qkv proj mlp  # 选择注入目标
  ../.venv/bin/python train_lora.py --epochs 30 --lr 1e-4     # 调整训练参数
  ../.venv/bin/python train_lora.py --head residual            # 使用残差 MLP 分类头
  ../.venv/bin/python train_lora.py --merge-and-save           # 训练后将 LoRA 合并进权重

LoRA 注入目标 (--target-modules):
  qkv   : 注意力 QKV 联合投影 (blocks.*.attn.qkv)
  proj  : 注意力输出投影 (blocks.*.attn.project.0)
  mlp   : MLP 层 (blocks.*.mlp.0, blocks.*.mlp.3)
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
from echo_paths import (  # noqa: E402
    DEFAULT_HOPE_DATASET,
    ECHO_ROOT,
    setup_echo_root_cwd,
)

setup_echo_root_cwd()
import utils  # noqa: E402

# ── 常量 ─────────────────────────────────────────────────────────────────────────
DATASET_ROOT = DEFAULT_HOPE_DATASET

FRAMES_TO_TAKE = 32
FRAME_STRIDE = 2
VIDEO_SIZE = 224
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

LABEL_MAP = {"Normal": 0, "PE": 1}
VIEWS = ["A4", "PSS"]


# ═══════════════════════════════════════════════════════════════════════════════
#  训练期视频增广
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AugmentationConfig:
    intensity_enabled: bool = False
    intensity_p: float = 0.5
    brightness: float = 0.10
    contrast_min: float = 0.90
    contrast_max: float = 1.10
    gamma_min: float = 0.90
    gamma_max: float = 1.10
    noise_std: float = 0.01

    affine_enabled: bool = False
    affine_p: float = 0.5
    translate_h: float = 0.05
    translate_w: float = 0.05
    scale_min: float = 0.95
    scale_max: float = 1.05
    rotate_deg: float = 8.0

    temporal_enabled: bool = False
    temporal_p: float = 0.3
    temporal_min_frac: float = 0.80
    temporal_random_start: bool = False

    def enabled(self) -> bool:
        return self.intensity_enabled or self.affine_enabled or self.temporal_enabled

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def add_augmentation_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--aug-intensity", action="store_true", dest="aug_intensity",
                        help="启用训练期强度增广（亮度/对比度/gamma/噪声）")
    parser.add_argument("--aug-intensity-p", type=float, default=0.5, dest="aug_intensity_p",
                        help="整段强度增广生效概率")
    parser.add_argument("--aug-brightness", type=float, default=0.10, dest="aug_brightness",
                        help="亮度扰动幅度（相对 255）")
    parser.add_argument("--aug-contrast", nargs=2, type=float, default=[0.90, 1.10], dest="aug_contrast",
                        help="对比度缩放范围 min max")
    parser.add_argument("--aug-gamma", nargs=2, type=float, default=[0.90, 1.10], dest="aug_gamma",
                        help="gamma 范围 min max")
    parser.add_argument("--aug-noise-std", type=float, default=0.01, dest="aug_noise_std",
                        help="高斯噪声标准差（相对 255）")

    parser.add_argument("--aug-affine", action="store_true", dest="aug_affine",
                        help="启用训练期 2D 视频仿射增广（全时间步共享同一参数）")
    parser.add_argument("--aug-affine-p", type=float, default=0.5, dest="aug_affine_p",
                        help="整段仿射增广生效概率")
    parser.add_argument("--aug-translate", nargs=2, type=float, default=[0.05, 0.05], dest="aug_translate",
                        help="仿射平移幅度（相对 H/W）")
    parser.add_argument("--aug-scale", nargs=2, type=float, default=[0.95, 1.05], dest="aug_scale",
                        help="各向同性缩放范围 min max")
    parser.add_argument("--aug-rotate", type=float, default=8.0, dest="aug_rotate",
                        help="旋转角度上限（±度）")

    parser.add_argument("--aug-temporal", action="store_true", dest="aug_temporal",
                        help="启用训练期时间裁剪后重采样")
    parser.add_argument("--aug-temporal-p", type=float, default=0.3, dest="aug_temporal_p",
                        help="时间裁剪生效概率")
    parser.add_argument("--aug-temporal-min-frac", type=float, default=0.80, dest="aug_temporal_min_frac",
                        help="最短时间裁剪比例")
    parser.add_argument("--aug-temporal-random-start", action="store_true", dest="aug_temporal_random_start",
                        help="训练期先随机截取连续 32 帧窗口，再执行其余增广")
    return parser


def build_augmentation_config(args: argparse.Namespace | None) -> AugmentationConfig:
    if args is None:
        return AugmentationConfig()
    contrast = list(getattr(args, "aug_contrast", [0.90, 1.10]))
    gamma = list(getattr(args, "aug_gamma", [0.90, 1.10]))
    translate = list(getattr(args, "aug_translate", [0.05, 0.05]))
    scale = list(getattr(args, "aug_scale", [0.95, 1.05]))
    return AugmentationConfig(
        intensity_enabled=bool(getattr(args, "aug_intensity", False)),
        intensity_p=float(getattr(args, "aug_intensity_p", 0.5)),
        brightness=float(getattr(args, "aug_brightness", 0.10)),
        contrast_min=float(contrast[0]),
        contrast_max=float(contrast[1]),
        gamma_min=float(gamma[0]),
        gamma_max=float(gamma[1]),
        noise_std=float(getattr(args, "aug_noise_std", 0.01)),
        affine_enabled=bool(getattr(args, "aug_affine", False)),
        affine_p=float(getattr(args, "aug_affine_p", 0.5)),
        translate_h=float(translate[0]),
        translate_w=float(translate[1]),
        scale_min=float(scale[0]),
        scale_max=float(scale[1]),
        rotate_deg=float(getattr(args, "aug_rotate", 8.0)),
        temporal_enabled=bool(getattr(args, "aug_temporal", False)),
        temporal_p=float(getattr(args, "aug_temporal_p", 0.3)),
        temporal_min_frac=float(getattr(args, "aug_temporal_min_frac", 0.80)),
        temporal_random_start=bool(getattr(args, "aug_temporal_random_start", False)),
    )


def _pad_or_repeat_video(video: torch.Tensor, target_frames: int, repeat: bool) -> torch.Tensor:
    if video.shape[1] >= target_frames:
        return video
    if repeat and video.shape[1] > 0:
        reps = math.ceil(target_frames / video.shape[1])
        return video.repeat(1, reps, 1, 1)[:, :target_frames, :, :]
    pad = torch.zeros(
        video.shape[0],
        target_frames - video.shape[1],
        video.shape[2],
        video.shape[3],
        dtype=video.dtype,
    )
    return torch.cat([video, pad], dim=1)


def _select_temporal_window(video: torch.Tensor, clip_length: int, random_start: bool, repeat_short: bool) -> torch.Tensor:
    video = _pad_or_repeat_video(video, clip_length, repeat=repeat_short)
    if video.shape[1] == clip_length:
        return video
    max_start = video.shape[1] - clip_length
    start = random.randint(0, max_start) if random_start and max_start > 0 else 0
    return video[:, start:start + clip_length, :, :]


def _apply_intensity_augmentation(video: torch.Tensor, cfg: AugmentationConfig) -> torch.Tensor:
    if not cfg.intensity_enabled or random.random() >= cfg.intensity_p:
        return video
    out = video.clone()
    if cfg.brightness > 0 and random.random() < 0.5:
        delta = random.uniform(-cfg.brightness, cfg.brightness) * 255.0
        out = out + delta
    if random.random() < 0.5:
        factor = random.uniform(cfg.contrast_min, cfg.contrast_max)
        mean_val = out.mean()
        out = (out - mean_val) * factor + mean_val
    if random.random() < 0.5:
        gamma = random.uniform(cfg.gamma_min, cfg.gamma_max)
        out = (out.clamp(0.0, 255.0) / 255.0).pow(gamma) * 255.0
    if cfg.noise_std > 0 and random.random() < 0.5:
        out = out + torch.randn_like(out) * cfg.noise_std * 255.0
    return out.clamp(0.0, 255.0)


def _apply_affine_augmentation(video: torch.Tensor, cfg: AugmentationConfig) -> torch.Tensor:
    if not cfg.affine_enabled or random.random() >= cfg.affine_p:
        return video
    _, depth, height, width = video.shape
    angle = random.uniform(-cfg.rotate_deg, cfg.rotate_deg)
    scale_factor = random.uniform(cfg.scale_min, cfg.scale_max)
    translate_h = random.uniform(-cfg.translate_h, cfg.translate_h) * height
    translate_w = random.uniform(-cfg.translate_w, cfg.translate_w) * width
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad) * scale_factor
    sin_a = math.sin(angle_rad) * scale_factor
    theta = torch.tensor(
        [
            [cos_a, -sin_a, translate_w / max(width / 2.0, 1.0)],
            [sin_a, cos_a, translate_h / max(height / 2.0, 1.0)],
        ],
        dtype=video.dtype,
    ).unsqueeze(0)
    frames = video.permute(1, 0, 2, 3)
    grid = F.affine_grid(theta.expand(depth, -1, -1), frames.size(), align_corners=False)
    warped = F.grid_sample(frames, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.permute(1, 0, 2, 3)


def _apply_temporal_crop_augmentation(video: torch.Tensor, cfg: AugmentationConfig) -> torch.Tensor:
    if not cfg.temporal_enabled or random.random() >= cfg.temporal_p:
        return video
    _, depth, height, width = video.shape
    if depth <= 1:
        return video
    min_frames = max(2, math.ceil(cfg.temporal_min_frac * depth))
    if min_frames >= depth:
        return video
    crop_frames = random.randint(min_frames, depth)
    start = random.randint(0, depth - crop_frames)
    cropped = video[:, start:start + crop_frames, :, :].unsqueeze(0)
    resized = F.interpolate(
        cropped,
        size=(depth, height, width),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _normalize_and_subsample(video: torch.Tensor) -> torch.Tensor:
    video = video.sub(MEAN).div(STD)
    return video[:, 0:FRAMES_TO_TAKE:FRAME_STRIDE, :, :]


# ═══════════════════════════════════════════════════════════════════════════════
#  LoRA 实现
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    在冻结的 nn.Linear 上叠加低秩适配器。

    前向计算：y = W_frozen @ x + (B @ A @ x) * scaling
    其中 A ∈ R^{in×r}, B ∈ R^{r×out}, scaling = alpha / rank。
    只有 A、B 参与梯度更新。
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """将 LoRA 增量合并回原始 Linear，返回普通 nn.Linear。"""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B) * self.scaling
            merged.weight.copy_(self.original.weight + delta.T)
            if self.original.bias is not None:
                merged.bias.copy_(self.original.bias)
        return merged


def _set_submodule(model: nn.Module, target_path: str, new_module: nn.Module):
    """按点分路径设置子模块（支持数字索引，如 'blocks.0.attn.qkv'）。"""
    parts = target_path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _get_submodule(model: nn.Module, target_path: str) -> nn.Module:
    parts = target_path.split(".")
    m = model
    for part in parts:
        if part.isdigit():
            m = m[int(part)]
        else:
            m = getattr(m, part)
    return m


TARGET_MODULE_PATTERNS = {
    "qkv": ".attn.qkv",
    "proj": ".attn.project.0",
    "mlp": (".mlp.0", ".mlp.3"),
}


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> list[str]:
    """
    向模型中符合条件的 Linear 层注入 LoRA 适配器。
    返回被注入的模块路径列表。
    """
    if target_modules is None:
        target_modules = ["qkv", "proj"]

    suffixes: list[str] = []
    for t in target_modules:
        pat = TARGET_MODULE_PATTERNS.get(t)
        if pat is None:
            raise ValueError(f"未知 target module: {t}，可选: {list(TARGET_MODULE_PATTERNS)}")
        if isinstance(pat, tuple):
            suffixes.extend(pat)
        else:
            suffixes.append(pat)

    injected: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) for s in suffixes):
            continue
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=lora_dropout)
        _set_submodule(model, name, lora_layer)
        injected.append(name)

    return injected


def merge_lora(model: nn.Module) -> None:
    """将模型中所有 LoRALinear 合并回普通 Linear（用于导出推理模型）。"""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            merged = module.merge_weights()
            _set_submodule(model, name, merged)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ═══════════════════════════════════════════════════════════════════════════════
#  视频预处理 & 数据集
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_single_video(
    path: str,
    training: bool = False,
    augmentation: AugmentationConfig | None = None,
) -> torch.Tensor:
    """读取单个视频并返回 (3, 16, 224, 224) 的预处理张量。"""
    pixels = utils.read_video_rgb_numpy(path)
    if pixels is None or len(pixels) == 0:
        raise RuntimeError(f"视频为空或无法读取: {path}")
    x = np.zeros((len(pixels), VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.float32)
    for i in range(len(pixels)):
        x[i] = utils.crop_and_scale(pixels[i])
    x = torch.as_tensor(x).permute(3, 0, 1, 2)
    cfg = augmentation or AugmentationConfig()
    if training and cfg.enabled():
        x = _select_temporal_window(
            x,
            clip_length=FRAMES_TO_TAKE,
            random_start=cfg.temporal_random_start,
            repeat_short=True,
        )
        x = _apply_intensity_augmentation(x, cfg)
        x = _apply_affine_augmentation(x, cfg)
        x = _apply_temporal_crop_augmentation(x, cfg)
        return _normalize_and_subsample(x)
    x = x.sub(MEAN).div(STD)
    if x.shape[1] < FRAMES_TO_TAKE:
        pad = torch.zeros(3, FRAMES_TO_TAKE - x.shape[1], VIDEO_SIZE, VIDEO_SIZE)
        x = torch.cat([x, pad], dim=1)
    return x[:, 0:FRAMES_TO_TAKE:FRAME_STRIDE, :, :]


def collect_samples(root: Path) -> list[tuple[str, int, str]]:
    """遍历 root/{Normal,PE}/{A4,PSS}/*.mkv，返回 [(path, label, view), ...]"""
    samples: list[tuple[str, int, str]] = []
    for cls_name, label in LABEL_MAP.items():
        for view in VIEWS:
            view_dir = root / cls_name / view
            if not view_dir.is_dir():
                print(f"[WARN] 目录不存在，跳过: {view_dir}")
                continue
            for p in sorted(view_dir.glob("*.mkv")):
                samples.append((str(p), label, view))
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
#  五折 CV 划分加载（与 experiments/utils/split_cv.py 产出对齐）
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CV_SPLITS_DIR = (
    EXP_ROOT / "data" / "splits" / "normal_vs_pe_cv" / "cartesian"
)


def _build_dataset_index(root: Path) -> dict[str, tuple[str, int, str]]:
    """扫描 root/{Normal,PE}/{A4,PSS}/* 建立 {file_stem: (path,label,view)} 索引。"""
    valid_exts = {".mp4", ".mkv", ".avi", ".mov"}
    index: dict[str, tuple[str, int, str]] = {}
    for cls_name, label in LABEL_MAP.items():
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            continue
        for view_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            view = view_dir.name
            for p in sorted(view_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in valid_exts:
                    index[p.stem] = (str(p), label, view)
    return index


def load_cv_fold_samples(
    splits_dir: Path,
    fold: int,
    dataset_root: Path,
) -> dict[str, list[tuple[str, int, str]]]:
    """读取 splits_dir/fold_{fold}/{train,val,test}.csv 并解析到本地数据集。

    CSV 由 experiments/utils/split_cv.py 产出，格式 `path,label`。
    通过 path 的文件名 stem 在 dataset_root 下定位真实视频（兼容 .mkv/.mp4 等）。
    返回 {"train"/"val"/"test": [(path,label,view), ...]}。
    """
    import csv

    splits_dir = Path(splits_dir)
    dataset_root = Path(dataset_root)
    fold_dir = splits_dir / f"fold_{fold}"
    if not fold_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 CV fold 目录: {fold_dir}\n"
            f"  请先运行 experiments/utils/split_cv.py 生成五折划分。"
        )

    index = _build_dataset_index(dataset_root)
    if not index:
        raise RuntimeError(f"数据集索引为空: {dataset_root}")

    out: dict[str, list[tuple[str, int, str]]] = {}
    for split in ("train", "val", "test"):
        csv_path = fold_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"缺少 CSV: {csv_path}")
        items: list[tuple[str, int, str]] = []
        missing: list[str] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = Path(str(row["path"])).stem
                hit = index.get(sid)
                if hit is None:
                    missing.append(sid)
                    continue
                items.append(hit)
        if missing:
            print(f"[WARN] fold {fold} {split}: {len(missing)} 个 id 在 "
                  f"{dataset_root} 下找不到（前 5: {missing[:5]}）")
        out[split] = items
    return out


def samples_from_feature_cache(cache: dict) -> list[tuple[str, int, str]] | None:
    """
    若 pe_run 特征缓存含 `paths`（与 embeddings/labels/views 行对齐），
    返回 [(path, label, view), ...]，供 LoRA 与冻结特征使用同一批视频。
    旧版缓存无 `paths` 时返回 None。
    """
    paths = cache.get("paths")
    if not paths:
        return None
    labels = cache["labels"]
    views = cache["views"]
    n = int(labels.shape[0]) if hasattr(labels, "shape") else len(labels)
    if len(paths) != n or len(views) != n:
        return None
    if isinstance(labels, torch.Tensor):
        labels_it = labels.tolist()
    else:
        labels_it = [int(x) for x in labels]
    return [(str(p), int(lbl), str(v)) for p, lbl, v in zip(paths, labels_it, views)]


class PEVideoDataset(Dataset):
    """端到端训练用视频数据集，每次按需加载和预处理。"""

    def __init__(
        self,
        samples: list[tuple[str, int, str]],
        training: bool = False,
        augmentation: AugmentationConfig | None = None,
    ):
        self.samples = samples
        self.training = training
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label, _view = self.samples[idx]
        video = preprocess_single_video(path, training=self.training, augmentation=self.augmentation)
        return video, label


# ═══════════════════════════════════════════════════════════════════════════════
#  分类头（与 pe_run 保持一致，方便对比）
# ═══════════════════════════════════════════════════════════════════════════════

class PEClassifier(nn.Module):
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
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256,
                 dropout: float = 0.3, num_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


HEAD_REGISTRY: dict[str, type] = {
    "mlp": PEClassifier,
    "residual": PEClassifierResidual,
}


def build_classifier(head: str, hidden: int, dropout: float, **kwargs) -> nn.Module:
    in_dim = int(kwargs.get("in_dim", 512))
    if head == "mlp":
        return PEClassifier(in_dim=in_dim, hidden_dim=hidden, dropout=dropout)
    elif head == "residual":
        return PEClassifierResidual(
            in_dim=in_dim, hidden_dim=hidden, dropout=dropout,
            num_blocks=kwargs.get("num_blocks", 3),
        )
    raise ValueError(f"未知分类头: {head}，可选: {list(HEAD_REGISTRY)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LoRA + 分类头 联合模型
# ═══════════════════════════════════════════════════════════════════════════════

class LoRAEchoPrimeClassifier(nn.Module):
    """
    封装 LoRA 微调的 MViT 编码器 + 分类头。
    encoder 输出 512 维嵌入，classifier 输出 2 类 logits。
    """

    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)  # (B, 512)
        return self.classifier(emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  训练 / 评估
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * len(y)
        total_samples += len(y)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(y.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


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


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA 微调 EchoPrime MViT 编码器进行 PE 二分类"
    )
    # 数据
    parser.add_argument(
        "--dataset",
        default=str(DATASET_ROOT),
        help="hope_dataset 根目录（默认：EchoPrime 同级目录下的 hope_dataset）",
    )
    parser.add_argument(
        "--feature-cache",
        default="",
        help="可选：pe_features_cache.pt；若内含 paths 则与其对齐使用同一批视频（否则仍扫描 --dataset）",
    )
    # LoRA 参数
    parser.add_argument("--rank", type=int, default=8, help="LoRA 秩 r")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA 缩放 alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout",
                        help="LoRA 层 dropout")
    parser.add_argument("--target-modules", nargs="+", default=["qkv", "proj"],
                        choices=list(TARGET_MODULE_PATTERNS), dest="target_modules",
                        help="LoRA 注入目标: qkv / proj / mlp")
    # 分类头
    parser.add_argument("--head", default="mlp", choices=list(HEAD_REGISTRY),
                        help="分类头类型")
    parser.add_argument("--hidden", type=int, default=256, help="分类头隐层维度")
    parser.add_argument("--head-dropout", type=float, default=0.3, dest="head_dropout",
                        help="分类头 dropout")
    parser.add_argument("--num-blocks", type=int, default=3, dest="num_blocks",
                        help="[residual] 残差块数量")
    # 训练
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5, help="LoRA 参数学习率")
    parser.add_argument("--head-lr", type=float, default=1e-3, dest="head_lr",
                        help="分类头学习率（独立于 LoRA lr）")
    parser.add_argument("--batch", type=int, default=4, help="batch size（视频占显存大）")
    parser.add_argument("--grad-accum", type=int, default=4, dest="grad_accum",
                        help="梯度累积步数，有效 batch = batch × grad_accum")
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, dest="warmup_epochs",
                        help="学习率 warmup 轮数")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（仅用于模型初始化/采样，不再用于数据划分）")
    parser.add_argument("--cv-splits", default=str(DEFAULT_CV_SPLITS_DIR),
                        dest="cv_splits",
                        help="五折 CV 划分目录（包含 fold_0..fold_4 子目录），"
                             "由 experiments/utils/split_cv.py 生成；"
                             "与其他库实验保持一致")
    parser.add_argument("--fold", type=int, default=0,
                        help="使用第几折（0..n_splits-1）")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers",
                        help="DataLoader 工作线程数")
    add_augmentation_args(parser)
    # 保存
    parser.add_argument("--merge-and-save", action="store_true", dest="merge_and_save",
                        help="训练后将 LoRA 合并进编码器权重并保存完整模型")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── 1. 加载五折 CV 划分（由 experiments/utils/split_cv.py 生成） ────────────
    print(f"\nCV 划分目录: {args.cv_splits}  fold={args.fold}")
    fold_samples = load_cv_fold_samples(
        Path(args.cv_splits), args.fold, Path(args.dataset),
    )
    train_samples = fold_samples["train"]
    val_samples = fold_samples["val"]
    test_samples = fold_samples["test"]
    if not (train_samples and val_samples and test_samples):
        print("[ERROR] CV fold 中存在空 split，请检查 --cv-splits / --dataset / --fold")
        sys.exit(1)
    print(f"数据划分: 训练={len(train_samples)}, 验证={len(val_samples)}, "
          f"测试={len(test_samples)}")

    val_views_list = [s[2] for s in val_samples]
    test_views_list = [s[2] for s in test_samples]

    # 加权采样
    train_labels = torch.tensor([s[1] for s in train_samples])
    class_counts = torch.bincount(train_labels)
    sample_weights = (1.0 / class_counts.float())[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
    aug_config = build_augmentation_config(args)

    train_loader = DataLoader(
        PEVideoDataset(train_samples, training=True, augmentation=aug_config), batch_size=args.batch,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        PEVideoDataset(val_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        PEVideoDataset(test_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ── 2. 加载编码器 + 注入 LoRA ────────────────────────────────────────────────
    print("\n加载 EchoPrime 编码器...")
    weights_path = ECHO_ROOT / "model_data" / "weights" / "echo_prime_encoder.pt"
    ckpt = torch.load(str(weights_path), map_location="cpu")
    encoder = torchvision.models.video.mvit_v2_s()
    encoder.head[-1] = nn.Linear(encoder.head[-1].in_features, 512)
    encoder.load_state_dict(ckpt)

    for p in encoder.parameters():
        p.requires_grad = False

    injected = inject_lora(
        encoder,
        rank=args.rank,
        alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    print(f"\nLoRA 注入完成，共 {len(injected)} 个层:")
    for name in injected:
        print(f"  • {name}")

    # ── 3. 构建联合模型 ──────────────────────────────────────────────────────────
    classifier = build_classifier(
        args.head, args.hidden, args.head_dropout, num_blocks=args.num_blocks,
    )
    model = LoRAEchoPrimeClassifier(encoder, classifier).to(device)

    trainable, total = count_parameters(model)
    print(f"\n参数量: 可训练={trainable:,} / 总计={total:,}  "
          f"({100 * trainable / total:.2f}%)")

    # ── 4. 优化器 & 调度器 ───────────────────────────────────────────────────────
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 1e-4},
    ])

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_total = len(train_labels)
    class_weight = (n_total / (2.0 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── 5. 训练循环 ──────────────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_state: dict | None = None
    no_improve = 0

    effective_batch = args.batch * args.grad_accum
    print(f"\n开始训练: epochs={args.epochs}, batch={args.batch}×{args.grad_accum}="
          f"{effective_batch}, lr_lora={args.lr}, lr_head={args.head_lr}")
    print(f"{'Epoch':>6}  {'Loss':>10}  {'ValAcc':>8}  {'ValAUC':>8}  {'BestAUC':>8}")
    print("-" * 52)

    step_count = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

            total_loss += loss.item() * args.grad_accum * len(y)
            total_n += len(y)
            pbar.set_postfix(loss=f"{total_loss / total_n:.4f}")

        epoch_loss = total_loss / total_n

        val_y, val_pred, val_prob = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_y, val_pred)
        try:
            val_auc = roc_auc_score(val_y, val_prob)
        except ValueError:
            val_auc = float("nan")

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        marker = " *" if improved else ""
        print(f"{epoch:6d}  {epoch_loss:10.4f}  {val_acc:8.4f}  {val_auc:8.4f}  "
              f"{best_val_auc:8.4f}{marker}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping（连续 {args.patience} epoch 无提升，停在 epoch {epoch}）")
            break

    # ── 6. 测试评估 ──────────────────────────────────────────────────────────────
    assert best_state is not None, "训练未产生有效模型"
    model.load_state_dict(best_state)
    model.to(device)

    val_y, val_pred, val_prob = evaluate(model, val_loader, device)
    val_views = val_views_list
    print_metrics("验证集", val_y, val_pred, val_prob, val_views)

    test_y, test_pred, test_prob = evaluate(model, test_loader, device)
    test_views = test_views_list
    print_metrics("测试集", test_y, test_pred, test_prob, test_views)

    # ── 7. 保存 ──────────────────────────────────────────────────────────────────
    # 保存 LoRA 适配器 + 分类头（轻量）
    lora_state = {k: v for k, v in best_state.items()
                  if "lora_A" in k or "lora_B" in k or "classifier" in k}
    lora_save_path = HERE / "lora_pe_best.pt"
    torch.save({
        "lora_state": lora_state,
        "config": {
            "rank": args.rank,
            "alpha": args.alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "head": args.head,
            "hidden": args.hidden,
            "head_dropout": args.head_dropout,
            "num_blocks": args.num_blocks,
        },
        "val_auc": best_val_auc,
        "args": vars(args),
    }, lora_save_path)
    print(f"\nLoRA 适配器已保存 → {lora_save_path}  "
          f"(大小: {lora_save_path.stat().st_size / 1024:.1f} KB)")

    # 可选：合并 LoRA 并保存完整编码器
    if args.merge_and_save:
        model.load_state_dict(best_state)
        merge_lora(model.encoder)
        merged_path = HERE / "merged_encoder_pe.pt"
        torch.save(model.encoder.state_dict(), merged_path)
        print(f"合并后编码器已保存 → {merged_path}")

        full_path = HERE / "merged_full_model_pe.pt"
        torch.save({
            "encoder_state": model.encoder.state_dict(),
            "classifier_state": model.classifier.state_dict(),
            "config": vars(args),
        }, full_path)
        print(f"完整模型已保存 → {full_path}")


if __name__ == "__main__":
    main()
