#!/usr/bin/env python3
"""
Raw 6-way view classification data-scaling experiment.

Task:
    Predict the original dataset folder label directly:
        A4 / IVC / MS / PSL / PSS / SX

Compared methods:
    1. pretrained_direct       : frozen original 11-way EchoPrime view head,
                                 no gradient updates; align 11->6 via majority
                                 vote on the current train subset.
    2. pretrained_linear_probe : pretrained backbone, freeze features, train
                                 only a new 6-way head.
    3. pretrained_partial_ft   : pretrained backbone, fine-tune only the last
                                 ConvNeXt stage (+ downsample) and the head.
    4. pretrained_full_ft      : pretrained backbone, fine-tune all layers.
    5. scratch_full_ft         : random initialization, full fine-tuning.

Design choices compared with the first script:
    - Keep the original 6-way raw labels (no 4-way merging).
    - Remove the previous double imbalance correction
      (WeightedRandomSampler + class-weighted CE).
      Here we use plain shuffled batches + class-weighted CE only.
    - Add mild warmup, label smoothing, and higher early-stopping patience.
    - Save split manifest, per-run metrics, training history, and per-sample
      test predictions for later analysis.

Typical usage:
    "/local/scratch/luming/echoprime/.venv/bin/python" view_raw_finetune_scaling.py \
      --ratios 0.15 0.30 0.50 0.75 1.00 \
      --output-dir ./view_raw_finetune_results_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys

# __echope_path_bootstrap__
import sys as _echope_sys
from pathlib import Path as _EchoPEPath
def _echope_setup_paths():
    here = _EchoPEPath(__file__).resolve().parent
    root = here
    for anc in [here, *here.parents]:
        if (anc / "echo_paths.py").exists():
            root = anc
            break
    exp = root / "experiment"
    candidates = [root, exp]
    if exp.is_dir():
        candidates += [p for p in sorted(exp.iterdir()) if p.is_dir()]
    for _p in candidates:
        _sp = str(_p)
        if _sp not in _echope_sys.path:
            _echope_sys.path.append(_sp)
_echope_setup_paths()
# __end_echope_path_bootstrap__
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ORIG_CWD = Path.cwd().resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from echo_paths import setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

import utils  # noqa: E402


RAW_VIEWS = ["A4", "IVC", "MS", "PSL", "PSS", "SX"]
RAW_TO_IDX = {v: i for i, v in enumerate(RAW_VIEWS)}
IDX_TO_RAW = {i: v for v, i in RAW_TO_IDX.items()}
PRETRAINED_VIEWS = list(utils.COARSE_VIEWS)
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}

MEAN = torch.tensor([29.110628, 28.076836, 29.096405], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083], dtype=torch.float32).view(3, 1, 1)

# Fallback for pretrained 11->raw mapping if a pretrained class never appears
# in the current train subset. This is only used for uncovered classes.
FALLBACK_PRETRAINED_TO_RAW = {
    "A2C": "A4",
    "A3C": "A4",
    "A4C": "A4",
    "A5C": "A4",
    "Apical_Doppler": "A4",
    "Doppler_Parasternal_Long": "PSL",
    "Parasternal_Long": "PSL",
    "Doppler_Parasternal_Short": "PSS",
    "Parasternal_Short": "PSS",
    "Subcostal": "SX",
    "SSN": "IVC",
}


@dataclass
class Sample:
    path: str
    source_group: str
    raw_view: str
    label: int
    strata: str


class CachedFrameDataset(Dataset):
    def __init__(self, frames_uint8: torch.Tensor, labels: np.ndarray, indices: np.ndarray):
        self.frames_uint8 = frames_uint8
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = self.frames_uint8[real_idx].float()
        x = x.sub(MEAN).div(STD)
        y = int(self.labels[real_idx])
        return x, y, real_idx


def resolve_user_path(path: Path) -> Path:
    return path if path.is_absolute() else (ORIG_CWD / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw-label view fine-tuning scaling experiment.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=HERE.parent.parent.parent / "dataset" / "preprocessed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "view_raw_finetune_results_v2",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.15, 0.30, 0.50, 0.75, 1.00],
        help="Train-pool sampling ratios.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional multi-seed run. If set, overrides --seed.",
    )
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--val-frac-total", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3)
    parser.add_argument("--partial-backbone-lr", type=float, default=5e-5)
    parser.add_argument("--partial-head-lr", type=float, default=5e-4)
    parser.add_argument("--full-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--full-head-lr", type=float, default=3e-4)
    parser.add_argument("--scratch-lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for source_group in ("Normal", "PE"):
        source_dir = dataset_root / source_group
        if not source_dir.is_dir():
            continue
        for raw_view_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            raw_view = raw_view_dir.name
            if raw_view not in RAW_TO_IDX:
                continue
            for path in sorted(raw_view_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    samples.append(
                        Sample(
                            path=str(path),
                            source_group=source_group,
                            raw_view=raw_view,
                            label=RAW_TO_IDX[raw_view],
                            strata=f"{source_group}_{raw_view}",
                        )
                    )
    if not samples:
        raise RuntimeError(f"No videos found under {dataset_root}")
    return samples


def preprocess_first_frame_uint8(video_path: str) -> torch.Tensor:
    frames = utils.read_video_rgb_numpy(video_path)
    frame = utils.crop_and_scale(frames[0])
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return torch.from_numpy(frame).permute(2, 0, 1).contiguous()


def build_or_load_frame_cache(samples: list[Sample], cache_path: Path, rebuild: bool) -> torch.Tensor:
    expected_paths = [s.path for s in samples]
    if cache_path.is_file() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("paths") == expected_paths:
            print(f"[cache] loaded: {cache_path}")
            return cache["frames"].to(torch.uint8)
        print("[cache] path mismatch, rebuilding cache.")

    frames = []
    for sample in tqdm(samples, desc="Caching first frames"):
        frames.append(preprocess_first_frame_uint8(sample.path))
    stacked = torch.stack(frames).to(torch.uint8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"paths": expected_paths, "frames": stacked}, cache_path)
    print(f"[cache] saved: {cache_path}")
    return stacked


def make_splits(samples: list[Sample], test_frac: float, val_frac_total: float, seed: int):
    all_indices = np.arange(len(samples))
    strata = np.array([s.strata for s in samples], dtype=object)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_frac,
        stratify=strata,
        random_state=seed,
        shuffle=True,
    )
    val_inner = val_frac_total / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_inner,
        stratify=strata[train_val_idx],
        random_state=seed + 1,
        shuffle=True,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), strata


def stratified_subsample(indices: np.ndarray, strata: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    if ratio >= 1.0:
        return np.sort(indices)
    sub_idx, _ = train_test_split(
        indices,
        train_size=ratio,
        stratify=strata[indices],
        random_state=seed,
        shuffle=True,
    )
    return np.sort(sub_idx)


def make_loader(
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    ds = CachedFrameDataset(frames_uint8, labels, indices)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True,
    )


def compute_per_class_ovr_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    total = len(y_true)
    for idx, name in IDX_TO_RAW.items():
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        tn = int(np.sum((y_true != idx) & (y_pred != idx)))
        out[name] = float((tp + tn) / total)
    return out


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(RAW_VIEWS))))
    per_class_recall = {}
    for idx, name in IDX_TO_RAW.items():
        denom = max(int(cm[idx].sum()), 1)
        per_class_recall[name] = float(cm[idx, idx] / denom)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_recall": per_class_recall,
        "per_class_accuracy_ovr": compute_per_class_ovr_accuracy(y_true, y_pred),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true = []
    all_pred = []
    all_real_idx = []
    for x, y, real_idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_true.extend(y.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_real_idx.extend(real_idx.numpy().tolist())
    return np.array(all_true), np.array(all_pred), np.array(all_real_idx)


def build_pretrained_11way(device: torch.device) -> nn.Module:
    model = torchvision.models.convnext_base(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 11)
    state = torch.load(HERE.parent.parent / "model_data" / "weights" / "view_classifier.pt", map_location="cpu")
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(device)


def build_finetune_model(init_mode: str, device: torch.device) -> nn.Module:
    model = torchvision.models.convnext_base(weights=None)
    if init_mode == "pretrained":
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 11)
        state = torch.load(HERE.parent.parent / "model_data" / "weights" / "view_classifier.pt", map_location="cpu")
        model.load_state_dict(state)
    elif init_mode != "scratch":
        raise ValueError(f"Unknown init_mode: {init_mode}")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(RAW_VIEWS))
    return model.to(device)


def configure_trainable(model: nn.Module, train_mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if train_mode == "linear_probe":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif train_mode == "partial":
        # Unfreeze the final downsample + final ConvNeXt stage + classifier.
        for p in model.features[6].parameters():
            p.requires_grad = True
        for p in model.features[7].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif train_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")


def build_optimizer(model: nn.Module, method: str, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    if method == "pretrained_linear_probe":
        return torch.optim.AdamW(
            [{"params": head_params, "lr": args.linear_probe_lr, "weight_decay": args.weight_decay}]
        )
    if method == "pretrained_partial_ft":
        groups = []
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": args.partial_backbone_lr, "weight_decay": args.weight_decay}
            )
        if head_params:
            groups.append({"params": head_params, "lr": args.partial_head_lr, "weight_decay": args.weight_decay})
        return torch.optim.AdamW(groups)
    if method == "pretrained_full_ft":
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.full_backbone_lr, "weight_decay": args.weight_decay},
                {"params": head_params, "lr": args.full_head_lr, "weight_decay": args.weight_decay},
            ]
        )
    if method == "scratch_full_ft":
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": args.scratch_lr, "weight_decay": args.weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": args.scratch_lr, "weight_decay": args.weight_decay})
        return torch.optim.AdamW(groups)

    raise ValueError(f"Unknown method: {method}")


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(current_epoch: int) -> float:
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = (current_epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def majority_vote_mapping_from_subset(
    model_11: nn.Module,
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    subset_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, int]:
    loader = make_loader(frames_uint8, labels, subset_indices, args.batch_size, args.num_workers, training=False)
    y_true, y_pred11, _ = predict_model(model_11, loader, device)

    bucket: dict[int, list[int]] = {i: [] for i in range(len(PRETRAINED_VIEWS))}
    for t, p in zip(y_true.tolist(), y_pred11.tolist()):
        bucket[p].append(t)

    mapping: dict[int, int] = {}
    for pred_idx in range(len(PRETRAINED_VIEWS)):
        if bucket[pred_idx]:
            raw_idx = Counter(bucket[pred_idx]).most_common(1)[0][0]
            mapping[pred_idx] = int(raw_idx)
        else:
            fallback_raw = FALLBACK_PRETRAINED_TO_RAW[PRETRAINED_VIEWS[pred_idx]]
            mapping[pred_idx] = RAW_TO_IDX[fallback_raw]
    return mapping


def evaluate_direct_pretrained(
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    subset_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict[int, int]]:
    model = build_pretrained_11way(device)
    mapping = majority_vote_mapping_from_subset(model, frames_uint8, labels, subset_indices, args, device)
    test_loader = make_loader(frames_uint8, labels, test_indices, args.batch_size, args.num_workers, training=False)
    y_true, y_pred11, real_idx = predict_model(model, test_loader, device)
    y_pred6 = np.array([mapping[int(p)] for p in y_pred11], dtype=np.int64)
    metrics = evaluate_predictions(y_true, y_pred6)
    return metrics, y_true, y_pred6, {int(k): int(v) for k, v in mapping.items()}


def save_test_predictions_csv(
    out_csv: Path,
    samples: list[Sample],
    real_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "source_group", "raw_view_true", "raw_view_pred"])
        for idx, t, p in zip(real_indices.tolist(), y_true.tolist(), y_pred.tolist()):
            sample = samples[int(idx)]
            writer.writerow([sample.path, sample.source_group, IDX_TO_RAW[int(t)], IDX_TO_RAW[int(p)]])


def run_trainable_method(
    method: str,
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    subset_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    if method == "pretrained_linear_probe":
        model = build_finetune_model("pretrained", device)
        configure_trainable(model, "linear_probe")
    elif method == "pretrained_partial_ft":
        model = build_finetune_model("pretrained", device)
        configure_trainable(model, "partial")
    elif method == "pretrained_full_ft":
        model = build_finetune_model("pretrained", device)
        configure_trainable(model, "full")
    elif method == "scratch_full_ft":
        model = build_finetune_model("scratch", device)
        configure_trainable(model, "full")
    else:
        raise ValueError(f"Unknown trainable method: {method}")

    optimizer = build_optimizer(model, method, args)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = make_loader(frames_uint8, labels, subset_indices, args.batch_size, args.num_workers, training=True)
    val_loader = make_loader(frames_uint8, labels, val_indices, args.batch_size, args.num_workers, training=False)
    test_loader = make_loader(frames_uint8, labels, test_indices, args.batch_size, args.num_workers, training=False)

    train_label_tensor = torch.tensor(labels[subset_indices], dtype=torch.long)
    class_counts = torch.bincount(train_label_tensor, minlength=len(RAW_VIEWS)).float().clamp(min=1)
    class_weights = (len(train_label_tensor) / (len(RAW_VIEWS) * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_val = -1.0
    best_epoch = 0
    best_state = None
    history: list[dict[str, float]] = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
        scheduler.step()

        val_y, val_pred, _ = predict_model(model, val_loader, device)
        val_metrics = evaluate_predictions(val_y, val_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(n_seen, 1),
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if float(val_metrics["macro_f1"]) > best_val:
            best_val = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    test_y, test_pred, test_real_idx = predict_model(model, test_loader, device)
    test_metrics = evaluate_predictions(test_y, test_pred)

    result = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val),
        "history": history,
        "metrics": test_metrics,
    }
    return result, test_y, test_pred, test_real_idx


def save_split_manifest(path: Path, samples: list[Sample], train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split_name, indices in (("train_pool", train_idx), ("val", val_idx), ("test", test_idx)):
        payload[split_name] = [
            {
                "path": samples[i].path,
                "source_group": samples[i].source_group,
                "raw_view": samples[i].raw_view,
                "label": samples[i].label,
                "strata": samples[i].strata,
            }
            for i in indices
        ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_user_path(args.dataset_root)
    args.output_dir = resolve_user_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = args.output_dir / "predictions"
    cache_path = args.output_dir / "first_frame_cache.pt"
    device = torch.device(args.device)

    seeds = args.seeds if args.seeds else [args.seed]
    set_seed(seeds[0])
    samples = collect_samples(args.dataset_root)
    frames_uint8 = build_or_load_frame_cache(samples, cache_path, rebuild=args.rebuild_cache)
    labels = np.array([s.label for s in samples], dtype=np.int64)

    print(f"[info] dataset_root={args.dataset_root}")
    print(f"[info] output_dir={args.output_dir}")
    print(f"[info] device={device}")
    print(f"[info] seeds={seeds}")

    all_results: list[dict[str, object]] = []
    method_order = [
        "pretrained_direct",
        "pretrained_linear_probe",
        "pretrained_partial_ft",
        "pretrained_full_ft",
        "scratch_full_ft",
    ]

    split_manifests: dict[str, str] = {}
    for seed in seeds:
        print(f"\n######## seed={seed} ########")
        set_seed(seed)
        train_pool_idx, val_idx, test_idx, strata = make_splits(
            samples, args.test_frac, args.val_frac_total, seed
        )
        split_manifest_path = args.output_dir / f"split_manifest_seed{seed}.json"
        save_split_manifest(split_manifest_path, samples, train_pool_idx, val_idx, test_idx)
        split_manifests[str(seed)] = str(split_manifest_path)
        print(f"[info] split sizes: train_pool={len(train_pool_idx)} val={len(val_idx)} test={len(test_idx)}")

        for ratio in args.ratios:
            subset_idx = stratified_subsample(
                train_pool_idx, strata, ratio, seed=seed + int(1000 * ratio)
            )
            subset_counter = Counter(labels[subset_idx].tolist())
            subset_counts = {IDX_TO_RAW[i]: int(subset_counter.get(i, 0)) for i in range(len(RAW_VIEWS))}
            print(f"\n=== seed={seed} ratio={ratio:.2f} n_train={len(subset_idx)} ===")

            for method in method_order:
                print(f"[run] {method}")
                if method == "pretrained_direct":
                    metrics, y_true, y_pred, mapping = evaluate_direct_pretrained(
                        frames_uint8, labels, subset_idx, test_idx, args, device
                    )
                    real_indices = test_idx
                    result = {
                        "seed": int(seed),
                        "method": method,
                        "ratio": float(ratio),
                        "n_train_subset": int(len(subset_idx)),
                        "subset_class_counts": subset_counts,
                        "mapping_11_to_6": {
                            PRETRAINED_VIEWS[k]: IDX_TO_RAW[v] for k, v in mapping.items()
                        },
                        "best_epoch": None,
                        "best_val_macro_f1": None,
                        "history": [],
                        "metrics": metrics,
                    }
                else:
                    train_result, y_true, y_pred, real_indices = run_trainable_method(
                        method, frames_uint8, labels, subset_idx, val_idx, test_idx, args, device
                    )
                    result = {
                        "seed": int(seed),
                        "method": method,
                        "ratio": float(ratio),
                        "n_train_subset": int(len(subset_idx)),
                        "subset_class_counts": subset_counts,
                        "mapping_11_to_6": None,
                        "best_epoch": train_result["best_epoch"],
                        "best_val_macro_f1": train_result["best_val_macro_f1"],
                        "history": train_result["history"],
                        "metrics": train_result["metrics"],
                    }

                pred_path = predictions_dir / f"seed-{seed}" / f"{method}__ratio-{ratio:.2f}.csv"
                save_test_predictions_csv(pred_path, samples, real_indices, y_true, y_pred)
                result["prediction_csv"] = str(pred_path)
                all_results.append(result)

                m = result["metrics"]
                print(
                    "      acc={:.4f} macro_f1={:.4f} bal_acc={:.4f}".format(
                        m["accuracy"], m["macro_f1"], m["balanced_accuracy"]
                    )
                )

    payload = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
            "ratios": [float(r) for r in args.ratios],
            "seed": args.seed,
            "seeds": [int(s) for s in seeds],
            "test_frac": args.test_frac,
            "val_frac_total": args.val_frac_total,
            "epochs": args.epochs,
            "patience": args.patience,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "label_smoothing": args.label_smoothing,
        },
        "class_names": RAW_VIEWS,
        "pretrained_class_names": PRETRAINED_VIEWS,
        "split_manifests": split_manifests,
        "results": all_results,
    }
    out_json = args.output_dir / "results.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_json}")


if __name__ == "__main__":
    main()
