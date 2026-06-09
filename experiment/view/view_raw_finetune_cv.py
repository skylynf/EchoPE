#!/usr/bin/env python3
"""
5-fold CV for the raw 6-way EchoPrime view classification task.

This script reuses the first-frame ConvNeXt pipeline from
`view_raw_finetune_scaling.py`, but replaces the random holdout split with an
external fold directory that already provides `fold_{0..4}/{train,val,test}.csv`.

The target remains the raw view label:
    A4 / IVC / MS / PSL / PSS / SX

Primary output metrics are reported as macro one-vs-rest metrics so that the
requested probability-based scores (Brier / AUROC / AUPR / F1@rec90) remain
well-defined for the 6-class task.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent.parent
ORIG_CWD = Path.cwd().resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from echo_paths import ECHO_ROOT, setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

import utils  # noqa: E402


RAW_VIEWS = ["A4", "IVC", "MS", "PSL", "PSS", "SX"]
RAW_TO_IDX = {name: idx for idx, name in enumerate(RAW_VIEWS)}
IDX_TO_RAW = {idx: name for name, idx in RAW_TO_IDX.items()}

MEAN = torch.tensor([29.110628, 28.076836, 29.096405], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083], dtype=torch.float32).view(3, 1, 1)
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
SUMMARY_HEADERS = [
    "fold",
    "test Brier ↓",
    "test F1",
    "test AUROC",
    "test AUPR",
    "test Acc",
    "test Sens",
    "test Spec",
    "test Prec",
    "F1@rec90",
    "val F1 (best)",
    "val AUROC (best)",
]
METHODS = (
    "pretrained_linear_probe",
    "pretrained_partial_ft",
    "pretrained_full_ft",
    "scratch_full_ft",
)


@dataclass
class Sample:
    path: str
    source_group: str
    raw_view: str
    label: int
    strata: str
    sample_id: str


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
    parser = argparse.ArgumentParser(description="Run 5-fold CV for 6-way raw view classification.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ECHO_ROOT.parent / "dataset" / "preprocessed",
        help="Root folder containing Normal/ and PE/ view folders.",
    )
    parser.add_argument(
        "--cv-splits",
        type=Path,
        default=ECHO_ROOT.parent / "dataset" / "preprocessed" / "all_dataset_preprocessed_cv_seed2026",
        help="Directory containing fold_0 ... fold_4 split CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "view_raw_finetune_cv_seed2026",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Fold indices to run.",
    )
    parser.add_argument(
        "--method",
        choices=list(METHODS),
        default="pretrained_full_ft",
        help="Training recipe for the 6-way classifier.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
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
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=ECHO_ROOT / "model_data" / "weights" / "view_classifier.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=2026)
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
                            path=str(path.resolve()),
                            source_group=source_group,
                            raw_view=raw_view,
                            label=RAW_TO_IDX[raw_view],
                            strata=f"{source_group}_{raw_view}",
                            sample_id=path.stem,
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
    expected_paths = [sample.path for sample in samples]
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


def load_fold_indices(
    cv_splits: Path,
    fold: int,
    sample_by_id: dict[str, int],
) -> dict[str, np.ndarray]:
    fold_dir = cv_splits / f"fold_{fold}"
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"Missing fold directory: {fold_dir}")

    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        csv_path = fold_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split csv: {csv_path}")
        indices: list[int] = []
        missing_ids: list[str] = []
        seen_ids: set[str] = set()
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = Path(str(row["path"])).stem
                if sample_id in seen_ids:
                    continue
                seen_ids.add(sample_id)
                idx = sample_by_id.get(sample_id)
                if idx is None:
                    missing_ids.append(sample_id)
                    continue
                indices.append(int(idx))
        if missing_ids:
            raise RuntimeError(
                f"Fold {fold} {split} has {len(missing_ids)} samples missing from dataset root. "
                f"First few ids: {missing_ids[:5]}"
            )
        out[split] = np.array(sorted(indices), dtype=np.int64)

    train_set = set(out["train"].tolist())
    val_set = set(out["val"].tolist())
    test_set = set(out["test"].tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError(f"Fold {fold} contains overlapping train/val/test samples.")
    return out


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


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_true: list[int] = []
    all_prob: list[np.ndarray] = []
    all_real_idx: list[int] = []
    for x, y, real_idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_true.extend(y.numpy().tolist())
        all_prob.extend(probs.tolist())
        all_real_idx.extend(real_idx.numpy().tolist())
    return (
        np.array(all_true, dtype=np.int64),
        np.array(all_prob, dtype=np.float64),
        np.array(all_real_idx, dtype=np.int64),
    )


def build_finetune_model(init_mode: str, device: torch.device, weights_path: Path) -> nn.Module:
    model = torchvision.models.convnext_base(weights=None)
    if init_mode == "pretrained":
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 11)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    elif init_mode != "scratch":
        raise ValueError(f"Unknown init_mode: {init_mode}")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(RAW_VIEWS))
    return model.to(device)


def configure_trainable(model: nn.Module, train_mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if train_mode == "linear_probe":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif train_mode == "partial":
        for param in model.features[6].parameters():
            param.requires_grad = True
        for param in model.features[7].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")


def build_optimizer(model: nn.Module, method: str, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_params = [param for param in model.classifier.parameters() if param.requires_grad]
    head_ids = {id(param) for param in head_params}
    backbone_params = [param for param in model.parameters() if param.requires_grad and id(param) not in head_ids]

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


def _safe_multiclass_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=list(range(len(RAW_VIEWS))),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def _safe_multiclass_aupr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true_oh = np.eye(len(RAW_VIEWS), dtype=np.float64)[y_true]
    try:
        return float(average_precision_score(y_true_oh, y_prob, average="macro"))
    except ValueError:
        return float("nan")


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true_oh = np.eye(len(RAW_VIEWS), dtype=np.float64)[y_true]
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))


def macro_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(RAW_VIEWS))))
    total = int(cm.sum())
    specificities = []
    for idx in range(len(RAW_VIEWS)):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = total - tp - fn - fp
        specificities.append(tn / max(tn + fp, 1))
    return float(np.mean(specificities))


def choose_threshold_for_target_recall_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> tuple[float, dict[str, float | str]]:
    thresholds = np.unique(np.round(y_prob, 6))
    if thresholds.size == 0:
        return 0.5, {
            "target_recall": float(target_recall),
            "selected_threshold": 0.5,
            "selection_rule": "fallback_fixed_threshold",
        }

    candidates: list[tuple[float, float, float, float]] = []
    fallback_rows: list[tuple[float, float, float, float]] = []
    for thr in sorted(thresholds.tolist(), reverse=True):
        pred = (y_prob >= thr).astype(np.int64)
        sens = float(recall_score(y_true, pred, zero_division=0))
        spec = float(macro_specificity(y_true, pred)) if len(np.unique(y_true)) > 2 else 0.0
        # Binary specificity is simpler and avoids dependence on helper logic.
        cm = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        tn, fp, _, _ = [int(x) for x in cm]
        spec = tn / max(tn + fp, 1)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        row = (float(thr), sens, spec, f1)
        fallback_rows.append(row)
        if sens >= target_recall:
            candidates.append(row)

    if candidates:
        threshold, sens, spec, f1 = max(candidates, key=lambda row: (row[2], row[3], row[0]))
        return threshold, {
            "target_recall": float(target_recall),
            "selected_threshold": float(threshold),
            "selected_val_recall": float(sens),
            "selected_val_specificity": float(spec),
            "selected_val_f1": float(f1),
            "selection_rule": "max_specificity_subject_to_recall",
        }

    threshold, sens, spec, f1 = max(fallback_rows, key=lambda row: (row[1], row[2], row[3], row[0]))
    return threshold, {
        "target_recall": float(target_recall),
        "selected_threshold": float(threshold),
        "selected_val_recall": float(sens),
        "selected_val_specificity": float(spec),
        "selected_val_f1": float(f1),
        "selection_rule": "max_recall_fallback",
    }


def macro_f1_at_target_recall(
    val_true: np.ndarray,
    val_prob: np.ndarray,
    test_true: np.ndarray,
    test_prob: np.ndarray,
    target_recall: float,
) -> tuple[float, dict[str, dict[str, float | str]]]:
    per_class: dict[str, dict[str, float | str]] = {}
    test_f1_scores: list[float] = []
    for class_idx, class_name in IDX_TO_RAW.items():
        val_true_bin = (val_true == class_idx).astype(np.int64)
        test_true_bin = (test_true == class_idx).astype(np.int64)
        val_prob_bin = val_prob[:, class_idx]
        test_prob_bin = test_prob[:, class_idx]
        threshold, info = choose_threshold_for_target_recall_binary(val_true_bin, val_prob_bin, target_recall)
        test_pred_bin = (test_prob_bin >= threshold).astype(np.int64)
        test_f1 = float(f1_score(test_true_bin, test_pred_bin, zero_division=0))
        per_class[class_name] = {
            **info,
            "test_f1": test_f1,
        }
        test_f1_scores.append(test_f1)
    return float(np.mean(test_f1_scores)), per_class


def evaluate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    val_true: np.ndarray | None = None,
    val_prob: np.ndarray | None = None,
    target_recall: float = 0.90,
) -> dict[str, object]:
    y_pred = y_prob.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_specificity": macro_specificity(y_true, y_pred),
        "macro_auroc_ovr": _safe_multiclass_auroc(y_true, y_prob),
        "macro_aupr_ovr": _safe_multiclass_aupr(y_true, y_prob),
        "multiclass_brier": multiclass_brier_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(RAW_VIEWS)))).tolist(),
        "confusion_label_order": RAW_VIEWS,
    }
    if val_true is not None and val_prob is not None:
        f1_rec90, per_class = macro_f1_at_target_recall(val_true, val_prob, y_true, y_prob, target_recall)
        result["macro_f1_at_recall90"] = f1_rec90
        result["f1_at_recall90_per_class"] = per_class
    return result


def save_predictions_csv(
    out_csv: Path,
    samples: list[Sample],
    split_name: str,
    real_indices: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "split",
        "sample_id",
        "path",
        "source_group",
        "raw_view_true",
        "raw_view_pred",
        *[f"prob_{name}" for name in RAW_VIEWS],
    ]
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(headers)
        for real_idx, true_idx, prob_row in zip(real_indices.tolist(), y_true.tolist(), y_prob.tolist()):
            sample = samples[int(real_idx)]
            pred_idx = int(np.argmax(np.asarray(prob_row, dtype=float)))
            writer.writerow(
                [
                    split_name,
                    sample.sample_id,
                    sample.path,
                    sample.source_group,
                    IDX_TO_RAW[int(true_idx)],
                    IDX_TO_RAW[pred_idx],
                    *[float(x) for x in prob_row],
                ]
            )


def build_model_for_method(method: str, device: torch.device, weights_path: Path) -> nn.Module:
    if method == "pretrained_linear_probe":
        model = build_finetune_model("pretrained", device, weights_path)
        configure_trainable(model, "linear_probe")
        return model
    if method == "pretrained_partial_ft":
        model = build_finetune_model("pretrained", device, weights_path)
        configure_trainable(model, "partial")
        return model
    if method == "pretrained_full_ft":
        model = build_finetune_model("pretrained", device, weights_path)
        configure_trainable(model, "full")
        return model
    if method == "scratch_full_ft":
        model = build_finetune_model("scratch", device, weights_path)
        configure_trainable(model, "full")
        return model
    raise ValueError(f"Unknown method: {method}")


def run_fold(
    fold: int,
    samples: list[Sample],
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sample_by_id: dict[str, int],
) -> dict[str, object]:
    split_indices = load_fold_indices(args.cv_splits, fold, sample_by_id)
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    run_dir = args.output_dir / f"fold-{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = run_dir / "predictions.csv"
    if predictions_csv.exists():
        predictions_csv.unlink()

    model = build_model_for_method(args.method, device, args.pretrained_weights)
    optimizer = build_optimizer(model, args.method, args)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = make_loader(frames_uint8, labels, train_idx, args.batch_size, args.num_workers, training=True)
    val_loader = make_loader(frames_uint8, labels, val_idx, args.batch_size, args.num_workers, training=False)
    test_loader = make_loader(frames_uint8, labels, test_idx, args.batch_size, args.num_workers, training=False)

    train_label_tensor = torch.tensor(labels[train_idx], dtype=torch.long)
    class_counts = torch.bincount(train_label_tensor, minlength=len(RAW_VIEWS)).float().clamp(min=1)
    class_weights = (len(train_label_tensor) / (len(RAW_VIEWS) * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    no_improve = 0

    print(
        f"[fold={fold}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} method={args.method}"
    )

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

        val_y, val_prob, _ = predict_model(model, val_loader, device)
        val_metrics = evaluate_probabilities(val_y, val_prob, target_recall=args.target_recall)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(n_seen, 1),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_auroc_ovr": float(val_metrics["macro_auroc_ovr"]),
                "val_macro_aupr_ovr": float(val_metrics["macro_aupr_ovr"]),
                "val_multiclass_brier": float(val_metrics["multiclass_brier"]),
            }
        )

        if float(val_metrics["macro_f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"Fold {fold} did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    val_y, val_prob, val_real_idx = predict_model(model, val_loader, device)
    test_y, test_prob, test_real_idx = predict_model(model, test_loader, device)
    val_metrics = evaluate_probabilities(val_y, val_prob, target_recall=args.target_recall)
    test_metrics = evaluate_probabilities(
        test_y,
        test_prob,
        val_true=val_y,
        val_prob=val_prob,
        target_recall=args.target_recall,
    )

    save_predictions_csv(predictions_csv, samples, "val", val_real_idx, val_y, val_prob)
    save_predictions_csv(predictions_csv, samples, "test", test_real_idx, test_y, test_prob)

    summary_row = {
        "fold": int(fold),
        "test Brier ↓": float(test_metrics["multiclass_brier"]),
        "test F1": float(test_metrics["macro_f1"]),
        "test AUROC": float(test_metrics["macro_auroc_ovr"]),
        "test AUPR": float(test_metrics["macro_aupr_ovr"]),
        "test Acc": float(test_metrics["accuracy"]),
        "test Sens": float(test_metrics["macro_recall"]),
        "test Spec": float(test_metrics["macro_specificity"]),
        "test Prec": float(test_metrics["macro_precision"]),
        "F1@rec90": float(test_metrics["macro_f1_at_recall90"]),
        "val F1 (best)": float(val_metrics["macro_f1"]),
        "val AUROC (best)": float(val_metrics["macro_auroc_ovr"]),
    }

    result = {
        "fold": int(fold),
        "method": args.method,
        "best_epoch": int(best_epoch),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "summary": summary_row,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "artifacts": {
            "predictions_csv": str(predictions_csv.resolve()),
        },
    }
    result_json = run_dir / "result.json"
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "[fold={}] test_f1={:.4f} test_auroc={:.4f} test_acc={:.4f}".format(
            fold,
            summary_row["test F1"],
            summary_row["test AUROC"],
            summary_row["test Acc"],
        )
    )
    return result


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_mean_std_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_headers = SUMMARY_HEADERS[1:]
    mean_row: dict[str, object] = {"fold": "mean"}
    std_row: dict[str, object] = {"fold": "std"}
    for header in metric_headers:
        values = [float(row[header]) for row in rows]
        mu, sd = _mean_std(values)
        mean_row[header] = mu
        std_row[header] = sd
    return [mean_row, std_row]


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_user_path(args.dataset_root)
    args.cv_splits = resolve_user_path(args.cv_splits)
    args.output_dir = resolve_user_path(args.output_dir)
    args.pretrained_weights = resolve_user_path(args.pretrained_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pretrained_weights.is_file() and args.method.startswith("pretrained_"):
        raise FileNotFoundError(f"Missing pretrained weights: {args.pretrained_weights}")

    device = torch.device(args.device)
    set_seed(args.seed)

    samples = collect_samples(args.dataset_root)
    sample_by_id = {sample.sample_id: idx for idx, sample in enumerate(samples)}
    if len(sample_by_id) != len(samples):
        raise RuntimeError("Duplicate sample ids detected in dataset root.")

    cache_path = args.output_dir / "first_frame_cache.pt"
    frames_uint8 = build_or_load_frame_cache(samples, cache_path, rebuild=args.rebuild_cache)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)

    print(f"[info] dataset_root={args.dataset_root}")
    print(f"[info] cv_splits={args.cv_splits}")
    print(f"[info] output_dir={args.output_dir}")
    print(f"[info] method={args.method}")
    print(f"[info] folds={args.folds}")
    print(f"[info] device={device}")

    fold_results = []
    summary_rows = []
    for fold in args.folds:
        set_seed(args.seed + int(fold))
        fold_result = run_fold(fold, samples, frames_uint8, labels, args, device, sample_by_id)
        fold_results.append(fold_result)
        summary_rows.append(fold_result["summary"])

    summary_csv = args.output_dir / "summary_metrics.csv"
    summary_mean_std_csv = args.output_dir / "summary_metrics_mean_std.csv"
    write_summary_csv(summary_csv, summary_rows)
    write_summary_csv(summary_mean_std_csv, build_mean_std_rows(summary_rows))

    payload = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "cv_splits": str(args.cv_splits),
            "output_dir": str(args.output_dir),
            "method": args.method,
            "folds": [int(fold) for fold in args.folds],
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "warmup_epochs": int(args.warmup_epochs),
            "batch_size": int(args.batch_size),
            "label_smoothing": float(args.label_smoothing),
            "target_recall": float(args.target_recall),
            "seed": int(args.seed),
        },
        "class_names": RAW_VIEWS,
        "summary_headers": SUMMARY_HEADERS,
        "summary_metrics_csv": str(summary_csv.resolve()),
        "summary_metrics_mean_std_csv": str(summary_mean_std_csv.resolve()),
        "fold_results": fold_results,
    }
    results_json = args.output_dir / "results.json"
    results_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {results_json}")


if __name__ == "__main__":
    main()
