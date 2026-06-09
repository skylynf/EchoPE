#!/usr/bin/env python3
"""
Data-scaling experiments for EchoPrime view classification fine-tuning.

Goal:
1. Fix a held-out test split once.
2. From the remaining training pool, subsample different ratios.
3. Compare how much EchoPrime view pretraining helps versus random init.
4. Save metrics, plots, and split manifests for reproducibility.

Label space:
    Use the raw dataset folders directly as 6-way classes:
        A4, IVC, MS, PSL, PSS, SX

Stratification:
    Although Normal/PE is NOT used as the prediction target, we preserve its
    composition by stratifying on "{top_folder}_{raw_view}".

Compared methods:
    - echo_pretrained: initialize ConvNeXt-Base backbone from
      model_data/weights/view_classifier.pt, replace final head with 6-way head,
      then fine-tune all layers.
    - scratch: same architecture, random initialization, fine-tune all layers.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ORIG_CWD = Path.cwd().resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from echo_paths import setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

import utils  # noqa: E402


RAW_VIEWS = ["A4", "IVC", "MS", "PSL", "PSS", "SX"]
VIEW_TO_IDX = {v: i for i, v in enumerate(RAW_VIEWS)}
IDX_TO_VIEW = {i: v for v, i in VIEW_TO_IDX.items()}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}

MEAN = torch.tensor([29.110628, 28.076836, 29.096405], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083], dtype=torch.float32).view(3, 1, 1)


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
        return x, y


def resolve_user_path(path: Path) -> Path:
    return path if path.is_absolute() else (ORIG_CWD / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoPrime view fine-tuning data-scaling experiment.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=HERE.parent.parent / "dataset" / "preprocessed",
        help="Dataset root containing Normal/ and PE/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "view_finetune_results",
        help="Directory for results, plots, and cached frames.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.15, 0.30, 0.50, 0.75, 1.00],
        help="Train-pool sampling ratios.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--test-frac", type=float, default=0.20, help="Held-out test fraction.")
    parser.add_argument(
        "--val-frac-total",
        type=float,
        default=0.10,
        help="Validation fraction of the total dataset.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Max fine-tuning epochs per run.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience on val macro-F1.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-pretrained", type=float, default=1e-4)
    parser.add_argument("--lr-scratch", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore on-disk frame cache and rebuild from videos.",
    )
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
            if raw_view not in VIEW_TO_IDX:
                continue
            for path in sorted(raw_view_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    samples.append(
                        Sample(
                            path=str(path),
                            source_group=source_group,
                            raw_view=raw_view,
                            label=VIEW_TO_IDX[raw_view],
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


def build_or_load_frame_cache(
    samples: list[Sample],
    cache_path: Path,
    rebuild: bool,
) -> torch.Tensor:
    expected_paths = [s.path for s in samples]
    if cache_path.is_file() and not rebuild:
        cache = torch.load(cache_path, map_location="cpu")
        cached_paths = cache.get("paths")
        if cached_paths == expected_paths:
            print(f"[cache] loaded frame cache: {cache_path}")
            return cache["frames"].to(torch.uint8)
        print("[cache] cache paths changed, rebuilding.")

    frames: list[torch.Tensor] = []
    for sample in tqdm(samples, desc="Caching first frames"):
        frames.append(preprocess_first_frame_uint8(sample.path))
    stacked = torch.stack(frames).to(torch.uint8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"paths": expected_paths, "frames": stacked}, cache_path)
    print(f"[cache] saved frame cache: {cache_path}")
    return stacked


def make_splits(samples: list[Sample], test_frac: float, val_frac_total: float, seed: int):
    n = len(samples)
    all_indices = np.arange(n)
    strata = np.array([s.strata for s in samples], dtype=object)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_frac,
        stratify=strata,
        random_state=seed,
        shuffle=True,
    )

    val_inner = val_frac_total / (1.0 - test_frac)
    train_val_strata = strata[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_inner,
        stratify=train_val_strata,
        random_state=seed + 1,
        shuffle=True,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


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
    loader_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if training:
        train_labels = torch.tensor(labels[indices], dtype=torch.long)
        class_counts = torch.bincount(train_labels, minlength=len(RAW_VIEWS)).float().clamp(min=1)
        sample_weights = (1.0 / class_counts)[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = False
    return DataLoader(**loader_kwargs)


def build_model(method: str, device: torch.device) -> nn.Module:
    if method == "echo_pretrained":
        model = torchvision.models.convnext_base()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 11)
        state = torch.load(HERE.parent / "model_data" / "weights" / "view_classifier.pt", map_location="cpu")
        model.load_state_dict(state)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(RAW_VIEWS))
    elif method == "scratch":
        model = torchvision.models.convnext_base(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(RAW_VIEWS))
    else:
        raise ValueError(f"Unknown method: {method}")
    return model.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    logits_all: list[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        logits_all.append(logits.float().cpu())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(RAW_VIEWS))))
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(RAW_VIEWS))),
        target_names=RAW_VIEWS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "logits": torch.cat(logits_all, dim=0).numpy(),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_f1": {k: float(v["f1-score"]) for k, v in report.items() if k in RAW_VIEWS},
    }


def train_one_experiment(
    method: str,
    ratio: float,
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    strata: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    subset_indices = stratified_subsample(train_indices, strata, ratio, seed=args.seed + int(ratio * 1000))
    train_loader = make_loader(
        frames_uint8, labels, subset_indices, args.batch_size, args.num_workers, training=True
    )
    val_loader = make_loader(
        frames_uint8, labels, val_indices, args.batch_size, args.num_workers, training=False
    )
    test_loader = make_loader(
        frames_uint8, labels, test_indices, args.batch_size, args.num_workers, training=False
    )

    model = build_model(method, device)
    backbone_lr = args.lr_pretrained if method == "echo_pretrained" else args.lr_scratch
    head_params = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_label_tensor = torch.tensor(labels[subset_indices], dtype=torch.long)
    class_counts = torch.bincount(train_label_tensor, minlength=len(RAW_VIEWS)).float().clamp(min=1)
    class_weights = (len(train_label_tensor) / (len(RAW_VIEWS) * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(n_seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_val:
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
    test_metrics = evaluate(model, test_loader, device)

    subset_counter = Counter(int(labels[i]) for i in subset_indices)
    subset_counts = {IDX_TO_VIEW[k]: int(v) for k, v in sorted(subset_counter.items())}
    return {
        "method": method,
        "ratio": ratio,
        "n_train_subset": int(len(subset_indices)),
        "subset_class_counts": subset_counts,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val),
        "history": history,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
        "per_class_f1": test_metrics["per_class_f1"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
    }


def save_split_manifest(
    path: Path,
    samples: list[Sample],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_map = {}
    for split_name, indices in (
        ("train_pool", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        split_map[split_name] = [
            {
                "path": samples[i].path,
                "source_group": samples[i].source_group,
                "raw_view": samples[i].raw_view,
                "label": samples[i].label,
                "strata": samples[i].strata,
            }
            for i in indices
        ]
    path.write_text(json.dumps(split_map, indent=2, ensure_ascii=False), encoding="utf-8")


def save_summary_csv(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "ratio",
                "n_train_subset",
                "best_epoch",
                "best_val_macro_f1",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
            ]
        )
        for row in results:
            tm = row["test_metrics"]
            writer.writerow(
                [
                    row["method"],
                    row["ratio"],
                    row["n_train_subset"],
                    row["best_epoch"],
                    row["best_val_macro_f1"],
                    tm["accuracy"],
                    tm["balanced_accuracy"],
                    tm["macro_f1"],
                    tm["weighted_f1"],
                ]
            )


def plot_metric_curves(results: list[dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        ("accuracy", "Top-1 Accuracy"),
        ("macro_f1", "Macro F1"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]
    methods = ["echo_pretrained", "scratch"]
    labels = {"echo_pretrained": "EchoPrime pretrained", "scratch": "Random init"}
    colors = {"echo_pretrained": "#1f77b4", "scratch": "#d62728"}

    for metric_key, metric_title in metric_names:
        plt.figure(figsize=(7.5, 4.8))
        for method in methods:
            rows = sorted([r for r in results if r["method"] == method], key=lambda x: x["ratio"])
            xs = [100 * float(r["ratio"]) for r in rows]
            ys = [float(r["test_metrics"][metric_key]) for r in rows]
            plt.plot(xs, ys, marker="o", linewidth=2.2, label=labels[method], color=colors[method])
        plt.xlabel("Train subset ratio (%)")
        plt.ylabel(metric_title)
        plt.title(f"{metric_title} vs train subset ratio")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric_key}_curve.png", dpi=180)
        plt.close()


def plot_pretraining_gain(results: list[dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    pretrained = {float(r["ratio"]): r for r in results if r["method"] == "echo_pretrained"}
    scratch = {float(r["ratio"]): r for r in results if r["method"] == "scratch"}
    ratios = sorted(set(pretrained) & set(scratch))

    metrics = [
        ("accuracy", "Accuracy gain"),
        ("macro_f1", "Macro-F1 gain"),
        ("balanced_accuracy", "Balanced-Acc gain"),
    ]
    plt.figure(figsize=(8.0, 5.0))
    for metric_key, title in metrics:
        ys = [
            float(pretrained[r]["test_metrics"][metric_key]) - float(scratch[r]["test_metrics"][metric_key])
            for r in ratios
        ]
        xs = [100 * r for r in ratios]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=title)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("Train subset ratio (%)")
    plt.ylabel("EchoPrime pretrained - random init")
    plt.title("Pretraining gain under different data budgets")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "pretraining_gain.png", dpi=180)
    plt.close()


def plot_per_class_f1(results: list[dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in results if float(r["ratio"]) == 1.0]
    rows.sort(key=lambda x: x["method"])
    if not rows:
        return
    labels = {"echo_pretrained": "EchoPrime pretrained", "scratch": "Random init"}
    colors = {"echo_pretrained": "#1f77b4", "scratch": "#d62728"}

    x = np.arange(len(RAW_VIEWS))
    width = 0.38
    plt.figure(figsize=(8.8, 4.8))
    for idx, row in enumerate(rows):
        ys = [float(row["per_class_f1"].get(view, 0.0)) for view in RAW_VIEWS]
        offset = (-0.5 + idx) * width
        plt.bar(x + offset, ys, width=width, label=labels[row["method"]], color=colors[row["method"]])
    plt.xticks(x, RAW_VIEWS)
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1 score")
    plt.title("Per-class F1 at 100% train pool")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "per_class_f1_full_ratio.png", dpi=180)
    plt.close()


def plot_confusion_matrices(results: list[dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in results if float(r["ratio"]) == 1.0]
    rows.sort(key=lambda x: x["method"])
    if not rows:
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(6.4 * len(rows), 5.3))
    if len(rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        cm = np.array(row["confusion_matrix"], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(RAW_VIEWS)), RAW_VIEWS, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(RAW_VIEWS)), RAW_VIEWS)
        title = "EchoPrime pretrained" if row["method"] == "echo_pretrained" else "Random init"
        ax.set_title(f"{title}\nratio=100%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(RAW_VIEWS)):
            for j in range(len(RAW_VIEWS)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    fig.suptitle("Normalized confusion matrices")
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion_matrices_full_ratio.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.dataset_root = resolve_user_path(args.dataset_root)
    args.output_dir = resolve_user_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    cache_path = args.output_dir / "first_frame_cache.pt"
    device = torch.device(args.device)

    set_seed(args.seed)
    samples = collect_samples(args.dataset_root)
    print(f"[info] dataset_root={args.dataset_root}")
    print(f"[info] output_dir={args.output_dir}")
    print(f"[info] total_samples={len(samples)}")
    print(f"[info] device={device}")

    print("[info] class distribution:")
    view_counts = Counter(s.raw_view for s in samples)
    for view in RAW_VIEWS:
        print(f"  {view:>3}: {view_counts.get(view, 0)}")

    frames_uint8 = build_or_load_frame_cache(samples, cache_path, rebuild=args.rebuild_cache)
    labels = np.array([s.label for s in samples], dtype=np.int64)
    strata = np.array([s.strata for s in samples], dtype=object)

    train_indices, val_indices, test_indices = make_splits(
        samples, test_frac=args.test_frac, val_frac_total=args.val_frac_total, seed=args.seed
    )

    save_split_manifest(args.output_dir / "split_manifest.json", samples, train_indices, val_indices, test_indices)

    print("\n[info] split sizes")
    print(f"  train_pool={len(train_indices)}")
    print(f"  val       ={len(val_indices)}")
    print(f"  test      ={len(test_indices)}")

    results: list[dict[str, object]] = []
    for ratio in args.ratios:
        print(f"\n=== Train subset ratio: {ratio:.2f} ===")
        for method in ("echo_pretrained", "scratch"):
            print(f"[run] method={method} ratio={ratio:.2f}")
            result = train_one_experiment(
                method=method,
                ratio=ratio,
                frames_uint8=frames_uint8,
                labels=labels,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                strata=strata,
                args=args,
                device=device,
            )
            tm = result["test_metrics"]
            print(
                "      test_acc={:.4f} test_macro_f1={:.4f} test_bal_acc={:.4f} n_train={}".format(
                    tm["accuracy"],
                    tm["macro_f1"],
                    tm["balanced_accuracy"],
                    result["n_train_subset"],
                )
            )
            results.append(result)

    results_json = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
            "ratios": [float(r) for r in args.ratios],
            "seed": args.seed,
            "test_frac": args.test_frac,
            "val_frac_total": args.val_frac_total,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr_pretrained": args.lr_pretrained,
            "lr_scratch": args.lr_scratch,
            "head_lr": args.head_lr,
        },
        "class_names": RAW_VIEWS,
        "results": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_summary_csv(args.output_dir / "summary.csv", results)
    plot_metric_curves(results, plot_dir)
    plot_pretraining_gain(results, plot_dir)
    plot_per_class_f1(results, plot_dir)
    plot_confusion_matrices(results, plot_dir)

    print("\n[done] wrote:")
    print(f"  {args.output_dir / 'results.json'}")
    print(f"  {args.output_dir / 'summary.csv'}")
    print(f"  {plot_dir}")


if __name__ == "__main__":
    main()
