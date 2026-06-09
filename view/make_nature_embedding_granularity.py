#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
ORIG_CWD = Path.cwd().resolve()
for path in (HERE, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import view_raw_finetune_scaling as raw_exp  # noqa: E402


RAW_ORDER = ["A4", "MS", "PSL", "PSS", "IVC", "SX"]

RAW_DISPLAY_LABELS = {
    "A4": "A4",
    "MS": "MS",
    "PSL": "PSL",
    "PSS": "PSS",
    "IVC": "IVC",
    "SX": "Subxiphoid",
}
GROUP4_ORDER = ["A4", "PSL", "PSS", "Subcostal"]
FAMILY3_ORDER = ["Apical", "Parasternal", "Subcostal"]

RAW_TO_GROUP4 = {
    "A4": "A4",
    "MS": "A4",
    "PSL": "PSL",
    "PSS": "PSS",
    "IVC": "Subcostal",
    "SX": "Subcostal",
}
RAW_TO_FAMILY3 = {
    "A4": "Apical",
    "MS": "Apical",
    "PSL": "Parasternal",
    "PSS": "Parasternal",
    "IVC": "Subcostal",
    "SX": "Subcostal",
}

RAW_COLORS = {
    "A4": "#3F74A3",
    "MS": "#8FBB8F",
    "PSL": "#B54646",
    "PSS": "#F18F2B",
    "IVC": "#794E47",
    "SX": "#C994AF",
}
GROUP4_COLORS = {
    "A4": "#3F74A3",
    "PSL": "#B54646",
    "PSS": "#F18F2B",
    "Subcostal": "#794E47",
}
FAMILY3_COLORS = {
    "Apical": "#3F74A3",
    "Parasternal": "#B54646",
    "Subcostal": "#794E47",
}

RAW_LEGEND_LABELS = {
    "A4": "A4 -> A4 / Apical",
    "MS": "MS -> A4 / Apical",
    "PSL": "PSL -> PSL / Parasternal",
    "PSS": "PSS -> PSS / Parasternal",
    "IVC": "IVC -> Subcostal",
    "SX": "Subxiphoid -> Subcostal",
}
GROUP4_LEGEND_LABELS = {
    "A4": "A4 (A4, MS)",
    "PSL": "PSL",
    "PSS": "PSS",
    "Subcostal": "Subcostal (IVC, Subxiphoid)",
}
FAMILY3_LEGEND_LABELS = {
    "Apical": "Apical (A4, MS)",
    "Parasternal": "Parasternal (PSL, PSS)",
    "Subcostal": "Subcostal (IVC, Subxiphoid)",
}

MS_LABEL_COLOR = "#F18F2B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Nature-style hidden-space figure for the final "
            "pretrained + full fine-tuned six-way view classifier."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Dataset root containing Normal/ and PE/ view folders. "
            "Defaults to the dataset_root recorded in experiment-dir/results.json."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=HERE / "view_raw_finetune_results_v3",
        help="6-way experiment directory containing results.json and frame cache.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to experiment-dir/analysis/nature_plots.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="best",
        help="Seed to visualize, or 'best' to pick the best full-ratio pretrained_full_ft seed.",
    )
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--method", type=str, default="pretrained_full_ft")
    parser.add_argument(
        "--projection",
        choices=["auto", "umap", "tsne"],
        default="auto",
        help="Projection method. 'auto' writes both UMAP and t-SNE when UMAP is installed.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--full-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--full-head-lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--test-frac", type=float, default=None)
    parser.add_argument("--val-frac-total", type=float, default=None)
    parser.add_argument(
        "--embedding-split",
        choices=["all", "train", "val", "test", "train_val"],
        default="all",
        help="Dataset split to embed and visualize. Default uses all available videos.",
    )
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=35)
    parser.add_argument("--min-dist", type=float, default=0.20)
    parser.add_argument("--perplexity", type=float, default=28.0)
    parser.add_argument("--projection-seed", type=int, default=17)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help=(
            "Maximum examples per raw class used to fit and plot the projection. "
            "0 uses every example in the selected embedding split."
        ),
    )
    parser.add_argument(
        "--no-balanced-projection",
        dest="balanced_projection",
        action="store_false",
        help="Use all test examples for the projection instead of class-capped sampling.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help=(
            "Path to EchoPrime model_data/weights/view_classifier.pt. "
            "Only required when training a new embedding checkpoint."
        ),
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-embedding", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def resolve_user_path(path: Path) -> Path:
    return path if path.is_absolute() else (ORIG_CWD / path).resolve()


def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.6,
            "figure.titlesize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results_payload(experiment_dir: Path) -> dict:
    results_json = experiment_dir / "results.json"
    if not results_json.is_file():
        raise FileNotFoundError(f"Missing results file: {results_json}")
    return json.loads(results_json.read_text(encoding="utf-8"))


def choose_seed(payload: dict, seed_arg: str, method: str, ratio: float) -> tuple[int, str]:
    if seed_arg != "best":
        return int(seed_arg), "user-selected"

    rows = [
        row
        for row in payload["results"]
        if row["method"] == method and abs(float(row["ratio"]) - float(ratio)) < 1e-9
    ]
    if not rows:
        raise ValueError(f"No {method} rows found for ratio={ratio}")
    best = max(rows, key=lambda row: float(row["metrics"]["macro_f1"]))
    reason = f"best full-ratio macro-F1={float(best['metrics']['macro_f1']):.4f}"
    return int(best["seed"]), reason


def training_args_from_payload(args: argparse.Namespace, payload: dict) -> argparse.Namespace:
    config = payload.get("config", {})
    return argparse.Namespace(
        batch_size=args.batch_size or int(config.get("batch_size", 64)),
        num_workers=args.num_workers,
        epochs=args.epochs or int(config.get("epochs", 40)),
        patience=args.patience or int(config.get("patience", 10)),
        warmup_epochs=args.warmup_epochs or int(config.get("warmup_epochs", 2)),
        weight_decay=args.weight_decay,
        label_smoothing=(
            args.label_smoothing
            if args.label_smoothing is not None
            else float(config.get("label_smoothing", 0.05))
        ),
        full_backbone_lr=args.full_backbone_lr,
        full_head_lr=args.full_head_lr,
        linear_probe_lr=1e-3,
        partial_backbone_lr=5e-5,
        partial_head_lr=5e-4,
        scratch_lr=3e-4,
        grad_clip=args.grad_clip,
    )


def split_params_from_payload(args: argparse.Namespace, payload: dict) -> tuple[float, float]:
    config = payload.get("config", {})
    test_frac = args.test_frac if args.test_frac is not None else float(config.get("test_frac", 0.20))
    val_frac = (
        args.val_frac_total
        if args.val_frac_total is not None
        else float(config.get("val_frac_total", 0.10))
    )
    return test_frac, val_frac


def default_checkpoint_path(experiment_dir: Path, method: str, seed: int, ratio: float) -> Path:
    stem = f"{method}__seed-{seed}__ratio-{ratio:.2f}.pt"
    return experiment_dir / "checkpoints_for_embedding" / stem


def resolve_pretrained_weights(args: argparse.Namespace) -> Path:
    candidates = []
    if args.pretrained_weights is not None:
        candidates.append(resolve_user_path(args.pretrained_weights))
    candidates.extend(
        [
            HERE.parent.parent / "model_data" / "weights" / "view_classifier.pt",
            HERE.parent / "model_data" / "weights" / "view_classifier.pt",
            ORIG_CWD / "model_data" / "weights" / "view_classifier.pt",
            ORIG_CWD / "EchoPrime" / "model_data" / "weights" / "view_classifier.pt",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Missing EchoPrime view-classifier weights. Download model_data.zip from "
        "the EchoPrime release and unzip it under EchoPrime/model_data, or pass "
        f"--pretrained-weights explicitly. Checked:\n{checked}"
    )


def build_six_way_model(device: torch.device) -> nn.Module:
    model = torchvision.models.convnext_base(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(raw_exp.RAW_VIEWS))
    return model.to(device)


def build_pretrained_six_way_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = torchvision.models.convnext_base(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 11)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(raw_exp.RAW_VIEWS))
    return model.to(device)


def train_pretrained_full_ft(
    pretrained_weights: Path,
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    train_args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    model = build_pretrained_six_way_model(pretrained_weights, device)
    raw_exp.configure_trainable(model, "full")

    optimizer = raw_exp.build_optimizer(model, "pretrained_full_ft", train_args)
    scheduler = raw_exp.build_scheduler(optimizer, train_args.epochs, train_args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_loader = raw_exp.make_loader(
        frames_uint8,
        labels,
        train_indices,
        train_args.batch_size,
        train_args.num_workers,
        training=True,
    )
    val_loader = raw_exp.make_loader(
        frames_uint8,
        labels,
        val_indices,
        train_args.batch_size,
        train_args.num_workers,
        training=False,
    )
    test_loader = raw_exp.make_loader(
        frames_uint8,
        labels,
        test_indices,
        train_args.batch_size,
        train_args.num_workers,
        training=False,
    )

    train_label_tensor = torch.tensor(labels[train_indices], dtype=torch.long)
    class_counts = torch.bincount(train_label_tensor, minlength=len(raw_exp.RAW_VIEWS)).float().clamp(min=1)
    class_weights = (len(train_label_tensor) / (len(raw_exp.RAW_VIEWS) * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=train_args.label_smoothing)

    best_val = -1.0
    best_epoch = 0
    best_state = None
    history: list[dict[str, float]] = []
    no_improve = 0

    for epoch in range(1, train_args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch:02d}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if train_args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
            pbar.set_postfix(loss=running_loss / max(n_seen, 1))
        scheduler.step()

        val_y, val_pred, _ = raw_exp.predict_model(model, val_loader, device)
        val_metrics = raw_exp.evaluate_predictions(val_y, val_pred)
        val_macro_f1 = float(val_metrics["macro_f1"])
        row = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(n_seen, 1),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": val_macro_f1,
        }
        history.append(row)
        print(
            "[train] "
            f"epoch={epoch:02d} loss={row['train_loss']:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val:
            best_val = val_macro_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= train_args.patience:
                print(f"[train] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    test_y, test_pred, _ = raw_exp.predict_model(model, test_loader, device)
    test_metrics = raw_exp.evaluate_predictions(test_y, test_pred)
    metadata = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val),
        "history": history,
        "test_metrics": test_metrics,
    }
    return model, metadata


def load_or_train_model(
    checkpoint_path: Path,
    args: argparse.Namespace,
    payload: dict,
    seed: int,
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    if checkpoint_path.is_file() and not args.force_retrain:
        print(f"[checkpoint] loading {checkpoint_path}")
        model = build_six_way_model(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model, state.get("metadata", {})

    if args.method != "pretrained_full_ft":
        raise ValueError("This embedding figure is designed for pretrained_full_ft.")

    print("[checkpoint] no reusable checkpoint found; training pretrained_full_ft")
    pretrained_weights = resolve_pretrained_weights(args)
    print(f"[weights] using {pretrained_weights}")
    train_args = training_args_from_payload(args, payload)
    raw_exp.set_seed(seed)
    model, metadata = train_pretrained_full_ft(
        pretrained_weights,
        frames_uint8,
        labels,
        train_indices,
        val_indices,
        test_indices,
        train_args,
        device,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "seed": seed,
            "ratio": args.ratio,
            "method": args.method,
            "class_names": raw_exp.RAW_VIEWS,
        },
        checkpoint_path,
    )
    print(f"[checkpoint] saved {checkpoint_path}")
    return model, metadata


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    embed_indices: np.ndarray,
    split_labels: np.ndarray,
    samples: list,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = raw_exp.make_loader(
        frames_uint8,
        labels,
        embed_indices,
        batch_size,
        num_workers,
        training=False,
    )
    model.eval()
    all_embeddings = []
    all_true = []
    all_pred = []
    all_indices = []

    for x, y, real_idx in tqdm(loader, desc="Extracting embeddings"):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            features = model.features(x)
            pooled = model.avgpool(features)
            normed = model.classifier[0](pooled)
            embedding = torch.flatten(normed, 1)
            logits = model.classifier[2](embedding)
        all_embeddings.append(embedding.float().cpu().numpy())
        all_true.extend(y.numpy().tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_indices.extend(real_idx.numpy().tolist())

    embeddings = np.concatenate(all_embeddings, axis=0)
    true_idx = np.asarray(all_true, dtype=np.int64)
    pred_idx = np.asarray(all_pred, dtype=np.int64)
    real_indices = np.asarray(all_indices, dtype=np.int64)
    raw_labels = np.asarray([raw_exp.IDX_TO_RAW[int(i)] for i in true_idx], dtype=object)
    pred_labels = np.asarray([raw_exp.IDX_TO_RAW[int(i)] for i in pred_idx], dtype=object)
    paths = np.asarray([samples[int(i)].path for i in real_indices], dtype=object)
    source_groups = np.asarray([samples[int(i)].source_group for i in real_indices], dtype=object)
    sample_splits = np.asarray([split_labels[int(i)] for i in real_indices], dtype=object)
    return {
        "embeddings": embeddings,
        "true_idx": true_idx,
        "pred_idx": pred_idx,
        "raw_labels": raw_labels,
        "pred_labels": pred_labels,
        "real_indices": real_indices,
        "paths": paths,
        "source_groups": source_groups,
        "sample_splits": sample_splits,
    }


def compute_projection(
    embeddings: np.ndarray,
    args: argparse.Namespace,
    projection_method: str,
) -> np.ndarray:
    scaled = StandardScaler().fit_transform(embeddings)
    n_components = min(args.pca_dim, scaled.shape[0] - 1, scaled.shape[1])
    basis = scaled
    if n_components >= 2:
        basis = PCA(n_components=n_components, random_state=args.projection_seed).fit_transform(scaled)

    if projection_method == "umap":
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(args.n_neighbors, max(2, basis.shape[0] - 1)),
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.projection_seed,
        )
        return reducer.fit_transform(basis)

    perplexity = min(args.perplexity, max(5.0, (basis.shape[0] - 1) / 3.0))
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": args.projection_seed,
        "metric": "euclidean",
    }
    try:
        reducer = TSNE(max_iter=1000, **tsne_kwargs)
    except TypeError:
        reducer = TSNE(n_iter=1000, **tsne_kwargs)
    return reducer.fit_transform(basis)


def projection_methods_for_args(args: argparse.Namespace) -> list[str]:
    if args.projection in {"umap", "tsne"}:
        return [args.projection]

    methods = ["tsne"]
    try:
        import umap  # noqa: F401

        methods.insert(0, "umap")
    except ImportError:
        print("[projection] umap-learn is not installed; auto mode will write t-SNE only")
    return methods


def embedding_cache_path(figure_dir: Path, method: str, seed: int, ratio: float, embedding_split: str) -> Path:
    return (
        figure_dir
        / f"fig5_embedding_granularity__{method}__seed-{seed}__ratio-{ratio:.2f}__split-{embedding_split}.npz"
    )


def load_or_compute_embeddings(
    cache_path: Path,
    model: nn.Module,
    frames_uint8: torch.Tensor,
    labels: np.ndarray,
    embed_indices: np.ndarray,
    split_labels: np.ndarray,
    samples: list,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if cache_path.is_file() and not args.force_embedding:
        print(f"[embedding] loading {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return {
            key: cached[key]
            for key in cached.files
            if key not in {"projection_method", "coords"}
        }

    data = extract_embeddings(
        model,
        frames_uint8,
        labels,
        embed_indices,
        split_labels,
        samples,
        args.batch_size or 64,
        args.num_workers,
        device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"[embedding] saved {cache_path}")
    return data


def select_plot_indices(raw_labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    n = len(raw_labels)
    if max_per_class <= 0:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in RAW_ORDER:
        idx = np.flatnonzero(raw_labels == label)
        if len(idx) == 0:
            continue
        selected.extend(rng.choice(idx, size=min(max_per_class, len(idx)), replace=False).tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def build_split_labels(n_samples: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    split_labels = np.full(n_samples, "unassigned", dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "validation"
    split_labels[test_idx] = "test"
    return split_labels


def indices_for_embedding_split(
    embedding_split: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    if embedding_split == "all":
        return np.arange(n_samples, dtype=np.int64)
    if embedding_split == "train":
        return np.sort(train_idx)
    if embedding_split == "val":
        return np.sort(val_idx)
    if embedding_split == "test":
        return np.sort(test_idx)
    if embedding_split == "train_val":
        return np.sort(np.concatenate([train_idx, val_idx]))
    raise ValueError(f"Unknown embedding split: {embedding_split}")


def style_embedding_axis(ax: plt.Axes, projection_label: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{projection_label} 1")
    ax.set_ylabel(f"{projection_label} 2")


def scatter_embedding(ax: plt.Axes, coords: np.ndarray, raw_labels: np.ndarray, projection_label: str) -> None:
    # Draw abundant classes first so smaller classes remain visible.
    label_counts = {label: int(np.sum(raw_labels == label)) for label in RAW_ORDER}
    draw_order = sorted(RAW_ORDER, key=lambda label: label_counts[label], reverse=True)
    for label in draw_order:
        mask = raw_labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8.2,
            c=RAW_COLORS[label],
            alpha=0.58,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
        )
    style_embedding_axis(ax, projection_label)


def draw_tree_legend(ax: plt.Axes, raw_labels: np.ndarray) -> None:
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    family_x, group_x, raw_x = 0.17, 0.50, 0.83
    header_y = 0.97
    branch_color = "#AEB8BF"
    text_color = "#202124"
    count_map = {label: int(np.sum(raw_labels == label)) for label in RAW_ORDER}

    ax.text(family_x, header_y, "3-class", ha="center", va="top", fontsize=6.0, color="#5F6368")
    ax.text(group_x, header_y, "4-class", ha="center", va="top", fontsize=6.0, color="#5F6368")
    ax.text(raw_x, header_y, "6-class", ha="center", va="top", fontsize=6.0, color="#5F6368")

    nodes = {
        "family_apical": ("Apical", family_x, 0.82, FAMILY3_COLORS["Apical"], "family"),
        "group_a4": ("A4", group_x, 0.82, GROUP4_COLORS["A4"], "group"),
        "raw_a4": ("A4", raw_x, 0.89, RAW_COLORS["A4"], "raw"),
        "raw_ms": ("MS", raw_x, 0.75, RAW_COLORS["MS"], "raw"),
        "family_parasternal": ("Parasternal", family_x, 0.52, FAMILY3_COLORS["Parasternal"], "family"),
        "group_psl": ("PSL", group_x, 0.60, GROUP4_COLORS["PSL"], "group"),
        "group_pss": ("PSS", group_x, 0.44, GROUP4_COLORS["PSS"], "group"),
        "raw_psl": ("PSL", raw_x, 0.60, RAW_COLORS["PSL"], "raw"),
        "raw_pss": ("PSS", raw_x, 0.44, RAW_COLORS["PSS"], "raw"),
        "family_subcostal": ("Subcostal", family_x, 0.22, FAMILY3_COLORS["Subcostal"], "family"),
        "group_subcostal": ("Subcostal", group_x, 0.22, GROUP4_COLORS["Subcostal"], "group"),
        "raw_ivc": ("IVC", raw_x, 0.29, RAW_COLORS["IVC"], "raw"),
        "raw_sx": ("SX", raw_x, 0.15, RAW_COLORS["SX"], "raw"),
    }
    node_half_width = {"family": 0.105, "group": 0.085, "raw": 0.075}

    def draw_branch(parent_key: str, child_keys: list[str]) -> None:
        parent = nodes[parent_key]
        children = [nodes[key] for key in child_keys]
        _, parent_x, parent_y, _, parent_kind = parent
        parent_right = parent_x + node_half_width[parent_kind]
        child_lefts = [child[1] - node_half_width[child[4]] for child in children]

        if len(children) == 1 and abs(parent_y - children[0][2]) < 1e-9:
            ax.plot(
                [parent_right, child_lefts[0]],
                [parent_y, parent_y],
                color=branch_color,
                linewidth=0.7,
                solid_capstyle="round",
                zorder=1,
            )
            return

        elbow_x = (parent_right + min(child_lefts)) / 2
        child_ys = [child[2] for child in children]
        ax.plot(
            [parent_right, elbow_x],
            [parent_y, parent_y],
            color=branch_color,
            linewidth=0.7,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [elbow_x, elbow_x],
            [min(child_ys + [parent_y]), max(child_ys + [parent_y])],
            color=branch_color,
            linewidth=0.7,
            solid_capstyle="round",
            zorder=1,
        )
        for child, child_left in zip(children, child_lefts):
            ax.plot(
                [elbow_x, child_left],
                [child[2], child[2]],
                color=branch_color,
                linewidth=0.7,
                solid_capstyle="round",
                zorder=1,
            )

    for parent_key, child_keys in [
        ("family_apical", ["group_a4"]),
        ("group_a4", ["raw_a4", "raw_ms"]),
        ("family_parasternal", ["group_psl", "group_pss"]),
        ("group_psl", ["raw_psl"]),
        ("group_pss", ["raw_pss"]),
        ("family_subcostal", ["group_subcostal"]),
        ("group_subcostal", ["raw_ivc", "raw_sx"]),
    ]:
        draw_branch(parent_key, child_keys)

    for label, x, y, color, kind in nodes.values():
        marker_size = 19 if kind == "raw" else 16
        ax.scatter([x], [y], s=marker_size, color=color, edgecolors="none", linewidths=0, alpha=0.92, zorder=3)
        if kind == "raw":
            text = f"{RAW_DISPLAY_LABELS[label]}\n(n={count_map[label]})"
        else:
            text = label
        weight = "bold" if kind == "family" else "normal"
        ax.text(
            x,
            y - 0.035,
            text,
            ha="center",
            va="top",
            fontsize=5.9 if kind == "raw" else 6.2,
            color=MS_LABEL_COLOR if label == "MS" else text_color,
            fontweight=weight,
            linespacing=0.95,
            zorder=3,
        )


def plot_embedding_figure(
    coords: np.ndarray,
    raw_labels: np.ndarray,
    projection_method: str,
    figure_dir: Path,
    method: str,
    seed: int,
    ratio: float,
    n_total: int,
    balanced_projection: bool,
    max_per_class: int,
    embedding_split: str,
) -> tuple[Path, Path, Path, str]:
    projection_label = "UMAP" if projection_method == "umap" else "t-SNE"
    fig = plt.figure(figsize=(6.8, 3.25), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.07)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])

    scatter_embedding(ax, coords, raw_labels, projection_label)
    draw_tree_legend(legend_ax, raw_labels)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.16)

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig5_view_embedding_granularity_{embedding_split}_{projection_method}"
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    svg_path = figure_dir / f"{stem}.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)

    caption = make_caption(
        projection_label,
        method,
        seed,
        ratio,
        n_total=n_total,
        n_plotted=len(raw_labels),
        balanced_projection=balanced_projection,
        max_per_class=max_per_class,
        embedding_split=embedding_split,
    )
    caption_path = figure_dir / f"{stem}_caption.txt"
    caption_path.write_text(textwrap.fill(caption, width=118), encoding="utf-8")
    return png_path, pdf_path, svg_path, caption


def make_caption(
    projection_label: str,
    method: str,
    seed: int,
    ratio: float,
    n_total: int,
    n_plotted: int,
    balanced_projection: bool,
    max_per_class: int,
    embedding_split: str,
) -> str:
    split_phrase = {
        "all": "all available videos",
        "train": "training videos",
        "val": "validation videos",
        "test": "held-out test videos",
        "train_val": "training and validation videos",
    }[embedding_split]
    if balanced_projection and max_per_class > 0:
        sampled_text = (
            f"The projection was fit and displayed on a class-capped subset of {n_plotted} videos from {split_phrase} "
            f"(up to {max_per_class} examples per raw view) to reduce layout dominance by abundant classes."
        )
    else:
        sampled_text = f"All {n_total} videos from {split_phrase} are shown."
    return (
        "Figure | Hidden-space organization of the final EchoPrime view classifier and its label hierarchy. "
        f"Penultimate ConvNeXt features were extracted from the six-way {method.replace('_', ' ')} model "
        f"trained at the full data ratio ({ratio:.2f}) with split seed {seed}, then projected with {projection_label}. "
        "Points are colored by the six raw view labels, and the adjacent tree legend shows how these labels collapse "
        "into the four-class coarse labels and three-class view families. Colors are intentionally separated between "
        "raw views while preserving the aggregation structure through the tree. "
        f"{sampled_text}"
    )


def save_projection_data(
    figure_dir: Path,
    projection_method: str,
    data: dict[str, np.ndarray],
    plot_idx: np.ndarray,
    coords: np.ndarray,
    method: str,
    seed: int,
    ratio: float,
    embedding_split: str,
) -> tuple[Path, Path]:
    stem = f"fig5_view_embedding_granularity_{embedding_split}_{projection_method}_data"
    npz_path = figure_dir / f"{stem}.npz"
    csv_path = figure_dir / f"{stem}.csv"

    raw_labels = data["raw_labels"][plot_idx]
    pred_labels = data["pred_labels"][plot_idx]
    group4_labels = np.asarray([RAW_TO_GROUP4[str(label)] for label in raw_labels], dtype=object)
    family3_labels = np.asarray([RAW_TO_FAMILY3[str(label)] for label in raw_labels], dtype=object)
    is_correct = raw_labels == pred_labels

    figure_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        coords=coords,
        projection_method=projection_method,
        method=method,
        seed=seed,
        ratio=ratio,
        embedding_split=embedding_split,
        selected_indices=plot_idx,
        embeddings=data["embeddings"][plot_idx],
        raw_labels=raw_labels,
        group4_labels=group4_labels,
        family3_labels=family3_labels,
        pred_labels=pred_labels,
        is_correct=is_correct,
        real_indices=data["real_indices"][plot_idx],
        sample_splits=data["sample_splits"][plot_idx],
        source_groups=data["source_groups"][plot_idx],
        paths=data["paths"][plot_idx],
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "projection_1",
                "projection_2",
                "raw_view",
                "group4_view",
                "family3_view",
                "pred_raw_view",
                "is_correct",
                "split",
                "source_group",
                "real_index",
                "path",
            ]
        )
        for i in range(len(plot_idx)):
            writer.writerow(
                [
                    float(coords[i, 0]),
                    float(coords[i, 1]),
                    str(raw_labels[i]),
                    str(group4_labels[i]),
                    str(family3_labels[i]),
                    str(pred_labels[i]),
                    bool(is_correct[i]),
                    str(data["sample_splits"][plot_idx][i]),
                    str(data["source_groups"][plot_idx][i]),
                    int(data["real_indices"][plot_idx][i]),
                    str(data["paths"][plot_idx][i]),
                ]
            )
    return npz_path, csv_path


def main() -> None:
    args = parse_args()
    args.experiment_dir = resolve_user_path(args.experiment_dir)
    args.figure_dir = (
        resolve_user_path(args.figure_dir)
        if args.figure_dir is not None
        else args.experiment_dir / "analysis" / "nature_plots"
    )
    device = torch.device(args.device)

    set_nature_style()
    payload = load_results_payload(args.experiment_dir)
    if args.dataset_root is None:
        config_dataset_root = payload.get("config", {}).get("dataset_root")
        if not config_dataset_root:
            raise ValueError("No --dataset-root provided and results.json has no config.dataset_root")
        args.dataset_root = Path(config_dataset_root).resolve()
    else:
        args.dataset_root = resolve_user_path(args.dataset_root)
    seed, seed_reason = choose_seed(payload, args.seed, args.method, args.ratio)
    print(f"[seed] using seed={seed} ({seed_reason})")

    test_frac, val_frac = split_params_from_payload(args, payload)
    raw_exp.set_seed(seed)
    samples = raw_exp.collect_samples(args.dataset_root)
    cache_path = args.experiment_dir / "first_frame_cache.pt"
    frames_uint8 = raw_exp.build_or_load_frame_cache(samples, cache_path, rebuild=args.rebuild_cache)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    train_pool_idx, val_idx, test_idx, strata = raw_exp.make_splits(samples, test_frac, val_frac, seed)
    train_idx = raw_exp.stratified_subsample(train_pool_idx, strata, args.ratio, seed + int(args.ratio * 1000))
    split_labels = build_split_labels(len(samples), train_idx, val_idx, test_idx)
    embed_idx = indices_for_embedding_split(args.embedding_split, train_idx, val_idx, test_idx, len(samples))

    checkpoint_path = (
        resolve_user_path(args.checkpoint)
        if args.checkpoint is not None
        else default_checkpoint_path(args.experiment_dir, args.method, seed, args.ratio)
    )
    emb_cache = embedding_cache_path(args.figure_dir, args.method, seed, args.ratio, args.embedding_split)
    if emb_cache.is_file() and not args.force_embedding:
        print(f"[embedding] loading {emb_cache}")
        cached = np.load(emb_cache, allow_pickle=True)
        data = {
            key: cached[key]
            for key in cached.files
            if key not in {"projection_method", "coords"}
        }
    else:
        model, metadata = load_or_train_model(
            checkpoint_path,
            args,
            payload,
            seed,
            frames_uint8,
            labels,
            train_idx,
            val_idx,
            test_idx,
            device,
        )
        if metadata:
            print(
                "[model] "
                f"best_epoch={metadata.get('best_epoch')} "
                f"best_val_macro_f1={metadata.get('best_val_macro_f1')}"
            )
        data = load_or_compute_embeddings(
            emb_cache,
            model,
            frames_uint8,
            labels,
            embed_idx,
            split_labels,
            samples,
            args,
            device,
        )

    if args.balanced_projection:
        plot_idx = select_plot_indices(data["raw_labels"], args.max_per_class, args.projection_seed)
    else:
        plot_idx = np.arange(len(data["raw_labels"]), dtype=np.int64)
    embeddings_plot = data["embeddings"][plot_idx]
    raw_labels_plot = data["raw_labels"][plot_idx]

    captions = []
    for projection_method in projection_methods_for_args(args):
        coords = compute_projection(embeddings_plot, args, projection_method)
        npz_path, csv_path = save_projection_data(
            args.figure_dir,
            projection_method,
            data,
            plot_idx,
            coords,
            args.method,
            seed,
            args.ratio,
            args.embedding_split,
        )
        png_path, pdf_path, svg_path, caption = plot_embedding_figure(
            coords,
            raw_labels_plot,
            projection_method,
            args.figure_dir,
            args.method,
            seed,
            args.ratio,
            n_total=len(data["raw_labels"]),
            balanced_projection=args.balanced_projection,
            max_per_class=args.max_per_class,
            embedding_split=args.embedding_split,
        )
        captions.append((projection_method, caption))
        print(f"[done] wrote {png_path}")
        print(f"[done] wrote {pdf_path}")
        print(f"[done] wrote {svg_path}")
        print(f"[done] wrote {npz_path}")
        print(f"[done] wrote {csv_path}")

    print("\n=== Figure captions ===")
    for projection_method, caption in captions:
        print(f"\n[{projection_method}]")
        print(textwrap.fill(caption, width=118))


if __name__ == "__main__":
    main()
