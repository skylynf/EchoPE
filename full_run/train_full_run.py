from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
ORIG_CWD = Path.cwd().resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from echo_paths import ECHO_ROOT, setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

from config import (  # noqa: E402
    BEST_MODEL_METRIC,
    DEFAULT_DATASET_ROOT,
    DEFAULT_FIXED_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_FRAC,
    DEFAULT_THRESHOLD_MODE,
    DEFAULT_VAL_FRAC_TOTAL,
)
from data import (  # noqa: E402
    TASKS,
    load_split_manifest,
    make_or_load_manifest,
    manifest_records,
    records_to_samples,
)
from evaluate_full_run import evaluate_prediction_rows  # noqa: E402
from lora_run.aux_heads import load_view_teacher  # noqa: E402
from lora_run.distill import load_frozen_teacher  # noqa: E402
from lora_run.train_lora import (  # noqa: E402
    HEAD_REGISTRY,
    TARGET_MODULE_PATTERNS,
    add_augmentation_args,
    build_augmentation_config,
    count_parameters,
)
from lora_run.train_lora_distill import (  # noqa: E402
    FT_MODES,
    HEAD_VIEW_MODES,
    NUM_COARSE_VIEWS,
    KD_MODES,
    PEVideoDatasetWithView,
    VIEW_AUX_MODES,
    LoRAEchoPrimeDistillModel,
    build_classifier,
    build_encoder,
    evaluate,
    train_one_epoch,
    view_collate,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_manifest_path(output_root: Path, seed: int) -> Path:
    return output_root / "manifests" / f"split_manifest_seed{seed}.json"


def resolve_cli_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ORIG_CWD / candidate).resolve()


def balanced_take(samples: list[tuple[str, int, str]], n_total: int) -> list[tuple[str, int, str]]:
    if len(samples) <= n_total:
        return samples
    cls0 = [sample for sample in samples if sample[1] == 0][: n_total // 2]
    cls1 = [sample for sample in samples if sample[1] == 1][: n_total - len(cls0)]
    reduced = cls0 + cls1
    return reduced if reduced else samples[:n_total]


def align_records_to_samples(
    records: list[dict[str, object]],
    samples: list[tuple[str, int, str] | tuple[str, int, str, int]],
) -> list[dict[str, object]]:
    by_path = {str(record["path"]): record for record in records}
    return [by_path[str(sample[0])] for sample in samples]


def task_run_name(task: str, head_view_mode: str) -> str:
    if head_view_mode == "none":
        return task
    return f"{task}__headview_{head_view_mode}"


def task_result_label(task: str, head_view_mode: str) -> str:
    if head_view_mode == "none":
        return task
    return f"{task}+headview-{head_view_mode}"


def build_loaders(
    train_samples: list[tuple[str, int, str] | tuple[str, int, str, int]],
    val_samples: list[tuple[str, int, str] | tuple[str, int, str, int]],
    test_samples: list[tuple[str, int, str] | tuple[str, int, str, int]],
    batch_size: int,
    num_workers: int,
    augmentation,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_labels = torch.tensor([sample[1] for sample in train_samples], dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=2).float().clamp(min=1)
    sample_weights = (1.0 / class_counts)[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    train_loader = DataLoader(
        PEVideoDatasetWithView(train_samples, training=True, augmentation=augmentation),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=view_collate,
    )
    val_loader = DataLoader(
        PEVideoDatasetWithView(val_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=view_collate,
    )
    test_loader = DataLoader(
        PEVideoDatasetWithView(test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=view_collate,
    )
    return train_loader, val_loader, test_loader, class_counts


def build_optimizer_and_scheduler(
    model: LoRAEchoPrimeDistillModel,
    args: argparse.Namespace,
    train_loader: DataLoader,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    aux_params: list[nn.Parameter] = []
    if model.view_ce_head is not None:
        aux_params.extend(model.view_ce_head.parameters())
    if model.view_kd_head is not None:
        aux_params.extend(model.view_kd_head.parameters())

    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": 1e-4})
    if aux_params:
        param_groups.append({"params": aux_params, "lr": args.head_lr, "weight_decay": 1e-4})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups)
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def save_prediction_csv(
    path: Path,
    split_name: str,
    records: list[dict[str, object]],
    probabilities: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(
                [
                    "split",
                    "id",
                    "path",
                    "label",
                    "label_name",
                    "source_group",
                    "raw_view",
                    "coarse_view",
                    "prob_pe",
                ]
            )
        for record, prob in zip(records, probabilities.tolist()):
            writer.writerow(
                [
                    split_name,
                    record["id"],
                    record["path"],
                    record["label"],
                    record["label_name"],
                    record["source_group"],
                    record["raw_view"],
                    record["coarse_view"],
                    float(prob),
                ]
            )


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    if args.head_view_mode != "none" and args.task != "pooled":
        raise ValueError(
            f"head_view_mode={args.head_view_mode!r} is only supported for task='pooled', got {args.task!r}."
        )
    output_root = resolve_cli_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_cli_path(args.manifest_path) if args.manifest_path else default_manifest_path(output_root, args.seed)
    manifest_path = make_or_load_manifest(
        manifest_path,
        dataset_root=resolve_cli_path(args.dataset_root),
        test_frac=args.test_frac,
        val_frac_total=args.val_frac_total,
        seed=args.seed,
        force_rebuild=args.force_manifest,
    )
    payload = load_split_manifest(manifest_path)

    train_records = manifest_records(payload, "train", task=args.task)
    val_records = manifest_records(payload, "val", task=args.task)
    test_records = manifest_records(payload, "test", task=args.task)
    include_coarse_view_idx = args.head_view_mode == "coarse4"
    train_samples = records_to_samples(train_records, include_coarse_view_idx=include_coarse_view_idx)
    val_samples = records_to_samples(val_records, include_coarse_view_idx=include_coarse_view_idx)
    test_samples = records_to_samples(test_records, include_coarse_view_idx=include_coarse_view_idx)

    if args.smoke:
        train_samples = balanced_take(train_samples, 8)
        val_samples = balanced_take(val_samples, 4)
        test_samples = balanced_take(test_samples, 4)
        train_records = align_records_to_samples(train_records, train_samples)
        val_records = align_records_to_samples(val_records, val_samples)
        test_records = align_records_to_samples(test_records, test_samples)
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.batch = min(args.batch, 2)
        args.grad_accum = 1
        args.warmup_epochs = 0
        args.kd_warmup_epochs = 0

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError(f"Task {args.task} has an empty split. Check the manifest at {manifest_path}.")

    if args.view_aux in ("ce", "both"):
        unsupported = sorted({sample[2] for sample in train_samples if sample[2] not in {"A4", "PSS"}})
        if unsupported:
            raise ValueError(
                f"view_aux={args.view_aux!r} only supports raw views A4/PSS, but task {args.task} includes {unsupported}."
            )

    run_name = task_run_name(args.task, args.head_view_mode)
    run_label = task_result_label(args.task, args.head_view_mode)
    run_dir = output_root / f"seed-{args.seed}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = run_dir / "predictions.csv"
    if predictions_csv.exists():
        predictions_csv.unlink()

    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device) if args.teacher_device else device
    aug_config = build_augmentation_config(args)

    train_loader, val_loader, test_loader, class_counts = build_loaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=args.batch,
        num_workers=args.num_workers,
        augmentation=aug_config,
    )

    weights_path = ECHO_ROOT / "model_data" / "weights" / "echo_prime_encoder.pt"
    encoder, injected = build_encoder(args, weights_path)
    classifier_in_dim = 512 + (NUM_COARSE_VIEWS if args.head_view_mode == "coarse4" else 0)
    classifier = build_classifier(
        args.head,
        args.hidden,
        args.head_dropout,
        num_blocks=args.num_blocks,
        in_dim=classifier_in_dim,
    )
    model = LoRAEchoPrimeDistillModel(
        encoder,
        classifier,
        view_aux=args.view_aux,
        head_view_mode=args.head_view_mode,
        head_dropout=args.head_dropout,
    ).to(device)

    teacher_kd = load_frozen_teacher(weights_path, teacher_device) if args.kd_mode != "none" else None
    teacher_view = None
    if args.view_aux in ("kd", "both"):
        teacher_view = load_view_teacher(ECHO_ROOT / "model_data" / "weights" / "view_classifier.pt", teacher_device)

    optimizer, scheduler = build_optimizer_and_scheduler(model, args, train_loader)
    n_total = int(class_counts.sum().item())
    class_weight = (n_total / (2.0 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    no_improve = 0

    print(f"[task={run_label} seed={args.seed}] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(
        f"[model] lora={args.lora} ft_mode={args.ft_mode} kd={args.kd_mode} "
        f"view_aux={args.view_aux} head_view_mode={args.head_view_mode}"
    )

    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(
            epoch,
            model,
            train_loader,
            teacher_kd,
            teacher_view,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
        )
        val_y, val_pred, val_prob = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_y, val_pred)
        try:
            val_auc = float(roc_auc_score(val_y, val_prob))
        except ValueError:
            val_auc = float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_ce": float(losses["ce"]),
                "train_kd": float(losses["kd"]),
                "train_view_ce": float(losses["view_ce"]),
                "train_view_kd": float(losses["view_kd"]),
                "train_total": float(losses["total"]),
                "val_accuracy": float(val_acc),
                "val_roc_auc": float(val_auc),
            }
        )
        improved = (val_auc > best_val_auc) if not math.isnan(val_auc) else False
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    val_y, _, val_prob = evaluate(model, val_loader, device)
    test_y, _, test_prob = evaluate(model, test_loader, device)
    save_prediction_csv(predictions_csv, "val", val_records, val_prob)
    save_prediction_csv(predictions_csv, "test", test_records, test_prob)

    eval_result = evaluate_prediction_rows(
        rows=[
            *[
                {
                    "split": "val",
                    "id": record["id"],
                    "path": record["path"],
                    "label": int(record["label"]),
                    "label_name": record["label_name"],
                    "source_group": record["source_group"],
                    "raw_view": record["raw_view"],
                    "coarse_view": record["coarse_view"],
                    "prob_pe": float(prob),
                }
                for record, prob in zip(val_records, val_prob.tolist())
            ],
            *[
                {
                    "split": "test",
                    "id": record["id"],
                    "path": record["path"],
                    "label": int(record["label"]),
                    "label_name": record["label_name"],
                    "source_group": record["source_group"],
                    "raw_view": record["raw_view"],
                    "coarse_view": record["coarse_view"],
                    "prob_pe": float(prob),
                }
                for record, prob in zip(test_records, test_prob.tolist())
            ],
        ],
        output_dir=run_dir / "evaluation",
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
        include_test_subgroups=(args.task == "pooled"),
    )

    checkpoint_path = run_dir / "best_checkpoint.pt"
    torch.save(
        {
            "model_state": best_state or model.state_dict(),
            "args": vars(args),
            "task": args.task,
            "task_label": run_label,
            "seed": args.seed,
            BEST_MODEL_METRIC: best_val_auc,
        },
        checkpoint_path,
    )

    trainable, total = count_parameters(model)
    result = {
        "task": args.task,
        "task_label": run_label,
        "head_view_mode": args.head_view_mode,
        "seed": int(args.seed),
        "manifest_path": str(manifest_path),
        "dataset_root": str(resolve_cli_path(args.dataset_root)),
        "run_dir": str(run_dir.resolve()),
        "augmentation": aug_config.to_dict(),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "train_positive_rate": float(np.mean([sample[1] for sample in train_samples])),
        "val_positive_rate": float(np.mean(val_y)),
        "test_positive_rate": float(np.mean(test_y)),
        "lora_injected_layers": injected,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "best_epoch": int(best_epoch),
        BEST_MODEL_METRIC: float(best_val_auc),
        "history": history,
        "artifacts": {
            "predictions_csv": str(predictions_csv.resolve()),
            "checkpoint_pt": str(checkpoint_path.resolve()),
            "metrics_json": eval_result["artifacts"]["metrics_json"],
        },
        "evaluation": eval_result,
    }

    result_json = run_dir / "result.json"
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {result_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune EchoPrime on the full PE dataset for pooled/per-view tasks.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--force-manifest", action="store_true", help="Regenerate the split manifest even if it exists.")
    parser.add_argument("--task", choices=TASKS, default="pooled")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--val-frac-total", type=float, default=DEFAULT_VAL_FRAC_TOTAL)
    parser.add_argument("--threshold-mode", choices=("youden", "f1", "fixed"), default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)

    parser.add_argument("--lora", choices=["on", "off"], default="off")
    parser.add_argument("--ft-mode", choices=list(FT_MODES), default="full")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    parser.add_argument("--target-modules", nargs="+", default=["qkv", "proj"], choices=list(TARGET_MODULE_PATTERNS))

    parser.add_argument("--head", choices=list(HEAD_REGISTRY), default="mlp")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.3, dest="head_dropout")
    parser.add_argument("--num-blocks", type=int, default=3, dest="num_blocks")
    parser.add_argument("--head-view-mode", choices=list(HEAD_VIEW_MODES), default="none", dest="head_view_mode")

    parser.add_argument("--kd-mode", choices=list(KD_MODES), default="none")
    parser.add_argument("--kd-weight", type=float, default=0.5, dest="kd_weight")
    parser.add_argument("--kd-warmup-epochs", type=int, default=2, dest="kd_warmup_epochs")
    parser.add_argument("--lam-rkd", type=float, default=1.0, dest="lam_rkd")

    parser.add_argument("--view-aux", choices=list(VIEW_AUX_MODES), default="none", dest="view_aux")
    parser.add_argument("--view-weight", type=float, default=0.3, dest="view_weight")
    parser.add_argument("--view-kd-weight", type=float, default=0.3, dest="view_kd_weight")
    parser.add_argument("--view-kd-temperature", type=float, default=2.0, dest="view_kd_temperature")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4, dest="head_lr")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, dest="warmup_epochs")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--teacher-device", default="", dest="teacher_device")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    add_augmentation_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

