from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

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
    DEFAULT_DATASET_ROOT,
    DEFAULT_FIXED_THRESHOLD,
    DEFAULT_THRESHOLD_MODE,
)
from data import RAW_TO_COARSE_VIEW  # noqa: E402
from evaluate_binary_cv import evaluate_predictions_extended  # noqa: E402
from finetuning.aux_heads import load_view_teacher  # noqa: E402
from finetuning.distill import load_frozen_teacher  # noqa: E402
from finetuning.train_lora import (  # noqa: E402
    HEAD_REGISTRY,
    TARGET_MODULE_PATTERNS,
    add_augmentation_args,
    build_augmentation_config,
    count_parameters,
    load_cv_fold_samples,
)
from finetuning.train_lora_distill import (  # noqa: E402
    FT_MODES,
    HEAD_VIEW_MODES,
    KD_MODES,
    NUM_COARSE_VIEWS,
    VIEW_AUX_MODES,
    LoRAEchoPrimeDistillModel,
    build_classifier,
    build_encoder,
    evaluate,
    train_one_epoch,
    view_collate,
    PEVideoDatasetWithView,
)
from train_full_run import (  # noqa: E402
    balanced_take,
    build_loaders,
    build_optimizer_and_scheduler,
    save_prediction_csv,
    set_seed,
)


def resolve_cli_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ORIG_CWD / candidate).resolve()


def sample_to_record(sample: tuple[str, int, str] | tuple[str, int, str, int]) -> dict[str, object]:
    path = str(sample[0])
    label = int(sample[1])
    raw_view = str(sample[2])
    coarse_view = RAW_TO_COARSE_VIEW.get(raw_view, raw_view)
    label_name = "PE" if label == 1 else "Normal"
    return {
        "id": Path(path).stem,
        "path": path,
        "label": label,
        "label_name": label_name,
        "source_group": label_name,
        "raw_view": raw_view,
        "coarse_view": coarse_view,
    }


def auto_model_label(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.lora == "on":
        parts.append("lora")
    else:
        parts.append(args.ft_mode)
    if args.kd_mode != "none":
        parts.append(f"kd-{args.kd_mode}")
    if args.view_aux != "none":
        parts.append(f"view-{args.view_aux}")
    return "_".join(parts)


def run_fold(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    output_root = resolve_cli_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fold_samples = load_cv_fold_samples(
        resolve_cli_path(args.cv_splits),
        args.fold,
        resolve_cli_path(args.dataset_root),
    )
    train_samples = fold_samples["train"]
    val_samples = fold_samples["val"]
    test_samples = fold_samples["test"]

    train_records = [sample_to_record(sample) for sample in train_samples]
    val_records = [sample_to_record(sample) for sample in val_samples]
    test_records = [sample_to_record(sample) for sample in test_samples]

    if args.smoke:
        train_samples = balanced_take(train_samples, 8)
        val_samples = balanced_take(val_samples, 4)
        test_samples = balanced_take(test_samples, 4)
        train_records = [sample_to_record(sample) for sample in train_samples]
        val_records = [sample_to_record(sample) for sample in val_samples]
        test_records = [sample_to_record(sample) for sample in test_samples]
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.batch = min(args.batch, 2)
        args.grad_accum = 1
        args.warmup_epochs = 0
        args.kd_warmup_epochs = 0

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError(f"Fold {args.fold} contains an empty split.")

    if args.view_aux in ("ce", "both"):
        unsupported = sorted({sample[2] for sample in train_samples if sample[2] not in {"A4", "PSS"}})
        if unsupported:
            raise ValueError(
                f"view_aux={args.view_aux!r} only supports raw views A4/PSS, got {unsupported}."
            )

    run_dir = output_root / f"fold-{args.fold}"
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

    history: list[dict[str, float]] = []
    best_val_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    print(
        f"[fold={args.fold}] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )
    print(
        f"[model] lora={args.lora} ft_mode={args.ft_mode} kd={args.kd_mode} view_aux={args.view_aux}"
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

    eval_result = evaluate_predictions_extended(
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
        target_recall=args.target_recall,
        include_test_subgroups=True,
    )

    checkpoint_path = run_dir / "best_checkpoint.pt"
    torch.save(
        {
            "model_state": best_state or model.state_dict(),
            "args": vars(args),
            "fold": int(args.fold),
            "model_label": auto_model_label(args),
            "val_roc_auc": float(best_val_auc),
        },
        checkpoint_path,
    )

    trainable, total = count_parameters(model)
    result = {
        "fold": int(args.fold),
        "model_label": auto_model_label(args),
        "cv_splits": str(resolve_cli_path(args.cv_splits)),
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
        "val_roc_auc": float(best_val_auc),
        "history": history,
        "artifacts": {
            "predictions_csv": str(predictions_csv.resolve()),
            "checkpoint_pt": str(checkpoint_path.resolve()),
            "metrics_standard_json": eval_result["standard"]["artifacts"]["metrics_json"],
            "metrics_extended_json": eval_result["artifacts"]["metrics_extended_json"],
            "auroc_curve_val_csv": eval_result["standard"]["artifacts"]["roc_val_csv"],
            "auroc_curve_test_csv": eval_result["standard"]["artifacts"]["roc_test_csv"],
            "aupr_curve_val_csv": eval_result["standard"]["artifacts"]["pr_val_csv"],
            "aupr_curve_test_csv": eval_result["standard"]["artifacts"]["pr_test_csv"],
            "threshold_sweep_val_csv": eval_result["standard"]["artifacts"]["threshold_sweep_val_csv"],
        },
        "evaluation": eval_result,
    }
    result_json = run_dir / "result.json"
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {result_json}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train binary PE baselines on external 5-fold CV splits."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cv-splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold-mode", choices=("youden", "f1", "fixed"), default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    parser.add_argument("--target-recall", type=float, default=0.90)

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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    add_augmentation_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_fold(args)


if __name__ == "__main__":
    main()
