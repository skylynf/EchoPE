#!/usr/bin/env python3
"""
Evaluate EchoPrime's standalone view classifier on dataset/preprocessed.

This script intentionally bypasses the full EchoPrime report / metric pipeline
and only runs the ConvNeXt-based view classifier used by `EchoPrime.get_views()`.

Ground truth comes from the dataset folder name:
  dataset/preprocessed/{Normal,PE}/{A4,IVC,MS,PSL,PSS,SX}/*.mp4

Because the pretrained classifier predicts 11 EchoPrime coarse views while this
dataset uses 6 folder names, the script reports two aligned evaluations:

1) 4-class coarse evaluation
   - A4 + MS              -> A4
   - PSL                  -> PSL
   - PSS                  -> PSS
   - IVC + SX             -> Subcostal

2) 3-class family evaluation
   - A4 + MS              -> Apical
   - PSL + PSS            -> Parasternal
   - IVC + SX             -> Subcostal

Predictions outside these aligned groups (e.g. SSN) are kept as "Other" so the
error pattern remains visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
ORIG_CWD = Path.cwd().resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from echo_paths import setup_echo_root_cwd  # noqa: E402


setup_echo_root_cwd()

import utils  # noqa: E402


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
RAW_VIEWS = ("A4", "IVC", "MS", "PSL", "PSS", "SX")

RAW_TO_GROUP = {
    "A4": "A4",
    "MS": "A4",
    "PSL": "PSL",
    "PSS": "PSS",
    "IVC": "Subcostal",
    "SX": "Subcostal",
}
GROUP_LABELS = ["A4", "PSL", "PSS", "Subcostal"]

RAW_TO_FAMILY = {
    "A4": "Apical",
    "MS": "Apical",
    "PSL": "Parasternal",
    "PSS": "Parasternal",
    "IVC": "Subcostal",
    "SX": "Subcostal",
}
FAMILY_LABELS = ["Apical", "Parasternal", "Subcostal"]

PRED_TO_GROUP = {
    "A2C": "A4",
    "A3C": "A4",
    "A4C": "A4",
    "A5C": "A4",
    "Apical_Doppler": "A4",
    "Doppler_Parasternal_Long": "PSL",
    "Parasternal_Long": "PSL",
    "Doppler_Parasternal_Short": "PSS",
    "Parasternal_Short": "PSS",
    "Subcostal": "Subcostal",
    "SSN": "Other",
}

PRED_TO_FAMILY = {
    "A2C": "Apical",
    "A3C": "Apical",
    "A4C": "Apical",
    "A5C": "Apical",
    "Apical_Doppler": "Apical",
    "Doppler_Parasternal_Long": "Parasternal",
    "Parasternal_Long": "Parasternal",
    "Doppler_Parasternal_Short": "Parasternal",
    "Parasternal_Short": "Parasternal",
    "Subcostal": "Subcostal",
    "SSN": "Other",
}

MEAN = torch.tensor([29.110628, 28.076836, 29.096405], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EchoPrime view classifier only.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=HERE.parent.parent / "dataset" / "preprocessed",
        help="Dataset root with Normal/ and PE/ subfolders.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size for first-frame tensors.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional sample limit for smoke testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=HERE / "view_classifier_eval.json",
        help="Where to write the evaluation summary JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=HERE / "view_classifier_predictions.csv",
        help="Where to write per-video predictions.",
    )
    return parser.parse_args()


def load_view_classifier(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = torchvision.models.convnext_base()
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 11)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def resolve_user_path(path: Path) -> Path:
    return path if path.is_absolute() else (ORIG_CWD / path).resolve()


def collect_samples(dataset_root: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for split_name in ("Normal", "PE"):
        split_dir = dataset_root / split_name
        if not split_dir.is_dir():
            continue
        for view_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            raw_view = view_dir.name
            for path in sorted(view_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    samples.append(
                        {
                            "split": split_name,
                            "raw_view": raw_view,
                            "path": str(path),
                            "file_name": path.name,
                        }
                    )
    return samples


def preprocess_first_frame(video_path: str) -> torch.Tensor:
    frames = utils.read_video_rgb_numpy(video_path)
    first_frame = utils.crop_and_scale(frames[0]).astype(np.float32)
    x = torch.from_numpy(first_frame).permute(2, 0, 1)
    x.sub_(MEAN).div_(STD)
    return x


def batch_predict(
    model: torch.nn.Module,
    items: list[dict[str, str]],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ok_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    batch_tensors: list[torch.Tensor] = []
    batch_meta: list[dict[str, str]] = []

    def flush() -> None:
        if not batch_tensors:
            return
        inputs = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu()
            pred_idx = torch.argmax(probs, dim=1).tolist()
        for meta, idx, prob in zip(batch_meta, pred_idx, probs.tolist()):
            pred_raw = utils.COARSE_VIEWS[idx]
            row = dict(meta)
            row["pred_raw"] = pred_raw
            row["pred_confidence"] = f"{max(prob):.6f}"
            row["gt_group"] = RAW_TO_GROUP.get(meta["raw_view"], "Unknown")
            row["pred_group"] = PRED_TO_GROUP.get(pred_raw, "Other")
            row["gt_family"] = RAW_TO_FAMILY.get(meta["raw_view"], "Unknown")
            row["pred_family"] = PRED_TO_FAMILY.get(pred_raw, "Other")
            ok_rows.append(row)
        batch_tensors.clear()
        batch_meta.clear()

    for item in tqdm(items, desc="Running standalone view classifier"):
        try:
            x = preprocess_first_frame(item["path"])
        except Exception as exc:  # noqa: BLE001
            failed_rows.append(
                {
                    **item,
                    "error": str(exc),
                }
            )
            continue
        batch_tensors.append(x)
        batch_meta.append(item)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()
    return ok_rows, failed_rows


def metric_block(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, object]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "other_prediction_rate": float(sum(p == "Other" for p in y_pred) / max(len(y_pred), 1)),
        "support": len(y_true),
        "label_order": labels + ["Other"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels + ["Other"]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def raw_view_breakdown(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    by_view: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_view[row["raw_view"]].append(row)

    out: dict[str, dict[str, object]] = {}
    for raw_view, sub_rows in sorted(by_view.items()):
        group_acc = sum(r["gt_group"] == r["pred_group"] for r in sub_rows) / len(sub_rows)
        family_acc = sum(r["gt_family"] == r["pred_family"] for r in sub_rows) / len(sub_rows)
        pred_group_counts = Counter(r["pred_group"] for r in sub_rows)
        pred_family_counts = Counter(r["pred_family"] for r in sub_rows)
        pred_raw_counts = Counter(r["pred_raw"] for r in sub_rows)
        out[raw_view] = {
            "count": len(sub_rows),
            "mapped_group_target": RAW_TO_GROUP.get(raw_view, "Unknown"),
            "family_target": RAW_TO_FAMILY.get(raw_view, "Unknown"),
            "group_accuracy": float(group_acc),
            "family_accuracy": float(family_acc),
            "top_pred_group_counts": dict(pred_group_counts.most_common()),
            "top_pred_family_counts": dict(pred_family_counts.most_common()),
            "top_pred_raw_counts": dict(pred_raw_counts.most_common(5)),
        }
    return out


def write_predictions_csv(path: Path, rows: list[dict[str, str]], failed_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "split",
        "raw_view",
        "gt_group",
        "gt_family",
        "pred_raw",
        "pred_group",
        "pred_family",
        "pred_confidence",
        "file_name",
        "path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "status": "ok",
                    **{k: row.get(k, "") for k in fieldnames if k != "status"},
                }
            )
        for row in failed_rows:
            writer.writerow(
                {
                    "status": "failed",
                    "split": row.get("split", ""),
                    "raw_view": row.get("raw_view", ""),
                    "gt_group": RAW_TO_GROUP.get(row.get("raw_view", ""), ""),
                    "gt_family": RAW_TO_FAMILY.get(row.get("raw_view", ""), ""),
                    "pred_raw": "",
                    "pred_group": "",
                    "pred_family": "",
                    "pred_confidence": "",
                    "file_name": row.get("file_name", ""),
                    "path": row.get("path", ""),
                    "error": row.get("error", ""),
                }
            )


def main() -> None:
    args = parse_args()
    dataset_root = resolve_user_path(args.dataset_root)
    output_json = resolve_user_path(args.output_json)
    output_csv = resolve_user_path(args.output_csv)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    device = torch.device(args.device)
    weights_path = HERE.parent / "model_data" / "weights" / "view_classifier.pt"
    if not weights_path.is_file():
        raise SystemExit(f"Missing view classifier weights: {weights_path}")

    samples = collect_samples(dataset_root)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No videos found under {dataset_root}")

    print(f"Dataset root: {dataset_root}")
    print(f"Device: {device}")
    print(f"Total discovered videos: {len(samples)}")

    model = load_view_classifier(weights_path, device)
    ok_rows, failed_rows = batch_predict(model, samples, args.batch_size, device)
    if not ok_rows:
        raise SystemExit("No videos were successfully processed.")

    y_true_group = [row["gt_group"] for row in ok_rows]
    y_pred_group = [row["pred_group"] for row in ok_rows]
    y_true_family = [row["gt_family"] for row in ok_rows]
    y_pred_family = [row["pred_family"] for row in ok_rows]

    summary = {
        "dataset_root": str(dataset_root),
        "device": str(device),
        "num_discovered": len(samples),
        "num_evaluated": len(ok_rows),
        "num_failed": len(failed_rows),
        "raw_view_counts": dict(Counter(row["raw_view"] for row in ok_rows)),
        "pred_raw_counts": dict(Counter(row["pred_raw"] for row in ok_rows)),
        "mapping_notes": {
            "raw_to_group": RAW_TO_GROUP,
            "raw_to_family": RAW_TO_FAMILY,
            "pred_to_group": PRED_TO_GROUP,
            "pred_to_family": PRED_TO_FAMILY,
        },
        "metrics_group_4way": metric_block(y_true_group, y_pred_group, GROUP_LABELS),
        "metrics_family_3way": metric_block(y_true_family, y_pred_family, FAMILY_LABELS),
        "per_raw_view": raw_view_breakdown(ok_rows),
        "failed_examples": failed_rows[:20],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions_csv(output_csv, ok_rows, failed_rows)

    print("\n=== 4-way grouped views ===")
    print(
        "accuracy={:.4f} balanced_acc={:.4f} macro_f1={:.4f} weighted_f1={:.4f} other_rate={:.4f}".format(
            summary["metrics_group_4way"]["accuracy"],
            summary["metrics_group_4way"]["balanced_accuracy"],
            summary["metrics_group_4way"]["macro_f1"],
            summary["metrics_group_4way"]["weighted_f1"],
            summary["metrics_group_4way"]["other_prediction_rate"],
        )
    )

    print("\n=== 3-way family views ===")
    print(
        "accuracy={:.4f} balanced_acc={:.4f} macro_f1={:.4f} weighted_f1={:.4f} other_rate={:.4f}".format(
            summary["metrics_family_3way"]["accuracy"],
            summary["metrics_family_3way"]["balanced_accuracy"],
            summary["metrics_family_3way"]["macro_f1"],
            summary["metrics_family_3way"]["weighted_f1"],
            summary["metrics_family_3way"]["other_prediction_rate"],
        )
    )

    print("\n=== Per raw dataset folder ===")
    for raw_view, stats in summary["per_raw_view"].items():
        print(
            f"{raw_view:>4}  n={stats['count']:>4}  "
            f"group_acc={stats['group_accuracy']:.4f}  "
            f"family_acc={stats['family_accuracy']:.4f}"
        )

    print(f"\nJSON written to: {output_json}")
    print(f"CSV written to:  {output_csv}")


if __name__ == "__main__":
    main()
