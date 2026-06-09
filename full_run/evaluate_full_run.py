from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import DEFAULT_FIXED_THRESHOLD, DEFAULT_THRESHOLD_MODE


THRESHOLD_MODES = ("youden", "f1", "fixed")


def rows_from_prediction_csv(path: str | Path) -> list[dict[str, object]]:
    in_path = Path(path).resolve()
    rows: list[dict[str, object]] = []
    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "split": row["split"],
                    "id": row["id"],
                    "path": row["path"],
                    "label": int(row["label"]),
                    "label_name": row["label_name"],
                    "source_group": row["source_group"],
                    "raw_view": row["raw_view"],
                    "coarse_view": row["coarse_view"],
                    "prob_pe": float(row["prob_pe"]),
                }
            )
    return rows


def split_rows(rows: Iterable[dict[str, object]], split_name: str) -> list[dict[str, object]]:
    return [row for row in rows if row["split"] == split_name]


def _arrays(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    y_prob = np.array([float(row["prob_pe"]) for row in rows], dtype=np.float64)
    return y_true, y_prob


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mode: str = DEFAULT_THRESHOLD_MODE,
    fixed_threshold: float = DEFAULT_FIXED_THRESHOLD,
) -> tuple[float, dict[str, object]]:
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"Unknown threshold mode: {mode}. Choices: {THRESHOLD_MODES}")
    if mode == "fixed":
        return float(fixed_threshold), {"mode": "fixed", "selected_threshold": float(fixed_threshold)}
    if len(np.unique(y_true)) < 2:
        return float(fixed_threshold), {
            "mode": mode,
            "selected_threshold": float(fixed_threshold),
            "warning": "Validation split has a single class; fallback to fixed threshold.",
        }

    if mode == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        scores = tpr - fpr
        best_idx = int(np.argmax(scores))
        return float(thresholds[best_idx]), {
            "mode": "youden",
            "selected_threshold": float(thresholds[best_idx]),
            "youden_j": float(scores[best_idx]),
            "selected_tpr": float(tpr[best_idx]),
            "selected_fpr": float(fpr[best_idx]),
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0:
        return float(fixed_threshold), {
            "mode": "f1",
            "selected_threshold": float(fixed_threshold),
            "warning": "No valid PR thresholds; fallback to fixed threshold.",
        }
    f1_scores = []
    for thr in thresholds:
        pred = (y_prob >= thr).astype(np.int64)
        f1_scores.append(f1_score(y_true, pred, zero_division=0))
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), {
        "mode": "f1",
        "selected_threshold": float(thresholds[best_idx]),
        "validation_f1": float(f1_scores[best_idx]),
        "curve_precision": float(precision[min(best_idx, len(precision) - 1)]),
        "curve_recall": float(recall[min(best_idx, len(recall) - 1)]),
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    prevalence = float(np.mean(y_true == 1))
    return {
        "n": int(len(y_true)),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "roc_auc": _safe_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_sweep_rows(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, object]]:
    thresholds = np.unique(np.round(y_prob, 6))
    rows = []
    for threshold in thresholds.tolist():
        metrics = compute_metrics(y_true, y_prob, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "ppv": metrics["ppv"],
                "npv": metrics["npv"],
            }
        )
    return rows


def roc_curve_rows(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, object]]:
    if len(np.unique(y_true)) < 2:
        return []
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return [
        {"threshold": float(thr), "fpr": float(x), "tpr": float(y)}
        for thr, x, y in zip(thresholds.tolist(), fpr.tolist(), tpr.tolist())
    ]


def pr_curve_rows(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, object]]:
    if len(np.unique(y_true)) < 2:
        return []
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    rows = []
    for idx in range(len(precision)):
        threshold = float(thresholds[idx]) if idx < len(thresholds) else None
        rows.append(
            {
                "threshold": threshold,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
            }
        )
    return rows


def subgroup_metrics(
    rows: list[dict[str, object]],
    threshold: float,
    group_key: str = "coarse_view",
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out = {}
    for group_name, items in sorted(grouped.items()):
        y_true, y_prob = _arrays(items)
        out[group_name] = compute_metrics(y_true, y_prob, threshold)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_prediction_rows(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    threshold_mode: str = DEFAULT_THRESHOLD_MODE,
    fixed_threshold: float = DEFAULT_FIXED_THRESHOLD,
    include_test_subgroups: bool = True,
) -> dict[str, object]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    val_rows = split_rows(rows, "val")
    test_rows = split_rows(rows, "test")
    if not val_rows or not test_rows:
        raise ValueError("Prediction rows must contain both val and test splits.")

    val_true, val_prob = _arrays(val_rows)
    test_true, test_prob = _arrays(test_rows)

    threshold, threshold_info = choose_threshold(
        val_true,
        val_prob,
        mode=threshold_mode,
        fixed_threshold=fixed_threshold,
    )

    val_metrics = compute_metrics(val_true, val_prob, threshold)
    test_metrics = compute_metrics(test_true, test_prob, threshold)

    sweep_rows = threshold_sweep_rows(val_true, val_prob)
    roc_val = roc_curve_rows(val_true, val_prob)
    roc_test = roc_curve_rows(test_true, test_prob)
    pr_val = pr_curve_rows(val_true, val_prob)
    pr_test = pr_curve_rows(test_true, test_prob)

    _write_csv(
        out_dir / "threshold_sweep_val.csv",
        sweep_rows,
        ["threshold", "accuracy", "balanced_accuracy", "f1", "sensitivity", "specificity", "ppv", "npv"],
    )
    _write_csv(out_dir / "roc_val.csv", roc_val, ["threshold", "fpr", "tpr"])
    _write_csv(out_dir / "roc_test.csv", roc_test, ["threshold", "fpr", "tpr"])
    _write_csv(out_dir / "pr_val.csv", pr_val, ["threshold", "precision", "recall"])
    _write_csv(out_dir / "pr_test.csv", pr_test, ["threshold", "precision", "recall"])

    grouped = {}
    if include_test_subgroups:
        grouped = subgroup_metrics(test_rows, threshold, group_key="coarse_view")
        subgroup_rows = []
        for subgroup_name, metrics in grouped.items():
            subgroup_rows.append(
                {
                    "coarse_view": subgroup_name,
                    "n": metrics["n"],
                    "prevalence": metrics["prevalence"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "ppv": metrics["ppv"],
                    "npv": metrics["npv"],
                    "brier_score": metrics["brier_score"],
                }
            )
        _write_csv(
            out_dir / "test_metrics_by_coarse_view.csv",
            subgroup_rows,
            [
                "coarse_view",
                "n",
                "prevalence",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "roc_auc",
                "pr_auc",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "brier_score",
            ],
        )

    result = {
        "threshold_selection": threshold_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_coarse_view": grouped,
        "artifacts": {
            "threshold_sweep_val_csv": str((out_dir / "threshold_sweep_val.csv").resolve()),
            "roc_val_csv": str((out_dir / "roc_val.csv").resolve()),
            "roc_test_csv": str((out_dir / "roc_test.csv").resolve()),
            "pr_val_csv": str((out_dir / "pr_val.csv").resolve()),
            "pr_test_csv": str((out_dir / "pr_test.csv").resolve()),
            "test_metrics_by_coarse_view_csv": str((out_dir / "test_metrics_by_coarse_view.csv").resolve()),
        },
    }
    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["artifacts"]["metrics_json"] = str(metrics_json.resolve())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PE full-run predictions with medical metrics.")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="CSV containing both val/test predictions.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-mode", choices=THRESHOLD_MODES, default=DEFAULT_THRESHOLD_MODE)
    parser.add_argument("--fixed-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    parser.add_argument("--no-subgroups", action="store_true", help="Disable test-set coarse-view subgroup metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = rows_from_prediction_csv(args.predictions_csv)
    result = evaluate_prediction_rows(
        rows,
        args.output_dir,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
        include_test_subgroups=not args.no_subgroups,
    )
    print(f"[done] metrics written to {result['artifacts']['metrics_json']}")


if __name__ == "__main__":
    main()

