from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_full_run import (
    compute_metrics,
    evaluate_prediction_rows,
    rows_from_prediction_csv,
    split_rows,
)


def _arrays(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    y_prob = np.array([float(row["prob_pe"]) for row in rows], dtype=np.float64)
    return y_true, y_prob


def choose_threshold_for_target_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
) -> tuple[float, dict[str, object]]:
    thresholds = np.unique(np.round(y_prob, 6))
    if thresholds.size == 0:
        return 0.5, {
            "target_recall": float(target_recall),
            "selected_threshold": 0.5,
            "selection_rule": "fallback_fixed_threshold",
            "warning": "No thresholds available; fallback to 0.5.",
        }

    candidates: list[tuple[float, float, float, float]] = []
    fallback_rows: list[tuple[float, float, float, float]] = []
    for thr in sorted(thresholds.tolist(), reverse=True):
        metrics = compute_metrics(y_true, y_prob, float(thr))
        sensitivity = float(metrics["sensitivity"])
        specificity = float(metrics["specificity"])
        f1 = float(metrics["f1"])
        row = (float(thr), sensitivity, specificity, f1)
        fallback_rows.append(row)
        if sensitivity >= target_recall:
            candidates.append(row)

    if candidates:
        selected_threshold, sensitivity, specificity, f1 = max(
            candidates,
            key=lambda row: (row[2], row[3], row[0]),
        )
        return selected_threshold, {
            "target_recall": float(target_recall),
            "selected_threshold": float(selected_threshold),
            "selection_rule": (
                "max_specificity_subject_to_recall; "
                "tie_breaker=f1_then_threshold"
            ),
            "selected_val_sensitivity": float(sensitivity),
            "selected_val_specificity": float(specificity),
            "selected_val_f1": float(f1),
            "n_candidate_thresholds": int(len(candidates)),
        }

    selected_threshold, sensitivity, specificity, f1 = max(
        fallback_rows,
        key=lambda row: (row[1], row[2], row[3], row[0]),
    )
    return selected_threshold, {
        "target_recall": float(target_recall),
        "selected_threshold": float(selected_threshold),
        "selection_rule": "max_recall_fallback_then_specificity_f1_threshold",
        "warning": (
            f"No validation threshold reaches recall >= {target_recall:.2f}; "
            "using the validation threshold with the highest recall."
        ),
        "selected_val_sensitivity": float(sensitivity),
        "selected_val_specificity": float(specificity),
        "selected_val_f1": float(f1),
        "n_candidate_thresholds": 0,
    }


def evaluate_predictions_extended(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    threshold_mode: str = "youden",
    fixed_threshold: float = 0.5,
    target_recall: float = 0.90,
    include_test_subgroups: bool = True,
) -> dict[str, object]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    standard = evaluate_prediction_rows(
        rows=rows,
        output_dir=out_dir,
        threshold_mode=threshold_mode,
        fixed_threshold=fixed_threshold,
        include_test_subgroups=include_test_subgroups,
    )

    val_rows = split_rows(rows, "val")
    test_rows = split_rows(rows, "test")
    val_true, val_prob = _arrays(val_rows)
    test_true, test_prob = _arrays(test_rows)

    recall_threshold, recall_info = choose_threshold_for_target_recall(
        val_true,
        val_prob,
        target_recall=target_recall,
    )
    recall_val_metrics = compute_metrics(val_true, val_prob, recall_threshold)
    recall_test_metrics = compute_metrics(test_true, test_prob, recall_threshold)

    combined = {
        "standard": standard,
        "recall_target": {
            "target_recall": float(target_recall),
            "selection_split": "val",
            "threshold_selection": recall_info,
            "val_metrics": recall_val_metrics,
            "test_metrics": recall_test_metrics,
            "test_specificity_at_target_recall": float(recall_test_metrics["specificity"]),
            "test_f1_at_target_recall": float(recall_test_metrics["f1"]),
        },
    }

    out_path = out_dir / "metrics_extended.json"
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    combined["artifacts"] = {"metrics_extended_json": str(out_path.resolve())}
    return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate binary PE predictions with standard and recall-target metrics."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-mode", choices=("youden", "f1", "fixed"), default="youden")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--no-subgroups", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = rows_from_prediction_csv(args.predictions_csv)
    result = evaluate_predictions_extended(
        rows=rows,
        output_dir=args.output_dir,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
        target_recall=args.target_recall,
        include_test_subgroups=not args.no_subgroups,
    )
    print(f"[done] extended metrics written to {result['artifacts']['metrics_extended_json']}")


if __name__ == "__main__":
    main()
