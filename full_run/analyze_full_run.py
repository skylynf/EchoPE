from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRIMARY_METRICS = [
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "balanced_accuracy",
    "accuracy",
    "f1",
    "brier_score",
]


def load_results(path: str | Path) -> dict[str, object]:
    in_path = Path(path).resolve()
    return json.loads(in_path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def result_task_label(row: dict[str, object]) -> str:
    if "task_label" in row:
        return str(row["task_label"])
    head_view_mode = str(row.get("head_view_mode", "none"))
    task = str(row["task"])
    return task if head_view_mode == "none" else f"{task}+headview-{head_view_mode}"


def write_raw_summary(results: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "task",
        "task_label",
        "head_view_mode",
        "train_size",
        "val_size",
        "test_size",
        "best_epoch",
        "selected_threshold",
        "val_roc_auc",
        "test_roc_auc",
        "test_pr_auc",
        "test_sensitivity",
        "test_specificity",
        "test_ppv",
        "test_npv",
        "test_balanced_accuracy",
        "test_accuracy",
        "test_f1",
        "test_brier_score",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r["task"], r["seed"])):
            threshold = row["evaluation"]["threshold_selection"]["selected_threshold"]
            val_metrics = row["evaluation"]["val_metrics"]
            test_metrics = row["evaluation"]["test_metrics"]
            writer.writerow(
                {
                    "seed": row["seed"],
                    "task": row["task"],
                    "task_label": result_task_label(row),
                    "head_view_mode": row.get("head_view_mode", "none"),
                    "train_size": row["train_size"],
                    "val_size": row["val_size"],
                    "test_size": row["test_size"],
                    "best_epoch": row["best_epoch"],
                    "selected_threshold": threshold,
                    "val_roc_auc": val_metrics["roc_auc"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_sensitivity": test_metrics["sensitivity"],
                    "test_specificity": test_metrics["specificity"],
                    "test_ppv": test_metrics["ppv"],
                    "test_npv": test_metrics["npv"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1": test_metrics["f1"],
                    "test_brier_score": test_metrics["brier_score"],
                }
            )


def write_aggregated_summary(results: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[result_task_label(row)].append(row)

    fields = ["task_label", "n_runs"]
    for metric in PRIMARY_METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task_label, task_rows in sorted(grouped.items()):
            out_row = {"task_label": task_label, "n_runs": len(task_rows)}
            for metric in PRIMARY_METRICS:
                values = [float(r["evaluation"]["test_metrics"][metric]) for r in task_rows]
                mu, sd = _mean_std(values)
                out_row[f"{metric}_mean"] = mu
                out_row[f"{metric}_std"] = sd
            writer.writerow(out_row)


def write_pooled_subgroup_summaries(results: list[dict[str, object]], raw_csv: Path, agg_csv: Path) -> None:
    pooled_rows = [row for row in results if row["task"] == "pooled"]
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_fields = [
        "seed", "task", "task_label", "head_view_mode", "coarse_view", "n",
        "roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy", "f1",
    ]
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        for row in sorted(pooled_rows, key=lambda r: r["seed"]):
            subgroup_metrics = row["evaluation"].get("test_metrics_by_coarse_view", {})
            for coarse_view, metrics in sorted(subgroup_metrics.items()):
                writer.writerow(
                    {
                        "seed": row["seed"],
                        "task": row["task"],
                        "task_label": result_task_label(row),
                        "head_view_mode": row.get("head_view_mode", "none"),
                        "coarse_view": coarse_view,
                        "n": metrics["n"],
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "sensitivity": metrics["sensitivity"],
                        "specificity": metrics["specificity"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pooled_rows:
        task_label = result_task_label(row)
        for coarse_view, metrics in row["evaluation"].get("test_metrics_by_coarse_view", {}).items():
            grouped[(task_label, coarse_view)].append(metrics)

    agg_fields = ["task_label", "coarse_view", "n_runs"]
    for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy", "f1"]:
        agg_fields.extend([f"{metric}_mean", f"{metric}_std"])
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for (task_label, coarse_view), rows in sorted(grouped.items()):
            out_row = {"task_label": task_label, "coarse_view": coarse_view, "n_runs": len(rows)}
            for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy", "f1"]:
                mu, sd = _mean_std([float(r[metric]) for r in rows])
                out_row[f"{metric}_mean"] = mu
                out_row[f"{metric}_std"] = sd
            writer.writerow(out_row)


def plot_metric_bars(results: list[dict[str, object]], metric: str, out_png: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in results:
        grouped[result_task_label(row)].append(float(row["evaluation"]["test_metrics"][metric]))
    if not grouped:
        return

    tasks = sorted(grouped)
    means = []
    stds = []
    for task in tasks:
        mu, sd = _mean_std(grouped[task])
        means.append(mu)
        stds.append(sd)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.8))
    xs = np.arange(len(tasks))
    plt.bar(xs, means, yerr=stds, capsize=4, color="#3F74A3")
    plt.xticks(xs, tasks)
    plt.ylim(0.0, 1.0)
    plt.ylabel(f"{metric} (mean±std)")
    plt.title(f"{metric} by task")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def run_analysis(results_json: str | Path, output_dir: str | Path | None = None) -> Path:
    payload = load_results(results_json)
    results = list(payload["results"])
    out_dir = Path(output_dir).resolve() if output_dir else Path(results_json).resolve().parent / "analysis"
    raw_dir = out_dir / "raw"
    agg_dir = out_dir / "aggregated"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_raw_summary(results, raw_dir / "summary_runs.csv")
    write_aggregated_summary(results, agg_dir / "summary_tasks_mean_std.csv")
    write_pooled_subgroup_summaries(
        results,
        raw_dir / "summary_pooled_test_by_coarse_view.csv",
        agg_dir / "summary_pooled_test_by_coarse_view_mean_std.csv",
    )
    for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "f1"]:
        plot_metric_bars(results, metric, plot_dir / f"{metric}_by_task.png")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate and visualize EchoPrime full-run results.")
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = run_analysis(args.results_json, args.output_dir)
    print(f"[done] analysis written to {out_dir}")


if __name__ == "__main__":
    main()

