#!/usr/bin/env python3
"""
Analyze results produced by `view_raw_finetune_scaling.py`.

Outputs:
    - raw/summary_*.csv
    - aggregated/summary_*_mean_std.csv
    - plots/*.png
"""
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


METHOD_LABELS = {
    "pretrained_direct": "Direct pretrained",
    "pretrained_linear_probe": "Pretrained + linear probe",
    "pretrained_partial_ft": "Pretrained + partial FT",
    "pretrained_full_ft": "Pretrained + full FT",
    "scratch_full_ft": "Scratch + full FT",
}

METHOD_COLORS = {
    "pretrained_direct": "#7f7f7f",
    "pretrained_linear_probe": "#1f77b4",
    "pretrained_partial_ft": "#2ca02c",
    "pretrained_full_ft": "#9467bd",
    "scratch_full_ft": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw view fine-tuning scaling results.")
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Path to results.json produced by view_raw_finetune_scaling.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional analysis output directory; defaults to sibling analysis/",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_overall_summary(results: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "method",
                "method_label",
                "ratio",
                "n_train_subset",
                "best_epoch",
                "best_val_macro_f1",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        )
        for row in sorted(results, key=lambda r: (r["ratio"], r["method"])):
            metrics = row["metrics"]
            writer.writerow(
                [
                    row.get("seed"),
                    row["method"],
                    METHOD_LABELS[row["method"]],
                    row["ratio"],
                    row["n_train_subset"],
                    row["best_epoch"],
                    row["best_val_macro_f1"],
                    metrics["accuracy"],
                    metrics["balanced_accuracy"],
                    metrics["macro_f1"],
                    metrics["weighted_f1"],
                ]
            )


def write_per_class_summary(results: list[dict], metric_key: str, out_csv: Path, class_names: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "method", "method_label", "ratio", "n_train_subset", *class_names])
        for row in sorted(results, key=lambda r: (r["ratio"], r["method"])):
            metric_map = row["metrics"][metric_key]
            writer.writerow(
                [
                    row.get("seed"),
                    row["method"],
                    METHOD_LABELS[row["method"]],
                    row["ratio"],
                    row["n_train_subset"],
                    *[metric_map.get(c, "") for c in class_names],
                ]
            )


def _group_results(results: list[dict]) -> dict[tuple[str, float], list[dict]]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in results:
        grouped[(row["method"], float(row["ratio"]))].append(row)
    return grouped


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def write_overall_aggregated_summary(results: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped = _group_results(results)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "method_label",
                "ratio",
                "n_seeds",
                "n_train_subset_mean",
                "n_train_subset_std",
                "accuracy_mean",
                "accuracy_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "weighted_f1_mean",
                "weighted_f1_std",
                "best_val_macro_f1_mean",
                "best_val_macro_f1_std",
            ]
        )
        for (method, ratio), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
            acc_mu, acc_sd = _mean_std([float(r["metrics"]["accuracy"]) for r in rows])
            bal_mu, bal_sd = _mean_std([float(r["metrics"]["balanced_accuracy"]) for r in rows])
            f1_mu, f1_sd = _mean_std([float(r["metrics"]["macro_f1"]) for r in rows])
            wf1_mu, wf1_sd = _mean_std([float(r["metrics"]["weighted_f1"]) for r in rows])
            n_mu, n_sd = _mean_std([float(r["n_train_subset"]) for r in rows])
            val_items = [r["best_val_macro_f1"] for r in rows if r["best_val_macro_f1"] is not None]
            if val_items:
                val_mu, val_sd = _mean_std([float(v) for v in val_items])
            else:
                val_mu, val_sd = float("nan"), float("nan")
            writer.writerow(
                [
                    method,
                    METHOD_LABELS[method],
                    ratio,
                    len(rows),
                    n_mu,
                    n_sd,
                    acc_mu,
                    acc_sd,
                    bal_mu,
                    bal_sd,
                    f1_mu,
                    f1_sd,
                    wf1_mu,
                    wf1_sd,
                    val_mu,
                    val_sd,
                ]
            )


def write_per_class_aggregated_summary(
    results: list[dict],
    metric_key: str,
    out_csv: Path,
    class_names: list[str],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped = _group_results(results)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        headers = ["method", "method_label", "ratio", "n_seeds"]
        for c in class_names:
            headers.extend([f"{c}_mean", f"{c}_std"])
        writer.writerow(headers)
        for (method, ratio), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
            out_row = [method, METHOD_LABELS[method], ratio, len(rows)]
            for c in class_names:
                mu, sd = _mean_std([float(r["metrics"][metric_key][c]) for r in rows])
                out_row.extend([mu, sd])
            writer.writerow(out_row)


def plot_metric_curves(results: list[dict], metric_key: str, metric_title: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 5.0))
    grouped = _group_results(results)
    methods = [m for m in METHOD_LABELS if any(r["method"] == m for r in results)]
    for method in methods:
        method_ratios = sorted({float(r["ratio"]) for r in results if r["method"] == method})
        xs = [100 * ratio for ratio in method_ratios]
        ys = []
        yerr = []
        for ratio in method_ratios:
            rows = grouped[(method, ratio)]
            mu, sd = _mean_std([float(r["metrics"][metric_key]) for r in rows])
            ys.append(mu)
            yerr.append(sd)
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
        plt.fill_between(
            xs,
            np.array(ys) - np.array(yerr),
            np.array(ys) + np.array(yerr),
            color=METHOD_COLORS[method],
            alpha=0.12,
        )
    plt.xlabel("Train subset ratio (%)")
    plt.ylabel(f"{metric_title} (mean±std)")
    plt.title(f"{metric_title} vs train subset ratio (mean±std)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_per_class_at_ratio(
    results: list[dict],
    metric_key: str,
    metric_title: str,
    ratio: float,
    class_names: list[str],
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        [r for r in results if abs(float(r["ratio"]) - float(ratio)) < 1e-9],
        key=lambda r: list(METHOD_LABELS).index(r["method"]),
    )
    if not rows:
        return

    x = np.arange(len(class_names))
    width = 0.15
    plt.figure(figsize=(11, 5.2))
    n = len(rows)
    for idx, row in enumerate(rows):
        ys = [float(row["metrics"][metric_key][c]) for c in class_names]
        offset = (idx - (n - 1) / 2.0) * width
        plt.bar(
            x + offset,
            ys,
            width=width,
            label=METHOD_LABELS[row["method"]],
            color=METHOD_COLORS[row["method"]],
        )
    plt.xticks(x, class_names)
    plt.ylim(0.0, 1.0)
    plt.ylabel(metric_title)
    plt.title(f"{metric_title} by class at ratio={ratio:.2f}")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_per_class_at_ratio_aggregated(
    results: list[dict],
    metric_key: str,
    metric_title: str,
    ratio: float,
    class_names: list[str],
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grouped = _group_results(results)
    methods = [m for m in METHOD_LABELS if (m, float(ratio)) in grouped]
    if not methods:
        return

    x = np.arange(len(class_names))
    width = 0.15
    n = len(methods)
    plt.figure(figsize=(11, 5.2))
    for idx, method in enumerate(methods):
        rows = grouped[(method, float(ratio))]
        ys = []
        yerr = []
        for c in class_names:
            mu, sd = _mean_std([float(r["metrics"][metric_key][c]) for r in rows])
            ys.append(mu)
            yerr.append(sd)
        offset = (idx - (n - 1) / 2.0) * width
        plt.bar(
            x + offset,
            ys,
            yerr=yerr,
            capsize=3,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    plt.xticks(x, class_names)
    plt.ylim(0.0, 1.0)
    plt.ylabel(f"{metric_title} (mean±std)")
    plt.title(f"{metric_title} by class at ratio={ratio:.2f} (mean±std)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_delta_vs_scratch(
    results: list[dict],
    metric_key: str,
    metric_title: str,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grouped = _group_results(results)
    scratch = {
        ratio: grouped[("scratch_full_ft", ratio)]
        for ratio in sorted({float(r["ratio"]) for r in results if r["method"] == "scratch_full_ft"})
    }
    compare_methods = [m for m in METHOD_LABELS if m != "scratch_full_ft"]

    plt.figure(figsize=(8.2, 5.0))
    for method in compare_methods:
        method_ratios = sorted({float(r["ratio"]) for r in results if r["method"] == method and float(r["ratio"]) in scratch})
        xs = [100 * ratio for ratio in method_ratios]
        ys = []
        yerr = []
        for ratio in method_ratios:
            rows = grouped[(method, ratio)]
            row_by_seed = {r.get("seed"): float(r["metrics"][metric_key]) for r in rows}
            scratch_by_seed = {r.get("seed"): float(r["metrics"][metric_key]) for r in scratch[ratio]}
            shared_seeds = sorted(set(row_by_seed) & set(scratch_by_seed))
            delta_vals = np.array([row_by_seed[s] - scratch_by_seed[s] for s in shared_seeds], dtype=float)
            mu, sd = _mean_std(delta_vals.tolist())
            ys.append(mu)
            yerr.append(sd)
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        plt.fill_between(
            xs,
            np.array(ys) - np.array(yerr),
            np.array(ys) + np.array(yerr),
            color=METHOD_COLORS[method],
            alpha=0.12,
        )
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("Train subset ratio (%)")
    plt.ylabel(f"Delta {metric_title} vs scratch (mean±std)")
    plt.title(f"{metric_title} gain over scratch (mean±std)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_json = args.results_json.resolve()
    payload = load_results(args.results_json)
    class_names = payload["class_names"]
    results = payload["results"]

    out_dir = args.output_dir.resolve() if args.output_dir else args.results_json.parent / "analysis"
    plots_dir = out_dir / "plots"
    raw_dir = out_dir / "raw"
    agg_dir = out_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_overall_summary(results, raw_dir / "summary_overall.csv")
    write_per_class_summary(results, "per_class_recall", raw_dir / "summary_per_class_recall.csv", class_names)
    write_per_class_summary(
        results, "per_class_accuracy_ovr", raw_dir / "summary_per_class_accuracy_ovr.csv", class_names
    )
    write_overall_aggregated_summary(results, agg_dir / "summary_overall_mean_std.csv")
    write_per_class_aggregated_summary(
        results, "per_class_recall", agg_dir / "summary_per_class_recall_mean_std.csv", class_names
    )
    write_per_class_aggregated_summary(
        results,
        "per_class_accuracy_ovr",
        agg_dir / "summary_per_class_accuracy_ovr_mean_std.csv",
        class_names,
    )

    plot_metric_curves(results, "macro_f1", "Macro F1", plots_dir / "macro_f1_curve.png")
    plot_metric_curves(results, "balanced_accuracy", "Balanced Accuracy", plots_dir / "balanced_accuracy_curve.png")
    plot_metric_curves(results, "accuracy", "Accuracy", plots_dir / "accuracy_curve.png")

    max_ratio = max(float(r["ratio"]) for r in results)
    plot_per_class_at_ratio_aggregated(
        results,
        "per_class_recall",
        "Per-class Recall",
        max_ratio,
        class_names,
        plots_dir / "per_class_recall_max_ratio.png",
    )
    plot_per_class_at_ratio_aggregated(
        results,
        "per_class_accuracy_ovr",
        "Per-class One-vs-Rest Accuracy",
        max_ratio,
        class_names,
        plots_dir / "per_class_accuracy_ovr_max_ratio.png",
    )

    plot_delta_vs_scratch(results, "macro_f1", "Macro F1", plots_dir / "macro_f1_gain_vs_scratch.png")
    plot_delta_vs_scratch(
        results,
        "balanced_accuracy",
        "Balanced Accuracy",
        plots_dir / "balanced_accuracy_gain_vs_scratch.png",
    )

    print(f"[done] analysis written to {out_dir}")


if __name__ == "__main__":
    main()
