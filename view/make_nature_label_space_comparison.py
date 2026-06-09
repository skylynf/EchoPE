#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

SPACE_COLORS = {
    "6-way": "#3F74A3",
    "4-way coarse": "#B54646",
    "3-way family": "#794E47",
}

LIGHT_SPACE_COLORS = {
    "6-way": "#699BC5",
    "4-way coarse": "#D18F90",
    "3-way family": "#A78F95",
}

PALETTE = {
    "blue": "#3F74A3",
    "light_blue": "#699BC5",
    "very_light_blue": "#D1DFE6",
    "red": "#B54646",
    "light_red": "#D18F90",
    "brown": "#794E47",
    "warm_brown": "#9C7B57",
    "soft_mauve": "#A78F95",
    "yellow": "#F18F2B",
    "purple": "#C994AF",
    "green": "#8FBB8F",
}

MS_LABEL_COLOR = PALETTE["yellow"]

RAW_VIEW_DISPLAY_NAMES = {
    "SX": "Subxiphoid",
}


def display_class_name(name: str) -> str:
    return RAW_VIEW_DISPLAY_NAMES.get(name, name)


CAPTIONS = {
    "fig1_fig2_label_space_summary": (
        "Figure 1 | Label-space summary for view classification. Overall metrics compare the pretrained EchoPrime view classifier "
        "with the pretrained model after full fine-tuning at the full training set ratio. Per-class recall is shown for the "
        "fine-tuned deployment model, with shaded regions denoting six-way raw-view classification, four-way coarse grouping, "
        "and three-way family grouping. Error bars denote s.d. across seeds. "
        "The six-way setting remains bottlenecked by fine-grained apical, subcostal, and subxiphoid subclasses, whereas the grouped settings stabilize "
        "recall while preserving clinically meaningful view structure."
    ),
    "fig3_confusion_matrices": (
        "Figure 3 | Row-normalized confusion matrices for the final deployment model at the full training set ratio. "
        "Values are percentages computed from the mean confusion matrix across seeds. The six-way pretrained fine-tuned "
        "model is shown alongside the six-way pretrained model before fine-tuning, plus the four-way and three-way grouped tasks. "
        "The six-way task shows persistent confusion between adjacent fine-grained views, especially within apical-related categories, "
        "whereas the grouped tasks are dominated by strong diagonal structure. "
        "The four-way setting preserves more view specificity than the three-way setting while remaining highly separable."
    ),
}


@dataclass
class LabelSpaceSummary:
    display_name: str
    short_name: str
    class_names: list[str]
    color: str
    light_color: str
    metric_mean: dict[str, float]
    metric_std: dict[str, float]
    per_class_recall_mean: dict[str, float]
    per_class_recall_std: dict[str, float]
    mean_ovr_accuracy: float
    mean_ovr_accuracy_std: float
    support_mean: dict[str, float]
    confusion_labels: list[str]
    confusion_counts_mean: np.ndarray
    confusion_row_norm: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Nature-style label-space comparison figures.")
    parser.add_argument(
        "--results-6way",
        type=Path,
        default=HERE / "view_raw_finetune_results_v3" / "results.json",
    )
    parser.add_argument(
        "--results-4way",
        type=Path,
        default=HERE / "view_grouped_finetune_results_coarse4" / "results.json",
    )
    parser.add_argument(
        "--results-3way",
        type=Path,
        default=HERE / "view_grouped_finetune_results_family3" / "results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "label_space_nature_plots",
    )
    parser.add_argument("--ratio", type=float, default=1.0, help="Use the final full-data ratio by default.")
    return parser.parse_args()


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
            "legend.fontsize": 6,
            "figure.titlesize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, ymin: float, ymax: float, ylabel: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", color=PALETTE["very_light_blue"], linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def percent_labels(values: np.ndarray) -> list[str]:
    return [f"{100 * v:.1f}" for v in values]


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wrap_text(text: str, width: int = 118) -> str:
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def load_summary(
    path: Path,
    display_name: str,
    short_name: str,
    color: str,
    light_color: str,
    ratio: float,
    method: str = "pretrained_full_ft",
    trim_confusion_to_class_names: bool = False,
) -> LabelSpaceSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["results"]
        if row["method"] == method and abs(float(row["ratio"]) - float(ratio)) < 1e-9
    ]
    if not rows:
        raise ValueError(f"No {method} rows found in {path} for ratio={ratio}")

    metric_keys = ["balanced_accuracy", "macro_f1"]
    metric_mean = {}
    metric_std = {}
    for key in metric_keys:
        mu, sd = _mean_std([float(row["metrics"][key]) for row in rows])
        metric_mean[key] = mu
        metric_std[key] = sd

    class_names = list(payload["class_names"])
    per_class_recall_mean = {}
    per_class_recall_std = {}
    for class_name in class_names:
        mu, sd = _mean_std([float(row["metrics"]["per_class_recall"][class_name]) for row in rows])
        per_class_recall_mean[class_name] = mu
        per_class_recall_std[class_name] = sd

    mean_ovr_per_seed = []
    support_mean = {}
    for class_idx, class_name in enumerate(class_names):
        support_mean[class_name] = float(
            np.mean([sum(float(v) for v in row["metrics"]["confusion_matrix"][class_idx]) for row in rows])
        )

    for row in rows:
        ovr_values = list(row["metrics"]["per_class_accuracy_ovr"].values())
        mean_ovr_per_seed.append(float(np.mean(ovr_values)))
    mean_ovr_accuracy, mean_ovr_accuracy_std = _mean_std(mean_ovr_per_seed)

    confusion_labels = rows[0]["metrics"].get("confusion_label_order", class_names)
    confusion_mats = [np.asarray(row["metrics"]["confusion_matrix"], dtype=float) for row in rows]
    if trim_confusion_to_class_names and all(class_name in confusion_labels for class_name in class_names):
        confusion_indices = [confusion_labels.index(class_name) for class_name in class_names]
        confusion_labels = class_names
        confusion_mats = [mat[np.ix_(confusion_indices, confusion_indices)] for mat in confusion_mats]
    confusion_counts_mean = np.mean(confusion_mats, axis=0)
    row_sums = confusion_counts_mean.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        confusion_row_norm = np.divide(
            confusion_counts_mean,
            row_sums,
            out=np.zeros_like(confusion_counts_mean, dtype=float),
            where=row_sums > 0,
        )

    return LabelSpaceSummary(
        display_name=display_name,
        short_name=short_name,
        class_names=class_names,
        color=color,
        light_color=light_color,
        metric_mean=metric_mean,
        metric_std=metric_std,
        per_class_recall_mean=per_class_recall_mean,
        per_class_recall_std=per_class_recall_std,
        mean_ovr_accuracy=mean_ovr_accuracy,
        mean_ovr_accuracy_std=mean_ovr_accuracy_std,
        support_mean=support_mean,
        confusion_labels=confusion_labels,
        confusion_counts_mean=confusion_counts_mean,
        confusion_row_norm=confusion_row_norm,
    )


def _metric_arrays(
    summaries: list[LabelSpaceSummary],
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if metric_key == "mean_ovr_accuracy":
        means = np.array([summary.mean_ovr_accuracy for summary in summaries], dtype=float)
        stds = np.array([summary.mean_ovr_accuracy_std for summary in summaries], dtype=float)
    else:
        means = np.array([summary.metric_mean[metric_key] for summary in summaries], dtype=float)
        stds = np.array([summary.metric_std[metric_key] for summary in summaries], dtype=float)
    return means, stds


def plot_metric_and_recall_summary(
    direct_summaries: list[LabelSpaceSummary],
    finetuned_summaries: list[LabelSpaceSummary],
    out_dir: Path,
) -> None:
    metric_defs = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
        ("mean_ovr_accuracy", "Mean OvR accuracy"),
    ]

    fig, axes = plt.subplots(
        1,
        len(metric_defs) + 1,
        figsize=(13, 3.0),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 2.85]},
    )
    title_y = 1.12
    title_fontsize = 8
    x = np.arange(len(finetuned_summaries))
    width = 0.34

    for ax, (metric_key, metric_label) in zip(axes[:3], metric_defs):
        direct_means, direct_stds = _metric_arrays(direct_summaries, metric_key)
        finetuned_means, finetuned_stds = _metric_arrays(finetuned_summaries, metric_key)

        for label, means, stds, offset, face_mode, hatch in [
            ("Pretrained", direct_means, direct_stds, -width / 2, "light", "///"),
            ("Fine-tuned", finetuned_means, finetuned_stds, width / 2, "solid", None),
        ]:
            for idx, summary in enumerate(finetuned_summaries):
                facecolor = summary.light_color if face_mode == "light" else summary.color
                ax.bar(
                    x[idx] + offset,
                    means[idx],
                    yerr=stds[idx],
                    width=width,
                    color=facecolor,
                    edgecolor=summary.color,
                    linewidth=0.8,
                    capsize=2.5,
                    hatch=hatch,
                    alpha=0.96,
                    label=label if idx == 0 else None,
                )
                ax.text(
                    x[idx] + offset,
                    means[idx] + stds[idx] + 0.007,
                    f"{100 * means[idx]:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                )

        all_values = np.concatenate(
            [
                direct_means - direct_stds,
                finetuned_means - finetuned_stds,
            ]
        )
        ymin = max(0.0, float(np.floor((all_values.min() - 0.03) * 20) / 20))
        style_axis(ax, ymin=ymin, ymax=1.0)
        ax.set_xticks(x, [summary.short_name for summary in finetuned_summaries])
        ax.set_title(metric_label, y=title_y, pad=0, fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=6.8)
        if metric_key == "mean_ovr_accuracy":
            ax.legend(
                frameon=False,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.08),
                borderaxespad=0.4,
                fontsize=6.8,
            )

    recall_ax = axes[3]
    current_x = 0.0
    gap = 0.25
    recall_positions = []
    recall_labels = []
    recall_means = []
    recall_stds = []
    recall_colors = []
    recall_edges = []
    group_spans = []
    for summary in finetuned_summaries:
        start_x = current_x
        for class_name in summary.class_names:
            recall_positions.append(current_x)
            recall_labels.append(display_class_name(class_name))
            recall_means.append(summary.per_class_recall_mean[class_name])
            recall_stds.append(summary.per_class_recall_std[class_name])
            recall_colors.append(summary.light_color)
            recall_edges.append(summary.color)
            current_x += 1.0
        end_x = current_x - 1.0
        group_spans.append((start_x, end_x, summary))
        current_x += gap

    for idx, (start_x, end_x, summary) in enumerate(group_spans):
        span_left = start_x - 0.5 if idx == 0 else (group_spans[idx - 1][1] + start_x) / 2
        span_right = end_x + 0.5 if idx == len(group_spans) - 1 else (end_x + group_spans[idx + 1][0]) / 2
        recall_ax.axvspan(span_left, span_right, color=summary.light_color, alpha=0.18, zorder=0)
        recall_ax.text(
            (start_x + end_x) / 2,
            1.01,
            summary.short_name,
            transform=recall_ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=summary.color,
        )

    for pos, mean, std, facecolor, edgecolor in zip(
        recall_positions,
        recall_means,
        recall_stds,
        recall_colors,
        recall_edges,
    ):
        recall_ax.bar(
            pos,
            mean,
            yerr=std,
            width=0.64,
            color=facecolor,
            edgecolor=edgecolor,
            linewidth=0.85,
            capsize=2.0,
            zorder=3,
        )

    style_axis(recall_ax, ymin=0.4, ymax=1.0)
    recall_ax.set_title("Per-class recall", y=title_y, pad=0, fontsize=title_fontsize)
    recall_ax.set_xticks(recall_positions, recall_labels)
    recall_ax.tick_params(axis="x", rotation=45, labelsize=6.8)
    recall_ax.tick_params(axis="y", labelsize=6.8)
    recall_ax.set_xlim(min(recall_positions) - 0.5, max(recall_positions) + 0.5)

    fig.tight_layout(w_pad=1.35)
    save_figure(fig, "fig1_fig2_label_space_summary", out_dir)


def make_confusion_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "nature_blue_scale",
        [PALETTE["very_light_blue"], PALETTE["light_blue"], PALETTE["blue"]],
    )


def color_ms_tick_labels(ax: plt.Axes) -> None:
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        if tick.get_text() == "MS":
            tick.set_color(MS_LABEL_COLOR)


def plot_confusion_matrices(summaries: list[LabelSpaceSummary], out_dir: Path) -> None:
    cmap = make_confusion_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=False)
    axes_flat = axes.ravel()

    for ax, summary in zip(axes_flat, summaries):
        matrix = summary.confusion_row_norm
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="equal")
        ax.set_title(summary.display_name, pad=5, fontsize=8.2)
        display_labels = [display_class_name(label) for label in summary.confusion_labels]
        ax.set_xticks(np.arange(len(summary.confusion_labels)), display_labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(summary.confusion_labels)), display_labels)
        color_ms_tick_labels(ax)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis="both", which="major", length=0, labelsize=6.8)
        ax.set_xticks(np.arange(-0.5, len(summary.confusion_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(summary.confusion_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                label = f"{100 * value:.1f}"
                text_color = "white" if value >= 0.55 else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=5.8, color=text_color)

    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.09, top=0.92, wspace=0.04, hspace=0.24)
    cbar_ax = fig.add_axes([0.91, 0.17, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Row-normalized confusion (%)", fontsize=6.8)
    ticks = np.linspace(0, 1, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{100 * tick:.0f}" for tick in ticks])
    cbar.ax.tick_params(labelsize=6.8)

    save_figure(fig, "fig3_confusion_matrices", out_dir)


def write_captions(out_dir: Path) -> None:
    caption_path = out_dir / "figure_captions.txt"
    lines = []
    for stem, text in CAPTIONS.items():
        lines.append(f"{stem}\n{wrap_text(text)}\n")
    caption_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Figure captions ===")
    for stem, text in CAPTIONS.items():
        print(f"\n[{stem}]")
        print(wrap_text(text))
    print(f"\n[done] captions written to {caption_path}")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()

    set_nature_style()

    label_space_specs = [
        (args.results_6way.resolve(), "6-way raw views", "6-way", SPACE_COLORS["6-way"], LIGHT_SPACE_COLORS["6-way"]),
        (
            args.results_4way.resolve(),
            "4-way coarse grouping",
            "4-way",
            SPACE_COLORS["4-way coarse"],
            LIGHT_SPACE_COLORS["4-way coarse"],
        ),
        (
            args.results_3way.resolve(),
            "3-way family grouping",
            "3-way",
            SPACE_COLORS["3-way family"],
            LIGHT_SPACE_COLORS["3-way family"],
        ),
    ]
    direct_summaries = [
        load_summary(path, display_name, short_name, color, light_color, args.ratio, method="pretrained_direct")
        for path, display_name, short_name, color, light_color in label_space_specs
    ]
    finetuned_summaries = [
        load_summary(
            path,
            display_name,
            short_name,
            color,
            light_color,
            args.ratio,
            method="pretrained_full_ft",
            trim_confusion_to_class_names=True,
        )
        for path, display_name, short_name, color, light_color in label_space_specs
    ]
    plot_metric_and_recall_summary(direct_summaries, finetuned_summaries, args.output_dir)
    direct_summaries[0].display_name = "6-way raw views (pretrained)"
    plot_confusion_matrices(
        [
            finetuned_summaries[0],
            direct_summaries[0],
            finetuned_summaries[1],
            finetuned_summaries[2],
        ],
        args.output_dir,
    )
    write_captions(args.output_dir)

    print(f"[done] Nature-style figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
