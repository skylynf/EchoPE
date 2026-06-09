from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_DIR = Path(__file__).resolve().parent / "raw"
OUT_DIR = Path(__file__).resolve().parent / "nature_plots"

METHOD_ORDER = [
    "Direct pretrained",
    "Pretrained + full FT",
    "Pretrained + linear probe",
    "Pretrained + partial FT",
    "Scratch + full FT",
]

METHOD_COLORS = {
    "Direct pretrained": "#3F74A3",
    "Pretrained + full FT": "#B54646",
    "Pretrained + linear probe": "#794E47",
    "Pretrained + partial FT": "#C994AF",
    "Scratch + full FT": "#8FBB8F",
}

# Marker shapes differ by method (Pretrained + full FT uses star).
METHOD_MARKERS = {
    "Direct pretrained": None,  # horizontal dashed baseline, no markers
    "Pretrained + full FT": "*",
    "Pretrained + linear probe": "^",
    "Pretrained + partial FT": "s",
    "Scratch + full FT": "D",
}

DIRECT_PRETRAINED_LEGEND_LABEL = "Direct pretrained (four-class aggregate)"

CLASS_ORDER = ["A4", "IVC", "MS", "PSL", "PSS", "SX"]
RATIO_ORDER = [0.15, 0.30, 0.50, 0.75, 1.00]
RATIO_XTICKLABELS = [".15", ".30", ".50", ".75", "1.00"]

CAPTIONS = {
    "fig_combined_metrics_recall_ovr": (
        "Figure 1 | Overall metrics, per-class recall, and one-vs-rest (OvR) accuracy across training subset ratios "
        "(fraction of training data). Rows show accuracy, balanced accuracy, and macro-F1 (mean ± s.d. over seeds); "
        "per-class recall; and per-class OvR accuracy. "
        "Pretrained + full fine-tuning (star) dominates overall transfer; direct pretrained inference is aggregate "
        "four-way accuracy (horizontal dashed baseline). Recall collapses on minority classes until fine-tuning, "
        "while OvR accuracy can remain high despite poor recall on rare positives."
    ),
    "fig2_gain_vs_scratch": (
        "Figure 2 | Performance gain over scratch training. "
        "Mean balanced-accuracy and macro-F1 improvements are shown relative to Scratch + full FT at each training ratio. "
        "Transfer learning provides the largest benefit in the low-data regime, with the advantage narrowing as the training set grows."
    ),
}


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


def load_overall() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "summary_overall.csv")
    df["method_label"] = pd.Categorical(df["method_label"], METHOD_ORDER, ordered=True)
    df["ratio"] = pd.Categorical(df["ratio"], RATIO_ORDER, ordered=True)
    return df


def load_per_class(filename: str, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename)
    df = df.melt(
        id_vars=["seed", "method", "method_label", "ratio", "n_train_subset"],
        value_vars=CLASS_ORDER,
        var_name="class_name",
        value_name=value_name,
    )
    df["method_label"] = pd.Categorical(df["method_label"], METHOD_ORDER, ordered=True)
    df["ratio"] = pd.Categorical(df["ratio"], RATIO_ORDER, ordered=True)
    df["class_name"] = pd.Categorical(df["class_name"], CLASS_ORDER, ordered=True)
    return df


def summarize_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols, observed=True)[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "value_mean", "std": "value_std"})
    )
    summary["value_std"] = summary["value_std"].fillna(0.0)
    return summary


def style_axis(ax: plt.Axes, ymin: float = 0.0, ymax: float = 1.0) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(ymin, ymax)
    # Faint horizontal grid only
    ax.grid(axis="y", color="#E8ECEF", linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)


def _direct_constant_from_summary(subset: pd.DataFrame) -> tuple[float, float]:
    """Use first ratio row for constant direct-pretrained line (user: values identical)."""
    if subset.empty:
        return 0.0, 0.0
    s = subset.sort_values("ratio").iloc[0]
    return float(s["value_mean"]), float(s["value_std"])


def _band_lo_hi(
    y: np.ndarray, yerr: np.ndarray, clip: tuple[float, float] | None
) -> tuple[np.ndarray, np.ndarray]:
    lo = y - yerr
    hi = y + yerr
    if clip is not None:
        lo = np.clip(lo, clip[0], clip[1])
        hi = np.clip(hi, clip[0], clip[1])
    return lo, hi


def add_mean_sd_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str = "value_mean",
    std_col: str = "value_std",
    *,
    direct_horizontal: bool = True,
    value_band_clip: tuple[float, float] | None = (0.0, 1.0),
) -> None:
    x = np.arange(len(RATIO_ORDER))
    x_left, x_right = float(x[0]), float(x[-1])

    for method in METHOD_ORDER:
        subset = (
            data[data["method_label"] == method]
            .sort_values("ratio")
            .reset_index(drop=True)
        )
        if subset.empty:
            continue
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]

        if method == "Direct pretrained":
            if direct_horizontal:
                y0, err0 = _direct_constant_from_summary(subset)
                ax.axhline(
                    y0,
                    color=color,
                    linewidth=1.5,
                    linestyle=(0, (4, 3)),
                    zorder=2,
                )
                if value_band_clip is None:
                    y_lo, y_hi = y0 - err0, y0 + err0
                else:
                    y_lo = float(np.clip(y0 - err0, *value_band_clip))
                    y_hi = float(np.clip(y0 + err0, *value_band_clip))
                ax.fill_between(
                    [x_left, x_right],
                    y_lo,
                    y_hi,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            else:
                y = subset[value_col].to_numpy()
                yerr = subset[std_col].to_numpy()
                lo, hi = _band_lo_hi(y, yerr, value_band_clip)
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.5,
                    linestyle=(0, (4, 3)),
                    marker=None,
                    zorder=2,
                )
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            continue

        y = subset[value_col].to_numpy()
        yerr = subset[std_col].to_numpy()
        lo, hi = _band_lo_hi(y, yerr, value_band_clip)
        star_ms = 5.5 if marker == "*" else 3.2
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.6,
            marker=marker,
            markersize=star_ms,
            label=method,
            zorder=3,
        )
        ax.fill_between(
            x,
            lo,
            hi,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

    ax.set_xticks(x, RATIO_XTICKLABELS)


def legend_handles_for_methods(*, direct_label: str | None = None) -> tuple[list, list]:
    """Figure-level legend entries in METHOD_ORDER (star / shapes / dashed direct)."""
    if direct_label is None:
        direct_label = DIRECT_PRETRAINED_LEGEND_LABEL
    handles: list = []
    labels: list[str] = []
    dc = METHOD_COLORS["Direct pretrained"]
    handles.append(
        mlines.Line2D(
            [0],
            [0],
            color=dc,
            linestyle=(0, (4, 3)),
            linewidth=1.5,
            label=direct_label,
        )
    )
    labels.append(direct_label)
    for method in METHOD_ORDER[1:]:
        mk = METHOD_MARKERS[method]
        handles.append(
            mlines.Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker=mk,
                linestyle="-",
                linewidth=1.6,
                markersize=5.5 if mk == "*" else 3.2,
                label=method,
            )
        )
        labels.append(method)
    return handles, labels


def wrap_caption(text: str, width: int = 118) -> str:
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined_metrics_recall_ovr(
    overall_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    acc_ovr_df: pd.DataFrame,
) -> None:
    """3x5 layout: col0 = overall metrics; cols 1-2 = per-class recall; cols 3-4 = per-class OvR accuracy."""
    recall_summary = summarize_metric(
        recall_df, ["method_label", "ratio", "class_name"], "recall"
    )
    ovr_summary = summarize_metric(
        acc_ovr_df, ["method_label", "ratio", "class_name"], "accuracy_ovr"
    )

    overall_metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(12.0, 7.15), constrained_layout=False)

    # Column 0: overall metrics (one per row)
    for row, (metric_key, metric_label) in enumerate(overall_metrics):
        ax = axes[row, 0]
        metric_summary = summarize_metric(overall_df, ["method_label", "ratio"], metric_key)
        add_mean_sd_lines(ax, metric_summary)
        style_axis(ax, ymin=0.15 if metric_key == "macro_f1" else 0.2, ymax=0.95)
        ax.set_title(metric_label, pad=3)
        ax.set_xlabel("Ratio")
        ax.set_ylabel(metric_label)

    # Columns 1-2: recall; columns 3-4: OvR accuracy (CLASS_ORDER in row-major pairs)
    for row in range(3):
        c_left, c_right = CLASS_ORDER[2 * row], CLASS_ORDER[2 * row + 1]
        for col, cname in enumerate((c_left, c_right), start=1):
            ax = axes[row, col]
            subset = recall_summary[recall_summary["class_name"] == cname]
            add_mean_sd_lines(ax, subset)
            style_axis(ax, ymin=0.0, ymax=1.02)
            ax.set_title(f"Recall · {cname}", pad=3)
            ax.set_xlabel("Ratio")
            ax.set_ylabel("Recall")
        for col, cname in enumerate((c_left, c_right), start=3):
            ax = axes[row, col]
            subset = ovr_summary[ovr_summary["class_name"] == cname]
            add_mean_sd_lines(ax, subset)
            style_axis(ax, ymin=0.0, ymax=1.02)
            ax.set_title(f"OvR · {cname}", pad=3)
            ax.set_xlabel("Ratio")
            ax.set_ylabel("OvR accuracy")

    # Leave room below for legend; no figure-level caption or column header row.
    fig.tight_layout(rect=[0, 0.068, 1, 0.94])

    h, lab = legend_handles_for_methods()
    fig.legend(
        h,
        lab,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.036),
        borderaxespad=0.3,
        columnspacing=1.0,
        handlelength=2.2,
    )

    save_figure(fig, "fig_combined_metrics_recall_ovr")


def plot_gain_vs_scratch(overall_df: pd.DataFrame) -> None:
    scratch = (
        overall_df[overall_df["method_label"] == "Scratch + full FT"]
        .groupby(["seed", "ratio"], observed=True)[["balanced_accuracy", "macro_f1"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "balanced_accuracy": "scratch_balanced_accuracy",
                "macro_f1": "scratch_macro_f1",
            }
        )
    )
    merged = overall_df.merge(scratch, on=["seed", "ratio"], how="left")
    merged["balanced_accuracy_gain"] = (
        merged["balanced_accuracy"] - merged["scratch_balanced_accuracy"]
    )
    merged["macro_f1_gain"] = merged["macro_f1"] - merged["scratch_macro_f1"]
    gain_df = merged[merged["method_label"] != "Scratch + full FT"].copy()

    metrics = [
        ("balanced_accuracy_gain", "Balanced accuracy gain"),
        ("macro_f1_gain", "Macro-F1 gain"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.25), constrained_layout=False)

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        metric_summary = summarize_metric(
            gain_df,
            ["method_label", "ratio"],
            metric_key,
        )
        add_mean_sd_lines(
            ax,
            metric_summary,
            direct_horizontal=False,
            value_band_clip=None,
        )
        style_axis(ax, ymin=-0.05, ymax=0.55)
        ax.axhline(0, color="#A78F95", linewidth=0.8, linestyle="--")
        ax.set_title(metric_label, pad=4)
        ax.set_xlabel("Ratio")
        ax.set_ylabel(metric_label)

    h, lab = legend_handles_for_methods(direct_label="Direct pretrained")
    # Gain figure excludes Scratch + full FT from data; trim legend accordingly
    skip = "Scratch + full FT"
    h_f = [hi for hi, li in zip(h, lab) if skip not in li]
    lab_f = [li for li in lab if skip not in li]
    fig.tight_layout(rect=[0, 0.095, 1, 0.78])
    panel_top = float(max(ax.get_position().y1 for ax in axes))
    fig.text(
        0.5,
        panel_top + 0.024,
        wrap_caption(CAPTIONS["fig2_gain_vs_scratch"], width=72),
        ha="center",
        va="bottom",
        fontsize=6,
        linespacing=1.2,
        color="#333333",
    )
    fig.legend(
        h_f,
        lab_f,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.048),
        borderaxespad=0.35,
        columnspacing=1.0,
        handlelength=2.2,
    )
    save_figure(fig, "fig2_gain_vs_scratch")


def write_caption_file() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, caption in CAPTIONS.items():
        lines.append(f"{key}\n{caption}\n")
    (OUT_DIR / "captions.txt").write_text("\n".join(lines), encoding="utf-8")


def print_key_findings(overall_df: pd.DataFrame, recall_df: pd.DataFrame) -> None:
    macro_summary = summarize_metric(overall_df, ["method_label", "ratio"], "macro_f1")
    best_rows = macro_summary.sort_values(["ratio", "value_mean"], ascending=[True, False])
    print("Key findings")
    for ratio in RATIO_ORDER:
        top_row = best_rows[best_rows["ratio"] == ratio].iloc[0]
        print(
            f"- Ratio {ratio:.2f}: best macro-F1 = {top_row['value_mean']:.3f} "
            f"({top_row['method_label']})"
        )

    direct_problem = (
        recall_df[recall_df["method_label"] == "Direct pretrained"]
        .groupby("class_name", observed=True)["recall"]
        .mean()
        .sort_values()
    )
    print("- Direct pretrained lowest mean recall classes:")
    for class_name, value in direct_problem.items():
        print(f"  {class_name}: {value:.3f}")


def main() -> None:
    set_nature_style()
    overall_df = load_overall()
    recall_df = load_per_class("summary_per_class_recall.csv", "recall")
    acc_ovr_df = load_per_class("summary_per_class_accuracy_ovr.csv", "accuracy_ovr")

    plot_combined_metrics_recall_ovr(overall_df, recall_df, acc_ovr_df)
    plot_gain_vs_scratch(overall_df)
    write_caption_file()

    print_key_findings(overall_df, recall_df)
    print("\nCaptions")
    for key, caption in CAPTIONS.items():
        print(f"- {key}: {caption}")
    print(f"\nSaved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
