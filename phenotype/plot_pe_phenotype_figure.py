#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FOCUS_PHENOTYPES = [
    "rv_systolic_function_depressed",
    "right_ventricle_dilation",
    "pulmonary_artery_pressure_continuous",
]

PHENOTYPE_META = {
    "rv_systolic_function_depressed": {
        "title": "RV systolic function",
        "ylabel": "Predicted dysfunction score",
    },
    "right_ventricle_dilation": {
        "title": "RV dilation",
        "ylabel": "Predicted dilation score",
    },
    "pulmonary_artery_pressure_continuous": {
        "title": "Pulmonary artery pressure",
        "ylabel": "Predicted pressure",
    },
}

COLOR_NORMAL = "#3F74A3"
COLOR_NORMAL_LIGHT = "#D1DFE6"
COLOR_PE = "#B54646"
COLOR_PE_LIGHT = "#D18F90"
COLOR_THRESHOLD = "#794E47"
COLOR_GRID = "#E6E6E6"
COLOR_TEXT = "#222222"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot Nature-style PE-vs-Normal figure from saved phenotype analysis results."
    )
    ap.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("experiments/phenotype/results/analysis"),
        help="Analysis output directory relative to the EchoPrime root.",
    )
    ap.add_argument(
        "--max-points-per-group",
        type=int,
        default=450,
        help="Maximum number of jittered points to display for each label in each panel.",
    )
    return ap.parse_args()


def _configure_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _sample_points(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(len(values), size=max_points, replace=False))
    return values[take]


def _draw_violin(ax: plt.Axes, values: np.ndarray, position: float, facecolor: str, edgecolor: str) -> None:
    violin = ax.violinplot(
        [values],
        positions=[position],
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor(edgecolor)
        body.set_alpha(0.85)
        body.set_linewidth(0.8)


def _draw_points(ax: plt.Axes, values: np.ndarray, position: float, color: str, seed: int, max_points: int) -> None:
    show_values = _sample_points(values, max_points=max_points, seed=seed)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.12, 0.12, size=len(show_values))
    ax.scatter(
        np.full(len(show_values), position, dtype=float) + jitter,
        show_values,
        s=6,
        color=color,
        alpha=0.24,
        linewidths=0,
        zorder=3,
    )


def _draw_summary(ax: plt.Axes, values: np.ndarray, position: float, color: str) -> None:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.vlines(position, q1, q3, color=color, linewidth=2.0, zorder=4)
    ax.hlines(med, position - 0.15, position + 0.15, color=color, linewidth=1.4, zorder=4)


def _format_number(value: float | None, digits: int) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _add_panel_annotations(ax: plt.Axes, summary_row: pd.Series, phenotype: str) -> None:
    digits = 1 if phenotype == "pulmonary_artery_pressure_continuous" else 3
    annotation = (
        f"Delta mean={_format_number(summary_row['mean_diff_pe_minus_normal'], digits)}\n"
        f"AUC={_format_number(summary_row['auc_for_pe'], 3)}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=COLOR_TEXT,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D9D9D9", "linewidth": 0.6},
    )


def _plot_distribution_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    summary_row: pd.Series,
    phenotype: str,
    threshold: float | None,
    max_points: int,
) -> None:
    work = plot_df.loc[plot_df["phenotype"].eq(phenotype)].copy()
    normal_scores = work.loc[work["label"].eq("normal"), "score"].to_numpy(dtype=float)
    pe_scores = work.loc[work["label"].eq("pe"), "score"].to_numpy(dtype=float)

    _draw_violin(ax, normal_scores, 0.0, COLOR_NORMAL_LIGHT, COLOR_NORMAL)
    _draw_violin(ax, pe_scores, 1.0, COLOR_PE_LIGHT, COLOR_PE)
    _draw_points(ax, normal_scores, 0.0, COLOR_NORMAL, seed=17, max_points=max_points)
    _draw_points(ax, pe_scores, 1.0, COLOR_PE, seed=29, max_points=max_points)
    _draw_summary(ax, normal_scores, 0.0, COLOR_NORMAL)
    _draw_summary(ax, pe_scores, 1.0, COLOR_PE)

    if threshold is not None and not pd.isna(threshold):
        ax.axhline(
            threshold,
            color=COLOR_THRESHOLD,
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            zorder=1,
            label="Official threshold",
        )
        ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=6,
            framealpha=0.95,
            edgecolor="#D9D9D9",
            handlelength=2.2,
        )

    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Normal", "PE"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_title(PHENOTYPE_META[phenotype]["title"], pad=5)
    ax.set_ylabel(PHENOTYPE_META[phenotype]["ylabel"])
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6)
    _style_axes(ax)
    _add_panel_annotations(ax, summary_row, phenotype)


def write_representative_reports_txt(examples_df: pd.DataFrame, output_path: Path, wrap_width: int = 88) -> None:
    lines: list[str] = [
        "Representative EchoPrime-generated report snippets (focus phenotypes)",
        "",
    ]
    if examples_df.empty:
        lines.append("Report generation was skipped or no representative_report_examples.json was found.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for phenotype in FOCUS_PHENOTYPES:
        meta_title = PHENOTYPE_META[phenotype]["title"]
        lines.append("=" * 72)
        lines.append(f"{meta_title} ({phenotype})")
        lines.append("")

        panel_df = examples_df.loc[examples_df["phenotype"].eq(phenotype)].copy()
        panel_df["label"] = pd.Categorical(panel_df["label"], categories=["normal", "pe"], ordered=True)
        panel_df = panel_df.sort_values("label").reset_index(drop=True)

        for row in panel_df.itertuples(index=False):
            label_text = "Normal" if row.label == "normal" else "PE"
            if phenotype == "pulmonary_artery_pressure_continuous":
                header = f"{label_text} | score={row.score:.1f}"
            else:
                header = f"{label_text} | score={row.score:.3f}"
            lines.append(f"--- {header} ---")
            lines.append(
                textwrap.fill(str(row.report_snippet), width=wrap_width, break_long_words=False, break_on_hyphens=False)
            )
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_caption(summary_df: pd.DataFrame) -> str:
    lookup = summary_df.set_index("phenotype")
    pap = lookup.loc["pulmonary_artery_pressure_continuous"]
    rvs = lookup.loc["rv_systolic_function_depressed"]
    rvd = lookup.loc["right_ventricle_dilation"]
    return (
        "Figure. Nature-style violin plots with jittered observations compare EchoPrime phenotype predictions "
        "between Normal and PE videos for RV systolic function, RV dilation, and pulmonary artery pressure. "
        "Each dot represents one video, the central line marks the median, and the thick vertical segment shows "
        "the interquartile range. Dashed horizontal lines indicate the official ROC thresholds for the binary "
        "phenotypes (see legend). Representative English report snippets are provided in the accompanying text "
        "file. Across successful videos, predicted pulmonary artery pressure was higher in PE "
        f"than in Normal (delta mean {_format_number(pap['mean_diff_pe_minus_normal'], 2)}, "
        f"AUC {_format_number(pap['auc_for_pe'], 3)}), RV systolic dysfunction scores were modestly higher in PE "
        f"(delta mean {_format_number(rvs['mean_diff_pe_minus_normal'], 3)}, AUC {_format_number(rvs['auc_for_pe'], 3)}), "
        f"and RV dilation scores were also higher in PE (delta mean {_format_number(rvd['mean_diff_pe_minus_normal'], 4)}, "
        f"AUC {_format_number(rvd['auc_for_pe'], 3)})."
    )


def build_figure(
    plot_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    max_points: int,
) -> None:
    _configure_nature_style()
    fig = plt.figure(figsize=(7.2, 2.85), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    for col, phenotype in enumerate(FOCUS_PHENOTYPES):
        ax = fig.add_subplot(gs[0, col])
        summary_row = summary_df.loc[summary_df["phenotype"].eq(phenotype)].iloc[0]
        _plot_distribution_panel(
            ax=ax,
            plot_df=plot_df,
            summary_row=summary_row,
            phenotype=phenotype,
            threshold=summary_row["threshold_used"],
            max_points=max_points,
        )

    fig.savefig(output_png, dpi=600, transparent=False)
    fig.savefig(output_svg, dpi=600, transparent=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    echo_root = Path(__file__).resolve().parents[2]
    analysis_dir = (echo_root / args.analysis_dir).resolve()

    plot_df = pd.read_csv(analysis_dir / "focus_phenotype_plot_data.csv")
    summary_df = pd.read_csv(analysis_dir / "focus_phenotype_summary.csv")

    examples_path = analysis_dir / "representative_report_examples.json"
    examples_df = pd.read_json(examples_path) if examples_path.is_file() else pd.DataFrame()

    output_png = analysis_dir / "focus_phenotype_nature_main.png"
    output_svg = analysis_dir / "focus_phenotype_nature_main.svg"
    build_figure(
        plot_df=plot_df,
        summary_df=summary_df,
        output_png=output_png,
        output_svg=output_svg,
        max_points=args.max_points_per_group,
    )

    representative_txt = analysis_dir / "focus_phenotype_representative_reports.txt"
    write_representative_reports_txt(examples_df, representative_txt)

    caption = build_caption(summary_df)
    caption_path = analysis_dir / "figure_caption.txt"
    caption_path.write_text(caption + "\n", encoding="utf-8")

    print(f"saved_figure_png={output_png}")
    print(f"saved_figure_svg={output_svg}")
    print(f"saved_representative_text={representative_txt}")
    print(f"saved_caption={caption_path}")
    print("caption:")
    print(caption)


if __name__ == "__main__":
    main()
