#!/usr/bin/env python3
"""Build a compact Nature-style figure for interpretability analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

RUN_ORDER = ["frozen_head", "lora_kd_combo", "aug_combo_fullft"]
RUN_LABELS = {
    "frozen_head": "Frozen head",
    "lora_kd_combo": "LoRA+KD",
    "aug_combo_fullft": "Aug+Full FT",
}
RUN_SHORT_LABELS = {
    "frozen_head": "P",
    "lora_kd_combo": "L",
    "aug_combo_fullft": "F",
}
RUN_COLORS = {
    "frozen_head": "#3F74A3",
    "lora_kd_combo": "#F18F2B",
    "aug_combo_fullft": "#B54646",
}
RUN_LIGHT_COLORS = {
    "frozen_head": "#699BC5",
    # Translucent pale yellow for Top-10 (pairs with RUN_COLORS[lora]); not the neutral blue-gray tint.
    "lora_kd_combo": (*to_rgb("#FFDFAE"), 0.48),
    "aug_combo_fullft": "#D18F90",
}

REGION_ORDER = ["foreground", "background", "sector_border", "probe_near_field"]
REGION_LABELS = {
    "foreground": "Foreground",
    "background": "Background",
    "sector_border": "Sector border",
    "probe_near_field": "Near field",
}
VIEW_ORDER = ["A4", "PSL", "PSS", "Subcostal"]

SEMANTIC_ROWS = [
    (r"$\mathrm{cos}_{\mathrm{Emb}}$", "embedding_cosine"),
    ("PAP", "pulmonary_artery_pressure_continuous"),
    ("RVSd", "rv_systolic_function_depressed"),
    ("RVd", "right_ventricle_dilation"),
    ("RAD", "right_atrium_dilation"),
    ("LAD", "left_atrium_dilation"),
]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "nature_blue_red",
    ["#3F74A3", "#D1DFE6", "#F7F3EF", "#D18F90", "#B54646"],
)

MAIN_CAPTION = (
    "Figure. Acquisition-context attribution, semantic alignment, and foreground occlusion. "
    "Top, weakly supervised context-ROI attribution summary across foreground, background, sector border, "
    "and probe near field for the three PE-head settings. ROI mass and top-10 overlap are shown on the left axis, "
    "and ROI enrichment is shown on the right axis. Bottom left, semantic probe alignment to frozen EchoPrime "
    "embeddings and phenotype readouts. Bottom right, median foreground-occlusion delta probability by view. "
    "LoRA+KD remains closely aligned to frozen EchoPrime, whereas Aug+Full FT shows stronger semantic drift "
    "and more view-dependent foreground-occlusion responses. The full fine-tuning branch uses the augmented-data "
    "pooled checkpoint and is therefore interpreted as a strong fine-tuning plus augmentation reference rather "
    "than a strict single-variable ablation."
)


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
        }
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_inputs(
    summary_json: Path,
    attribution_csv: Path,
    perturbation_csv: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    return _read_json(summary_json), pd.read_csv(attribution_csv), pd.read_csv(perturbation_csv)


def _semantic_frame(summary: dict) -> pd.DataFrame:
    semantic = summary.get("semantic_summary", {})
    corr = semantic.get("correlations", {})
    rows: list[dict[str, float | str]] = []
    for label, key in SEMANTIC_ROWS:
        if key == "embedding_cosine":
            lora = corr.get("embedding_cosine_lora_kd_combo_vs_frozen", {}).get("median")
            full = corr.get("embedding_cosine_full_finetune_vs_frozen", {}).get("median")
        else:
            lora = corr.get(f"spearman_{key}_lora_kd_combo_vs_frozen")
            full = corr.get(f"spearman_{key}_full_finetune_vs_frozen")
        rows.append(
            {
                "metric": label,
                "lora_kd_combo": float(lora) if lora is not None else np.nan,
                "aug_combo_fullft": float(full) if full is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _attribution_region_run_frame(attr: pd.DataFrame) -> pd.DataFrame:
    cols = ["roi_mass_median", "roi_enrichment_median", "top10_overlap_median"]
    summary = (
        attr.groupby(["region", "run"], observed=False, dropna=False)[cols]
        .median()
        .reset_index()
    )
    summary["region"] = pd.Categorical(summary["region"], categories=REGION_ORDER, ordered=True)
    summary["run"] = pd.Categorical(summary["run"], categories=RUN_ORDER, ordered=True)
    return summary.sort_values(["region", "run"]).reset_index(drop=True)


def _foreground_occlusion_frame(pert: pd.DataFrame) -> pd.DataFrame:
    sub = pert.loc[pert["region"].astype(str).eq("foreground")].copy()
    pivot = (
        sub.groupby(["coarse_view", "run"], observed=False, dropna=False)["delta_prob_median"]
        .median()
        .unstack("run")
    )
    pivot = pivot.reindex(VIEW_ORDER)
    for run in RUN_ORDER:
        if run not in pivot.columns:
            pivot[run] = np.nan
    return pivot[RUN_ORDER]


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _style_colorbar_no_outline(cb) -> None:
    cb.outline.set_visible(False)


def _save_transparent_svg(fig: plt.Figure, output_svg: Path) -> None:
    """Vector export with transparent figure and axes patches (Inkscape/design tools)."""
    output_svg = Path(output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
    fig.savefig(
        output_svg,
        format="svg",
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )


def plot_combined_attribution(ax: plt.Axes, attr_summary: pd.DataFrame) -> None:
    attr_summary = attr_summary.copy()
    attr_summary["x"] = np.arange(len(attr_summary), dtype=float)
    ax2 = ax.twinx()
    bar_width = 0.28

    for run in RUN_ORDER:
        run_sub = attr_summary.loc[attr_summary["run"].astype(str).eq(run)].copy()
        xpos = run_sub["x"].to_numpy(dtype=float)
        ax.bar(
            xpos - bar_width / 2.0,
            run_sub["roi_mass_median"].to_numpy(dtype=float),
            width=bar_width,
            color=RUN_COLORS[run],
            edgecolor=RUN_COLORS[run],
            linewidth=0.6,
            zorder=2,
        )
        ax.bar(
            xpos + bar_width / 2.0,
            run_sub["top10_overlap_median"].to_numpy(dtype=float),
            width=bar_width,
            color=RUN_LIGHT_COLORS[run],
            edgecolor=RUN_COLORS[run],
            linewidth=0.6,
            zorder=2,
        )
        ax2.plot(
            xpos,
            run_sub["roi_enrichment_median"].to_numpy(dtype=float),
            color=RUN_COLORS[run],
            marker="o",
            markersize=3.0,
            linewidth=0.9,
            zorder=3,
        )

    left_max = float(
        np.nanmax(
            np.concatenate(
                [
                    attr_summary["roi_mass_median"].to_numpy(dtype=float),
                    attr_summary["top10_overlap_median"].to_numpy(dtype=float),
                ]
            )
        )
    )
    right_max = float(np.nanmax(attr_summary["roi_enrichment_median"].to_numpy(dtype=float)))
    ax.set_ylim(0.0, max(0.9, left_max + 0.08))
    ax2.set_ylim(0.0, max(3.0, right_max + 0.25))
    ax.set_ylabel("ROI Mass / ROI Top-10")
    ax2.set_ylabel("ROI Enrichment")
    ax.set_xlim(-0.7, len(attr_summary) - 0.3)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    xticks = attr_summary["x"].to_numpy(dtype=float)
    ax.set_xticks(xticks)
    ax.set_xticklabels([RUN_SHORT_LABELS[str(run)] for run in attr_summary["run"]])

    for ridx, region in enumerate(REGION_ORDER):
        start = ridx * len(RUN_ORDER)
        end = start + len(RUN_ORDER) - 1
        center = (start + end) / 2.0
        if ridx < len(REGION_ORDER) - 1:
            ax.axvline(end + 0.5, color="#D9D9D9", linewidth=0.6, zorder=1)
        ax.text(
            center,
            -0.30,
            REGION_LABELS[region],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.8,
        )

    legend_handles = [
        Patch(facecolor="#7F7F7F", edgecolor="#7F7F7F", label="ROI mass"),
        Patch(facecolor="#D9D9D9", edgecolor="#7F7F7F", label="Top-10 overlap"),
        Line2D([0], [0], color="#7F7F7F", marker="o", linewidth=0.9, markersize=3.0, label="ROI enrichment"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        ncols=3,
        handlelength=1.5,
        columnspacing=0.6,
        borderpad=0.2,
        handletextpad=0.4,
        fontsize=5.6,
    )
    _style_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(direction="out")


def plot_semantic_dumbbell(ax: plt.Axes, sem: pd.DataFrame) -> None:
    ypos = np.arange(len(sem))[::-1]
    for i, row in enumerate(sem.itertuples(index=False)):
        y = ypos[i]
        full = float(row.aug_combo_fullft)
        lora = float(row.lora_kd_combo)
        ax.plot([full, lora], [y, y], color="#BFBFBF", linewidth=0.9, zorder=1)
        ax.scatter(
            full,
            y,
            s=18,
            color=RUN_COLORS["aug_combo_fullft"],
            label="Aug+Full FT" if i == 0 else None,
            zorder=3,
        )
        ax.scatter(
            lora,
            y,
            s=18,
            color=RUN_COLORS["lora_kd_combo"],
            label="LoRA+KD" if i == 0 else None,
            zorder=3,
        )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.6, len(sem) - 0.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(list(sem["metric"]))
    ax.set_xlabel("Alignment to frozen EchoPrime")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.legend(
        frameon=False,
        loc="upper left",
        handletextpad=0.3,
        borderpad=0.2,
        fontsize=5.4,
    )
    _style_axes(ax)


def plot_foreground_heatmap(ax: plt.Axes, occ: pd.DataFrame):
    data = occ.to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(data))), 0.12)
    im = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(RUN_ORDER)))
    ax.set_xticklabels([RUN_SHORT_LABELS[r] for r in RUN_ORDER])
    ax.set_yticks(np.arange(len(VIEW_ORDER)))
    ax.set_yticklabels(VIEW_ORDER)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "NA"
                color = "#4D4D4D"
            else:
                text = f"{val:+.2f}"
                color = "white" if abs(val) > vmax * 0.42 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return im


def build_main_figure(
    summary: dict,
    attr: pd.DataFrame,
    pert: pd.DataFrame,
    output_path: Path,
) -> None:
    _configure_nature_style()
    sem = _semantic_frame(summary)
    attr_summary = _attribution_region_run_frame(attr)
    occ = _foreground_occlusion_frame(pert)

    # "compressed" + small w_space: gridspec wspace stacks with constrained_layout paddings;
    # previously wspace=0.14 made the B/C gutter visibly wide.
    fig = plt.figure(figsize=(4.0, 3.0), layout="compressed")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.0, 3.0],
        width_ratios=[1.0, 1.0],
        hspace=0.18,
        wspace=0.03,
    )

    ax_attr = fig.add_subplot(gs[0, :])
    plot_combined_attribution(ax_attr, attr_summary)

    ax_sem = fig.add_subplot(gs[1, 0])
    plot_semantic_dumbbell(ax_sem, sem)

    ax_occ = fig.add_subplot(gs[1, 1])
    im = plot_foreground_heatmap(ax_occ, occ)
    cbar = fig.colorbar(im, ax=ax_occ, fraction=0.055, pad=-0.18)
    _style_colorbar_no_outline(cbar)
    cbar.set_label(r"$\Delta$ prob.", rotation=90, labelpad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    _save_transparent_svg(fig, output_path.with_suffix(".svg"))
    plt.close(fig)


def build_spatial_only_figure(
    attr: pd.DataFrame,
    pert: pd.DataFrame,
    output_path: Path,
) -> None:
    _configure_nature_style()
    attr_summary = _attribution_region_run_frame(attr)
    occ = _foreground_occlusion_frame(pert)

    fig = plt.figure(figsize=(6.0, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.16)

    ax_attr = fig.add_subplot(gs[0, 0])
    plot_combined_attribution(ax_attr, attr_summary)

    ax_occ = fig.add_subplot(gs[1, 0])
    im = plot_foreground_heatmap(ax_occ, occ)
    cbar = fig.colorbar(im, ax=ax_occ, fraction=0.028, pad=-0.018)
    _style_colorbar_no_outline(cbar)
    cbar.set_label(r"$\Delta$ prob.", rotation=90)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    _save_transparent_svg(fig, output_path.with_suffix(".svg"))
    plt.close(fig)


def build_semantic_only_figure(summary: dict, output_path: Path) -> None:
    _configure_nature_style()
    sem = _semantic_frame(summary)
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    plot_semantic_dumbbell(ax, sem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    _save_transparent_svg(fig, output_path.with_suffix(".svg"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate compact paper figures for interpretability analysis.")
    p.add_argument("--summary-json", type=Path, default=HERE / "summary.json")
    p.add_argument("--attribution-csv", type=Path, default=HERE / "table_roi_attribution_summary.csv")
    p.add_argument("--perturbation-csv", type=Path, default=HERE / "table_roi_perturbation_summary.csv")
    p.add_argument("--output-dir", type=Path, default=HERE)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary, attr, pert = _read_inputs(
        args.summary_json.resolve(),
        args.attribution_csv.resolve(),
        args.perturbation_csv.resolve(),
    )
    output_dir = args.output_dir.resolve()
    build_main_figure(summary, attr, pert, output_dir / "fig_interpretability_compact.png")
    build_spatial_only_figure(attr, pert, output_dir / "fig_spatial_context_compact.png")
    build_semantic_only_figure(summary, output_dir / "fig_semantic_drift_compact.png")
    print(output_dir / "fig_interpretability_compact.png")
    print(output_dir / "fig_interpretability_compact.svg")
    print(output_dir / "fig_spatial_context_compact.png")
    print(output_dir / "fig_spatial_context_compact.svg")
    print(output_dir / "fig_semantic_drift_compact.png")
    print(output_dir / "fig_semantic_drift_compact.svg")
    print("\nCaption:\n")
    print(MAIN_CAPTION)


if __name__ == "__main__":
    main()
