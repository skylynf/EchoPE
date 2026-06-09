#!/usr/bin/env python3
"""Visualization for summarize_for_paper CSV tables (ROI attribution & perturbation)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Order for consistent plot axes (weak-context masks + four-way partition)
REGION_ORDER = ["foreground", "background", "sector_border", "probe_near_field"]
RUN_ORDER = ["frozen_head", "lora_kd_combo", "aug_combo_fullft"]

RUN_LABELS = {
    "frozen_head": "Frozen head",
    "lora_kd_combo": "LoRA + KD combo",
    "aug_combo_fullft": "Aug + full FT",
}


def _read_tables(
    attribution_csv: Path,
    perturbation_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attr = pd.read_csv(attribution_csv)
    pert = pd.read_csv(perturbation_csv)
    return attr, pert


def _agg_attr_stratum_medians(attr: pd.DataFrame, map_type: str, metric_median_col: str) -> pd.DataFrame:
    """Across coarse_view × label_name strata: take median of stratum-level medians per (run, region)."""
    sub = attr.loc[attr["map_type"].astype(str).eq(map_type)].copy()
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["run", "region"], observed=False, dropna=False)[metric_median_col].median()
    return g.unstack("run")


def _agg_pert_stratum_medians(pert: pd.DataFrame, metric_median_col: str) -> pd.DataFrame:
    sub = pert.copy()
    g = sub.groupby(["run", "region"], observed=False, dropna=False)[metric_median_col].median()
    return g.unstack("run")


def _reindex_region_run(pivot: pd.DataFrame) -> pd.DataFrame:
    if pivot.empty:
        return pivot
    regions = [r for r in REGION_ORDER if r in pivot.index]
    if not regions:
        regions = list(pivot.index)
    out = pivot.reindex(regions)
    for c in RUN_ORDER:
        if c not in out.columns:
            out[c] = np.nan
    return out[[c for c in RUN_ORDER]]


def plot_grouped_bars(
    pivot: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    figsize: tuple[float, float] = (9.0, 4.5),
) -> None:
    if pivot.empty:
        return
    regions = [r for r in REGION_ORDER if r in pivot.index]
    if not regions:
        regions = list(pivot.index)
    pivot = pivot.reindex(regions)
    runs = [c for c in RUN_ORDER if c in pivot.columns]
    if not runs:
        runs = list(pivot.columns)

    x = np.arange(len(regions), dtype=float)
    n = len(runs)
    width = min(0.22, 0.75 / max(n, 1))

    fig, ax = plt.subplots(figsize=figsize)
    for i, run in enumerate(runs):
        offset = (i - (n - 1) / 2.0) * width
        vals = []
        for r in regions:
            v = pivot.loc[r, run] if run in pivot.columns else np.nan
            vals.append(float(v) if pd.notna(v) else np.nan)
        ax.bar(x + offset, vals, width=width, label=RUN_LABELS.get(run, run), edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ") for r in regions], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_all(
    attribution_csv: Path,
    perturbation_csv: Path,
    output_dir: Path,
) -> list[Path]:
    attr, pert = _read_tables(attribution_csv, perturbation_csv)
    written: list[Path] = []

    # --- Attribution: grad & attention, three y-metrics
    for mt in ("grad", "attention"):
        for metric, ylabel, suffix in (
            ("roi_mass_median", "Median ROI mass (across view–label strata)", "roi_mass"),
            ("roi_enrichment_median", "Median ROI enrichment", "roi_enrichment"),
            ("top10_overlap_median", "Median top-10% overlap", "top10_overlap"),
        ):
            col = metric
            if col not in attr.columns:
                continue
            piv = _agg_attr_stratum_medians(attr, mt, col)
            if piv.empty:
                continue
            piv = _reindex_region_run(piv)
            fname = output_dir / f"fig_attribution_{mt}_{suffix}.png"
            plot_grouped_bars(
                piv,
                title=f"{mt.title()} · {suffix.replace('_', ' ')} (median of stratum medians)",
                ylabel=ylabel,
                output_path=fname,
            )
            written.append(fname)

    # --- Perturbation
    for col, ylabel, suffix in (
        ("delta_prob_median", "Median Δ P(PE)", "delta_prob"),
        ("delta_logit_median", "Median Δ logit(PE)", "delta_logit"),
    ):
        if col not in pert.columns:
            continue
        piv = _agg_pert_stratum_medians(pert, col)
        if piv.empty:
            continue
        piv = _reindex_region_run(piv)
        fname = output_dir / f"fig_perturbation_{suffix}.png"
        plot_grouped_bars(
            piv,
            title=f"Perturbation · {suffix.replace('_', ' ')} (median of stratum medians)",
            ylabel=ylabel,
            output_path=fname,
        )
        written.append(fname)

    # --- 2×2 overview: grad mass, grad enrichment, Δprob, Δlogit
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=False)
    piv_mass = _reindex_region_run(_agg_attr_stratum_medians(attr, "grad", "roi_mass_median"))
    piv_enr = _reindex_region_run(_agg_attr_stratum_medians(attr, "grad", "roi_enrichment_median"))
    piv_dp = _reindex_region_run(_agg_pert_stratum_medians(pert, "delta_prob_median"))
    piv_dl = _reindex_region_run(_agg_pert_stratum_medians(pert, "delta_logit_median"))

    def _subplot_bar(ax, pivot: pd.DataFrame, title: str, ylabel: str) -> None:
        if pivot.empty:
            ax.set_visible(False)
            return
        regions = [r for r in REGION_ORDER if r in pivot.index]
        pivot = pivot.reindex(regions)
        runs = [c for c in RUN_ORDER if c in pivot.columns]
        x = np.arange(len(regions), dtype=float)
        n = len(runs)
        width = min(0.22, 0.75 / max(n, 1))
        for i, run in enumerate(runs):
            offset = (i - (n - 1) / 2.0) * width
            vals = [float(pivot.loc[r, run]) if pd.notna(pivot.loc[r, run]) else np.nan for r in regions]
            ax.bar(x + offset, vals, width=width, label=RUN_LABELS.get(run, run), edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", " ") for r in regions], rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axhline(0.0, color="0.45", linewidth=0.7, linestyle="--", zorder=0)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    _subplot_bar(axes[0, 0], piv_mass, "Grad · ROI mass (median of stratum medians)", "ROI mass")
    _subplot_bar(axes[0, 1], piv_enr, "Grad · ROI enrichment", "Enrichment")
    _subplot_bar(axes[1, 0], piv_dp, "Perturbation · Δ P(PE)", "Δ probability")
    _subplot_bar(axes[1, 1], piv_dl, "Perturbation · Δ logit", "Δ logit")

    fig.suptitle("ROI tables overview (weak-context regions)", fontsize=12, y=1.02)
    fig.tight_layout()
    overview = output_dir / "fig_tables_overview.png"
    fig.savefig(overview, dpi=200, bbox_inches="tight")
    plt.close(fig)
    written.append(overview)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot ROI attribution & perturbation summary CSVs.")
    p.add_argument(
        "--attribution-csv",
        type=Path,
        default=HERE / "table_roi_attribution_summary.csv",
    )
    p.add_argument(
        "--perturbation-csv",
        type=Path,
        default=HERE / "table_roi_perturbation_summary.csv",
    )
    p.add_argument("--output-dir", type=Path, default=HERE)
    return p


def main() -> None:
    args = build_parser().parse_args()
    paths = draw_all(args.attribution_csv.resolve(), args.perturbation_csv.resolve(), args.output_dir.resolve())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
