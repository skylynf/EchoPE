#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from roi_utils import resolve_path


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as name=/path/to/roi_run_output")
    name, path = value.split("=", 1)
    return name.strip(), resolve_path(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def summarize_roi(run_name: str, run_dir: Path) -> pd.DataFrame:
    df = read_optional_csv(run_dir / "roi_attribution_metrics.csv")
    if df.empty:
        return pd.DataFrame()
    group_cols = [col for col in ("map_type", "region", "coarse_view", "label_name") if col in df.columns]
    summary = (
        df.groupby(group_cols, dropna=False)[["roi_mass", "roi_enrichment", "top10_overlap"]]
        .agg(["count", "median", "mean"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.insert(0, "run", run_name)
    return summary


def summarize_perturbation(run_name: str, run_dir: Path) -> pd.DataFrame:
    df = read_optional_csv(run_dir / "roi_perturbation_metrics.csv")
    if df.empty:
        return pd.DataFrame()
    group_cols = [col for col in ("region", "coarse_view", "label_name") if col in df.columns]
    summary = df.groupby(group_cols, dropna=False)[["delta_prob", "delta_logit"]].agg(["count", "median", "mean"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.insert(0, "run", run_name)
    return summary


def plot_roi_mass(roi_df: pd.DataFrame, output_path: Path) -> None:
    if roi_df.empty or "roi_mass_median" not in roi_df.columns:
        return
    subset = roi_df.loc[roi_df.get("map_type", "grad").astype(str).eq("grad")].copy() if "map_type" in roi_df else roi_df.copy()
    if subset.empty:
        return
    subset["label"] = subset["run"].astype(str) + " | " + subset["region"].astype(str)
    subset = subset.sort_values("roi_mass_median", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(subset))))
    ax.barh(subset["label"], subset["roi_mass_median"])
    ax.set_xlabel("Median ROI Attribution Mass")
    ax.set_title("Grad x Input ROI Attribution")
    ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_semantics(semantic_dir: Path | None) -> dict[str, object]:
    if semantic_dir is None:
        return {}
    summary_path = semantic_dir / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper-ready summaries from ROI, perturbation, and semantic probe outputs.")
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--semantic-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_summaries = [summarize_roi(name, path) for name, path in args.run]
    roi_summary = pd.concat([df for df in roi_summaries if not df.empty], ignore_index=True) if any(not df.empty for df in roi_summaries) else pd.DataFrame()
    roi_path = output_dir / "table_roi_attribution_summary.csv"
    roi_summary.to_csv(roi_path, index=False)

    perturb_summaries = [summarize_perturbation(name, path) for name, path in args.run]
    perturb_summary = (
        pd.concat([df for df in perturb_summaries if not df.empty], ignore_index=True)
        if any(not df.empty for df in perturb_summaries)
        else pd.DataFrame()
    )
    perturb_path = output_dir / "table_roi_perturbation_summary.csv"
    perturb_summary.to_csv(perturb_path, index=False)

    plot_path = output_dir / "fig_roi_mass_grad.png"
    plot_roi_mass(roi_summary, plot_path)

    semantic_summary = summarize_semantics(resolve_path(args.semantic_dir) if args.semantic_dir else None)
    summary = {
        "runs": {name: str(path) for name, path in args.run},
        "artifacts": {
            "roi_attribution_summary": str(roi_path),
            "roi_perturbation_summary": str(perturb_path),
            "roi_mass_figure": str(plot_path) if plot_path.is_file() else "",
        },
        "semantic_summary": semantic_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
