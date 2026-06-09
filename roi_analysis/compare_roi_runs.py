#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roi_utils import jaccard, load_attribution_payloads, rank_correlation, resolve_path, top_indices


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as name=/path/to/roi_output_or_interpret_dir")
    name, path = value.split("=", 1)
    return name.strip(), resolve_path(path)


def maybe_read_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_roi_metrics(runs: list[tuple[str, Path]], filename: str) -> pd.DataFrame:
    frames = []
    for name, run_dir in runs:
        df = maybe_read_csv(run_dir, filename)
        if df.empty:
            continue
        value_cols = [
            col
            for col in df.columns
            if col
            not in {
                "case_id",
                "attribution_case_id",
                "split",
                "path",
                "label",
                "label_name",
                "raw_view",
                "coarse_view",
                "pred_label",
                "correct",
                "map_type",
                "region",
            }
        ]
        df = df.rename(columns={col: f"{col}_{name}" for col in value_cols})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    keys = [
        col
        for col in ("case_id", "path", "label", "label_name", "raw_view", "coarse_view", "map_type", "region")
        if col in frames[0].columns
    ]
    joined = frames[0]
    for frame in frames[1:]:
        join_keys = [col for col in keys if col in frame.columns]
        joined = joined.merge(frame, on=join_keys, how="inner")
    return joined


def add_pairwise_deltas(df: pd.DataFrame, runs: list[str], metrics: list[str]) -> pd.DataFrame:
    out = df.copy()
    if len(runs) < 2 or out.empty:
        return out
    baseline = runs[0]
    for candidate in runs[1:]:
        for metric in metrics:
            base_col = f"{metric}_{baseline}"
            cand_col = f"{metric}_{candidate}"
            if base_col in out.columns and cand_col in out.columns:
                out[f"delta_{metric}_{candidate}_minus_{baseline}"] = out[cand_col] - out[base_col]
    return out


def load_temporal_cases(output_dir: Path, run_name: str) -> pd.DataFrame:
    rows = []
    try:
        payloads = load_attribution_payloads(output_dir)
    except FileNotFoundError:
        return pd.DataFrame()
    for payload in payloads:
        record = payload["record"]
        curve_tensor = payload.get("temporal_occlusion_raw_prob", payload.get("temporal_occlusion"))
        curve = curve_tensor.float().cpu().numpy() if curve_tensor is not None else np.empty(0)
        rows.append(
            {
                "path": str(record["path"]),
                "split": str(payload["split"]),
                f"case_id_{run_name}": str(payload["case_id"]),
                f"temporal_curve_{run_name}": curve.tolist(),
                f"top_frames_{run_name}": sorted(top_indices(curve, k=3)),
                f"temporal_aopc_prob_{run_name}": float(np.clip(curve, a_min=0.0, a_max=None).mean()) if curve.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_temporal(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = [load_temporal_cases(path, name) for name, path in runs]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.merge(frame, on=["split", "path"], how="inner")
    if joined.empty or len(runs) < 2:
        return joined
    baseline = runs[0][0]
    rows = []
    for _, row in joined.iterrows():
        base_curve = np.asarray(row[f"temporal_curve_{baseline}"], dtype=float)
        base_top = set(row[f"top_frames_{baseline}"])
        out = {"split": row["split"], "path": row["path"]}
        for candidate, _path in runs[1:]:
            cand_curve = np.asarray(row[f"temporal_curve_{candidate}"], dtype=float)
            n = min(base_curve.size, cand_curve.size)
            out[f"top_frame_jaccard_{candidate}_vs_{baseline}"] = jaccard(base_top, set(row[f"top_frames_{candidate}"]))
            out[f"temporal_rank_corr_{candidate}_vs_{baseline}"] = rank_correlation(base_curve[:n], cand_curve[:n]) if n else np.nan
            out[f"delta_temporal_aopc_prob_{candidate}_minus_{baseline}"] = (
                float(row[f"temporal_aopc_prob_{candidate}"]) - float(row[f"temporal_aopc_prob_{baseline}"])
            )
        rows.append(out)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pairwise ROI and temporal comparison across model strategies.")
    parser.add_argument("--run", action="append", required=True, type=parse_run, help="name=/path/to/run_output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-file", default="roi_attribution_metrics.csv")
    parser.add_argument("--perturb-file", default="roi_perturbation_metrics.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = list(args.run)
    run_names = [name for name, _ in runs]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi = load_roi_metrics(runs, args.roi_file)
    roi = add_pairwise_deltas(roi, run_names, ["roi_mass", "roi_enrichment", "top10_overlap"])
    roi_path = output_dir / "paired_roi_attribution.csv"
    roi.to_csv(roi_path, index=False)

    perturb = load_roi_metrics(runs, args.perturb_file)
    perturb = add_pairwise_deltas(perturb, run_names, ["delta_prob", "delta_logit"])
    perturb_path = output_dir / "paired_roi_perturbation.csv"
    perturb.to_csv(perturb_path, index=False)

    temporal = compare_temporal(runs)
    temporal_path = output_dir / "paired_temporal_evidence.csv"
    temporal.to_csv(temporal_path, index=False)

    summary = {
        "runs": {name: str(path) for name, path in runs},
        "paired_roi_rows": int(len(roi)),
        "paired_perturbation_rows": int(len(perturb)),
        "paired_temporal_rows": int(len(temporal)),
        "artifacts": {
            "paired_roi_attribution": str(roi_path),
            "paired_roi_perturbation": str(perturb_path),
            "paired_temporal_evidence": str(temporal_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
