#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from roi_utils import (
    ANATOMY_MASK_NAMES,
    CONTEXT_MASK_NAMES,
    load_annotations,
    load_attribution_payloads,
    load_or_generate_mask_payload,
    normalized_entropy,
    resolve_path,
    roi_metrics,
    spatialize_attribution,
    summarize_metric_rows,
)


def mask_case_lookup(mask_dir: Path) -> dict[str, str]:
    manifest_path = mask_dir / "mask_manifest.csv"
    if not manifest_path.is_file():
        return {}
    manifest = pd.read_csv(manifest_path)
    if "path" not in manifest.columns or "case_id" not in manifest.columns:
        return {}
    return {str(row["path"]): str(row["case_id"]) for _, row in manifest.iterrows()}


def compute_anatomy_context_ratios(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    ratio_rows = []
    group_cols = ["case_id", "path", "map_type"]
    for keys, group in df.groupby(group_cols, dropna=False):
        anatomy = group.loc[group["region"].isin(ANATOMY_MASK_NAMES), "roi_mass"].sum()
        context = group.loc[group["region"].isin(CONTEXT_MASK_NAMES), "roi_mass"].sum()
        ratio_rows.append(
            {
                "case_id": keys[0],
                "path": keys[1],
                "map_type": keys[2],
                "anatomy_mass": float(anatomy),
                "context_mass": float(context),
                "anatomy_to_context_ratio": float(anatomy / context) if context > 0 else float("nan"),
            }
        )
    return pd.DataFrame(ratio_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantify ROI attribution mass, enrichment, and top-k overlap.")
    parser.add_argument("--interpret-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None, help="Optional anatomical ROI JSON/JSONL annotations.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=None, help="Optional fixed frame for 3D maps. Default averages over time.")
    parser.add_argument("--top-ratio", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    interpret_dir = resolve_path(args.interpret_dir)
    mask_dir = resolve_path(args.mask_dir)
    output_csv = resolve_path(args.output_csv)
    annotations = load_annotations(args.annotations)
    path_to_mask_case = mask_case_lookup(mask_dir)
    rows: list[dict[str, object]] = []
    skipped = []

    for payload in load_attribution_payloads(interpret_dir):
        record = payload["record"]
        attr_case_id = str(payload["case_id"])
        mask_case_id = path_to_mask_case.get(str(record["path"]), attr_case_id)
        mask_payload, generated_mask = load_or_generate_mask_payload(
            mask_dir,
            case_id=mask_case_id,
            record={**record, "split": payload["split"]},
        )
        masks = dict(mask_payload["masks"])
        masks.update(annotations.get(mask_case_id, {}))
        masks.update(annotations.get(attr_case_id, {}))
        if not masks:
            skipped.append({"case_id": attr_case_id, "path": str(record["path"]), "reason": "no_masks"})
            continue
        maps = {
            "grad": spatialize_attribution(payload["grad_map"], frame_index=args.frame_index),
            "attention": spatialize_attribution(payload["attention_map"], frame_index=args.frame_index),
        }
        for map_type, score_map in maps.items():
            entropy = normalized_entropy(score_map)
            for region, mask in masks.items():
                stats = roi_metrics(score_map, mask.bool(), top_ratio=args.top_ratio)
                rows.append(
                    {
                        "case_id": mask_case_id,
                        "attribution_case_id": attr_case_id,
                        "split": str(payload["split"]),
                        "path": str(record["path"]),
                        "label": int(record["label"]),
                        "label_name": str(record["label_name"]),
                        "raw_view": str(record["raw_view"]),
                        "coarse_view": str(record["coarse_view"]),
                        "pred_label": int(record.get("pred_label", -1)),
                        "correct": int(record.get("correct", -1)),
                        "mask_source": str(mask_payload.get("source", "unknown")),
                        "generated_mask": int(generated_mask),
                        "map_type": map_type,
                        "region": str(region),
                        "map_entropy": entropy,
                        **stats,
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    ratio_df = compute_anatomy_context_ratios(rows)
    ratio_path = output_csv.with_name(output_csv.stem + "_anatomy_context_ratio.csv")
    ratio_df.to_csv(ratio_path, index=False)
    summary = {
        "n_rows": int(len(df)),
        "n_cases": int(df["case_id"].nunique()) if not df.empty else 0,
        "output_csv": str(output_csv),
        "anatomy_context_ratio_csv": str(ratio_path),
        "skipped": skipped,
        "metric_summary": summarize_metric_rows(rows),
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
