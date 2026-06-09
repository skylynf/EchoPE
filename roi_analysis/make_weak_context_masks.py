#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from roi_utils import (
    case_rows_from_attributions,
    generate_context_masks,
    load_preprocessed_video,
    resolve_path,
    save_mask_payload,
)


def load_case_panel(interpret_dir: Path, cases_csv: Path | None) -> pd.DataFrame:
    if cases_csv is not None:
        df = pd.read_csv(resolve_path(cases_csv))
        if "case_id" not in df.columns:
            df.insert(0, "case_id", [f"roi_{idx:04d}" for idx in range(len(df))])
        return df
    return case_rows_from_attributions(interpret_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate weak foreground/background/border/probe masks for ROI analysis.")
    parser.add_argument("--interpret-dir", type=Path, required=True, help="An existing interpretability output directory.")
    parser.add_argument("--cases-csv", type=Path, default=None, help="Optional selected case panel CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--border-width", type=int, default=5)
    parser.add_argument("--probe-fraction", type=float, default=0.18)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    interpret_dir = resolve_path(args.interpret_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_case_panel(interpret_dir, args.cases_csv)
    rows: list[dict[str, object]] = []
    for _, row in tqdm(cases.iterrows(), total=len(cases), desc="weak masks"):
        video = load_preprocessed_video(str(row["path"]))
        masks = generate_context_masks(
            video,
            threshold=args.threshold,
            border_width=args.border_width,
            probe_fraction=args.probe_fraction,
        )
        case = row.to_dict()
        save_mask_payload(output_dir / f"{row['case_id']}.pt", case_row=case, masks=masks, source="weak_context")
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "path": str(row["path"]),
                "foreground_area": float(masks["foreground"].float().mean().item()),
                "background_area": float(masks["background"].float().mean().item()),
                "sector_border_area": float(masks["sector_border"].float().mean().item()),
                "probe_near_field_area": float(masks["probe_near_field"].float().mean().item()),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "mask_manifest.csv", index=False)
    summary = {
        "n_cases": int(len(rows)),
        "output_dir": str(output_dir),
        "mask_manifest": str(output_dir / "mask_manifest.csv"),
        "threshold": float(args.threshold),
        "border_width": int(args.border_width),
        "probe_fraction": float(args.probe_fraction),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
