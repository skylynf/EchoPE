#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from roi_utils import load_predictions, resolve_path


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as name=/path/to/interpretable_output")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Run name cannot be empty")
    return name, resolve_path(path)


def load_joined_runs(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    joined: pd.DataFrame | None = None
    keys = ["split", "path", "label", "label_name", "raw_view", "coarse_view"]
    for name, out_dir in runs:
        df = load_predictions(out_dir)
        keep = keys + ["prob_pe", "pred_label", "correct"]
        missing = [col for col in keep if col not in df.columns]
        if missing:
            raise ValueError(f"{out_dir} is missing columns: {missing}")
        df = df[keep].rename(
            columns={
                "prob_pe": f"prob_pe_{name}",
                "pred_label": f"pred_label_{name}",
                "correct": f"correct_{name}",
            }
        )
        joined = df if joined is None else joined.merge(df, on=keys, how="inner")
    if joined is None:
        raise ValueError("At least one --run is required")
    return joined


def stratified_select(
    rows: pd.DataFrame,
    *,
    split: str,
    views: list[str] | None,
    per_group: int,
    max_cases: int,
    seed: int,
    reference_run: str,
) -> pd.DataFrame:
    df = rows.loc[rows["split"].astype(str).eq(split)].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split!r}. Available splits: {sorted(rows['split'].astype(str).unique())}")
    if views:
        keep = {view.upper() for view in views}
        df = df.loc[df["coarse_view"].astype(str).str.upper().isin(keep)].copy()
        if df.empty:
            raise ValueError(f"No rows left after filtering views={sorted(keep)}")
    correct_col = f"correct_{reference_run}"
    if correct_col not in df.columns:
        correct_col = next((col for col in df.columns if col.startswith("correct_")), "")
    group_cols = ["coarse_view", "label_name"]
    if correct_col:
        group_cols.append(correct_col)
    sampled_groups = []
    for _group_key, group in df.groupby(group_cols, dropna=False, sort=True):
        sampled_groups.append(group.sample(n=min(per_group, len(group)), random_state=seed))
    if not sampled_groups:
        raise ValueError("No groups were available for ROI case selection.")
    sampled = pd.concat(sampled_groups, ignore_index=True)
    if max_cases > 0 and len(sampled) > max_cases:
        sampled = sampled.sample(n=max_cases, random_state=seed).reset_index(drop=True)
    sampled = sampled.sort_values(["coarse_view", "label_name", "path"]).reset_index(drop=True)
    sampled.insert(0, "case_id", [f"{split}_roi_{idx:04d}" for idx in range(len(sampled))])
    return sampled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a shared ROI case panel from one or more interpretability outputs.")
    parser.add_argument("--run", action="append", required=True, type=parse_run, help="name=/path/to/interpretability/output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--view", action="append", default=None, help="Keep a coarse view, e.g. A4 or PSS. Can repeat.")
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--reference-run", default="", help="Run name whose correctness is used for stratification.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = list(args.run)
    reference = args.reference_run or runs[0][0]
    joined = load_joined_runs(runs)
    selected = stratified_select(
        joined,
        split=args.split,
        views=args.view,
        per_group=args.per_group,
        max_cases=args.max_cases,
        seed=args.seed,
        reference_run=reference,
    )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    summary = {
        "runs": {name: str(path) for name, path in runs},
        "reference_run": reference,
        "n_joined": int(len(joined)),
        "n_selected": int(len(selected)),
        "output": str(output),
        "group_counts": {
            f"{coarse_view}/{label_name}": int(count)
            for (coarse_view, label_name), count in selected.groupby(["coarse_view", "label_name"], dropna=False).size().items()
        },
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
