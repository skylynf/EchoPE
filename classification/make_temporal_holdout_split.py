#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


LABEL_TO_INT = {"normal": 0, "pe": 1}
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov")
TIME_PREFIX_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})")


@dataclass(frozen=True)
class SampleRecord:
    id: str
    label: str
    label_int: int
    parsed_view: str
    path: str
    txt_path: str
    strata: str
    parsed_time: str
    time_precision: str
    year_month: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single temporal holdout split for the preprocessed "
            "EchoPrime dataset. Samples with parsed_time >= holdout_start "
            "become the test split; earlier samples are stratified into "
            "train/val."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/local/scratch/luming/echoprime/dataset/preprocessed"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output directory that will contain manifest/summary and fold_0/*.csv",
    )
    parser.add_argument(
        "--holdout-start",
        default="2025-05",
        help="Inclusive holdout boundary in YYYY-MM format (default: 2025-05).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.10,
        help="Validation fraction within the pre-holdout pool (default: 0.10).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_key_values(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_year_month(parsed_time: str) -> str:
    match = TIME_PREFIX_RE.match(parsed_time.strip())
    if not match:
        raise ValueError(f"Unsupported parsed_time format: {parsed_time!r}")
    return f"{match.group('year')}-{match.group('month')}"


def resolve_video_path(txt_path: Path) -> Path:
    for suffix in VIDEO_EXTS:
        candidate = txt_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing paired video for metadata file: {txt_path}")


def collect_samples(dataset_root: Path) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    for label_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name in {"Normal", "PE"}):
        for view_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            for txt_path in sorted(view_dir.glob("*.txt")):
                fields = parse_key_values(txt_path)
                parsed_label = fields.get("label", "").strip().lower()
                parsed_view = fields.get("parsed_view", "").strip()
                parsed_time = fields.get("parsed_time", "").strip()
                time_precision = fields.get("time_precision", "").strip()
                sample_id = fields.get("id", txt_path.stem).strip()
                if parsed_label not in LABEL_TO_INT:
                    raise ValueError(f"Unexpected label in {txt_path}: {parsed_label!r}")
                if not parsed_view:
                    raise ValueError(f"Missing parsed_view in {txt_path}")
                if not parsed_time:
                    raise ValueError(f"Missing parsed_time in {txt_path}")

                video_path = resolve_video_path(txt_path)
                year_month = parse_year_month(parsed_time)
                samples.append(
                    SampleRecord(
                        id=sample_id,
                        label=parsed_label,
                        label_int=LABEL_TO_INT[parsed_label],
                        parsed_view=parsed_view,
                        path=str(video_path),
                        txt_path=str(txt_path),
                        strata=f"{parsed_label}_{parsed_view}",
                        parsed_time=parsed_time,
                        time_precision=time_precision or "unknown",
                        year_month=year_month,
                    )
                )
    if not samples:
        raise RuntimeError(f"No samples found under {dataset_root}")
    return samples


def can_stratify(labels: list[str], test_size: float) -> bool:
    counts = Counter(labels)
    if len(counts) <= 1:
        return False
    if min(counts.values()) < 2:
        return False
    n_samples = len(labels)
    n_test = math.ceil(n_samples * test_size)
    n_train = n_samples - n_test
    return len(counts) <= n_test and len(counts) <= n_train


def split_pre_holdout(
    records: list[SampleRecord],
    val_frac: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord], str]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if len(records) < 2:
        raise ValueError("Need at least two pre-holdout samples to make train/val.")

    indices = list(range(len(records)))
    strata = [record.strata for record in records]
    labels = [record.label for record in records]

    stratify_values: list[str] | None = None
    strategy = "random"
    if can_stratify(strata, val_frac):
        stratify_values = strata
        strategy = "strata=label+parsed_view"
    elif can_stratify(labels, val_frac):
        stratify_values = labels
        strategy = "strata=label_only"

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify_values,
    )
    train_records = [records[idx] for idx in sorted(train_idx)]
    val_records = [records[idx] for idx in sorted(val_idx)]
    return train_records, val_records, strategy


def write_split_csv(path: Path, records: list[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        for record in records:
            writer.writerow({"path": record.path, "label": record.label_int})


def summarize_records(records: list[SampleRecord]) -> dict[str, object]:
    label_counts = Counter(record.label for record in records)
    view_counts = Counter(record.parsed_view for record in records)
    ym_counts = Counter(record.year_month for record in records)
    strata_counts = Counter(record.strata for record in records)
    return {
        "n": len(records),
        "positive_rate": float(sum(record.label_int for record in records) / len(records)) if records else 0.0,
        "label_counts": dict(sorted(label_counts.items())),
        "parsed_view_counts": dict(sorted(view_counts.items())),
        "year_month_counts": dict(sorted(ym_counts.items())),
        "strata_counts": dict(sorted(strata_counts.items())),
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Manifest rows cannot be empty.")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, *, dataset_root: Path, holdout_start: str, val_frac: float, seed: int) -> None:
    content = (
        "# Temporal Holdout Split\n\n"
        f"- Source data: `{dataset_root}`\n"
        f"- Holdout rule: samples with `parsed_time >= {holdout_start}` go to `test`\n"
        f"- Validation fraction within pre-holdout pool: `{val_frac}`\n"
        f"- Seed: `{seed}`\n"
        "- Training CSV format: `path,label` with `normal=0`, `pe=1`\n"
        "- Compatible with `train_cv_binary.py` via `--fold 0`\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_root = args.out_root.resolve()

    samples = collect_samples(dataset_root)
    test_records = [record for record in samples if record.year_month >= args.holdout_start]
    pre_holdout_records = [record for record in samples if record.year_month < args.holdout_start]

    if not test_records:
        raise RuntimeError(f"No holdout samples found with parsed_time >= {args.holdout_start}")
    if not pre_holdout_records:
        raise RuntimeError(f"No pre-holdout samples found with parsed_time < {args.holdout_start}")

    train_records, val_records, val_strategy = split_pre_holdout(
        pre_holdout_records,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    fold_dir = out_root / "fold_0"
    write_split_csv(fold_dir / "train.csv", train_records)
    write_split_csv(fold_dir / "val.csv", val_records)
    write_split_csv(fold_dir / "test.csv", test_records)

    assignment_rows = []
    split_lookup = {
        "train": {record.id for record in train_records},
        "val": {record.id for record in val_records},
        "test": {record.id for record in test_records},
    }
    for record in sorted(samples, key=lambda item: (item.year_month, item.label, item.parsed_view, item.id)):
        split_name = "train"
        if record.id in split_lookup["val"]:
            split_name = "val"
        elif record.id in split_lookup["test"]:
            split_name = "test"
        row = asdict(record)
        row["split"] = split_name
        assignment_rows.append(row)

    write_manifest(out_root / "manifest.csv", assignment_rows)

    summary = {
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "holdout_start": args.holdout_start,
        "holdout_rule": f"parsed_time >= {args.holdout_start}",
        "seed": int(args.seed),
        "val_frac_within_pre_holdout": float(args.val_frac),
        "val_split_strategy": val_strategy,
        "summary": {
            "all": summarize_records(samples),
            "train": summarize_records(train_records),
            "val": summarize_records(val_records),
            "test": summarize_records(test_records),
        },
        "artifacts": {
            "manifest_csv": str((out_root / "manifest.csv").resolve()),
            "train_csv": str((fold_dir / "train.csv").resolve()),
            "val_csv": str((fold_dir / "val.csv").resolve()),
            "test_csv": str((fold_dir / "test.csv").resolve()),
        },
    }
    (out_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(
        out_root / "README.md",
        dataset_root=dataset_root,
        holdout_start=args.holdout_start,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print(f"[done] temporal holdout split written to {out_root}")
    print(
        "[counts] "
        f"train={len(train_records)} "
        f"val={len(val_records)} "
        f"test={len(test_records)} "
        f"(holdout_start={args.holdout_start}, val_strategy={val_strategy})"
    )


if __name__ == "__main__":
    main()
