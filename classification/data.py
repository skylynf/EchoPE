from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split

from config import DEFAULT_DATASET_ROOT, DEFAULT_TEST_FRAC, DEFAULT_VAL_FRAC_TOTAL


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
LABEL_NAME_TO_INT = {"Normal": 0, "PE": 1}
LABEL_INT_TO_NAME = {v: k for k, v in LABEL_NAME_TO_INT.items()}

RAW_TO_COARSE_VIEW = {
    "A4": "A4",
    "MS": "A4",
    "PSL": "PSL",
    "PSS": "PSS",
    "IVC": "Subcostal",
    "SX": "Subcostal",
}
COARSE_VIEWS = ["A4", "PSL", "PSS", "Subcostal"]
COARSE_VIEW_TO_IDX = {name: idx for idx, name in enumerate(COARSE_VIEWS)}
TASKS = ["pooled", *COARSE_VIEWS]


@dataclass(frozen=True)
class SampleRecord:
    id: str
    path: str
    label: int
    label_name: str
    source_group: str
    raw_view: str
    coarse_view: str
    strata: str


def resolve_user_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else candidate.resolve()


def collect_samples(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> list[SampleRecord]:
    root = resolve_user_path(dataset_root)
    samples: list[SampleRecord] = []
    for group_name, label in LABEL_NAME_TO_INT.items():
        group_dir = root / group_name
        if not group_dir.is_dir():
            continue
        for raw_view_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            raw_view = raw_view_dir.name
            coarse_view = RAW_TO_COARSE_VIEW.get(raw_view)
            if coarse_view is None:
                continue
            for path in sorted(raw_view_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                    continue
                samples.append(
                    SampleRecord(
                        id=path.stem,
                        path=str(path),
                        label=label,
                        label_name=group_name,
                        source_group=group_name,
                        raw_view=raw_view,
                        coarse_view=coarse_view,
                        # Split at the pooled-task granularity requested in plan.
                        strata=f"{group_name}_{coarse_view}",
                    )
                )
    if not samples:
        raise RuntimeError(f"No videos found under {root}")
    return samples


def summarize_samples(samples: Iterable[SampleRecord]) -> dict[str, object]:
    items = list(samples)
    label_counts = Counter(item.label_name for item in items)
    coarse_counts = Counter(item.coarse_view for item in items)
    raw_counts = Counter((item.coarse_view, item.raw_view, item.label_name) for item in items)
    return {
        "n_total": len(items),
        "label_counts": dict(sorted(label_counts.items())),
        "coarse_view_counts": dict(sorted(coarse_counts.items())),
        "raw_view_label_counts": {
            f"{coarse}/{raw}/{label}": count
            for (coarse, raw, label), count in sorted(raw_counts.items())
        },
    }


def split_indices(
    samples: list[SampleRecord],
    test_frac: float = DEFAULT_TEST_FRAC,
    val_frac_total: float = DEFAULT_VAL_FRAC_TOTAL,
    seed: int = 2024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(samples))
    strata = np.array([sample.strata for sample in samples], dtype=object)
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_frac,
        stratify=strata,
        random_state=seed,
        shuffle=True,
    )
    val_inner = val_frac_total / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_inner,
        stratify=strata[train_val_idx],
        random_state=seed + 1,
        shuffle=True,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _serialize_split(
    samples: list[SampleRecord],
    indices: np.ndarray,
    split_name: str,
) -> dict[str, object]:
    split_samples = [samples[int(i)] for i in indices]
    return {
        "name": split_name,
        "n": len(split_samples),
        "label_counts": dict(sorted(Counter(item.label_name for item in split_samples).items())),
        "coarse_view_counts": dict(sorted(Counter(item.coarse_view for item in split_samples).items())),
        "records": [asdict(item) for item in split_samples],
    }


def save_split_manifest(
    path: str | Path,
    samples: list[SampleRecord],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset_root: str | Path,
    seed: int,
    test_frac: float,
    val_frac_total: float,
) -> Path:
    out_path = resolve_user_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_root": str(resolve_user_path(dataset_root)),
        "seed": int(seed),
        "test_frac": float(test_frac),
        "val_frac_total": float(val_frac_total),
        "tasks": TASKS,
        "summary": summarize_samples(samples),
        "splits": {
            "train": _serialize_split(samples, train_idx, "train"),
            "val": _serialize_split(samples, val_idx, "val"),
            "test": _serialize_split(samples, test_idx, "test"),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_split_manifest(path: str | Path) -> dict[str, object]:
    in_path = resolve_user_path(path)
    return json.loads(in_path.read_text(encoding="utf-8"))


def manifest_records(payload: dict[str, object], split_name: str, task: str = "pooled") -> list[dict[str, object]]:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Choices: {TASKS}")
    records = list(payload["splits"][split_name]["records"])
    if task == "pooled":
        return records
    return [record for record in records if record["coarse_view"] == task]


def records_to_samples(
    records: list[dict[str, object]],
    *,
    include_coarse_view_idx: bool = False,
) -> list[tuple[str, int, str] | tuple[str, int, str, int]]:
    # Keep raw_view so the existing PE training utilities remain compatible.
    if not include_coarse_view_idx:
        return [(str(r["path"]), int(r["label"]), str(r["raw_view"])) for r in records]
    return [
        (
            str(r["path"]),
            int(r["label"]),
            str(r["raw_view"]),
            int(COARSE_VIEW_TO_IDX[str(r["coarse_view"])]),
        )
        for r in records
    ]


def make_or_load_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    test_frac: float = DEFAULT_TEST_FRAC,
    val_frac_total: float = DEFAULT_VAL_FRAC_TOTAL,
    seed: int = 2024,
    force_rebuild: bool = False,
) -> Path:
    out_path = resolve_user_path(manifest_path)
    if out_path.is_file() and not force_rebuild:
        return out_path
    samples = collect_samples(dataset_root)
    train_idx, val_idx, test_idx = split_indices(samples, test_frac=test_frac, val_frac_total=val_frac_total, seed=seed)
    return save_split_manifest(
        out_path,
        samples,
        train_idx,
        val_idx,
        test_idx,
        dataset_root=dataset_root,
        seed=seed,
        test_frac=test_frac,
        val_frac_total=val_frac_total,
    )

