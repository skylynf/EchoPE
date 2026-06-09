#!/usr/bin/env python3
"""Generate deterministic 5-fold CV splits for the HOPE normal-vs-PE benchmark.

Both `data/hope_clean_polar/` (nested as `{class}/{view}/{ID}.mp4`) and
`data/hope_clean_cartesian/` (currently flat as `{ID}.mp4`, but auto-detected
fall back to nested) share the **same** numeric IDs. The split files are
generated for both modalities so that polar vs cartesian experiments are
evaluated on identical patient/sample partitions across all folds.

Layout produced (relative to repo root):

    data/splits/normal_vs_pe_cv/
        manifest.csv                              # one row per ID with metadata
        cartesian/
            fold_0/{train.csv,val.csv,test.csv}
            ...
            fold_4/{train.csv,val.csv,test.csv}
        polar/
            fold_0/{train.csv,val.csv,test.csv}
            ...
            fold_4/{train.csv,val.csv,test.csv}

CSV format (matches the existing repo convention):

    path,label
    data/hope_clean_polar/PE/A4/1909.mp4,1
    ...

Stratification:
    Outer K-fold is stratified on the joint label = "{class}_{view}", which
    keeps PE/A4, PE/PSS, Normal/A4, Normal/PSS proportions stable across
    folds. Within each fold's train portion, the validation set is carved
    with the same stratification key.

Determinism:
    A single integer `--seed` controls everything (default 42). Re-running
    the script with the same seed reproduces the exact same splits.

Usage:
    python script/make_cv_splits.py
    python script/make_cv_splits.py --n_splits 5 --val_frac 0.125 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_manifest(polar_root: Path, cartesian_root: Path) -> pd.DataFrame:
    """Scan `polar_root` for nested `{class}/{view}/{id}.mp4` files and pair
    each with a cartesian counterpart (auto-detected: nested first, then flat).

    Returns a DataFrame with columns:
        id, label_str, view, label, polar_path, cartesian_path, strata
    """
    if not polar_root.is_dir():
        raise FileNotFoundError(f"Polar root not found: {polar_root}")
    if not cartesian_root.is_dir():
        raise FileNotFoundError(f"Cartesian root not found: {cartesian_root}")

    label_str_to_int = {"Normal": 0, "PE": 1}

    rows: List[dict] = []
    missing_cartesian: List[str] = []

    for mp4 in sorted(polar_root.rglob("*.mp4")):
        rel = mp4.relative_to(polar_root)
        parts = rel.parts
        if len(parts) != 3:
            print(
                f"[warn] skip unexpected polar path layout: {rel}", file=sys.stderr
            )
            continue
        label_str, view, fname = parts
        if label_str not in label_str_to_int:
            print(f"[warn] skip unknown class folder: {label_str}", file=sys.stderr)
            continue
        sample_id = Path(fname).stem

        nested_cart = cartesian_root / label_str / view / fname
        flat_cart = cartesian_root / fname
        if nested_cart.is_file():
            cart_path = nested_cart
        elif flat_cart.is_file():
            cart_path = flat_cart
        else:
            missing_cartesian.append(sample_id)
            continue

        rows.append(
            {
                "id": sample_id,
                "label_str": label_str,
                "view": view,
                "label": label_str_to_int[label_str],
                "polar_path": str(mp4.relative_to(REPO_ROOT)),
                "cartesian_path": str(cart_path.relative_to(REPO_ROOT)),
                "strata": f"{label_str}_{view}",
            }
        )

    if missing_cartesian:
        print(
            f"[warn] {len(missing_cartesian)} samples missing in cartesian root, "
            f"first few: {missing_cartesian[:5]}",
            file=sys.stderr,
        )

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Manifest is empty - no samples found.")
    return df


def write_split_csv(
    out_path: Path, df: pd.DataFrame, path_column: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[[path_column, "label"]].rename(
        columns={path_column: "path"}
    ).to_csv(out_path, index=False)


def make_fold_splits(
    manifest: pd.DataFrame,
    n_splits: int,
    val_frac: float,
    seed: int,
) -> Iterable[Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold_idx, train_df, val_df, test_df) tuples.

    Outer split: StratifiedKFold(n_splits) on `strata` => test fold.
    Inner split: stratified train/val on the remaining (1 - 1/K) portion.
    `val_frac` is the **fraction of the entire dataset** that goes to val,
    so train fraction = 1 - 1/K - val_frac. Defaults give 70/10/20.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strata = manifest["strata"].to_numpy()

    # `train_test_split` `test_size` is relative to the input array, not the
    # full dataset. Convert: val_size_within_remaining = val_frac / (1 - 1/K).
    test_share = 1.0 / n_splits
    if val_frac >= 1.0 - test_share:
        raise ValueError(
            f"val_frac={val_frac} too large for n_splits={n_splits} "
            f"(must be < {1.0 - test_share:.4f})"
        )
    val_size_inner = val_frac / (1.0 - test_share)

    for fold_idx, (train_val_idx, test_idx) in enumerate(
        skf.split(manifest, strata)
    ):
        train_val_df = manifest.iloc[train_val_idx]
        test_df = manifest.iloc[test_idx].reset_index(drop=True)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_inner,
            stratify=train_val_df["strata"],
            random_state=seed + fold_idx,  # different inner seed per fold
            shuffle=True,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        yield fold_idx, train_df, val_df, test_df


def summarize(name: str, df: pd.DataFrame) -> str:
    counts = df.groupby(["label_str", "view"]).size().to_dict()
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return f"{name:>5}: n={len(df):4d}  pos={int(df['label'].sum()):4d}  " \
           f"neg={int((df['label'] == 0).sum()):4d}  [{counts_str}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--polar_root",
        type=Path,
        default=REPO_ROOT / "data" / "hope_clean_polar",
    )
    parser.add_argument(
        "--cartesian_root",
        type=Path,
        default=REPO_ROOT / "data" / "hope_clean_cartesian",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "normal_vs_pe_cv",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.125,
        help="Fraction of TOTAL dataset reserved for validation per fold "
             "(default 0.125 -> 70/10/20 with n_splits=5).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[info] polar_root     = {args.polar_root}")
    print(f"[info] cartesian_root = {args.cartesian_root}")
    print(f"[info] out_root       = {args.out_root}")
    print(f"[info] n_splits={args.n_splits}  val_frac={args.val_frac}  seed={args.seed}")

    manifest = build_manifest(args.polar_root, args.cartesian_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[info] manifest written: {manifest_path} ({len(manifest)} rows)")
    print("[info] global class/view distribution:")
    print(summarize(" all ", manifest))

    for fold_idx, train_df, val_df, test_df in make_fold_splits(
        manifest, args.n_splits, args.val_frac, args.seed
    ):
        # Sanity: ensure no overlap between splits
        ids = {
            "train": set(train_df["id"]),
            "val": set(val_df["id"]),
            "test": set(test_df["id"]),
        }
        assert ids["train"].isdisjoint(ids["val"]), \
            f"fold {fold_idx}: train/val overlap"
        assert ids["train"].isdisjoint(ids["test"]), \
            f"fold {fold_idx}: train/test overlap"
        assert ids["val"].isdisjoint(ids["test"]), \
            f"fold {fold_idx}: val/test overlap"

        print(f"\n[fold {fold_idx}]")
        print(summarize("train", train_df))
        print(summarize("  val", val_df))
        print(summarize(" test", test_df))

        for modality, path_col in (
            ("cartesian", "cartesian_path"),
            ("polar", "polar_path"),
        ):
            fold_dir = args.out_root / modality / f"fold_{fold_idx}"
            for split_name, split_df in (
                ("train", train_df),
                ("val", val_df),
                ("test", test_df),
            ):
                write_split_csv(fold_dir / f"{split_name}.csv", split_df, path_col)
            print(f"[info]   wrote {modality}/fold_{fold_idx}/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
