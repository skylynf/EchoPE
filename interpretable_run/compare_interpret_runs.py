#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from load_interpret_model import resolve_cli_path  # noqa: E402


def maybe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def load_predictions(output_dir: Path, run_name: str) -> pd.DataFrame:
    path = output_dir / "embedding_analysis" / "sample_predictions.csv"
    df = pd.read_csv(path)
    rename = {
        "prob_pe": f"prob_pe_{run_name}",
        "pred_label": f"pred_label_{run_name}",
        "correct": f"correct_{run_name}",
    }
    keep = [
        "split",
        "path",
        "label",
        "label_name",
        "raw_view",
        "coarse_view",
        "prob_pe",
        "pred_label",
        "correct",
    ]
    return df[keep].rename(columns=rename)


def compute_per_view_delta(joined: pd.DataFrame, baseline_name: str, candidate_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, coarse_view), group in joined.groupby(["split", "coarse_view"], dropna=False):
        labels = group["label"].to_numpy(dtype=int)
        base_prob = group[f"prob_pe_{baseline_name}"].to_numpy(dtype=float)
        cand_prob = group[f"prob_pe_{candidate_name}"].to_numpy(dtype=float)
        base_pred = group[f"pred_label_{baseline_name}"].to_numpy(dtype=int)
        cand_pred = group[f"pred_label_{candidate_name}"].to_numpy(dtype=int)
        rows.append(
            {
                "split": split,
                "coarse_view": coarse_view,
                "n": int(len(group)),
                f"accuracy_{baseline_name}": float(accuracy_score(labels, base_pred)),
                f"accuracy_{candidate_name}": float(accuracy_score(labels, cand_pred)),
                "delta_accuracy": float(accuracy_score(labels, cand_pred) - accuracy_score(labels, base_pred)),
                f"auc_{baseline_name}": maybe_auc(labels, base_prob),
                f"auc_{candidate_name}": maybe_auc(labels, cand_prob),
                "mean_delta_prob": float(np.mean(cand_prob - base_prob)),
            }
        )
    return pd.DataFrame(rows)


def top_indices(curve: np.ndarray, k: int = 3) -> set[int]:
    if curve.size == 0:
        return set()
    values = np.clip(curve.astype(float), a_min=0.0, a_max=None)
    if values.size == 0 or float(values.max()) <= 0.0:
        values = np.abs(curve.astype(float))
    k = min(k, values.size)
    return set(int(idx) for idx in np.argsort(-values)[:k])


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return float("nan")
    return float(len(left & right) / max(1, len(left | right)))


def load_case_metrics(output_dir: Path, run_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "attributions").glob("*.pt")):
        payload = torch.load(str(path), map_location="cpu")
        record = payload["record"]
        curve_tensor = payload.get("temporal_occlusion_raw_prob", payload.get("temporal_occlusion"))
        curve = curve_tensor.float().cpu().numpy() if curve_tensor is not None else np.empty(0)
        rows.append(
            {
                "split": str(payload["split"]),
                "path": str(record["path"]),
                f"case_id_{run_name}": str(payload["case_id"]),
                f"top_frames_{run_name}": sorted(top_indices(curve, k=3)),
                f"temporal_curve_{run_name}": curve.tolist(),
                **{f"{key}_{run_name}": float(value) for key, value in payload.get("metrics", {}).items()},
            }
        )
    return pd.DataFrame(rows)


def compare_temporal_cases(baseline_dir: Path, candidate_dir: Path, baseline_name: str, candidate_name: str) -> pd.DataFrame:
    baseline = load_case_metrics(baseline_dir, baseline_name)
    candidate = load_case_metrics(candidate_dir, candidate_name)
    if baseline.empty or candidate.empty:
        return pd.DataFrame()
    joined = baseline.merge(candidate, on=["split", "path"], how="inner")
    rows: list[dict[str, object]] = []
    for _, row in joined.iterrows():
        base_curve = np.asarray(row[f"temporal_curve_{baseline_name}"], dtype=float)
        cand_curve = np.asarray(row[f"temporal_curve_{candidate_name}"], dtype=float)
        n = min(base_curve.size, cand_curve.size)
        base_top = set(row[f"top_frames_{baseline_name}"])
        cand_top = set(row[f"top_frames_{candidate_name}"])
        out = {
            "split": row["split"],
            "path": row["path"],
            "top_frame_jaccard": jaccard(base_top, cand_top),
            "temporal_rank_corr": rank_correlation(base_curve[:n], cand_curve[:n]) if n else float("nan"),
        }
        for metric in (
            "temporal_entropy_occlusion",
            "temporal_occlusion_aopc_prob",
            "spatial_entropy_grad",
            "spatial_entropy_attention",
        ):
            base_key = f"{metric}_{baseline_name}"
            cand_key = f"{metric}_{candidate_name}"
            if base_key in row and cand_key in row:
                out[f"delta_{metric}"] = float(row[cand_key] - row[base_key])
        rows.append(out)
    return pd.DataFrame(rows)


def plot_probability_scatter(joined: pd.DataFrame, baseline_name: str, candidate_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = joined["label"].map({0: "tab:blue", 1: "tab:orange"}).fillna("gray")
    ax.scatter(joined[f"prob_pe_{baseline_name}"], joined[f"prob_pe_{candidate_name}"], s=16, alpha=0.7, c=colors)
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8)
    ax.set_xlabel(f"{baseline_name} prob PE")
    ax.set_ylabel(f"{candidate_name} prob PE")
    ax.set_title("Prediction Probability Agreement")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    baseline_dir = resolve_cli_path(args.baseline_dir)
    candidate_dir = resolve_cli_path(args.candidate_dir)
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else candidate_dir / "compare_to_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_predictions(baseline_dir, args.baseline_name)
    candidate = load_predictions(candidate_dir, args.candidate_name)
    joined = baseline.merge(
        candidate,
        on=["split", "path", "label", "label_name", "raw_view", "coarse_view"],
        how="inner",
    )
    joined["prob_delta_candidate_minus_baseline"] = joined[f"prob_pe_{args.candidate_name}"] - joined[f"prob_pe_{args.baseline_name}"]
    joined["prediction_agreement"] = (
        joined[f"pred_label_{args.baseline_name}"] == joined[f"pred_label_{args.candidate_name}"]
    ).astype(int)
    joined.to_csv(output_dir / "sample_comparison.csv", index=False)

    per_view = compute_per_view_delta(joined, args.baseline_name, args.candidate_name)
    per_view.to_csv(output_dir / "per_view_delta.csv", index=False)

    temporal = compare_temporal_cases(baseline_dir, candidate_dir, args.baseline_name, args.candidate_name)
    if not temporal.empty:
        temporal.to_csv(output_dir / "temporal_case_comparison.csv", index=False)

    plot_probability_scatter(joined, args.baseline_name, args.candidate_name, output_dir / "figures" / "probability_scatter.png")

    summary = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "output_dir": str(output_dir),
        "n_joined_samples": int(len(joined)),
        "prediction_agreement": float(joined["prediction_agreement"].mean()) if len(joined) else float("nan"),
        "mean_abs_prob_delta": float(joined["prob_delta_candidate_minus_baseline"].abs().mean()) if len(joined) else float("nan"),
        "n_joined_temporal_cases": int(len(temporal)),
        "artifacts": {
            "sample_comparison": str((output_dir / "sample_comparison.csv").resolve()),
            "per_view_delta": str((output_dir / "per_view_delta.csv").resolve()),
            "temporal_case_comparison": str((output_dir / "temporal_case_comparison.csv").resolve()) if not temporal.empty else "",
            "probability_scatter": str((output_dir / "figures" / "probability_scatter.png").resolve()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two interpretable_run output directories sample-by-sample.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-name", type=str, default="full_finetune")
    parser.add_argument("--candidate-name", type=str, default="frozen_head")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_comparison(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
