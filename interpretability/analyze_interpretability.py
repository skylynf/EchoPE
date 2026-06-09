#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from load_interpret_model import default_interpret_output_dir, resolve_cli_path  # noqa: E402


def maybe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.unique(y_true)
    if labels.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def plot_embedding_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, group in df.groupby(color_col):
        ax.scatter(group[x_col], group[y_col], s=16, alpha=0.7, label=str(key))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_by_view_and_label(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = sorted(df["label_name"].dropna().unique().tolist())
    views = sorted(df["coarse_view"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    for label_idx, label in enumerate(labels):
        for view_idx, view in enumerate(views):
            group = df[(df["label_name"] == label) & (df["coarse_view"] == view)]
            if group.empty:
                continue
            ax.scatter(
                group[x_col],
                group[y_col],
                s=22,
                alpha=0.75,
                color=cmap(label_idx % 10),
                marker=markers[view_idx % len(markers)],
                label=f"{label} | {view}",
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca(df: pd.DataFrame, color_col: str, output_path: Path, title: str) -> None:
    plot_embedding_scatter(df, "pc1", "pc2", color_col, output_path, title, "PC1", "PC2")


def plot_pca_by_view_and_label(df: pd.DataFrame, output_path: Path) -> None:
    plot_embedding_by_view_and_label(
        df,
        "pc1",
        "pc2",
        output_path,
        "Embedding PCA by Label and Coarse View",
        "PC1",
        "PC2",
    )


def compute_umap_tsne_coords(
    embeddings: torch.Tensor,
    *,
    pre_pca_components: int = 50,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """50D PCA (after scaling) → 2D UMAP and 2D t-SNE, matching prior frozen_head artifacts."""
    emb_np = embeddings.detach().cpu().numpy().astype(np.float64)
    n = emb_np.shape[0]
    if n < 5:
        raise ValueError("too few samples for UMAP/t-SNE")

    scaled = StandardScaler().fit_transform(emb_np)
    n_pca = min(int(pre_pca_components), scaled.shape[1], max(1, n - 1))
    basis = PCA(n_components=n_pca, random_state=random_state).fit_transform(scaled)

    meta: dict[str, object] = {"pre_pca_components": int(n_pca)}
    import umap  # type: ignore

    nn = min(30, max(2, n - 1))
    umapper = umap.UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    umap_xy = umapper.fit_transform(basis)
    meta["umap"] = {"n_neighbors": int(nn), "min_dist": 0.1, "metric": "euclidean", "random_state": random_state}

    perplexity = float(min(30.0, max(5.0, (n - 1) / 3.0)))
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": random_state,
    }
    try:
        tsne_model = TSNE(max_iter=1000, **tsne_kwargs)
    except TypeError:
        tsne_model = TSNE(n_iter=1000, **tsne_kwargs)
    tsne_xy = tsne_model.fit_transform(basis)
    meta["tsne"] = {
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": random_state,
        "max_iter": 1000,
    }
    return umap_xy, tsne_xy, meta


def load_embedding_payload(output_dir: Path) -> tuple[pd.DataFrame, torch.Tensor]:
    payload = torch.load(str(output_dir / "embedding_analysis" / "embeddings.pt"), map_location="cpu")
    rows = pd.DataFrame(payload["rows"])
    embeddings = payload["embeddings"].float()
    if len(rows) != int(embeddings.shape[0]):
        raise RuntimeError("Embedding payload row count does not match tensor count.")
    return rows, embeddings


def load_detailed_cases(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "attributions").glob("*.pt")):
        payload = torch.load(str(path), map_location="cpu")
        row = {
            "case_id": payload["case_id"],
            "split": payload["split"],
            "path": str(payload["record"]["path"]),
            "label": int(payload["record"]["label"]),
            "label_name": str(payload["record"]["label_name"]),
            "pred_label": int(payload["record"].get("pred_label", -1)),
            "correct": int(payload["record"].get("correct", -1)),
            "raw_view": str(payload["record"]["raw_view"]),
            "coarse_view": str(payload["record"]["coarse_view"]),
            **{key: float(value) for key, value in payload["metrics"].items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_temporal_metrics(detailed_df: pd.DataFrame) -> pd.DataFrame:
    if detailed_df.empty:
        return pd.DataFrame()
    group_cols = [col for col in ("split", "coarse_view", "label_name", "correct") if col in detailed_df.columns]
    metric_cols = [
        col
        for col in detailed_df.columns
        if col.startswith("temporal_")
        or col.startswith("spatial_entropy")
        or col in {"attention_top10_mass", "grad_top10_mass", "foreground_mass_grad", "foreground_mass_attention"}
    ]
    if not group_cols or not metric_cols:
        return pd.DataFrame()
    return detailed_df.groupby(group_cols, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()


def compute_performance_by_view(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for (split, coarse_view), group in rows.groupby(["split", "coarse_view"], dropna=False):
        labels = group["label"].to_numpy(dtype=int)
        probs = group["prob_pe"].to_numpy(dtype=float)
        preds = group["pred_label"].to_numpy(dtype=int)
        metrics.append(
            {
                "split": split,
                "coarse_view": coarse_view,
                "n": int(len(group)),
                "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
                "accuracy": float(accuracy_score(labels, preds)) if len(group) else float("nan"),
                "auc": maybe_auc(labels, probs),
                "mean_prob_pe": float(np.mean(probs)) if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(metrics)


def compute_embedding_projection(rows: pd.DataFrame, embeddings: torch.Tensor) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(embeddings.numpy())
    out = rows.copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    out["pca_explained_var_1"] = float(pca.explained_variance_ratio_[0])
    out["pca_explained_var_2"] = float(pca.explained_variance_ratio_[1])
    return out


def fit_view_probe(rows: pd.DataFrame, embeddings: torch.Tensor) -> dict[str, object]:
    train_mask = rows["split"] == "train"
    test_mask = rows["split"] != "train"
    if int(train_mask.sum()) < 4 or int(test_mask.sum()) < 2:
        return {"status": "skipped", "reason": "not enough train/test samples"}
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(rows.loc[train_mask, "coarse_view"])
    y_test = encoder.transform(rows.loc[test_mask, "coarse_view"])
    if int(np.unique(y_train).size) < 2:
        return {"status": "skipped", "reason": "train split has only one coarse_view"}
    clf = LogisticRegression(max_iter=2000)
    train_index = train_mask.to_numpy(dtype=bool, copy=True)
    test_index = test_mask.to_numpy(dtype=bool, copy=True)
    X_train = embeddings[train_index].detach().cpu().numpy().copy()
    X_test = embeddings[test_index].detach().cpu().numpy().copy()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "status": "ok",
        "n_classes": int(len(encoder.classes_)),
        "classes": encoder.classes_.tolist(),
        "test_accuracy": float(accuracy_score(y_test, pred)),
    }


def fit_label_probe(rows: pd.DataFrame, embeddings: torch.Tensor) -> dict[str, object]:
    train_mask = rows["split"] == "train"
    test_mask = rows["split"] != "train"
    if int(train_mask.sum()) < 4 or int(test_mask.sum()) < 2:
        return {"status": "skipped", "reason": "not enough train/test samples"}
    y_train = rows.loc[train_mask, "label"].to_numpy(dtype=int)
    y_test = rows.loc[test_mask, "label"].to_numpy(dtype=int)
    if int(np.unique(y_train).size) < 2 or int(np.unique(y_test).size) < 2:
        return {"status": "skipped", "reason": "train or test split has one label"}
    clf = LogisticRegression(max_iter=2000)
    train_index = train_mask.to_numpy(dtype=bool, copy=True)
    test_index = test_mask.to_numpy(dtype=bool, copy=True)
    X_train = embeddings[train_index].detach().cpu().numpy().copy()
    X_test = embeddings[test_index].detach().cpu().numpy().copy()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    return {
        "status": "ok",
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_auc": maybe_auc(y_test, prob),
    }


def compute_label_auc_by_view(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for (split, coarse_view), group in rows.groupby(["split", "coarse_view"], dropna=False):
        labels = group["label"].to_numpy(dtype=int)
        probs = group["prob_pe"].to_numpy(dtype=float)
        preds = group["pred_label"].to_numpy(dtype=int)
        metrics.append(
            {
                "split": split,
                "coarse_view": coarse_view,
                "n": int(len(group)),
                "label_auc": maybe_auc(labels, probs),
                "label_accuracy": float(accuracy_score(labels, preds)) if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(metrics)


def compute_embedding_separation(rows: pd.DataFrame, embeddings: torch.Tensor) -> dict[str, object]:
    emb_np = embeddings.detach().cpu().numpy()
    out: dict[str, object] = {}
    for column in ("label_name", "coarse_view"):
        labels = rows[column].astype(str).to_numpy()
        unique = np.unique(labels)
        if len(rows) < 3 or unique.size < 2 or unique.size >= len(rows):
            out[f"{column}_silhouette"] = {"status": "skipped", "reason": "not enough classes or samples"}
            continue
        out[f"{column}_silhouette"] = {
            "status": "ok",
            "score": float(silhouette_score(emb_np, labels)),
        }
    return out


def compute_centroid_scores(rows: pd.DataFrame, embeddings: torch.Tensor) -> pd.DataFrame:
    emb_np = embeddings.numpy()
    out = rows.copy()
    train_mask = rows["split"] == "train"
    if int(train_mask.sum()) < 2:
        out["dist_to_normal_centroid"] = np.nan
        out["dist_to_pe_centroid"] = np.nan
        return out
    train_labels = rows.loc[train_mask, "label"].to_numpy(dtype=int)
    train_emb = emb_np[train_mask.to_numpy(dtype=bool, copy=True)]
    normal_centroid = train_emb[train_labels == 0].mean(axis=0)
    pe_centroid = train_emb[train_labels == 1].mean(axis=0)
    out["dist_to_normal_centroid"] = np.linalg.norm(emb_np - normal_centroid[None, :], axis=1)
    out["dist_to_pe_centroid"] = np.linalg.norm(emb_np - pe_centroid[None, :], axis=1)
    return out


def cosine_topk(query: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    query = query / np.linalg.norm(query)
    candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True).clip(min=1e-8)
    sims = candidates @ query
    top_idx = np.argsort(-sims)[:k]
    return np.stack([top_idx, sims[top_idx]], axis=1)


def build_prototype_report(rows: pd.DataFrame, embeddings: torch.Tensor, k: int = 3) -> list[dict[str, object]]:
    train_mask = rows["split"] == "train"
    query_mask = rows["split"] != "train"
    if int(train_mask.sum()) < k or int(query_mask.sum()) == 0:
        return []
    train_rows = rows.loc[train_mask].reset_index(drop=True)
    train_emb = embeddings[train_mask.to_numpy(dtype=bool, copy=True)].numpy()
    reports: list[dict[str, object]] = []
    for row_idx, query_row in rows.loc[query_mask].iterrows():
        query_emb = embeddings[row_idx].numpy()
        global_hits = cosine_topk(query_emb, train_emb, k=min(k, len(train_rows)))
        same_view_mask = train_rows["coarse_view"] == query_row["coarse_view"]
        same_view_rows = train_rows.loc[same_view_mask].reset_index(drop=True)
        same_view_emb = train_emb[same_view_mask.to_numpy(dtype=bool, copy=True)]
        same_view_hits = cosine_topk(query_emb, same_view_emb, k=min(k, len(same_view_rows))) if len(same_view_rows) else np.empty((0, 2))
        reports.append(
            {
                "query_path": str(query_row["path"]),
                "query_split": str(query_row["split"]),
                "query_label_name": str(query_row["label_name"]),
                "query_coarse_view": str(query_row["coarse_view"]),
                "global_neighbors": [
                    {
                        "path": str(train_rows.iloc[int(idx)]["path"]),
                        "label_name": str(train_rows.iloc[int(idx)]["label_name"]),
                        "coarse_view": str(train_rows.iloc[int(idx)]["coarse_view"]),
                        "cosine_similarity": float(score),
                    }
                    for idx, score in global_hits
                ],
                "same_view_neighbors": [
                    {
                        "path": str(same_view_rows.iloc[int(idx)]["path"]),
                        "label_name": str(same_view_rows.iloc[int(idx)]["label_name"]),
                        "coarse_view": str(same_view_rows.iloc[int(idx)]["coarse_view"]),
                        "cosine_similarity": float(score),
                    }
                    for idx, score in same_view_hits
                ],
            }
        )
    return reports


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else default_interpret_output_dir(args.checkpoint)
    rows, embeddings = load_embedding_payload(output_dir)
    detailed_df = load_detailed_cases(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    performance_df = compute_performance_by_view(rows)
    performance_df.to_csv(output_dir / "performance_by_view.csv", index=False)
    label_auc_by_view = compute_label_auc_by_view(rows)
    label_auc_by_view.to_csv(output_dir / "label_auc_by_view.csv", index=False)

    pca_df = compute_embedding_projection(rows, embeddings)
    pca_df = compute_centroid_scores(pca_df, embeddings)
    pca_df.to_csv(output_dir / "embedding_analysis" / "embedding_projection.csv", index=False)
    plot_pca(pca_df, "label_name", figures_dir / "embedding_pca_by_label.png", "Embedding PCA by Label")
    plot_pca(pca_df, "coarse_view", figures_dir / "embedding_pca_by_view.png", "Embedding PCA by Coarse View")
    plot_pca_by_view_and_label(pca_df, figures_dir / "embedding_pca_by_view_and_label.png")

    emb_analysis_dir = output_dir / "embedding_analysis"
    umap_tsne_artifact: dict[str, object] = {"status": "skipped", "reason": "not run"}
    try:
        umap_xy, tsne_xy, proj_meta = compute_umap_tsne_coords(embeddings)
        proj_df = rows.copy().reset_index(drop=True)
        proj_df["umap1"] = umap_xy[:, 0]
        proj_df["umap2"] = umap_xy[:, 1]
        proj_df["tsne1"] = tsne_xy[:, 0]
        proj_df["tsne2"] = tsne_xy[:, 1]
        umap_csv = emb_analysis_dir / "embedding_umap_tsne_projection.csv"
        proj_df.to_csv(umap_csv, index=False)

        umap_figures = [
            figures_dir / "embedding_umap_by_label.png",
            figures_dir / "embedding_umap_by_view.png",
            figures_dir / "embedding_umap_by_view_and_label.png",
            figures_dir / "embedding_tsne_by_label.png",
            figures_dir / "embedding_tsne_by_view.png",
            figures_dir / "embedding_tsne_by_view_and_label.png",
        ]
        plot_embedding_scatter(
            proj_df, "umap1", "umap2", "label_name", umap_figures[0], "Embedding UMAP by Label", "UMAP1", "UMAP2"
        )
        plot_embedding_scatter(
            proj_df, "umap1", "umap2", "coarse_view", umap_figures[1], "Embedding UMAP by Coarse View", "UMAP1", "UMAP2"
        )
        plot_embedding_by_view_and_label(
            proj_df,
            "umap1",
            "umap2",
            umap_figures[2],
            "Embedding UMAP by Label and Coarse View",
            "UMAP1",
            "UMAP2",
        )
        plot_embedding_scatter(
            proj_df, "tsne1", "tsne2", "label_name", umap_figures[3], "Embedding t-SNE by Label", "t-SNE 1", "t-SNE 2"
        )
        plot_embedding_scatter(
            proj_df, "tsne1", "tsne2", "coarse_view", umap_figures[4], "Embedding t-SNE by Coarse View", "t-SNE 1", "t-SNE 2"
        )
        plot_embedding_by_view_and_label(
            proj_df,
            "tsne1",
            "tsne2",
            umap_figures[5],
            "Embedding t-SNE by Label and Coarse View",
            "t-SNE 1",
            "t-SNE 2",
        )

        umap_tsne_artifact = {
            "status": "ok",
            "projection_csv": str(umap_csv.resolve()),
            "figures": [str(p.resolve()) for p in umap_figures],
            "n_samples": int(len(proj_df)),
            "pre_pca_components": proj_meta["pre_pca_components"],
            "umap": proj_meta["umap"],
            "tsne": proj_meta["tsne"],
        }
        (emb_analysis_dir / "umap_tsne_summary.json").write_text(
            json.dumps({k: v for k, v in umap_tsne_artifact.items() if k != "status"}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except ImportError:
        umap_tsne_artifact = {"status": "skipped", "reason": "umap-learn not installed"}
    except Exception as exc:  # noqa: BLE001
        umap_tsne_artifact = {"status": "failed", "reason": repr(exc)}

    detailed_summary = (
        detailed_df.groupby(["split", "coarse_view"], dropna=False).mean(numeric_only=True).reset_index()
        if not detailed_df.empty
        else pd.DataFrame()
    )
    if not detailed_summary.empty:
        detailed_summary.to_csv(output_dir / "detailed_case_metrics.csv", index=False)
    temporal_summary = summarize_temporal_metrics(detailed_df)
    if not temporal_summary.empty:
        temporal_summary.to_csv(output_dir / "temporal_summary_by_group.csv", index=False)

    probe = fit_view_probe(rows, embeddings)
    label_probe = fit_label_probe(rows, embeddings)
    embedding_separation = compute_embedding_separation(rows, embeddings)
    prototypes = build_prototype_report(rows, embeddings, k=args.prototype_topk)
    (output_dir / "embedding_analysis" / "prototype_neighbors.json").write_text(
        json.dumps(prototypes[: args.max_prototype_reports], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "output_dir": str(output_dir),
        "n_samples": int(len(rows)),
        "n_detailed_cases": int(len(detailed_df)),
        "splits": sorted(rows["split"].unique().tolist()),
        "coarse_views": sorted(rows["coarse_view"].unique().tolist()),
        "performance_artifact": str((output_dir / "performance_by_view.csv").resolve()),
        "label_auc_by_view_artifact": str((output_dir / "label_auc_by_view.csv").resolve()),
        "embedding_artifact": str((output_dir / "embedding_analysis" / "embedding_projection.csv").resolve()),
        "view_probe": probe,
        "label_probe": label_probe,
        "embedding_separation": embedding_separation,
        "temporal_summary_artifact": str((output_dir / "temporal_summary_by_group.csv").resolve()) if not temporal_summary.empty else "",
        "detailed_metric_columns": detailed_df.columns.tolist() if not detailed_df.empty else [],
        "prototype_reports": min(len(prototypes), int(args.max_prototype_reports)),
        "umap_tsne": umap_tsne_artifact,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate view-level and representation-level interpretability outputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prototype-topk", type=int, default=3)
    parser.add_argument("--max-prototype-reports", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
