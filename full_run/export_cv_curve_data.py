from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_full_run import pr_curve_rows, roc_curve_rows, rows_from_prediction_csv, split_rows, _arrays


ROC_GRID = np.linspace(0.0, 1.0, 101)
RECALL_GRID = np.linspace(0.0, 1.0, 101)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _discover_folds(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("fold-*"), key=lambda p: int(p.name.split("-")[-1]))


def _setting_label(run_dir: Path) -> str:
    return run_dir.name


def _load_fold_curves(fold_dir: Path, split: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    eval_dir = fold_dir / "evaluation"
    roc_path = eval_dir / f"roc_{split}.csv"
    pr_path = eval_dir / f"pr_{split}.csv"
    if roc_path.exists() and pr_path.exists():
        roc_rows: list[dict[str, object]] = []
        pr_rows: list[dict[str, object]] = []
        with roc_path.open(encoding="utf-8", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                roc_rows.append(
                    {
                        "point_index": idx,
                        "threshold": float(row["threshold"]) if row["threshold"] not in ("", "inf") else row["threshold"],
                        "fpr": float(row["fpr"]),
                        "tpr": float(row["tpr"]),
                    }
                )
        with pr_path.open(encoding="utf-8", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                thr = row["threshold"]
                pr_rows.append(
                    {
                        "point_index": idx,
                        "threshold": float(thr) if thr not in ("", "None") else None,
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                    }
                )
        return roc_rows, pr_rows

    pred_path = fold_dir / "predictions.csv"
    rows = rows_from_prediction_csv(pred_path)
    split_rows_list = split_rows(rows, split)
    y_true, y_prob = _arrays(split_rows_list)
    roc_rows = [{"point_index": i, **r} for i, r in enumerate(roc_curve_rows(y_true, y_prob))]
    pr_rows = [{"point_index": i, **r} for i, r in enumerate(pr_curve_rows(y_true, y_prob))]
    return roc_rows, pr_rows


def _interpolate_roc_at_fpr(roc_rows: list[dict[str, object]], grid: np.ndarray) -> np.ndarray:
    fpr = np.array([float(r["fpr"]) for r in roc_rows], dtype=float)
    tpr = np.array([float(r["tpr"]) for r in roc_rows], dtype=float)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    uniq_fpr, uniq_idx = np.unique(fpr, return_index=True)
    return np.interp(grid, uniq_fpr, tpr[uniq_idx])


def _interpolate_pr_at_recall(pr_rows: list[dict[str, object]], grid: np.ndarray) -> np.ndarray:
    recall = np.array([float(r["recall"]) for r in pr_rows], dtype=float)
    precision = np.array([float(r["precision"]) for r in pr_rows], dtype=float)
    order = np.argsort(recall)
    recall, precision = recall[order], precision[order]
    uniq_recall, uniq_idx = np.unique(recall, return_index=True)
    return np.interp(grid, uniq_recall, precision[uniq_idx])


def export_setting(run_dir: Path, out_dir: Path | None = None, splits: tuple[str, ...] = ("test", "val")) -> Path:
    run_dir = run_dir.resolve()
    export_dir = (out_dir or (run_dir / "curve_exports")).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    setting = _setting_label(run_dir)
    fold_dirs = _discover_folds(run_dir)

    for split in splits:
        roc_long: list[dict[str, object]] = []
        pr_long: list[dict[str, object]] = []
        pooled_y: list[np.ndarray] = []
        pooled_p: list[np.ndarray] = []
        fold_auc_rows: list[dict[str, object]] = []
        fold_roc_curves: list[list[dict[str, object]]] = []
        fold_pr_curves: list[list[dict[str, object]]] = []

        for fold_dir in fold_dirs:
            fold = int(fold_dir.name.split("-")[-1])
            roc_rows, pr_rows = _load_fold_curves(fold_dir, split)
            fold_roc_curves.append(roc_rows)
            fold_pr_curves.append(pr_rows)

            for row in roc_rows:
                roc_long.append({"setting": setting, "fold": fold, **row})
            for row in pr_rows:
                pr_long.append({"setting": setting, "fold": fold, **row})

            pred_rows = split_rows(rows_from_prediction_csv(fold_dir / "predictions.csv"), split)
            y_true, y_prob = _arrays(pred_rows)
            pooled_y.append(y_true)
            pooled_p.append(y_prob)
            fold_auc_rows.append(
                {
                    "setting": setting,
                    "fold": fold,
                    "split": split,
                    "n": len(y_true),
                    "prevalence": float(np.mean(y_true)) if len(y_true) else float("nan"),
                    "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
                    "pr_auc": float(average_precision_score(y_true, y_prob))
                    if len(np.unique(y_true)) > 1
                    else float("nan"),
                }
            )

        y_pool = np.concatenate(pooled_y)
        p_pool = np.concatenate(pooled_p)
        roc_pooled = [{"point_index": i, **r} for i, r in enumerate(roc_curve_rows(y_pool, p_pool))]
        pr_pooled = [{"point_index": i, **r} for i, r in enumerate(pr_curve_rows(y_pool, p_pool))]

        roc_mean_rows: list[dict[str, object]] = []
        pr_mean_rows: list[dict[str, object]] = []
        if fold_roc_curves:
            tpr_stack = np.stack([_interpolate_roc_at_fpr(c, ROC_GRID) for c in fold_roc_curves], axis=0)
            prec_stack = np.stack([_interpolate_pr_at_recall(c, RECALL_GRID) for c in fold_pr_curves], axis=0)
            for i, fpr in enumerate(ROC_GRID):
                roc_mean_rows.append(
                    {
                        "grid_index": i,
                        "fpr": float(fpr),
                        "tpr_mean": float(np.mean(tpr_stack[:, i])),
                        "tpr_std": float(np.std(tpr_stack[:, i])),
                    }
                )
            for i, recall in enumerate(RECALL_GRID):
                pr_mean_rows.append(
                    {
                        "grid_index": i,
                        "recall": float(recall),
                        "precision_mean": float(np.mean(prec_stack[:, i])),
                        "precision_std": float(np.std(prec_stack[:, i])),
                    }
                )

        _write_csv(
            export_dir / f"roc_{split}_by_fold.csv",
            roc_long,
            ["setting", "fold", "point_index", "threshold", "fpr", "tpr"],
        )
        _write_csv(
            export_dir / f"pr_{split}_by_fold.csv",
            pr_long,
            ["setting", "fold", "point_index", "threshold", "precision", "recall"],
        )
        _write_csv(
            export_dir / f"roc_{split}_pooled.csv",
            [{"setting": setting, **r} for r in roc_pooled],
            ["setting", "point_index", "threshold", "fpr", "tpr"],
        )
        _write_csv(
            export_dir / f"pr_{split}_pooled.csv",
            [{"setting": setting, **r} for r in pr_pooled],
            ["setting", "point_index", "threshold", "precision", "recall"],
        )
        _write_csv(
            export_dir / f"roc_{split}_mean_std_grid.csv",
            [{"setting": setting, **r} for r in roc_mean_rows],
            ["setting", "grid_index", "fpr", "tpr_mean", "tpr_std"],
        )
        _write_csv(
            export_dir / f"pr_{split}_mean_std_grid.csv",
            [{"setting": setting, **r} for r in pr_mean_rows],
            ["setting", "grid_index", "recall", "precision_mean", "precision_std"],
        )
        pooled_auc = {
            "setting": setting,
            "fold": "pooled",
            "split": split,
            "n": int(y_pool.size),
            "prevalence": float(np.mean(y_pool)),
            "roc_auc": float(roc_auc_score(y_pool, p_pool)),
            "pr_auc": float(average_precision_score(y_pool, p_pool)),
        }
        fold_auc_rows.append(pooled_auc)
        _write_csv(
            export_dir / f"auc_{split}_summary.csv",
            fold_auc_rows,
            ["setting", "fold", "split", "n", "prevalence", "roc_auc", "pr_auc"],
        )

    meta = {
        "setting": setting,
        "run_dir": str(run_dir),
        "n_folds": len(fold_dirs),
        "export_dir": str(export_dir),
        "splits": list(splits),
    }
    (export_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return export_dir


def export_comparison(run_dirs: list[Path], out_dir: Path, split: str = "test") -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    roc_compare: list[dict[str, object]] = []
    pr_compare: list[dict[str, object]] = []
    auc_rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        export_dir = run_dir / "curve_exports"
        if not (export_dir / f"roc_{split}_pooled.csv").exists():
            export_setting(run_dir)
        setting = _setting_label(run_dir)

        with (export_dir / f"roc_{split}_pooled.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                roc_compare.append(
                    {
                        "setting": setting,
                        "point_index": int(row["point_index"]),
                        "threshold": row["threshold"],
                        "fpr": float(row["fpr"]),
                        "tpr": float(row["tpr"]),
                    }
                )
        with (export_dir / f"pr_{split}_pooled.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pr_compare.append(
                    {
                        "setting": setting,
                        "point_index": int(row["point_index"]),
                        "threshold": row["threshold"] if row["threshold"] not in ("", "None") else None,
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                    }
                )
        with (export_dir / f"auc_{split}_summary.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row["fold"]) == "pooled":
                    auc_rows.append(row)

    _write_csv(out_dir / f"roc_{split}_pooled_compare.csv", roc_compare, ["setting", "point_index", "threshold", "fpr", "tpr"])
    _write_csv(
        out_dir / f"pr_{split}_pooled_compare.csv",
        pr_compare,
        ["setting", "point_index", "threshold", "precision", "recall"],
    )
    _write_csv(out_dir / f"auc_{split}_pooled_compare.csv", auc_rows, list(auc_rows[0].keys()) if auc_rows else ["setting"])

    for run_dir in run_dirs:
        setting = _setting_label(run_dir)
        export_dir = run_dir / "curve_exports"
        grid_path = export_dir / f"roc_{split}_mean_std_grid.csv"
        dest = out_dir / f"roc_{split}_mean_std_grid_{setting}.csv"
        dest.write_text(grid_path.read_text(encoding="utf-8"), encoding="utf-8")
        pr_grid_path = export_dir / f"pr_{split}_mean_std_grid.csv"
        (out_dir / f"pr_{split}_mean_std_grid_{setting}.csv").write_text(
            pr_grid_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    return out_dir


def _display_name(setting: str) -> str:
    if "frozen" in setting:
        return "EchoPrime+head"
    if "full_finetune" in setting:
        return "EchoPE"
    return setting


SETTING_COLORS = {
    "outputs_cv_frozen_mlp_ep50": "#3F74A3",
    "outputs_cv_full_finetune_residual_ep10": "#B54646",
}
RECALL_LINE_COLORS = {
    0.9: "#9C7B57",
    0.95: "#794E47",
}
GUIDE_COLOR = "#D1DFE6"
PR_RECALL_MARKS = (0.9, 0.95)


def _apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def _style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if grid:
        ax.grid(True, axis="both", color=GUIDE_COLOR, linewidth=0.5, alpha=0.9)
        ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.6, colors="black")


def _precision_at_recall(recall: np.ndarray, precision: np.ndarray, target: float) -> float:
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    uniq_r, uniq_idx = np.unique(r, return_index=True)
    return float(np.interp(target, uniq_r, p[uniq_idx]))


def _patch_svg_pixel_size(svg_path: Path, dpi: int) -> None:
    """Match SVG display size to PNG at the same DPI (avoids 96 dpi viewer scaling)."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width = root.get("width")
    height = root.get("height")
    if not width or not height:
        return
    w_pt = float(str(width).replace("pt", ""))
    h_pt = float(str(height).replace("pt", ""))
    w_px = w_pt / 72.0 * dpi
    h_px = h_pt / 72.0 * dpi
    root.set("width", f"{w_px:g}px")
    root.set("height", f"{h_px:g}px")
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def _save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    """Save PDF/PNG/SVG with identical layout and typographic scale."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(dpi)
    save_kwargs = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
        "edgecolor": "none",
    }
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".png"), **save_kwargs)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", **save_kwargs)
    _patch_svg_pixel_size(svg_path, dpi=dpi)


def _load_compare_curves(compare_dir: Path, split: str) -> tuple[list[str], dict, dict, dict]:
    roc_path = compare_dir / f"roc_{split}_pooled_compare.csv"
    pr_path = compare_dir / f"pr_{split}_pooled_compare.csv"
    auc_path = compare_dir / f"auc_{split}_pooled_compare.csv"

    with roc_path.open(encoding="utf-8", newline="") as f:
        settings = sorted({row["setting"] for row in csv.DictReader(f)})

    roc_by_setting: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pr_by_setting: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    auc_by_setting: dict[str, dict[str, float]] = {}

    for setting in settings:
        with roc_path.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["setting"] == setting]
        roc_by_setting[setting] = (
            np.array([float(r["fpr"]) for r in rows]),
            np.array([float(r["tpr"]) for r in rows]),
        )
        with pr_path.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["setting"] == setting]
        pr_by_setting[setting] = (
            np.array([float(r["recall"]) for r in rows]),
            np.array([float(r["precision"]) for r in rows]),
        )

    with auc_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row["fold"]) == "pooled":
                auc_by_setting[row["setting"]] = {
                    "roc_auc": float(row["roc_auc"]),
                    "pr_auc": float(row["pr_auc"]),
                    "n": int(row["n"]),
                }
    return settings, roc_by_setting, pr_by_setting, auc_by_setting


def _scatter_pr_recall_points(
    ax: plt.Axes, recall: np.ndarray, precision: np.ndarray, color: str
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for target in PR_RECALL_MARKS:
        prec = _precision_at_recall(recall, precision, target)
        points.append({"recall": target, "precision": prec})
        ax.scatter(
            [target],
            [prec],
            s=22,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
    return points


def _draw_pr_recall_guides(ax: plt.Axes) -> list[Line2D]:
    handles: list[Line2D] = []
    for target in PR_RECALL_MARKS:
        line_color = RECALL_LINE_COLORS[target]
        ax.axvline(target, color=line_color, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
        handles.append(
            Line2D(
                [0],
                [0],
                color=line_color,
                linestyle=(0, (3, 2)),
                linewidth=0.9,
                label=f"Recall = {target:.2f}",
            )
        )
    return handles


def _legend_with_recall_guides(ax: plt.Axes, recall_handles: list[Line2D], *, loc: str = "lower left") -> None:
    curve_handles, curve_labels = ax.get_legend_handles_labels()
    guide_labels = [h.get_label() for h in recall_handles]
    ax.legend(
        curve_handles + recall_handles,
        curve_labels + guide_labels,
        loc=loc,
        frameon=False,
        handlelength=1.8,
    )


def _write_pr_recall_markers_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _write_csv(
        path,
        rows,
        ["setting", "display_name", "recall_target", "precision_at_recall", "split", "curve_type"],
    )


def _write_caption(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_nature_figures(compare_dir: Path, split: str = "test") -> Path:
    """Nature-style ROC/PR figures; captions written separately."""
    _apply_nature_style()
    compare_dir = compare_dir.resolve()
    plot_dir = compare_dir / "nature_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    settings, roc_by_setting, pr_by_setting, auc_by_setting = _load_compare_curves(compare_dir, split)
    pr_marker_rows: list[dict[str, object]] = []
    caption_lines: list[str] = [
        f"Figure. Pooled five-fold cross-validation ROC and precision-recall curves ({split} split).",
        "Left, receiver operating characteristic (ROC) curves for EchoPrime+head and EchoPE.",
        "Right, precision-recall (PR) curves; dashed vertical lines mark recall 0.90 (warm brown) and 0.95 (brown),",
        "with filled markers showing interpolated precision on each model curve.",
        "",
    ]

    # --- Pooled curves (primary figure) ---
    fig_w_in = 7.08  # ~180 mm double-column
    fig_h_in = 2.95
    fig, axes = plt.subplots(1, 2, figsize=(fig_w_in, fig_h_in))

    for setting in settings:
        color = SETTING_COLORS.get(setting, "#3F74A3")
        label = _display_name(setting)
        auc = auc_by_setting[setting]
        fpr, tpr = roc_by_setting[setting]
        recall, precision = pr_by_setting[setting]

        axes[0].plot(
            fpr,
            tpr,
            color=color,
            label=f"{label} (AUROC {auc['roc_auc']:.3f})",
            zorder=3,
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            label=f"{label} (AUPR {auc['pr_auc']:.3f})",
            zorder=3,
        )
        mark_pts = _scatter_pr_recall_points(axes[1], recall, precision, color)
        for pt in mark_pts:
            pr_marker_rows.append(
                {
                    "setting": setting,
                    "display_name": label,
                    "recall_target": pt["recall"],
                    "precision_at_recall": pt["precision"],
                    "split": split,
                    "curve_type": "pooled",
                }
            )
        p90 = next(p["precision"] for p in mark_pts if p["recall"] == 0.9)
        p95 = next(p["precision"] for p in mark_pts if p["recall"] == 0.95)
        caption_lines.append(
            f"{label}: n={auc['n']}, AUROC={auc['roc_auc']:.4f}, AUPR={auc['pr_auc']:.4f}; "
            f"precision at recall 0.90={p90:.4f}, at recall 0.95={p95:.4f}."
        )

    axes[0].plot([0, 1], [0, 1], color="#A78F95", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        _style_axes(ax)
    recall_guide_handles = _draw_pr_recall_guides(axes[1])
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.8)
    _legend_with_recall_guides(axes[1], recall_guide_handles)

    fig.subplots_adjust(wspace=0.32)
    stem = plot_dir / f"roc_pr_{split}_pooled_compare"
    _save_figure(fig, stem)
    plt.close(fig)

    # --- Mean ± std over folds ---
    fig, axes = plt.subplots(1, 2, figsize=(fig_w_in, fig_h_in))
    caption_std: list[str] = [
        "",
        f"Supplementary-style panel. Mean ± s.d. of ROC and PR curves across five folds ({split} split).",
        "Shaded bands indicate ±1 s.d. at each grid point.",
        "",
    ]

    for setting in settings:
        color = SETTING_COLORS.get(setting, "#3F74A3")
        label = _display_name(setting)
        grid_roc = compare_dir / f"roc_{split}_mean_std_grid_{setting}.csv"
        grid_pr = compare_dir / f"pr_{split}_mean_std_grid_{setting}.csv"
        if not grid_roc.exists():
            continue

        with grid_roc.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        fpr = np.array([float(r["fpr"]) for r in rows])
        tpr_m = np.array([float(r["tpr_mean"]) for r in rows])
        tpr_s = np.array([float(r["tpr_std"]) for r in rows])
        axes[0].plot(fpr, tpr_m, color=color, label=label, zorder=3)
        axes[0].fill_between(
            fpr,
            tpr_m - tpr_s,
            tpr_m + tpr_s,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

        with grid_pr.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        recall = np.array([float(r["recall"]) for r in rows])
        prec_m = np.array([float(r["precision_mean"]) for r in rows])
        prec_s = np.array([float(r["precision_std"]) for r in rows])
        axes[1].plot(recall, prec_m, color=color, label=label, zorder=3)
        axes[1].fill_between(
            recall,
            prec_m - prec_s,
            prec_m + prec_s,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        for target in PR_RECALL_MARKS:
            prec = _precision_at_recall(recall, prec_m, target)
            axes[1].scatter([target], [prec], s=22, color=color, edgecolors="white", linewidths=0.5, zorder=4)
            pr_marker_rows.append(
                {
                    "setting": setting,
                    "display_name": label,
                    "recall_target": target,
                    "precision_at_recall": prec,
                    "split": split,
                    "curve_type": "mean_over_folds",
                }
            )
            caption_std.append(f"{label} at recall {target:.2f}: mean precision={prec:.4f}.")

    axes[0].plot([0, 1], [0, 1], color="#A78F95", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        _style_axes(ax)
    recall_guide_handles = _draw_pr_recall_guides(axes[1])
    axes[0].legend(loc="lower right", frameon=False, handlelength=1.8)
    _legend_with_recall_guides(axes[1], recall_guide_handles)

    fig.subplots_adjust(wspace=0.32)
    stem_std = plot_dir / f"roc_pr_{split}_mean_std_compare"
    _save_figure(fig, stem_std)
    plt.close(fig)

    _write_pr_recall_markers_csv(plot_dir / f"pr_{split}_recall_markers.csv", pr_marker_rows)
    _write_caption(plot_dir / f"roc_pr_{split}_pooled_compare.caption.txt", caption_lines)
    _write_caption(plot_dir / f"roc_pr_{split}_mean_std_compare.caption.txt", caption_std)
    return plot_dir


def plot_examples(compare_dir: Path, split: str = "test") -> None:
    plot_nature_figures(compare_dir, split=split)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ROC/PR curve CSVs for CV binary runs.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_cv_curve_comparison",
    )
    parser.add_argument("--splits", nargs="+", default=["test", "val"])
    parser.add_argument("--plot", action="store_true", help="Write Nature-style comparison figures (PDF/PNG/SVG).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    for run_dir in run_dirs:
        out = export_setting(run_dir, splits=tuple(args.splits))
        print(f"[done] {run_dir.name} -> {out}")
    compare_dir = export_comparison(run_dirs, args.compare_dir.resolve())
    print(f"[done] comparison -> {compare_dir}")
    if args.plot:
        plot_examples(compare_dir, split="test")
        print(f"[done] nature plots -> {compare_dir / 'nature_plots'}")


if __name__ == "__main__":
    main()
