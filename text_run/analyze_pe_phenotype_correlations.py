#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


FOCUS_PHENOTYPES = [
    "rv_systolic_function_depressed",
    "right_ventricle_dilation",
    "pulmonary_artery_pressure_continuous",
]

BINARY_PHENOTYPES = [
    "rv_systolic_function_depressed",
    "right_ventricle_dilation",
]

PHENOTYPE_META = {
    "rv_systolic_function_depressed": {
        "title": "RV systolic function",
        "ylabel": "Predicted dysfunction score",
    },
    "right_ventricle_dilation": {
        "title": "RV dilation",
        "ylabel": "Predicted dilation score",
    },
    "pulmonary_artery_pressure_continuous": {
        "title": "Pulmonary artery pressure",
        "ylabel": "Predicted pressure",
    },
}

REPORT_SECTION_HEADERS = {
    "rv_systolic_function_depressed": ["Right Ventricle:"],
    "right_ventricle_dilation": ["Right Ventricle:"],
    "pulmonary_artery_pressure_continuous": ["Pulmonary Artery:", "Tricuspid Valve:", "Pulmonic Valve:"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze three EchoPrime PE phenotypes and save plot-ready result files."
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=Path("experiments/text_run/results/pe_phenotype_predictions.csv"),
        help="Per-video phenotype CSV relative to the EchoPrime root.",
    )
    ap.add_argument(
        "--thresholds",
        type=Path,
        default=Path("assets/roc_thresholds.csv"),
        help="Official ROC thresholds CSV relative to the EchoPrime root.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/text_run/results/analysis"),
        help="Analysis output directory relative to the EchoPrime root.",
    )
    ap.add_argument(
        "--report-max-chars",
        type=int,
        default=220,
        help="Maximum number of characters shown for each representative report snippet.",
    )
    ap.add_argument(
        "--skip-report-examples",
        action="store_true",
        help="Skip generating actual EchoPrime report examples.",
    )
    return ap.parse_args()


def safe_auc(y_true: pd.Series, scores: pd.Series) -> float | None:
    work = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    if work.empty or work["y"].nunique() < 2 or work["score"].nunique() < 2:
        return None
    return float(roc_auc_score(work["y"], work["score"]))


def safe_corr(y_true: pd.Series, scores: pd.Series) -> float | None:
    work = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    if work.empty or work["y"].nunique() < 2 or work["score"].nunique() < 2:
        return None
    corr = np.corrcoef(work["y"].to_numpy(dtype=float), work["score"].to_numpy(dtype=float))[0, 1]
    return float(corr)


def risk_metrics(mask: pd.Series, y_true: pd.Series) -> tuple[float | None, float | None, float | None, float | None]:
    work = pd.DataFrame({"mask": mask, "y": y_true}).dropna()
    if work.empty or work["y"].nunique() < 2 or work["mask"].nunique() < 2:
        return (None, None, None, None)

    pe = work["y"].eq(1)
    normal = work["y"].eq(0)

    pe_pos = float(work.loc[pe, "mask"].sum())
    pe_neg = float(pe.sum() - pe_pos)
    normal_pos = float(work.loc[normal, "mask"].sum())
    normal_neg = float(normal.sum() - normal_pos)

    pe_rate = pe_pos / float(pe.sum()) if pe.sum() else None
    normal_rate = normal_pos / float(normal.sum()) if normal.sum() else None
    risk_ratio = None
    if pe_rate is not None and normal_rate not in (None, 0):
        risk_ratio = float(pe_rate / normal_rate)

    pe_pos += 0.5
    pe_neg += 0.5
    normal_pos += 0.5
    normal_neg += 0.5
    odds_ratio = float((pe_pos * normal_neg) / (pe_neg * normal_pos))
    return (pe_rate, normal_rate, risk_ratio, odds_ratio)


def phenotype_summary(df: pd.DataFrame, phenotype: str, threshold_map: dict[str, float]) -> dict[str, object]:
    values = pd.to_numeric(df[phenotype], errors="coerce")
    work = df.assign(score=values).dropna(subset=["score"]).copy()

    pe_scores = work.loc[work["label_binary"].eq(1), "score"]
    normal_scores = work.loc[work["label_binary"].eq(0), "score"]
    auc = safe_auc(work["label_binary"], work["score"])
    corr = safe_corr(work["label_binary"], work["score"])
    mean_diff = None
    if not pe_scores.empty and not normal_scores.empty:
        mean_diff = float(pe_scores.mean() - normal_scores.mean())

    threshold = threshold_map.get(phenotype)
    pe_positive_rate = None
    normal_positive_rate = None
    risk_ratio = None
    odds_ratio = None
    if threshold is not None:
        mask = work["score"].ge(threshold)
        pe_positive_rate, normal_positive_rate, risk_ratio, odds_ratio = risk_metrics(
            mask=mask,
            y_true=work["label_binary"],
        )

    return {
        "phenotype": phenotype,
        "display_name": PHENOTYPE_META[phenotype]["title"],
        "ylabel": PHENOTYPE_META[phenotype]["ylabel"],
        "n_non_null": int(len(work)),
        "n_pe": int(work["label_binary"].sum()),
        "n_normal": int((work["label_binary"] == 0).sum()),
        "pe_mean": float(pe_scores.mean()) if not pe_scores.empty else None,
        "normal_mean": float(normal_scores.mean()) if not normal_scores.empty else None,
        "pe_median": float(pe_scores.median()) if not pe_scores.empty else None,
        "normal_median": float(normal_scores.median()) if not normal_scores.empty else None,
        "pe_std": float(pe_scores.std(ddof=1)) if len(pe_scores) > 1 else None,
        "normal_std": float(normal_scores.std(ddof=1)) if len(normal_scores) > 1 else None,
        "mean_diff_pe_minus_normal": mean_diff,
        "pearson_with_pe_label": corr,
        "auc_for_pe": auc,
        "signed_rank_biserial": (2.0 * auc - 1.0) if auc is not None else None,
        "threshold_used": threshold,
        "pe_positive_rate": pe_positive_rate,
        "normal_positive_rate": normal_positive_rate,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
    }


def _format_number(value: float | None, digits: int) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _trim_report(report: str, max_chars: int) -> str:
    one_line = " ".join(str(report).replace("[SEP]", " ").split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 1].rstrip() + "..."


def _target_example_score(scores: pd.Series, phenotype: str, label: str, threshold: float | None) -> float:
    if phenotype in BINARY_PHENOTYPES and label == "pe":
        if threshold is not None and scores.ge(threshold).any():
            return float(threshold)
        return float(scores.quantile(0.75))
    return float(scores.median())


def select_representative_case(
    df: pd.DataFrame,
    phenotype: str,
    label: str,
    threshold: float | None,
) -> pd.Series:
    work = df.loc[df["label"].eq(label), ["id", "video_path", phenotype]].copy()
    work["score"] = pd.to_numeric(work[phenotype], errors="coerce")
    work = work.dropna(subset=["score"])
    if work.empty:
        raise ValueError(f"No valid rows found for {phenotype} / {label}")
    target_score = _target_example_score(work["score"], phenotype=phenotype, label=label, threshold=threshold)
    work["distance_to_target"] = (work["score"] - target_score).abs()
    work = work.sort_values(["distance_to_target", "score", "id"], ascending=[True, False, True])
    return work.iloc[0]


def _extract_report_focus(utils_module, report: str, phenotype: str, max_chars: int) -> str:
    for header in REPORT_SECTION_HEADERS[phenotype]:
        section = utils_module.extract_section(report, header)
        if section != "Section not found.":
            return _trim_report(section, max_chars=max_chars)
    return _trim_report(report, max_chars=max_chars)


def generate_report_examples(df: pd.DataFrame, threshold_map: dict[str, float], max_chars: int) -> pd.DataFrame:
    experiments_dir = Path(__file__).resolve().parent.parent
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))

    from echo_paths import setup_echo_root_cwd  # noqa: WPS433

    setup_echo_root_cwd()
    import utils  # noqa: WPS433
    from echo_prime import EchoPrime  # noqa: WPS433
    from text_run.run_echoprime_pe_phenotypes import preprocess_single_video  # noqa: WPS433

    ep = EchoPrime()
    cache: dict[str, str] = {}
    rows: list[dict[str, object]] = []

    for phenotype in FOCUS_PHENOTYPES:
        for label in ["normal", "pe"]:
            case = select_representative_case(
                df=df,
                phenotype=phenotype,
                label=label,
                threshold=threshold_map.get(phenotype),
            )
            sample_id = str(case["id"])
            video_path = Path(str(case["video_path"]))

            if sample_id not in cache:
                stack = preprocess_single_video(ep, utils, video_path)
                encoding = ep.encode_study(stack, visualize=False)
                cache[sample_id] = ep.generate_report(encoding)

            report = cache[sample_id]
            rows.append(
                {
                    "phenotype": phenotype,
                    "display_name": PHENOTYPE_META[phenotype]["title"],
                    "label": label,
                    "id": sample_id,
                    "video_path": str(video_path),
                    "score": float(case["score"]),
                    "report": report,
                    "report_snippet": _extract_report_focus(
                        utils_module=utils,
                        report=report,
                        phenotype=phenotype,
                        max_chars=max_chars,
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_examples_markdown(examples_df: pd.DataFrame, output_path: Path) -> None:
    if examples_df.empty:
        output_path.write_text("# Representative generated reports\n\nReport generation skipped.\n", encoding="utf-8")
        return

    lines = ["# Representative generated reports", ""]
    for phenotype in FOCUS_PHENOTYPES:
        lines.append(f"## {PHENOTYPE_META[phenotype]['title']}")
        subset = examples_df.loc[examples_df["phenotype"].eq(phenotype)].copy()
        subset["label"] = pd.Categorical(subset["label"], categories=["normal", "pe"], ordered=True)
        subset = subset.sort_values("label")
        for _, row in subset.iterrows():
            label_name = "Normal" if row["label"] == "normal" else "PE"
            digits = 1 if phenotype == "pulmonary_artery_pressure_continuous" else 3
            lines.append(f"### {label_name}")
            lines.append(f"- ID: `{row['id']}`")
            lines.append(f"- Score: {float(row['score']):.{digits}f}")
            lines.append(f"- Snippet: {row['report_snippet']}")
            lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_markdown_report(
    summary_df: pd.DataFrame,
    counts_by_view: pd.Series,
    n_successful_videos: int,
    output_path: Path,
) -> None:
    ordered = summary_df.sort_values("auc_for_pe", ascending=False, na_position="last")
    lines = [
        "# EchoPrime PE phenotype analysis",
        "",
        "## Dataset snapshot",
        f"- Successful videos: {n_successful_videos}",
        "- View counts:",
    ]
    for view, count in counts_by_view.items():
        lines.append(f"  - {view}: {int(count)}")

    lines.extend(["", "## Focus phenotypes"])
    for _, row in ordered.iterrows():
        lines.append(
            f"- {row['display_name']}: "
            f"delta mean={_format_number(row['mean_diff_pe_minus_normal'], 4)}, "
            f"AUC={_format_number(row['auc_for_pe'], 3)}, "
            f"Pearson={_format_number(row['pearson_with_pe_label'], 3)}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_plot_ready_data(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phenotype in FOCUS_PHENOTYPES:
        values = pd.to_numeric(df[phenotype], errors="coerce")
        work = df.assign(score=values).dropna(subset=["score"]).copy()
        for row in work[["id", "label", "label_binary", "parsed_view", "score"]].itertuples(index=False):
            rows.append(
                {
                    "id": row.id,
                    "label": row.label,
                    "label_binary": int(row.label_binary),
                    "parsed_view": row.parsed_view,
                    "phenotype": phenotype,
                    "display_name": PHENOTYPE_META[phenotype]["title"],
                    "ylabel": PHENOTYPE_META[phenotype]["ylabel"],
                    "score": float(row.score),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    echo_root = Path(__file__).resolve().parents[2]

    predictions_path = (echo_root / args.predictions).resolve()
    thresholds_path = (echo_root / args.thresholds).resolve()
    output_dir = (echo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)
    df = df.loc[df["status"].eq("ok")].copy()
    df["label"] = df["label"].astype(str).str.lower()
    df["label_binary"] = df["label_binary"].astype(int)

    threshold_df = pd.read_csv(thresholds_path)
    threshold_map = dict(zip(threshold_df["feature"], threshold_df["threshold"], strict=False))

    counts_by_view = df["parsed_view"].value_counts().sort_index()
    label_counts = df["label"].value_counts().sort_index()

    summary_rows = [phenotype_summary(df, phenotype, threshold_map) for phenotype in FOCUS_PHENOTYPES]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "focus_phenotype_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_df = build_plot_ready_data(df)
    plot_data_path = output_dir / "focus_phenotype_plot_data.csv"
    plot_df.to_csv(plot_data_path, index=False)

    examples_df = pd.DataFrame()
    if not args.skip_report_examples:
        examples_df = generate_report_examples(
            df=df,
            threshold_map=threshold_map,
            max_chars=args.report_max_chars,
        )
        examples_df.to_json(output_dir / "representative_report_examples.json", orient="records", indent=2)
    build_examples_markdown(examples_df, output_dir / "representative_report_examples.md")

    build_markdown_report(summary_df, counts_by_view, int(len(df)), output_dir / "analysis_summary.md")

    dataset_summary = {
        "n_successful_videos": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "view_counts": {str(k): int(v) for k, v in counts_by_view.items()},
        "focus_phenotypes": FOCUS_PHENOTYPES,
        "binary_phenotypes": BINARY_PHENOTYPES,
        "saved_files": {
            "summary_csv": str(summary_path),
            "plot_data_csv": str(plot_data_path),
            "examples_json": str(output_dir / "representative_report_examples.json"),
        },
    }
    dataset_summary_path = output_dir / "dataset_summary.json"
    dataset_summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    print(json.dumps(dataset_summary, indent=2))
    print(f"saved_summary={summary_path}")
    print(f"saved_plot_data={plot_data_path}")
    if not args.skip_report_examples:
        print(f"saved_examples={output_dir / 'representative_report_examples.json'}")
    print("ready_for_plot=EchoPrime/experiments/text_run/plot_pe_phenotype_figure.py")


if __name__ == "__main__":
    main()
