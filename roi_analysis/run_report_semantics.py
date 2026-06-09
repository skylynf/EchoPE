#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
INTERPRET_ROOT = EXP_ROOT / "interpretability"
FULL_RUN_ROOT = EXP_ROOT / "classification"
LAUNCH_CWD = Path(os.environ.get("PWD", str(Path.cwd()))).resolve()
for path in (EXP_ROOT, INTERPRET_ROOT, FULL_RUN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from echo_paths import setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

from echo_prime import EchoPrime  # noqa: E402
from classification.data import COARSE_VIEW_TO_IDX  # noqa: E402
from load_interpret_model import build_model_from_checkpoint  # noqa: E402
from roi_utils import load_preprocessed_video  # noqa: E402


FOCUS_PHENOTYPES = [
    "pulmonary_artery_pressure_continuous",
    "rv_systolic_function_depressed",
    "right_ventricle_dilation",
    "right_atrium_dilation",
    "left_atrium_dilation",
]


def resolve_launch_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (LAUNCH_CWD / candidate).resolve()


def _json_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


def jaccard_ids(left: list[int], right: list[int]) -> float:
    left_set = set(int(v) for v in left)
    right_set = set(int(v) for v in right)
    if not left_set and not right_set:
        return float("nan")
    return float(len(left_set & right_set) / max(1, len(left_set | right_set)))


def section_index(ep: EchoPrime, section_name: str) -> int:
    sections = [str(sec) for sec in ep.non_empty_sections]
    if section_name in sections:
        return sections.index(section_name)
    matches = [idx for idx, sec in enumerate(sections) if section_name.lower() in sec.lower()]
    if matches:
        return matches[0]
    raise ValueError(f"Section {section_name!r} not found. Available sections include: {sections[:8]}")


def section_embedding(ep: EchoPrime, study_embedding: torch.Tensor, section_name: str) -> torch.Tensor:
    study_embedding = study_embedding.detach().cpu()
    s_idx = section_index(ep, section_name)
    weights = [float(ep.section_weights[s_idx][int(torch.argmax(view_encoding).item())]) for view_encoding in study_embedding[:, 512:]]
    weighted = study_embedding[:, :512] * torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    emb = weighted.mean(dim=0)
    return F.normalize(emb.float(), dim=0)


def retrieve_section_topk(ep: EchoPrime, study_embedding: torch.Tensor, section_name: str, k: int) -> list[int]:
    emb = section_embedding(ep, study_embedding, section_name)
    sims = emb @ ep.candidate_embeddings.float().T
    return [int(idx) for idx in torch.topk(sims, k=min(k, sims.numel())).indices.tolist()]


@torch.no_grad()
def frozen_study_embedding(ep: EchoPrime, video: torch.Tensor) -> torch.Tensor:
    stack = video.to(ep.device)
    features = ep.embed_videos(stack)
    views = ep.get_views(stack, visualize=False)
    return torch.cat((features, views), dim=1).detach().cpu()


@torch.no_grad()
def checkpoint_study_embedding(
    ep: EchoPrime,
    model: torch.nn.Module,
    checkpoint_args,
    video: torch.Tensor,
    coarse_view: str,
    device: torch.device,
) -> torch.Tensor:
    stack = video.to(device)
    coarse_view_idx = None
    if getattr(checkpoint_args, "head_view_mode", "none") == "coarse4":
        coarse_view_idx = torch.tensor([COARSE_VIEW_TO_IDX[str(coarse_view)]], dtype=torch.long, device=device)
    emb = model(stack, coarse_view_idx=coarse_view_idx)["emb"].detach().cpu()
    views = ep.get_views(video.to(ep.device), visualize=False).detach().cpu()
    return torch.cat((emb, views), dim=1)


def run_predict_metrics(ep: EchoPrime, study_embedding: torch.Tensor, k: int) -> dict[str, float | None]:
    preds = ep.predict_metrics(study_embedding, k=k)
    return {name: _json_float(preds.get(name)) for name in FOCUS_PHENOTYPES}


def correlation_summary(df: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    if df.empty:
        return out
    for pheno in FOCUS_PHENOTYPES:
        frozen_col = f"{pheno}_frozen"
        for tag in ("lora_kd_combo", "full_finetune"):
            other_col = f"{pheno}_{tag}"
            if frozen_col in df.columns and other_col in df.columns:
                valid = df[[frozen_col, other_col]].dropna()
                if len(valid) >= 3:
                    out[f"spearman_{pheno}_{tag}_vs_frozen"] = float(valid[frozen_col].corr(valid[other_col], method="spearman"))
    for tag in ("lora_kd_combo", "full_finetune"):
        col = f"embedding_cosine_{tag}_vs_frozen"
        if col in df.columns:
            out[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
            }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen EchoPrime and LoRA+KD semantic probes on the ROI case panel.")
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-kd-checkpoint", type=Path, default=None)
    parser.add_argument("--full-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--retrieval-k", type=int, default=50)
    parser.add_argument("--section", default="Right Ventricle")
    parser.add_argument("--save-reports", action="store_true")
    parser.add_argument("--max-reports", type=int, default=12)
    return parser


def maybe_load_checkpoint(
    checkpoint_path: Path | None,
    *,
    tag: str,
    device: torch.device,
    skipped: list[dict[str, str]],
) -> tuple[torch.nn.Module, object] | None:
    if checkpoint_path is None:
        return None
    resolved = resolve_launch_path(checkpoint_path)
    if not resolved.is_file():
        message = f"checkpoint not found; skipping {tag}: {resolved}"
        print(f"[WARN] {message}", file=sys.stderr)
        skipped.append({"model_tag": tag, "checkpoint": str(resolved), "reason": "missing_checkpoint"})
        return None
    _path, _payload, ckpt_args, model = build_model_from_checkpoint(resolved, device=device)
    return model, ckpt_args


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_launch_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(resolve_launch_path(args.cases_csv))
    if "case_id" not in cases.columns:
        cases.insert(0, "case_id", [f"semantic_{idx:04d}" for idx in range(len(cases))])

    device = torch.device(args.device)
    ep = EchoPrime()
    checkpoint_models: dict[str, tuple[torch.nn.Module, object]] = {}
    skipped_checkpoints: list[dict[str, str]] = []
    lora_kd = maybe_load_checkpoint(
        args.lora_kd_checkpoint,
        tag="lora_kd_combo",
        device=device,
        skipped=skipped_checkpoints,
    )
    if lora_kd is not None:
        checkpoint_models["lora_kd_combo"] = lora_kd
    full = maybe_load_checkpoint(
        args.full_checkpoint,
        tag="full_finetune",
        device=device,
        skipped=skipped_checkpoints,
    )
    if full is not None:
        checkpoint_models["full_finetune"] = full

    phenotype_rows = []
    drift_rows = []
    report_rows = []
    for _, case in tqdm(cases.iterrows(), total=len(cases), desc="semantic probe"):
        video = load_preprocessed_video(str(case["path"])).unsqueeze(0)
        frozen_emb = frozen_study_embedding(ep, video)
        embeddings = {"frozen": frozen_emb}
        for tag, (model, ckpt_args) in checkpoint_models.items():
            embeddings[tag] = checkpoint_study_embedding(ep, model, ckpt_args, video, str(case.get("coarse_view", "")), device)

        phenotype_by_tag = {tag: run_predict_metrics(ep, emb, k=args.k) for tag, emb in embeddings.items()}
        topk_by_tag = {tag: retrieve_section_topk(ep, emb, args.section, k=args.retrieval_k) for tag, emb in embeddings.items()}
        for tag, preds in phenotype_by_tag.items():
            phenotype_rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "path": str(case["path"]),
                    "label": int(case["label"]) if "label" in case else None,
                    "label_name": str(case.get("label_name", "")),
                    "coarse_view": str(case.get("coarse_view", "")),
                    "model_tag": tag,
                    **preds,
                }
            )

        drift = {
            "case_id": str(case["case_id"]),
            "path": str(case["path"]),
            "label_name": str(case.get("label_name", "")),
            "coarse_view": str(case.get("coarse_view", "")),
        }
        for tag in checkpoint_models:
            drift[f"embedding_cosine_{tag}_vs_frozen"] = float(
                F.cosine_similarity(embeddings[tag][:, :512], frozen_emb[:, :512], dim=1).mean().item()
            )
            drift[f"{args.section}_top{args.retrieval_k}_jaccard_{tag}_vs_frozen"] = jaccard_ids(
                topk_by_tag[tag],
                topk_by_tag["frozen"],
            )
            for pheno in FOCUS_PHENOTYPES:
                drift[f"{pheno}_frozen"] = phenotype_by_tag["frozen"].get(pheno)
                drift[f"{pheno}_{tag}"] = phenotype_by_tag[tag].get(pheno)
        drift_rows.append(drift)

        if args.save_reports and len(report_rows) < args.max_reports:
            entry = {
                "case_id": str(case["case_id"]),
                "path": str(case["path"]),
                "frozen_report": ep.generate_report(frozen_emb),
            }
            if "lora_kd_combo" in embeddings:
                entry["lora_kd_combo_report"] = ep.generate_report(embeddings["lora_kd_combo"])
            report_rows.append(entry)

    phenotype_df = pd.DataFrame(phenotype_rows)
    phenotype_path = output_dir / "phenotype_predictions.csv"
    phenotype_df.to_csv(phenotype_path, index=False)
    drift_df = pd.DataFrame(drift_rows)
    drift_path = output_dir / "semantic_drift.csv"
    drift_df.to_csv(drift_path, index=False)
    if report_rows:
        with (output_dir / "generated_reports.jsonl").open("w", encoding="utf-8") as fh:
            for row in report_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_cases": int(len(cases)),
        "models": ["frozen", *checkpoint_models.keys()],
        "skipped_checkpoints": skipped_checkpoints,
        "focus_phenotypes": FOCUS_PHENOTYPES,
        "section": args.section,
        "artifacts": {
            "phenotype_predictions": str(phenotype_path),
            "semantic_drift": str(drift_path),
            "generated_reports": str(output_dir / "generated_reports.jsonl") if report_rows else "",
        },
        "correlations": correlation_summary(drift_df),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
