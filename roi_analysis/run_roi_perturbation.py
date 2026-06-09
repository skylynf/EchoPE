#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
INTERPRET_ROOT = EXP_ROOT / "interpretability"
FULL_RUN_ROOT = EXP_ROOT / "classification"
for path in (EXP_ROOT, INTERPRET_ROOT, FULL_RUN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from classification.data import COARSE_VIEW_TO_IDX  # noqa: E402
from load_interpret_model import build_model_from_checkpoint  # noqa: E402
from roi_utils import (  # noqa: E402
    load_annotations,
    load_attribution_payloads,
    load_or_generate_mask_payload,
    load_preprocessed_video,
    mask_case_lookup,
    resolve_path,
    summarize_metric_rows,
)


def softmax_prob(logits: torch.Tensor) -> float:
    return float(torch.softmax(logits.float(), dim=1)[0, 1].detach().cpu())


def pe_logit(logits: torch.Tensor) -> float:
    return float(logits.float()[0, 1].detach().cpu())


def maybe_coarse_view_idx(checkpoint_args, record: dict[str, object], device: torch.device) -> torch.Tensor | None:
    if getattr(checkpoint_args, "head_view_mode", "none") != "coarse4":
        return None
    return torch.tensor([COARSE_VIEW_TO_IDX[str(record["coarse_view"])]], dtype=torch.long, device=device)


def apply_spatial_occlusion(video: torch.Tensor, mask: torch.Tensor, fill_value: float) -> torch.Tensor:
    out = video.detach().clone()
    mask = mask.bool()
    if mask.shape != tuple(out.shape[-2:]):
        mask = torch.nn.functional.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=tuple(out.shape[-2:]),
            mode="nearest",
        )[0, 0].bool()
    out[:, :, :, mask] = float(fill_value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure PE probability/logit changes after ROI or context occlusion.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--interpret-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fill-value", type=float, default=0.0, help="Occlusion value in normalized video space.")
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint_path, _payload, checkpoint_args, model = build_model_from_checkpoint(args.checkpoint, device=device)
    interpret_dir = resolve_path(args.interpret_dir)
    mask_dir = resolve_path(args.mask_dir)
    output_csv = resolve_path(args.output_csv)
    annotations = load_annotations(args.annotations)
    path_to_mask_case = mask_case_lookup(mask_dir)

    rows: list[dict[str, object]] = []
    skipped = []
    for payload in tqdm(load_attribution_payloads(interpret_dir), desc="roi perturbation"):
        record = payload["record"]
        attr_case_id = str(payload["case_id"])
        mask_case_id = path_to_mask_case.get(str(record["path"]), attr_case_id)
        mask_payload, generated_mask = load_or_generate_mask_payload(
            mask_dir,
            case_id=mask_case_id,
            record={**record, "split": payload["split"]},
        )
        masks = dict(mask_payload["masks"])
        masks.update(annotations.get(mask_case_id, {}))
        masks.update(annotations.get(attr_case_id, {}))
        if not masks:
            skipped.append({"case_id": attr_case_id, "path": str(record["path"]), "reason": "no_masks"})
            continue
        video = load_preprocessed_video(str(record["path"])).unsqueeze(0).to(device)
        coarse_view_idx = maybe_coarse_view_idx(checkpoint_args, record, device)
        base_logits = model(video, coarse_view_idx=coarse_view_idx)["logits"]
        base_prob = softmax_prob(base_logits)
        base_logit = pe_logit(base_logits)
        for region, mask in masks.items():
            occluded = apply_spatial_occlusion(video, mask.to(device), fill_value=args.fill_value)
            occ_logits = model(occluded, coarse_view_idx=coarse_view_idx)["logits"]
            occ_prob = softmax_prob(occ_logits)
            occ_logit = pe_logit(occ_logits)
            rows.append(
                {
                    "case_id": mask_case_id,
                    "attribution_case_id": attr_case_id,
                    "split": str(payload["split"]),
                    "path": str(record["path"]),
                    "label": int(record["label"]),
                    "label_name": str(record["label_name"]),
                    "raw_view": str(record["raw_view"]),
                    "coarse_view": str(record["coarse_view"]),
                    "mask_source": str(mask_payload.get("source", "unknown")),
                    "generated_mask": int(generated_mask),
                    "region": str(region),
                    "base_prob": base_prob,
                    "occluded_prob": occ_prob,
                    "delta_prob": float(base_prob - occ_prob),
                    "base_logit": base_logit,
                    "occluded_logit": occ_logit,
                    "delta_logit": float(base_logit - occ_logit),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "n_rows": int(len(df)),
        "n_cases": int(df["case_id"].nunique()) if not df.empty else 0,
        "output_csv": str(output_csv),
        "skipped": skipped,
        "metric_summary": summarize_metric_rows(rows),
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
