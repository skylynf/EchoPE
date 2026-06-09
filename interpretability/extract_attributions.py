#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import types
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video.mvit import _add_rel_pos, _add_shortcut

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
FULL_RUN_ROOT = EXP_ROOT / "classification"
LORA_RUN_ROOT = EXP_ROOT / "finetuning"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(FULL_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(FULL_RUN_ROOT))
if str(LORA_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(LORA_RUN_ROOT))

from classification.data import COARSE_VIEW_TO_IDX  # noqa: E402
from finetuning.train_lora import MEAN, STD, preprocess_single_video  # noqa: E402

from load_interpret_model import (  # noqa: E402
    default_interpret_output_dir,
    load_interpret_artifacts,
    model_summary,
    resolve_cli_path,
)


def softmax_prob(logits: torch.Tensor) -> float:
    return float(torch.softmax(logits.float(), dim=1)[0, 1].detach().cpu())


def pe_logit_value(logits: torch.Tensor) -> float:
    return float(logits.float()[0, 1].detach().cpu())


def positive_normalized(tensor: torch.Tensor) -> torch.Tensor:
    return normalize_map(tensor.float().clamp(min=0))


def rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_np = left.detach().float().cpu().numpy()
    right_np = right.detach().float().cpu().numpy()
    if left_np.size < 2 or float(np.std(left_np)) == 0.0 or float(np.std(right_np)) == 0.0:
        return float("nan")
    left_rank = np.argsort(np.argsort(left_np)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right_np)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def resolve_occlusion_spans(args: argparse.Namespace) -> list[int]:
    spans = args.occlusion_spans if args.occlusion_spans else [args.occlusion_span]
    resolved: list[int] = []
    for span in spans:
        span = max(1, int(span))
        if span not in resolved:
            resolved.append(span)
    return resolved


def build_coarse_view_idx(record: dict[str, object], device: torch.device) -> torch.Tensor:
    coarse_view = str(record["coarse_view"])
    return torch.tensor([COARSE_VIEW_TO_IDX[coarse_view]], dtype=torch.long, device=device)


def safe_case_id(record: dict[str, object], split: str) -> str:
    digest = hashlib.md5(str(record["path"]).encode("utf-8")).hexdigest()[:8]
    stem = str(record["id"]).replace(" ", "_")
    return f"{split}_{stem}_{digest}"


def normalize_map(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor.float()
    tensor = tensor - tensor.min()
    denom = tensor.max().clamp(min=eps)
    return tensor / denom


def normalized_entropy(vector: torch.Tensor, eps: float = 1e-8) -> float:
    probs = vector.float().clamp(min=0)
    total = probs.sum().item()
    if total <= eps:
        return 0.0
    probs = probs / probs.sum()
    entropy = -(probs * (probs + eps).log()).sum().item()
    return float(entropy / math.log(len(probs) + eps))


def topk_mass_ratio(tensor: torch.Tensor, ratio: float = 0.1) -> float:
    flat = tensor.flatten().float().clamp(min=0)
    if flat.numel() == 0:
        return 0.0
    k = max(1, int(round(flat.numel() * ratio)))
    total = flat.sum().item()
    if total <= 0:
        return 0.0
    values, _ = torch.topk(flat, k=k)
    return float(values.sum().item() / total)


def mean_positive_delta(tensor: torch.Tensor) -> float:
    positive = tensor.float().clamp(min=0)
    if positive.numel() == 0:
        return 0.0
    return float(positive.mean().item())


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(video.device, dtype=video.dtype)
    std = STD.to(video.device, dtype=video.dtype)
    return video * std + mean


def foreground_mass_ratio(video: torch.Tensor, score_map: torch.Tensor) -> float:
    denorm = denormalize_video(video).float().clamp(min=0)
    mask = denorm.mean(dim=0) > 8.0
    total = score_map.float().sum().item()
    if total <= 0:
        return 0.0
    inside = score_map.float()[mask].sum().item()
    return float(inside / total)


def overlay_frame(frame: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    frame_np = frame.detach().cpu().numpy()
    frame_np = frame_np - frame_np.min()
    if frame_np.max() > 0:
        frame_np = frame_np / frame_np.max()
    heatmap_np = normalize_map(heatmap).detach().cpu().numpy()
    colored = plt.get_cmap("jet")(heatmap_np)[..., :3]
    overlay = (1 - alpha) * np.repeat(frame_np[..., None], 3, axis=-1) + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


def save_case_figure(
    output_path: Path,
    video: torch.Tensor,
    grad_map: torch.Tensor,
    attention_map: torch.Tensor,
    temporal_grad: torch.Tensor,
    temporal_attention: torch.Tensor,
    temporal_occlusion: torch.Tensor,
    temporal_occlusion_raw: torch.Tensor,
    top_frame_indices: list[int],
    title: str,
) -> None:
    frame_gray = denormalize_video(video).mean(dim=0)
    grad_idx = int(torch.argmax(temporal_grad).item())
    attn_idx = int(torch.argmax(temporal_attention).item())
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.reshape(3, 3)
    axes[0, 0].imshow(frame_gray[grad_idx].cpu(), cmap="gray")
    axes[0, 0].set_title(f"Grad peak frame {grad_idx}")
    axes[0, 1].imshow(overlay_frame(frame_gray[grad_idx], grad_map[grad_idx]))
    axes[0, 1].set_title("Grad x Input")
    axes[0, 2].imshow(overlay_frame(frame_gray[attn_idx], attention_map[attn_idx]))
    axes[0, 2].set_title(f"Attention ref frame {attn_idx}")
    avg_overlay = overlay_frame(frame_gray.mean(dim=0), grad_map.mean(dim=0))
    axes[1, 0].imshow(avg_overlay)
    axes[1, 0].set_title("Mean Grad Overlay")
    axes[1, 1].plot(temporal_grad.cpu().numpy(), label="grad")
    axes[1, 1].plot(temporal_attention.cpu().numpy(), label="attn ref")
    axes[1, 1].plot(temporal_occlusion.cpu().numpy(), label="occlusion")
    axes[1, 1].legend()
    axes[1, 1].set_title("Temporal Evidence (normalized)")
    axes[1, 2].plot(temporal_occlusion_raw.cpu().numpy(), label="delta prob")
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].legend()
    axes[1, 2].set_title("Occlusion Raw Delta")
    for axis_idx, frame_idx in enumerate(top_frame_indices[:2]):
        axes[2, axis_idx].imshow(overlay_frame(frame_gray[frame_idx], grad_map[frame_idx]))
        axes[2, axis_idx].set_title(f"Top occlusion frame {frame_idx}")
    for axis_idx in range(len(top_frame_indices[:2]), 2):
        axes[2, axis_idx].axis("off")
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.0,
        1.0,
        title,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def enable_attention_capture(encoder: torch.nn.Module) -> list[tuple[torch.nn.Module, object]]:
    patched: list[tuple[torch.nn.Module, object]] = []

    def forward_with_cache(self, x: torch.Tensor, thw: tuple[int, int, int]):  # type: ignore[no-untyped-def]
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(dim=2)

        if self.pool_k is not None:
            k, k_thw = self.pool_k(k, thw)
        else:
            k_thw = thw
        if self.pool_v is not None:
            v = self.pool_v(v, thw)[0]
        if self.pool_q is not None:
            q, thw = self.pool_q(q, thw)

        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        if self.rel_pos_h is not None and self.rel_pos_w is not None and self.rel_pos_t is not None:
            attn = _add_rel_pos(
                attn,
                q,
                thw,
                k_thw,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)
        self.last_attention = attn.detach()
        self.last_query_thw = tuple(int(v) for v in thw)
        self.last_key_thw = tuple(int(v) for v in k_thw)

        x = torch.matmul(attn, v)
        if self.residual_pool:
            _add_shortcut(x, q, self.residual_with_cls_embed)
        x = x.transpose(1, 2).reshape(B, -1, self.output_dim)
        x = self.project(x)
        return x, thw

    for block in encoder.blocks:
        attn = block.attn
        original_forward = attn.forward
        attn.forward = types.MethodType(forward_with_cache, attn)
        patched.append((attn, original_forward))
    return patched


def disable_attention_capture(patched: list[tuple[torch.nn.Module, object]]) -> None:
    for module, original_forward in patched:
        module.forward = original_forward  # type: ignore[assignment]


def collect_attention_map(
    encoder: torch.nn.Module,
    target_shape: tuple[int, int, int],
    block_start_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    maps: list[torch.Tensor] = []
    temporal_curves: list[torch.Tensor] = []
    n_blocks = len(encoder.blocks)
    start_idx = min(n_blocks - 1, max(0, int(n_blocks * block_start_ratio)))
    for block in encoder.blocks[start_idx:]:
        attn = getattr(block.attn, "last_attention", None)
        k_thw = getattr(block.attn, "last_key_thw", None)
        if attn is None or k_thw is None:
            continue
        cls_to_patch = attn.mean(dim=1)[0, 0, 1:]
        if cls_to_patch.numel() != int(np.prod(k_thw)):
            continue
        grid = cls_to_patch.reshape(1, 1, *k_thw)
        upsampled = F.interpolate(
            grid.float(),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        maps.append(normalize_map(upsampled))
        temporal_curves.append(normalize_map(upsampled.mean(dim=(1, 2))))
    if not maps:
        empty = torch.zeros(target_shape, dtype=torch.float32)
        return empty, torch.zeros(target_shape[0], dtype=torch.float32)
    stacked = torch.stack(maps)
    temporal = torch.stack(temporal_curves).mean(dim=0)
    return normalize_map(stacked.mean(dim=0)), normalize_map(temporal)


def compute_occlusion_evidence(
    artifacts,
    input_video: torch.Tensor,
    coarse_view_idx: torch.Tensor | None,
    base_prob: float,
    base_logit: float,
    occlusion_spans: list[int],
) -> dict[str, dict[str, object]]:
    temporal_evidence: dict[str, dict[str, object]] = {}
    n_frames = int(input_video.shape[2])
    for span in occlusion_spans:
        delta_prob = torch.zeros(n_frames, dtype=torch.float32)
        delta_logit = torch.zeros(n_frames, dtype=torch.float32)
        windows: list[dict[str, object]] = []
        for start in range(0, n_frames, span):
            stop = min(n_frames, start + span)
            occluded = input_video.detach().clone()
            occluded[:, :, start:stop] = 0.0
            occ_logits = artifacts.model(occluded, coarse_view_idx=coarse_view_idx)["logits"]
            occ_prob = softmax_prob(occ_logits)
            occ_logit = pe_logit_value(occ_logits)
            prob_drop = float(base_prob - occ_prob)
            logit_drop = float(base_logit - occ_logit)
            delta_prob[start:stop] = prob_drop
            delta_logit[start:stop] = logit_drop
            windows.append(
                {
                    "start": int(start),
                    "stop": int(stop),
                    "center_frame": int((start + stop - 1) // 2),
                    "delta_prob": prob_drop,
                    "delta_logit": logit_drop,
                    "occluded_prob": float(occ_prob),
                    "occluded_logit": float(occ_logit),
                }
            )
        temporal_evidence[str(span)] = {
            "span": int(span),
            "delta_prob": delta_prob,
            "delta_logit": delta_logit,
            "delta_prob_norm": positive_normalized(delta_prob),
            "delta_logit_norm": positive_normalized(delta_logit),
            "windows": windows,
        }
    return temporal_evidence


def top_frame_indices(curve: torch.Tensor, k: int = 2) -> list[int]:
    if curve.numel() == 0:
        return []
    values = curve.float().clamp(min=0)
    if float(values.max().item()) <= 0.0:
        values = curve.float().abs()
    k = min(k, int(values.numel()))
    return [int(idx) for idx in torch.topk(values, k=k).indices.tolist()]


def build_temporal_frame_rows(
    temporal_grad: torch.Tensor,
    temporal_attention: torch.Tensor,
    primary_occlusion: dict[str, object],
) -> list[dict[str, float | int]]:
    delta_prob = primary_occlusion["delta_prob"]
    delta_logit = primary_occlusion["delta_logit"]
    delta_prob_norm = primary_occlusion["delta_prob_norm"]
    delta_logit_norm = primary_occlusion["delta_logit_norm"]
    rows: list[dict[str, float | int]] = []
    for frame_idx in range(int(temporal_grad.numel())):
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "grad_sensitivity": float(temporal_grad[frame_idx].item()),
                "attn_mass": float(temporal_attention[frame_idx].item()),
                "occlusion_delta_prob": float(delta_prob[frame_idx].item()),
                "occlusion_delta_logit": float(delta_logit[frame_idx].item()),
                "occlusion_delta_prob_norm": float(delta_prob_norm[frame_idx].item()),
                "occlusion_delta_logit_norm": float(delta_logit_norm[frame_idx].item()),
            }
        )
    return rows


def temporal_evidence_to_json(
    temporal_evidence: dict[str, dict[str, object]],
    frame_rows: list[dict[str, float | int]],
) -> dict[str, object]:
    spans: dict[str, object] = {}
    for span_key, payload in temporal_evidence.items():
        spans[span_key] = {
            "span": int(payload["span"]),
            "delta_prob": [float(v) for v in payload["delta_prob"].tolist()],
            "delta_logit": [float(v) for v in payload["delta_logit"].tolist()],
            "delta_prob_norm": [float(v) for v in payload["delta_prob_norm"].tolist()],
            "delta_logit_norm": [float(v) for v in payload["delta_logit_norm"].tolist()],
            "windows": payload["windows"],
        }
    return {
        "frames": frame_rows,
        "occlusion_spans": spans,
    }


@torch.no_grad()
def run_embedding_pass(artifacts, splits: list[str]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    embeddings: list[torch.Tensor] = []
    for split in splits:
        loader = artifacts.loaders_by_split[split]
        records = artifacts.records_by_split[split]
        cursor = 0
        for batch in loader:
            videos, labels, views = batch[:3]
            coarse_view_idx = batch[3] if len(batch) >= 4 else None
            videos = videos.to(artifacts.device, non_blocking=True)
            coarse_view_idx = coarse_view_idx.to(artifacts.device, non_blocking=True) if coarse_view_idx is not None else None
            out = artifacts.model(videos, coarse_view_idx=coarse_view_idx)
            logits = out["logits"].float().cpu()
            emb = out["emb"].detach().cpu()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            batch_records = records[cursor : cursor + len(labels)]
            cursor += len(labels)
            for idx, record in enumerate(batch_records):
                rows.append(
                    {
                        "split": split,
                        "id": str(record["id"]),
                        "path": str(record["path"]),
                        "label": int(record["label"]),
                        "label_name": str(record["label_name"]),
                        "source_group": str(record["source_group"]),
                        "raw_view": str(record["raw_view"]),
                        "coarse_view": str(record["coarse_view"]),
                        "prob_pe": float(probs[idx].item()),
                        "pred_label": int(preds[idx].item()),
                        "correct": int(preds[idx].item() == int(record["label"])),
                    }
                )
            embeddings.append(emb)
    return {
        "rows": rows,
        "embeddings": torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 512),
    }


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_detailed_cases(rows: list[dict[str, object]], per_group: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        pred = int(row["pred_label"])
        if label == 1 and pred == 1:
            key = "tp"
        elif label == 0 and pred == 0:
            key = "tn"
        elif label == 0 and pred == 1:
            key = "fp"
        else:
            key = "fn"
        grouped[key].append(row)
    selected: list[dict[str, object]] = []
    for key, items in grouped.items():
        if key in {"tp", "fp"}:
            items = sorted(items, key=lambda row: float(row["prob_pe"]), reverse=True)
        elif key in {"tn", "fn"}:
            items = sorted(items, key=lambda row: float(row["prob_pe"]))
        selected.extend(items[:per_group])
    uncertain = sorted(rows, key=lambda row: abs(float(row["prob_pe"]) - 0.5))[:per_group]
    seen = {str((row["split"], row["path"])) for row in selected}
    for row in uncertain:
        marker = str((row["split"], row["path"]))
        if marker not in seen:
            selected.append(row)
            seen.add(marker)
    return selected


def compute_detailed_case(
    artifacts,
    record: dict[str, object],
    split: str,
    occlusion_spans: list[int],
    save_figures: bool,
    case_dir: Path,
) -> dict[str, object]:
    video = preprocess_single_video(str(record["path"])).to(artifacts.device)
    input_video = video.unsqueeze(0).detach()
    input_video.requires_grad_(True)
    artifacts.model.zero_grad(set_to_none=True)
    coarse_view_idx = None
    if getattr(artifacts.checkpoint_args, "head_view_mode", "none") == "coarse4":
        coarse_view_idx = build_coarse_view_idx(record, artifacts.device)
    out = artifacts.model(input_video, coarse_view_idx=coarse_view_idx)
    logits = out["logits"]
    emb = out["emb"][0].detach().cpu()
    pe_logit = logits[:, 1].sum()
    pe_logit.backward()
    grad = input_video.grad.detach()[0]
    grad_map = normalize_map((grad * input_video.detach()[0]).abs().mean(dim=0).cpu())
    temporal_grad = normalize_map(grad_map.sum(dim=(1, 2)))
    base_prob = softmax_prob(logits)
    base_logit = pe_logit_value(logits)

    with torch.no_grad():
        attention_map, temporal_attention = collect_attention_map(
            artifacts.model.encoder,
            target_shape=(video.shape[1], video.shape[2], video.shape[3]),
        )

    with torch.no_grad():
        temporal_evidence = compute_occlusion_evidence(
            artifacts,
            input_video=input_video,
            coarse_view_idx=coarse_view_idx,
            base_prob=base_prob,
            base_logit=base_logit,
            occlusion_spans=occlusion_spans,
        )
    primary_span = str(occlusion_spans[0])
    primary_occlusion = temporal_evidence[primary_span]
    temporal_occlusion_raw = primary_occlusion["delta_prob"]
    temporal_occlusion_logit_raw = primary_occlusion["delta_logit"]
    temporal_occlusion = primary_occlusion["delta_prob_norm"]
    frame_rows = build_temporal_frame_rows(temporal_grad, temporal_attention, primary_occlusion)
    temporal_json = temporal_evidence_to_json(temporal_evidence, frame_rows)

    attention_spatial = normalize_map(attention_map.mean(dim=0))
    grad_spatial = normalize_map(grad_map.mean(dim=0))
    metrics = {
        "prob_pe": float(base_prob),
        "pe_logit": float(base_logit),
        "occlusion_primary_span": int(occlusion_spans[0]),
        "spatial_entropy_grad": normalized_entropy(grad_spatial.flatten()),
        "spatial_entropy_attention": normalized_entropy(attention_spatial.flatten()),
        "temporal_entropy_grad": normalized_entropy(temporal_grad),
        "temporal_entropy_attention": normalized_entropy(temporal_attention),
        "temporal_entropy_occlusion": normalized_entropy(temporal_occlusion),
        "temporal_occlusion_aopc_prob": mean_positive_delta(temporal_occlusion_raw),
        "temporal_occlusion_aopc_logit": mean_positive_delta(temporal_occlusion_logit_raw),
        "temporal_occlusion_max_delta_prob": float(temporal_occlusion_raw.max().item()),
        "temporal_occlusion_min_delta_prob": float(temporal_occlusion_raw.min().item()),
        "temporal_occlusion_top25_mass": topk_mass_ratio(temporal_occlusion, ratio=0.25),
        "temporal_corr_grad_occlusion": rank_correlation(temporal_grad, temporal_occlusion),
        "temporal_corr_attention_occlusion": rank_correlation(temporal_attention, temporal_occlusion),
        "attention_top10_mass": topk_mass_ratio(attention_map, ratio=0.10),
        "grad_top10_mass": topk_mass_ratio(grad_map, ratio=0.10),
        "foreground_mass_grad": foreground_mass_ratio(video.cpu(), grad_map),
        "foreground_mass_attention": foreground_mass_ratio(video.cpu(), attention_map.cpu()),
    }

    case_id = safe_case_id(record, split)
    payload = {
        "case_id": case_id,
        "split": split,
        "record": record,
        "metrics": metrics,
        "embedding": emb,
        "grad_map": grad_map,
        "attention_map": attention_map.cpu(),
        "temporal_grad": temporal_grad,
        "temporal_attention": temporal_attention.cpu(),
        "temporal_occlusion": temporal_occlusion,
        "temporal_occlusion_raw_prob": temporal_occlusion_raw,
        "temporal_occlusion_raw_logit": temporal_occlusion_logit_raw,
        "temporal_evidence": temporal_evidence,
    }
    case_path = case_dir / f"{case_id}.pt"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, case_path)

    if save_figures:
        fig_title = (
            f"{case_id}\n"
            f"label={record['label_name']} pred={int(base_prob >= 0.5)} prob={base_prob:.3f}\n"
            f"raw_view={record['raw_view']} coarse_view={record['coarse_view']}"
        )
        save_case_figure(
            case_dir.parent / "figures" / f"{case_id}.png",
            video.cpu(),
            grad_map,
            attention_map.cpu(),
            temporal_grad,
            temporal_attention.cpu(),
            temporal_occlusion,
            temporal_occlusion_raw,
            top_frame_indices(temporal_occlusion_raw, k=2),
            fig_title,
        )
    return {"case_id": case_id, "path": str(case_path), "metrics": metrics, "temporal_evidence": temporal_json}


def run_extraction(args: argparse.Namespace) -> dict[str, object]:
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    occlusion_spans = resolve_occlusion_spans(args)
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else default_interpret_output_dir(args.checkpoint)
    artifacts = load_interpret_artifacts(
        args.checkpoint,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest_path,
        splits=splits,
        device=args.device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = output_dir / "embedding_analysis"
    attribution_dir = output_dir / "attributions"
    temporal_dir = output_dir / "temporal_curves"
    figures_dir = output_dir / "figures"
    for path in (embedding_dir, attribution_dir, temporal_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    patched = enable_attention_capture(artifacts.model.encoder)
    try:
        embedding_pass = run_embedding_pass(artifacts, splits=splits)
        rows = embedding_pass["rows"]
        embeddings = embedding_pass["embeddings"]
        write_rows_csv(embedding_dir / "sample_predictions.csv", rows)
        torch.save(
            {
                "rows": rows,
                "embeddings": embeddings,
            },
            embedding_dir / "embeddings.pt",
        )

        selected = select_detailed_cases(rows, per_group=args.detailed_samples_per_group)
        if args.max_detailed_cases > 0:
            selected = selected[: args.max_detailed_cases]

        case_summaries = []
        for row in selected:
            case_result = compute_detailed_case(
                artifacts,
                record=row,
                split=str(row["split"]),
                occlusion_spans=occlusion_spans,
                save_figures=args.save_figures,
                case_dir=attribution_dir,
            )
            case_summaries.append(case_result)
            temporal_payload = {
                "case_id": case_result["case_id"],
                "metrics": case_result["metrics"],
                "temporal_evidence": case_result["temporal_evidence"],
            }
            (temporal_dir / f"{case_result['case_id']}.json").write_text(
                json.dumps(temporal_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    finally:
        disable_attention_capture(patched)

    summary = {
        "checkpoint_path": str(artifacts.checkpoint_path),
        "manifest_path": str(artifacts.manifest_path),
        "task": str(artifacts.checkpoint_payload.get("task", args.task or "pooled")),
        "seed": int(artifacts.checkpoint_payload.get("seed", getattr(artifacts.checkpoint_args, "seed", 2024))),
        "splits": splits,
        "occlusion_spans": occlusion_spans,
        "model_summary": model_summary(artifacts.model),
        "n_rows": len(rows),
        "n_detailed_cases": len(case_summaries),
        "artifacts": {
            "predictions_csv": str((embedding_dir / "sample_predictions.csv").resolve()),
            "embeddings_pt": str((embedding_dir / "embeddings.pt").resolve()),
            "attributions_dir": str(attribution_dir.resolve()),
            "temporal_dir": str(temporal_dir.resolve()),
            "figures_dir": str(figures_dir.resolve()),
        },
    }
    (output_dir / "extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract embedding, attention, gradient, and temporal interpretability artifacts for a full-run EchoPrime checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to classification best_checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, default=None, help="Interpretability output directory. Defaults to interpretability/outputs/<seed>/<task>.")
    parser.add_argument("--task", choices=["pooled", "A4", "PSL", "PSS", "Subcostal"], default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detailed-samples-per-group", type=int, default=6)
    parser.add_argument("--max-detailed-cases", type=int, default=40)
    parser.add_argument("--occlusion-span", type=int, default=1)
    parser.add_argument("--occlusion-spans", type=int, nargs="*", default=None)
    parser.add_argument("--save-figures", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_extraction(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
